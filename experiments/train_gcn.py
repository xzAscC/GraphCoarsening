"""Train GCN backbone models for link prediction on various graph datasets."""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExperimentConfig

from src.models.gcn import GCN

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.utils import negative_sampling, train_test_split_edges

try:
    from ogb.linkproppred import LinkPropPredDataset
except ImportError:
    LinkPropPredDataset = None


PLANETOID_DATASETS = {"Cora", "Citeseer", "PubMed"}
COAUTHOR_DATASETS = {"Coauthor-CS", "Coauthor-Physics"}
AMAZON_DATASETS = {"Amazon-Computers", "Amazon-Photo"}
OGB_LINK_DATASETS = {"ogbl-ppa", "ogbl-collab", "ogbl-ddi"}


def load_dataset(name: str, root: str = "data"):
    """Load a graph dataset and return a PyG Data object with edge splits.

    For citation networks, co-authorship, and Amazon datasets the helper
    applies ``train_test_split_edges``.  For OGB link-prediction datasets
    it converts the OGB container to a PyG Data object preserving the
    built-in train/val/test splits.
    """
    os.makedirs(root, exist_ok=True)

    if name in PLANETOID_DATASETS:
        dataset = Planetoid(
            root=os.path.join(root, name),
            name=name,
            transform=T.NormalizeFeatures(),
        )
        data = dataset[0]
        data = train_test_split_edges(data)
        data.num_features = dataset.num_features
        return data

    if name in COAUTHOR_DATASETS:
        dataset_name = name.split("-")[1]
        dataset = Coauthor(
            root=os.path.join(root, "Coauthor"),
            name=dataset_name,
            transform=T.NormalizeFeatures(),
        )
        data = dataset[0]
        data = train_test_split_edges(data)
        data.num_features = dataset.num_features
        return data

    if name in AMAZON_DATASETS:
        dataset_name = name.split("-")[1]
        dataset = Amazon(
            root=os.path.join(root, "Amazon"),
            name=dataset_name,
            transform=T.NormalizeFeatures(),
        )
        data = dataset[0]
        data = train_test_split_edges(data)
        data.num_features = dataset.num_features
        return data

    if name in OGB_LINK_DATASETS:
        if LinkPropPredDataset is None:
            raise ImportError("ogb package required for OGB datasets")
        dataset = LinkPropPredDataset(name=name, root=os.path.join(root, "ogb"))
        split_edge = dataset.get_edge_split()
        edge_index = torch.tensor(split_edge["train"]["edge"], dtype=torch.long).t().contiguous()
        num_nodes = edge_index.max().item() + 1

        from torch_geometric.data import Data

        data = Data(edge_index=edge_index, num_nodes=num_nodes)
        data.num_features = dataset.num_features if hasattr(dataset, "num_features") else 1

        if hasattr(dataset, "graph") and isinstance(dataset.graph, dict) and "node_feat" in dataset.graph:
            feat = dataset.graph["node_feat"]
            if feat is not None:
                data.x = torch.tensor(feat, dtype=torch.float)
                data.num_features = data.x.size(1)

        if data.x is None and (data.num_features is None or data.num_features == 0):
            data.x = torch.eye(num_nodes)
            data.num_features = num_nodes

        has_val = "valid" in split_edge and "edge" in split_edge["valid"]
        has_test = "test" in split_edge and "edge" in split_edge["test"]

        data.val_pos_edge_index = (
            torch.tensor(split_edge["valid"]["edge"], dtype=torch.long).t().contiguous()
            if has_val
            else torch.empty(2, 0, dtype=torch.long)
        )
        data.val_neg_edge_index = (
            torch.tensor(split_edge["valid"]["edge_neg"], dtype=torch.long).t().contiguous()
            if "valid" in split_edge and "edge_neg" in split_edge["valid"]
            else torch.empty(2, 0, dtype=torch.long)
        )
        data.test_pos_edge_index = (
            torch.tensor(split_edge["test"]["edge"], dtype=torch.long).t().contiguous()
            if has_test
            else torch.empty(2, 0, dtype=torch.long)
        )
        data.test_neg_edge_index = (
            torch.tensor(split_edge["test"]["edge_neg"], dtype=torch.long).t().contiguous()
            if "test" in split_edge and "edge_neg" in split_edge["test"]
            else torch.empty(2, 0, dtype=torch.long)
        )
        data.train_pos_edge_index = edge_index
        return data

    raise ValueError(f"Unknown dataset: {name}")


class DotProductLinkPredictor(torch.nn.Module):
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


class MLPLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = z[edge_index[0]] * z[edge_index[1]]
        h = F.relu(self.fc1(h))
        return self.fc2(h).squeeze(-1)


def train_epoch(
    model: GCN,
    predictor: torch.nn.Module,
    data: torch_geometric.data.Data,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    neg_ratio: float = 1.0,
):
    model.train()
    predictor.train()

    pos_edge = data.train_pos_edge_index.to(device)
    num_neg = int(pos_edge.size(1) * neg_ratio)
    neg_edge = negative_sampling(
        edge_index=pos_edge,
        num_nodes=data.num_nodes,
        num_neg_samples=num_neg,
    ).to(device)

    x = (data.x.to(device) if data.x is not None else torch.eye(data.num_nodes, device=device))
    optimizer.zero_grad()
    z = model(x, pos_edge)

    pos_score = predictor(z, pos_edge)
    neg_score = predictor(z, neg_edge)

    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([
        torch.ones(pos_score.size(0), device=device),
        torch.zeros(neg_score.size(0), device=device),
    ])
    loss = F.binary_cross_entropy_with_logits(scores, labels)

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(
    model: GCN,
    predictor: torch.nn.Module,
    data: torch_geometric.data.Data,
    device: torch.device,
    split: str = "val",
):
    from sklearn.metrics import roc_auc_score

    model.eval()
    predictor.eval()

    x = (data.x.to(device) if data.x is not None else torch.eye(data.num_nodes, device=device))
    z = model(x, data.train_pos_edge_index.to(device))

    pos_key = f"{split}_pos_edge_index"
    neg_key = f"{split}_neg_edge_index"
    pos_edge = getattr(data, pos_key, None)
    neg_edge = getattr(data, neg_key, None)
    if pos_edge is None or neg_edge is None:
        return 0.0

    pos_edge = pos_edge.to(device)
    neg_edge = neg_edge.to(device)

    pos_score = torch.sigmoid(predictor(z, pos_edge)).cpu().numpy()
    neg_score = torch.sigmoid(predictor(z, neg_edge)).cpu().numpy()

    scores = np.concatenate([pos_score, neg_score])
    labels = np.concatenate([
        np.ones_like(pos_score),
        np.zeros_like(neg_score),
    ])
    return roc_auc_score(labels, scores)


def main():
    parser = argparse.ArgumentParser(description="Train GCN for link prediction")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--hidden", type=int, default=None, help="Override hidden channels")
    parser.add_argument("--layers", type=int, default=None, help="Override number of layers")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    cfg = ExperimentConfig()

    if not torch.cuda.is_available():
        cfg.device = "cpu"
    if args.device is not None:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.model.epochs = args.epochs
    if args.lr is not None:
        cfg.model.lr = args.lr
    if args.hidden is not None:
        cfg.model.hidden_channels = args.hidden
    if args.layers is not None:
        cfg.model.num_layers = args.layers
    if args.seed is not None:
        cfg.seed = args.seed

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg.seed)

    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    print(f"  Nodes: {data.num_nodes}, Edges: {data.train_pos_edge_index.size(1)}, Features: {data.num_features}")

    in_channels = data.num_features
    out_channels = cfg.model.hidden_channels

    model = GCN(
        in_channels=in_channels,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=out_channels,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    ).to(device)
    predictor = MLPLinkPredictor(out_channels, cfg.model.hidden_channels).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
    )

    print(f"Training on {device} for {cfg.model.epochs} epochs...")
    for epoch in range(1, cfg.model.epochs + 1):
        loss = train_epoch(model, predictor, data, optimizer, device, cfg.model.neg_ratio)
        if epoch % 10 == 0 or epoch == 1:
            val_auc = evaluate(model, predictor, data, device, "val")
            print(f"  Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{args.dataset}_gcn.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "config": {
            "in_channels": in_channels,
            "hidden_channels": cfg.model.hidden_channels,
            "out_channels": out_channels,
            "num_layers": cfg.model.num_layers,
            "dataset": args.dataset,
        },
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    test_auc = evaluate(model, predictor, data, device, "test")
    print(f"Test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()
