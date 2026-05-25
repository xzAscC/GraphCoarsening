"""Multi-backbone experiment (Priority 10).

Tests explanation quality across GCN, GraphSAGE, and GAT backbones
to address reviewer concern about inconsistent backbone settings.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExperimentConfig
from experiments.train_gcn import load_dataset, MLPLinkPredictor, train_epoch, evaluate
from src.models.gcn import GCN, GraphSAGEModel
from torch_geometric.nn import SAGEConv
from src.models.link_predictor import LinkPredictionModel

try:
    from torch_geometric.nn import GATConv
    HAS_GAT = True
except ImportError:
    HAS_GAT = False

try:
    from src.evaluation.fidelity import fidelity_plus, fidelity_minus
except ImportError:
    fidelity_plus = None
    fidelity_minus = None

try:
    from src.explainers.coarsen_explainer import CoarsenExplainer
except ImportError:
    CoarsenExplainer = None

try:
    from src.explainers.baselines import OcclusionExplainer, SaliencyExplainer
except ImportError:
    OcclusionExplainer = None
    SaliencyExplainer = None

BACKBONES = ["GCN", "GraphSAGE"] + (["GAT"] if HAS_GAT else [])
METHODS = ["Occlusion", "Saliency", "Ours"]


class GraphSAGEBNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def get_encoder(backbone_name, in_channels, hidden_channels, out_channels, num_layers, dropout):
    if backbone_name == "GCN":
        return GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
    elif backbone_name == "GraphSAGE":
        return GraphSAGEBNModel(in_channels, hidden_channels, out_channels, num_layers, dropout)
    elif backbone_name == "GAT" and HAS_GAT:
        return GATModel(in_channels, hidden_channels, out_channels, num_layers, dropout)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


def train_backbone(backbone_name, data, device, epochs=100, lr=0.01, hidden=128, layers=3, seed=42):
    """Train a backbone model and return the LinkPredictionModel."""
    torch.manual_seed(seed)
    in_channels = data.num_features

    if backbone_name == "GraphSAGE":
        lr = 0.005
        dropout = 0.3
        epochs = max(epochs, 100)
        layers = 2
    elif backbone_name == "GAT":
        lr = 0.005
        hidden = max(hidden, 256)
        dropout = 0.3
        epochs = max(epochs, 100)
        layers = 2
    else:
        dropout = 0.5

    out_channels = hidden
    encoder = get_encoder(backbone_name, in_channels, hidden, out_channels, layers, dropout).to(device)
    predictor = MLPLinkPredictor(out_channels, hidden).to(device)
    model = LinkPredictionModel(encoder, predictor).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for epoch in range(1, epochs + 1):
        loss = train_epoch(encoder, predictor, data, optimizer, device)
        if epoch % 20 == 0:
            val_auc = evaluate(encoder, predictor, data, device, "val")
            print(f"    Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

    test_auc = evaluate(encoder, predictor, data, device, "test")
    return model, test_auc


def load_gcn_checkpoint(dataset, data, device):
    """Load a pre-trained GCN model from checkpoint.

    Uses the same checkpoint that the baselines comparison experiment uses,
    so that GCN fidelity results are directly comparable.
    """
    ckpt_path = os.path.join("checkpoints", f"{dataset}_gcn.pt")
    print(f"  Loading GCN from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']

    encoder = GCN(cfg['in_channels'], cfg['hidden_channels'], cfg['out_channels'], cfg['num_layers'])
    predictor = MLPLinkPredictor(cfg['out_channels'])
    encoder.load_state_dict(ckpt['model_state_dict'])
    predictor.load_state_dict(ckpt['predictor_state_dict'])

    encoder = encoder.to(device)
    predictor = predictor.to(device)
    model = LinkPredictionModel(encoder, predictor).to(device)

    test_auc = evaluate(encoder, predictor, data, device, "test")
    return model, test_auc


def run_method(method_name, model, data, test_edges, device):
    """Run a single explanation method and return results."""
    if method_name == "Occlusion" and OcclusionExplainer is not None:
        explainer = OcclusionExplainer(model, device=device)
    elif method_name == "Saliency" and SaliencyExplainer is not None:
        explainer = SaliencyExplainer(model, device=device)
    elif method_name == "Ours" and CoarsenExplainer is not None:
        explainer = CoarsenExplainer(model, device=device)
    else:
        return None

    fid_p_list = []
    fid_m_list = []
    times = []

    for i in range(test_edges.size(1)):
        a = int(test_edges[0, i].item())
        b = int(test_edges[1, i].item())

        t0 = time.time()
        try:
            explanation = explainer.explain_link(data, a, b)
        except Exception:
            continue
        times.append(time.time() - t0)

        if fidelity_plus is not None:
            fp = fidelity_plus(model, data, explanation, a, b, device)
            fm = fidelity_minus(model, data, explanation, a, b, device)
            fid_p_list.append(fp)
            fid_m_list.append(fm)

    if not fid_p_list:
        return None

    return {
        "mean_fidelity_plus": float(np.mean(fid_p_list)),
        "std_fidelity_plus": float(np.std(fid_p_list)),
        "mean_fidelity_minus": float(np.mean(fid_m_list)),
        "std_fidelity_minus": float(np.std(fid_m_list)),
        "mean_time": float(np.mean(times)),
        "num_samples": len(fid_p_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--backbones", type=str, default=",".join(BACKBONES))
    parser.add_argument("--methods", type=str, default=",".join(METHODS))
    parser.add_argument("--num_edges", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    if args.device:
        cfg.device = args.device
    device = torch.device(cfg.device)
    torch.manual_seed(args.seed)

    backbones = [b.strip() for b in args.backbones.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]

    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    if data.edge_index is None:
        data.edge_index = data.train_pos_edge_index
    print(f"  Nodes: {data.num_nodes}, Edges: {data.train_pos_edge_index.size(1)}")

    if hasattr(data, "test_pos_edge_index") and data.test_pos_edge_index is not None:
        pos = data.test_pos_edge_index
    else:
        pos = data.train_pos_edge_index
    rng = np.random.RandomState(args.seed)
    n = min(args.num_edges, pos.size(1))
    indices = rng.choice(pos.size(1), size=n, replace=False)
    test_edges = pos[:, indices]

    results = {}
    for backbone in backbones:
        print(f"\n{'='*60}")
        action = "Loading checkpoint for" if backbone == "GCN" else "Training"
        print(f"{action} {backbone} on {args.dataset}")
        print(f"{'='*60}")

        if backbone == "GCN":
            model, test_auc = load_gcn_checkpoint(args.dataset, data, device)
        else:
            model, test_auc = train_backbone(backbone, data, device, seed=args.seed)
        print(f"  Test AUC: {test_auc:.4f}")

        backbone_results = {"test_auc": test_auc}

        for method in methods:
            print(f"  Running {method}...")
            result = run_method(method, model, data, test_edges, device)
            if result:
                backbone_results[method] = result
                print(f"    Fid+: {result['mean_fidelity_plus']:.4f} ± {result['std_fidelity_plus']:.4f}")
                print(f"    Fid-: {result['mean_fidelity_minus']:.4f} ± {result['std_fidelity_minus']:.4f}")
                print(f"    Time: {result['mean_time']:.4f}s")

        results[backbone] = backbone_results

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"multibackbone_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    try:
        _plot_multibackbone(args.dataset, results, methods)
    except Exception as e:
        print(f"Plotting skipped: {e}")


def _plot_multibackbone(dataset, results, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    backbones = list(results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(backbones))
    width = 0.8 / max(len(methods), 1)

    for mi, method in enumerate(methods):
        fid_p = [results[b].get(method, {}).get("mean_fidelity_plus", 0) for b in backbones]
        fid_p_err = [results[b].get(method, {}).get("std_fidelity_plus", 0) for b in backbones]
        ax1.bar(x + mi * width, fid_p, width, yerr=fid_p_err, label=method, capsize=3)

    ax1.set_xticks(x + width * len(methods) / 2)
    ax1.set_xticklabels(backbones)
    ax1.set_ylabel("Necessity (Fid+)")
    ax1.set_title(f"Necessity by Backbone ({dataset})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for mi, method in enumerate(methods):
        fid_m = [results[b].get(method, {}).get("mean_fidelity_minus", 0) for b in backbones]
        fid_m_err = [results[b].get(method, {}).get("std_fidelity_minus", 0) for b in backbones]
        ax2.bar(x + mi * width, fid_m, width, yerr=fid_m_err, label=method, capsize=3)

    ax2.set_xticks(x + width * len(methods) / 2)
    ax2.set_xticklabels(backbones)
    ax2.set_ylabel("Sufficiency (Fid-)")
    ax2.set_title(f"Sufficiency by Backbone ({dataset})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig_path = os.path.join("figures", f"multibackbone_{dataset}.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
