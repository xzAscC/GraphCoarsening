"""Link prediction heads and training utilities.

Provides dot-product and MLP-based link predictors, a wrapper model combining
a GNN encoder with a predictor head, and a training/evaluation loop for
standard link prediction tasks.
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score


class LinkPredictor(torch.nn.Module):
    """Dot-product link predictor: score(u,v) = z_u^T z_v."""

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Score edges via dot product of endpoint embeddings.

        Args:
            z: (N, d) node embeddings.
            edge_index: (2, E) edges to score.

        Returns:
            (E,) prediction logits.
        """
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)


class MLPLinkPredictor(torch.nn.Module):
    """2-layer MLP link predictor operating on concatenated embeddings."""

    def __init__(self, hidden_channels: int, out_channels: int = 1):
        super().__init__()
        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Score edges by concatenating endpoint embeddings and passing through MLP.

        Args:
            z: (N, d) node embeddings (d must equal hidden_channels).
            edge_index: (2, E) edges to score.

        Returns:
            (E,) prediction logits.
        """
        src, dst = edge_index
        h = torch.cat([z[src], z[dst]], dim=-1)
        h = F.relu(self.lin1(h))
        h = self.lin2(h)
        return h.squeeze(-1)


class LinkPredictionModel(torch.nn.Module):
    """Combined GNN encoder + link predictor."""

    def __init__(self, encoder: torch.nn.Module, predictor: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode nodes with GNN, then score target edges.

        Args:
            x: (N, in_channels) node features.
            edge_index: (2, E) message-passing edges.
            target_edge_index: (2, T) edges to score.
            edge_weight: (E,) optional edge weights for coarsened graphs.

        Returns:
            (T,) prediction logits for target edges.
        """
        z = self.encoder(x, edge_index, edge_weight=edge_weight)
        return self.predictor(z, target_edge_index)


def train_link_prediction(
    model: torch.nn.Module,
    data: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    num_epochs: int = 100,
    neg_ratio: float = 1.0,
    patience: int = 10,
    device: str = "cpu",
) -> dict:
    """Train a LinkPredictionModel for link prediction.

    Uses binary cross-entropy with negative sampling at the given ratio.
    Supports early stopping based on training loss plateau.

    Args:
        model: LinkPredictionModel instance.
        data: PyG Data object with `x`, `edge_index`, and
            `train_pos_edge_index` attributes.
        optimizer: Optimizer. Defaults to Adam(lr=0.01, weight_decay=5e-4).
        num_epochs: Maximum training epochs.
        neg_ratio: Ratio of negative to positive edges per batch.
        patience: Early stopping patience (epochs without improvement).
        device: 'cpu' or 'cuda'.

    Returns:
        Dict with 'losses' (list of per-epoch losses), 'model' (state_dict),
        and 'best_epoch'.
    """
    model = model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.01, weight_decay=5e-4
        )

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    train_pos = data.train_pos_edge_index.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    losses = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()

        num_neg = int(train_pos.size(1) * neg_ratio)
        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=x.size(0),
            num_neg_samples=num_neg,
        )

        pos_score = model(x, edge_index, train_pos)
        neg_score = model(x, edge_index, neg_edge_index)

        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([
            torch.ones(pos_score.size(0), device=device),
            torch.zeros(neg_score.size(0), device=device),
        ])

        loss = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss = loss.item()
        losses.append(epoch_loss)

        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "losses": losses,
        "model": model.state_dict(),
        "best_epoch": best_epoch,
    }


def evaluate_link_prediction(
    model: torch.nn.Module,
    data: torch.Tensor,
    eval_edges: tuple[torch.Tensor, torch.Tensor],
    device: str = "cpu",
) -> float:
    """Evaluate link prediction using AUC-ROC.

    Args:
        model: Trained LinkPredictionModel.
        data: PyG Data object with `x` and `edge_index`.
        eval_edges: Tuple of (pos_edge_index, neg_edge_index), each (2, E).
        device: 'cpu' or 'cuda'.

    Returns:
        AUC-ROC score (float).
    """
    model = model.to(device)
    model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    pos_edge_index = eval_edges[0].to(device)
    neg_edge_index = eval_edges[1].to(device)

    with torch.no_grad():
        pos_score = model(x, edge_index, pos_edge_index)
        neg_score = model(x, edge_index, neg_edge_index)

    scores = torch.cat([pos_score, neg_score]).sigmoid().cpu().numpy()
    labels = torch.cat([
        torch.ones(pos_score.size(0)),
        torch.zeros(neg_score.size(0)),
    ]).numpy()

    return roc_auc_score(labels, scores)
