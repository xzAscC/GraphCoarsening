"""Fidelity evaluation metrics for GNN link-prediction explanations."""

from typing import Dict, List

import torch
from torch_geometric.data import Data


def _predict_binary(model, x, edge_index, node_a, node_b, edge_weight=None, device="cpu"):
    target = torch.tensor([[node_a], [node_b]], device=device)
    with torch.no_grad():
        score = model(x, edge_index, target, edge_weight=edge_weight).squeeze()
    binary = int((score > 0.5).item())
    return binary, score.item()


def _is_coarse_explanation(explanation: Data) -> bool:
    """Check if explanation is a coarse graph (has aggregated features, not original)."""
    return getattr(explanation, "is_coarse_graph", False)


def fidelity_plus(
    model: torch.nn.Module,
    data: Data,
    explanation: Data,
    node_a: int,
    node_b: int,
    device: str = "cpu",
) -> float:
    """Identity-drop fidelity (fidelity+) — tests NECESSITY.

    Remove the explanation subgraph from the original graph. If the
    prediction changes, the explanation captures something necessary.

    Returns 1.0 if prediction changed (good), 0.0 if unchanged.
    For coarse-graph explanations, uses prediction-score distance as proxy.
    """
    model = model.to(device)
    model.eval()
    data = data.to(device)

    original_pred, original_score = _predict_binary(
        model, data.x, data.edge_index, node_a, node_b,
        edge_weight=getattr(data, "edge_weight", None), device=device,
    )

    if _is_coarse_explanation(explanation):
        return _fidelity_plus_coarse(model, data, explanation, node_a, node_b, original_score, device)

    exp_edge_index = _to_global_edges(data, explanation)

    mask = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=device)
    for i in range(exp_edge_index.size(1)):
        src, dst = exp_edge_index[0, i], exp_edge_index[1, i]
        matches = (
            ((data.edge_index[0] == src) & (data.edge_index[1] == dst))
            | ((data.edge_index[0] == dst) & (data.edge_index[1] == src))
        )
        mask[matches] = False

    modified_edge_index = data.edge_index[:, mask]
    modified_weight = None
    if hasattr(data, "edge_weight") and data.edge_weight is not None:
        modified_weight = data.edge_weight[mask]

    new_pred, _ = _predict_binary(
        model, data.x, modified_edge_index, node_a, node_b,
        edge_weight=modified_weight, device=device,
    )

    return 1.0 if new_pred != original_pred else 0.0


def _fidelity_plus_coarse(model, data, explanation, node_a, node_b, original_score, device):
    exp_x = explanation.x.to(device) if explanation.x is not None else data.x
    exp_edges = explanation.edge_index.to(device)
    exp_weight = getattr(explanation, "edge_weight", None)
    if exp_weight is not None:
        exp_weight = exp_weight.to(device)

    target_a = getattr(explanation, "target_a", 0)
    target_b = getattr(explanation, "target_b", 1)
    target = torch.tensor([[target_a], [target_b]], device=device)
    with torch.no_grad():
        coarse_score = model(exp_x, exp_edges, target, edge_weight=exp_weight).squeeze().item()

    original_pred = int(original_score > 0.5)
    coarse_pred = int(coarse_score > 0.5)
    return 1.0 if original_pred != coarse_pred else 0.0


def fidelity_minus(
    model: torch.nn.Module,
    data: Data,
    explanation: Data,
    node_a: int,
    node_b: int,
    device: str = "cpu",
) -> float:
    """Insufficiency fidelity (fidelity-) — tests SUFFICIENCY.

    Run the model on the explanation alone. If the prediction
    is preserved, the explanation is self-sufficient.

    Returns 0.0 if prediction preserved (good), 1.0 if lost (bad).
    For coarse-graph explanations, uses prediction-score distance as proxy.
    """
    model = model.to(device)
    model.eval()
    data = data.to(device)

    original_pred, original_score = _predict_binary(
        model, data.x, data.edge_index, node_a, node_b,
        edge_weight=getattr(data, "edge_weight", None), device=device,
    )

    if _is_coarse_explanation(explanation):
        return _fidelity_minus_coarse(model, data, explanation, original_score, device)

    exp_edge_index = _to_global_edges(data, explanation)

    involved_nodes = torch.unique(exp_edge_index)
    target_nodes = torch.tensor([node_a, node_b], device=device)
    involved_nodes = torch.unique(torch.cat([involved_nodes, target_nodes]))
    x_sub = data.x[involved_nodes]
    node_map = torch.empty(data.x.size(0), dtype=torch.long, device=device)
    node_map[involved_nodes] = torch.arange(involved_nodes.size(0), device=device)
    relabeled = node_map[exp_edge_index]

    exp_weight = None
    if hasattr(explanation, "edge_weight") and explanation.edge_weight is not None:
        exp_weight = explanation.edge_weight.to(device)

    local_a = int(node_map[node_a].item())
    local_b = int(node_map[node_b].item())
    new_pred, _ = _predict_binary(
        model, x_sub, relabeled, local_a, local_b,
        edge_weight=exp_weight, device=device,
    )

    return 0.0 if new_pred == original_pred else 1.0


def _fidelity_minus_coarse(model, data, explanation, original_score, device):
    exp_x = explanation.x.to(device) if explanation.x is not None else data.x
    exp_edges = explanation.edge_index.to(device)
    exp_weight = getattr(explanation, "edge_weight", None)
    if exp_weight is not None:
        exp_weight = exp_weight.to(device)

    target_a = getattr(explanation, "target_a", 0)
    target_b = getattr(explanation, "target_b", 1)
    target = torch.tensor([[target_a], [target_b]], device=device)
    with torch.no_grad():
        coarse_score = model(exp_x, exp_edges, target, edge_weight=exp_weight).squeeze().item()

    original_pred = int(original_score > 0.5)
    coarse_pred = int(coarse_score > 0.5)
    return 0.0 if original_pred == coarse_pred else 1.0


def fidelity_plus_continuous(
    model: torch.nn.Module,
    data: Data,
    explanation: Data,
    node_a: int,
    node_b: int,
    device: str = "cpu",
) -> float:
    """Continuous fidelity+ — measures prediction score drop when removing explanation.

    Returns |original_score - modified_score| instead of binary flip.
    More discriminative than binary fidelity for statistical testing.
    """
    model = model.to(device)
    model.eval()
    data = data.to(device)

    _, original_score = _predict_binary(
        model, data.x, data.edge_index, node_a, node_b,
        edge_weight=getattr(data, "edge_weight", None), device=device,
    )

    exp_edge_index = _to_global_edges(data, explanation)

    mask = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=device)
    for i in range(exp_edge_index.size(1)):
        src, dst = exp_edge_index[0, i], exp_edge_index[1, i]
        matches = (
            ((data.edge_index[0] == src) & (data.edge_index[1] == dst))
            | ((data.edge_index[0] == dst) & (data.edge_index[1] == src))
        )
        mask[matches] = False

    modified_edge_index = data.edge_index[:, mask]
    modified_weight = None
    if hasattr(data, "edge_weight") and data.edge_weight is not None:
        modified_weight = data.edge_weight[mask]

    _, modified_score = _predict_binary(
        model, data.x, modified_edge_index, node_a, node_b,
        edge_weight=modified_weight, device=device,
    )

    return abs(original_score - modified_score)


def _to_global_edges(data: Data, explanation: Data) -> torch.Tensor:
    if hasattr(explanation, "original_node_indices"):
        orig = explanation.original_node_indices
        return orig[explanation.edge_index]
    return explanation.edge_index


def evaluate_fidelity(
    model: torch.nn.Module,
    explainer,
    data: Data,
    test_edges: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, object]:
    data = data.to(device)
    model = model.to(device)
    model.eval()

    per_instance: List[tuple] = []
    fid_plus_sum = 0.0
    fid_minus_sum = 0.0
    num_edges = test_edges.size(1)

    for i in range(num_edges):
        a = int(test_edges[0, i].item())
        b = int(test_edges[1, i].item())

        explanation = explainer.explain_link(data, a, b)

        fp = fidelity_plus(model, data, explanation, a, b, device=device)
        fm = fidelity_minus(model, data, explanation, a, b, device=device)

        per_instance.append((fp, fm))
        fid_plus_sum += fp
        fid_minus_sum += fm

    return {
        "fidelity_plus": fid_plus_sum / max(num_edges, 1),
        "fidelity_minus": fid_minus_sum / max(num_edges, 1),
        "per_instance": per_instance,
    }


def compute_sparsity(explanation: Data, original_data: Data) -> float:
    num_exp = explanation.edge_index.size(1)
    num_orig = original_data.edge_index.size(1)
    if num_orig == 0:
        return 0.0
    return 1.0 - num_exp / num_orig
