"""Comprehensive evaluation metrics for GNN link-prediction explanations.

Extends the basic fidelity metrics with continuous probability-based metrics
(sufficiency, necessity, comprehensiveness), local sparsity, deletion and
insertion AUC curves, and batch evaluation utilities.

Handles both regular subgraph explanations (Data with original_node_indices)
and coarse graph explanations (Data with is_coarse_graph=True).
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import auc
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from src.evaluation.fidelity import (
    _is_coarse_explanation,
    _to_global_edges,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_probability(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_a: int,
    node_b: int,
    edge_weight: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> float:
    """Return the sigmoid probability for the target edge (node_a, node_b)."""
    target = torch.tensor([[node_a], [node_b]], device=device)
    with torch.no_grad():
        logit = model(x, edge_index, target, edge_weight=edge_weight).squeeze()
    return torch.sigmoid(logit).item()


def _get_coarse_target_nodes(explanation: Data) -> Tuple[int, int]:
    """Return (target_a, target_b) from a coarse explanation."""
    return (
        int(getattr(explanation, "target_a", 0)),
        int(getattr(explanation, "target_b", 1)),
    )


def _build_removal_mask(
    data_edge_index: torch.Tensor,
    edges_to_remove: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Boolean mask that is *False* for every edge in *edges_to_remove*.

    Matching is **undirected**: an edge (u, v) in *data_edge_index* is
    masked out when either (u, v) or (v, u) appears in *edges_to_remove*.
    """
    mask = torch.ones(data_edge_index.size(1), dtype=torch.bool, device=device)

    remove_set = set()
    for i in range(edges_to_remove.size(1)):
        s = edges_to_remove[0, i].item()
        d = edges_to_remove[1, i].item()
        remove_set.add((min(s, d), max(s, d)))

    for i in range(data_edge_index.size(1)):
        s = data_edge_index[0, i].item()
        d = data_edge_index[1, i].item()
        if (min(s, d), max(s, d)) in remove_set:
            mask[i] = False

    return mask


def _sort_edges_by_importance(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sort edges by importance (descending weight).

    If *edge_weight* is ``None`` or has the wrong size, edges are shuffled
    randomly (no importance information available).
    """
    num_edges = edge_index.size(1)
    if edge_weight is not None and edge_weight.numel() == num_edges:
        sorted_idx = edge_weight.argsort(descending=True)
    else:
        sorted_idx = torch.randperm(num_edges)

    sorted_edges = edge_index[:, sorted_idx]
    sorted_weights = (
        edge_weight[sorted_idx] if edge_weight is not None else None
    )
    return sorted_edges, sorted_weights


def _compute_necessity_components(
    model: torch.nn.Module,
    data: Data,
    explanation: Data,
    node_a: int,
    node_b: int,
    device: str,
) -> Tuple[float, float]:
    """Return ``(p_full, p_remove)`` shared by *necessity* and *comprehensiveness*."""
    model = model.to(device)
    model.eval()
    data = data.to(device)

    p_full = _get_probability(
        model, data.x, data.edge_index, node_a, node_b,
        edge_weight=getattr(data, "edge_weight", None), device=device,
    )

    if _is_coarse_explanation(explanation):
        involved_nodes = getattr(explanation, "original_node_indices", None)
        if involved_nodes is not None and involved_nodes.numel() > 0:
            involved_nodes = involved_nodes.to(device)
            node_mask = torch.zeros(data.x.size(0), dtype=torch.bool, device=device)
            node_mask[involved_nodes] = True
            src_in = node_mask[data.edge_index[0]]
            dst_in = node_mask[data.edge_index[1]]
            keep_mask = ~(src_in & dst_in)

            modified_edges = data.edge_index[:, keep_mask]
            modified_weight = None
            if hasattr(data, "edge_weight") and data.edge_weight is not None:
                modified_weight = data.edge_weight[keep_mask]

            p_remove = _get_probability(
                model, data.x, modified_edges, node_a, node_b,
                edge_weight=modified_weight, device=device,
            )
        else:
            # Fallback: compare prediction on coarse graph vs full graph.
            # For coarse explanations without node mapping, use the coarse
            # graph prediction as p_remove.  This measures how much the
            # coarse representation changes the model's prediction.
            target_a, target_b = _get_coarse_target_nodes(explanation)
            exp_x = explanation.x.to(device) if explanation.x is not None else data.x
            exp_edges = explanation.edge_index.to(device)
            exp_weight = getattr(explanation, "edge_weight", None)
            if exp_weight is not None:
                exp_weight = exp_weight.to(device)
            p_remove = _get_probability(
                model, exp_x, exp_edges, target_a, target_b,
                edge_weight=exp_weight, device=device,
            )
    else:
        exp_edge_index = _to_global_edges(data, explanation)
        keep_mask = _build_removal_mask(data.edge_index, exp_edge_index, device)

        modified_edges = data.edge_index[:, keep_mask]
        modified_weight = None
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            modified_weight = data.edge_weight[keep_mask]

        p_remove = _get_probability(
            model, data.x, modified_edges, node_a, node_b,
            edge_weight=modified_weight, device=device,
        )

    return p_full, p_remove


# ---------------------------------------------------------------------------
# Subgraph helpers for deletion / insertion (subgraph explanations)
# ---------------------------------------------------------------------------


def _deletion_auc_subgraph(
    model: torch.nn.Module,
    data: Data,
    explanation: Data,
    node_a: int,
    node_b: int,
    num_steps: int,
    device: str,
) -> float:
    p_full = _get_probability(
        model, data.x, data.edge_index, node_a, node_b,
        edge_weight=getattr(data, "edge_weight", None), device=device,
    )

    exp_edge_index = _to_global_edges(data, explanation)
    exp_weight = getattr(explanation, "edge_weight", None)
    if exp_weight is not None:
        exp_weight = exp_weight.to(device)

    sorted_edges, _ = _sort_edges_by_importance(exp_edge_index.to(device), exp_weight)
    num_exp = sorted_edges.size(1)

    fractions = [0.0]
    drops = [0.0]

    for step in range(1, num_steps + 1):
        k = min(int(np.ceil(step / num_steps * num_exp)), num_exp)
        edges_to_remove = sorted_edges[:, :k]
        keep_mask = _build_removal_mask(data.edge_index, edges_to_remove, device)

        mod_edges = data.edge_index[:, keep_mask]
        mod_weight = None
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            mod_weight = data.edge_weight[keep_mask]

        p_k = _get_probability(
            model, data.x, mod_edges, node_a, node_b,
            edge_weight=mod_weight, device=device,
        )
        fractions.append(k / num_exp)
        drops.append(p_full - p_k)

    return float(auc(np.array(fractions), np.array(drops)))


def _deletion_auc_coarse(
    model: torch.nn.Module,
    data: Data,
    explanation: Data,
    node_a: int,
    node_b: int,
    num_steps: int,
    device: str,
) -> float:
    target_a, target_b = _get_coarse_target_nodes(explanation)
    exp_x = explanation.x.to(device) if explanation.x is not None else data.x
    exp_edges = explanation.edge_index.to(device)
    exp_weight = getattr(explanation, "edge_weight", None)
    if exp_weight is not None:
        exp_weight = exp_weight.to(device)

    p_coarse = _get_probability(
        model, exp_x, exp_edges, target_a, target_b,
        edge_weight=exp_weight, device=device,
    )

    sorted_edges, sorted_weights = _sort_edges_by_importance(exp_edges, exp_weight)
    num_exp = sorted_edges.size(1)

    fractions = [0.0]
    drops = [0.0]

    for step in range(1, num_steps + 1):
        k = min(int(np.ceil(step / num_steps * num_exp)), num_exp)
        keep_mask = torch.ones(num_exp, dtype=torch.bool, device=device)
        keep_mask[:k] = False

        mod_edges = sorted_edges[:, keep_mask]
        mod_weight = sorted_weights[keep_mask] if sorted_weights is not None else None

        p_k = _get_probability(
            model, exp_x, mod_edges, target_a, target_b,
            edge_weight=mod_weight, device=device,
        )
        fractions.append(k / num_exp)
        drops.append(p_coarse - p_k)

    return float(auc(np.array(fractions), np.array(drops)))


def _insertion_auc_subgraph(
    model: torch.nn.Module,
    data: Data,
    explanation: Data,
    node_a: int,
    node_b: int,
    num_steps: int,
    device: str,
) -> float:
    exp_edge_index = _to_global_edges(data, explanation)
    exp_weight = getattr(explanation, "edge_weight", None)
    if exp_weight is not None:
        exp_weight = exp_weight.to(device)

    sorted_edges, sorted_weights = _sort_edges_by_importance(
        exp_edge_index.to(device), exp_weight,
    )
    num_exp = sorted_edges.size(1)

    x_base = data.x[[node_a, node_b]]
    empty_edges = torch.zeros(2, 0, dtype=torch.long, device=device)
    p_empty = _get_probability(model, x_base, empty_edges, 0, 1, device=device)

    fractions = [0.0]
    scores = [p_empty]

    for step in range(1, num_steps + 1):
        k = min(int(np.ceil(step / num_steps * num_exp)), num_exp)
        current_edges = sorted_edges[:, :k]

        involved = torch.unique(torch.cat([
            current_edges.reshape(-1),
            torch.tensor([node_a, node_b], device=device),
        ]))
        x_sub = data.x[involved]
        node_map = torch.empty(data.x.size(0), dtype=torch.long, device=device)
        node_map[involved] = torch.arange(involved.size(0), device=device)
        relabeled = node_map[current_edges]
        local_a = int(node_map[node_a].item())
        local_b = int(node_map[node_b].item())

        cur_weight = sorted_weights[:k] if sorted_weights is not None else None

        p_k = _get_probability(
            model, x_sub, relabeled, local_a, local_b,
            edge_weight=cur_weight, device=device,
        )
        fractions.append(k / num_exp)
        scores.append(p_k)

    return float(auc(np.array(fractions), np.array(scores)))


def _insertion_auc_coarse(
    model: torch.nn.Module,
    data: Data,
    explanation: Data,
    node_a: int,
    node_b: int,
    num_steps: int,
    device: str,
) -> float:
    target_a, target_b = _get_coarse_target_nodes(explanation)
    exp_x = explanation.x.to(device) if explanation.x is not None else data.x
    exp_edges = explanation.edge_index.to(device)
    exp_weight = getattr(explanation, "edge_weight", None)
    if exp_weight is not None:
        exp_weight = exp_weight.to(device)

    sorted_edges, sorted_weights = _sort_edges_by_importance(exp_edges, exp_weight)
    num_exp = sorted_edges.size(1)

    num_coarse = exp_x.size(0)
    x_base = exp_x[[target_a, target_b]]
    empty_edges = torch.zeros(2, 0, dtype=torch.long, device=device)
    p_empty = _get_probability(
        model, x_base, empty_edges, 0, 1, device=device,
    )

    fractions = [0.0]
    scores = [p_empty]

    for step in range(1, num_steps + 1):
        k = min(int(np.ceil(step / num_steps * num_exp)), num_exp)
        current_edges = sorted_edges[:, :k]

        involved = torch.unique(torch.cat([
            current_edges.reshape(-1),
            torch.tensor([target_a, target_b], device=device),
        ]))
        x_sub = exp_x[involved]
        node_map = torch.empty(num_coarse, dtype=torch.long, device=device)
        node_map[involved] = torch.arange(involved.size(0), device=device)
        relabeled = node_map[current_edges]
        local_a = int(node_map[target_a].item())
        local_b = int(node_map[target_b].item())

        cur_weight = sorted_weights[:k] if sorted_weights is not None else None

        p_k = _get_probability(
            model, x_sub, relabeled, local_a, local_b,
            edge_weight=cur_weight, device=device,
        )
        fractions.append(k / num_exp)
        scores.append(p_k)

    return float(auc(np.array(fractions), np.array(scores)))


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------


def sufficiency(
    model: torch.nn.Module,
    data: Data,
    explanation: Data,
    node_a: int,
    node_b: int,
    device: str = "cpu",
) -> float:
    """Sufficiency: ``|p_full - p_exp|``.

    Measures how well the explanation alone preserves the original prediction.
    Lower is better (0 = perfect).

    Args:
        model: Trained link-prediction model.
        data: Original graph as PyG ``Data``.
        explanation: Explanation subgraph or coarse graph.
        node_a: Source node of the target edge.
        node_b: Destination node of the target edge.
        device: ``'cpu'`` or ``'cuda'``.

    Returns:
        Absolute difference between full-graph and explanation probabilities.
    """
    model = model.to(device)
    model.eval()
    data = data.to(device)

    p_full = _get_probability(
        model, data.x, data.edge_index, node_a, node_b,
        edge_weight=getattr(data, "edge_weight", None), device=device,
    )

    if _is_coarse_explanation(explanation):
        target_a, target_b = _get_coarse_target_nodes(explanation)
        exp_x = explanation.x.to(device) if explanation.x is not None else data.x
        exp_edges = explanation.edge_index.to(device)
        exp_weight = getattr(explanation, "edge_weight", None)
        if exp_weight is not None:
            exp_weight = exp_weight.to(device)
        p_exp = _get_probability(
            model, exp_x, exp_edges, target_a, target_b,
            edge_weight=exp_weight, device=device,
        )
    else:
        exp_edge_index = _to_global_edges(data, explanation)
        involved_nodes = torch.unique(torch.cat([
            exp_edge_index.reshape(-1),
            torch.tensor([node_a, node_b], device=device),
        ]))
        x_sub = data.x[involved_nodes]
        node_map = torch.empty(data.x.size(0), dtype=torch.long, device=device)
        node_map[involved_nodes] = torch.arange(involved_nodes.size(0), device=device)
        relabeled = node_map[exp_edge_index]
        local_a = int(node_map[node_a].item())
        local_b = int(node_map[node_b].item())

        exp_weight = None
        if hasattr(explanation, "edge_weight") and explanation.edge_weight is not None:
            exp_weight = explanation.edge_weight.to(device)

        p_exp = _get_probability(
            model, x_sub, relabeled, local_a, local_b,
            edge_weight=exp_weight, device=device,
        )

    return abs(p_full - p_exp)


def necessity(
    model: torch.nn.Module,
    data: Data,
    explanation: Data,
    node_a: int,
    node_b: int,
    device: str = "cpu",
) -> float:
    """Necessity: ``p_full - p_remove``.

    Measures how much the prediction drops when explanation edges are removed
    from the original graph.  Higher is better.

    For coarse explanations, all original edges between the involved nodes
    (``original_node_indices``) are removed.

    Args:
        model: Trained link-prediction model.
        data: Original graph as PyG ``Data``.
        explanation: Explanation subgraph or coarse graph.
        node_a: Source node of the target edge.
        node_b: Destination node of the target edge.
        device: ``'cpu'`` or ``'cuda'``.

    Returns:
        Difference ``p_full - p_remove``.
    """
    p_full, p_remove = _compute_necessity_components(
        model, data, explanation, node_a, node_b, device,
    )
    return p_full - p_remove


def sparsity(
    data: Data,
    explanation: Data,
    node_a: int,
    node_b: int,
    num_hops: int = 2,
    device: str = "cpu",
) -> Tuple[float, int]:
    """Sparsity: ``|E_exp| / |E_local|``.

    Uses the *k*-hop enclosing subgraph around the target edge as the
    reference rather than the full graph, giving a more informative ratio
    for large graphs.

    Args:
        data: Original graph as PyG ``Data``.
        explanation: Explanation subgraph or coarse graph.
        node_a: Source node of the target edge.
        node_b: Destination node of the target edge.
        num_hops: Number of hops for the enclosing subgraph (default 2).
        device: ``'cpu'`` or ``'cuda'``.

    Returns:
        ``(ratio, absolute_edge_count)`` where *ratio* is
        ``num_exp_edges / num_local_edges``.
    """
    data = data.to(device)

    subset, sub_edge_index, _, _ = k_hop_subgraph(
        node_idx=torch.tensor([node_a, node_b], device=device),
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=False,
        num_nodes=data.x.size(0),
    )

    num_local = sub_edge_index.size(1)
    num_exp = explanation.edge_index.size(1)

    if num_local == 0:
        return 0.0, num_exp

    return num_exp / num_local, num_exp


def comprehensiveness(
    model: torch.nn.Module,
    data: Data,
    explanation: Data,
    node_a: int,
    node_b: int,
    device: str = "cpu",
) -> float:
    """Comprehensiveness: ``(p_full - p_remove) / p_full``.

    Normalised version of :func:`necessity`.  Higher is better.  Returns 0
    when the full-graph probability is near zero.

    Args:
        model: Trained link-prediction model.
        data: Original graph as PyG ``Data``.
        explanation: Explanation subgraph or coarse graph.
        node_a: Source node of the target edge.
        node_b: Destination node of the target edge.
        device: ``'cpu'`` or ``'cuda'``.

    Returns:
        Normalised necessity score in ``[-inf, 1]``.
    """
    p_full, p_remove = _compute_necessity_components(
        model, data, explanation, node_a, node_b, device,
    )
    if abs(p_full) < 1e-8:
        return 0.0
    return (p_full - p_remove) / p_full


def deletion_auc(
    model: torch.nn.Module,
    data: Data,
    explanation: Data,
    node_a: int,
    node_b: int,
    num_steps: int = 20,
    device: str = "cpu",
) -> float:
    """Deletion AUC: progressively remove explanation edges and measure drop.

    Explanation edges are sorted by importance (``edge_weight`` if present,
    otherwise randomly).  The top-*k* edges are removed at each step and the
    prediction score drop is recorded.  Higher AUC means the importance
    ranking is accurate.

    Args:
        model: Trained link-prediction model.
        data: Original graph as PyG ``Data``.
        explanation: Explanation subgraph or coarse graph.
        node_a: Source node of the target edge.
        node_b: Destination node of the target edge.
        num_steps: Number of removal steps (default 20).
        device: ``'cpu'`` or ``'cuda'``.

    Returns:
        AUC of the deletion curve.
    """
    model = model.to(device)
    model.eval()
    data = data.to(device)

    num_exp = explanation.edge_index.size(1)
    if num_exp == 0:
        return 0.0

    if _is_coarse_explanation(explanation):
        return _deletion_auc_coarse(
            model, data, explanation, node_a, node_b, num_steps, device,
        )
    return _deletion_auc_subgraph(
        model, data, explanation, node_a, node_b, num_steps, device,
    )


def insertion_auc(
    model: torch.nn.Module,
    data: Data,
    explanation: Data,
    node_a: int,
    node_b: int,
    num_steps: int = 20,
    device: str = "cpu",
) -> float:
    """Insertion AUC: progressively add explanation edges and measure recovery.

    Start from an empty graph (just the two target nodes) and progressively
    add explanation edges sorted by importance.  Higher AUC means the
    explanation quickly recovers the original prediction.

    Args:
        model: Trained link-prediction model.
        data: Original graph as PyG ``Data``.
        explanation: Explanation subgraph or coarse graph.
        node_a: Source node of the target edge.
        node_b: Destination node of the target edge.
        num_steps: Number of insertion steps (default 20).
        device: ``'cpu'`` or ``'cuda'``.

    Returns:
        AUC of the insertion curve.
    """
    model = model.to(device)
    model.eval()
    data = data.to(device)

    num_exp = explanation.edge_index.size(1)
    if num_exp == 0:
        return 0.0

    if _is_coarse_explanation(explanation):
        return _insertion_auc_coarse(
            model, data, explanation, node_a, node_b, num_steps, device,
        )
    return _insertion_auc_subgraph(
        model, data, explanation, node_a, node_b, num_steps, device,
    )


# ---------------------------------------------------------------------------
# Convenience aggregators
# ---------------------------------------------------------------------------


def compute_all_metrics(
    model: torch.nn.Module,
    data: Data,
    explanation: Data,
    node_a: int,
    node_b: int,
    device: str = "cpu",
    num_steps: int = 20,
    num_hops: int = 2,
) -> Dict[str, float]:
    """Compute all explanation quality metrics for a single target edge.

    Args:
        model: Trained link-prediction model.
        data: Original graph as PyG ``Data``.
        explanation: Explanation subgraph or coarse graph.
        node_a: Source node of the target edge.
        node_b: Destination node of the target edge.
        device: ``'cpu'`` or ``'cuda'``.
        num_steps: Number of steps for AUC metrics.
        num_hops: Number of hops for local sparsity.

    Returns:
        Dictionary with keys ``sufficiency``, ``necessity``,
        ``comprehensiveness``, ``sparsity``, ``sparsity_abs_edges``,
        ``deletion_auc``, ``insertion_auc``.
    """
    model = model.to(device)
    model.eval()
    data = data.to(device)

    p_full, p_remove = _compute_necessity_components(
        model, data, explanation, node_a, node_b, device,
    )
    nec = p_full - p_remove
    comp = (p_full - p_remove) / p_full if abs(p_full) > 1e-8 else 0.0

    suff = sufficiency(model, data, explanation, node_a, node_b, device)

    sp_ratio, sp_count = sparsity(
        data, explanation, node_a, node_b, num_hops=num_hops, device=device,
    )

    del_auc = deletion_auc(
        model, data, explanation, node_a, node_b,
        num_steps=num_steps, device=device,
    )
    ins_auc = insertion_auc(
        model, data, explanation, node_a, node_b,
        num_steps=num_steps, device=device,
    )

    return {
        "sufficiency": suff,
        "necessity": nec,
        "comprehensiveness": comp,
        "sparsity": sp_ratio,
        "sparsity_abs_edges": float(sp_count),
        "deletion_auc": del_auc,
        "insertion_auc": ins_auc,
    }


def batch_evaluate(
    model: torch.nn.Module,
    data: Data,
    explanations: List[Data],
    node_pairs: Union[List[Tuple[int, int]], torch.Tensor],
    device: str = "cpu",
    num_steps: int = 20,
    num_hops: int = 2,
) -> Dict[str, object]:
    """Evaluate a batch of explanations and return aggregate statistics.

    Args:
        model: Trained link-prediction model.
        data: Original graph as PyG ``Data``.
        explanations: List of explanation ``Data`` objects.
        node_pairs: Either a list of ``(node_a, node_b)`` tuples or a
            ``(2, E)`` tensor of edge indices.
        device: ``'cpu'`` or ``'cuda'``.
        num_steps: Number of steps for AUC metrics.
        num_hops: Number of hops for local sparsity.

    Returns:
        Dictionary with keys:
        - ``mean``: Dict of metric name → mean value.
        - ``std``: Dict of metric name → standard deviation.
        - ``per_instance``: List of per-instance metric dicts.
    """
    if isinstance(node_pairs, torch.Tensor):
        pairs = [
            (int(node_pairs[0, i].item()), int(node_pairs[1, i].item()))
            for i in range(node_pairs.size(1))
        ]
    else:
        pairs = list(node_pairs)

    if len(pairs) != len(explanations):
        raise ValueError(
            "Number of node pairs ({}) must match number of explanations ({})".format(
                len(pairs), len(explanations),
            )
        )

    per_instance: List[Dict[str, float]] = []
    for (a, b), exp in zip(pairs, explanations):
        metrics = compute_all_metrics(
            model, data, exp, a, b,
            device=device, num_steps=num_steps, num_hops=num_hops,
        )
        per_instance.append(metrics)

    if not per_instance:
        return {"mean": {}, "std": {}, "per_instance": []}

    keys = list(per_instance[0].keys())
    mean_stats: Dict[str, float] = {}
    std_stats: Dict[str, float] = {}
    for key in keys:
        values = np.array([m[key] for m in per_instance], dtype=np.float64)
        mean_stats[key] = float(np.mean(values))
        std_stats[key] = float(np.std(values))

    return {
        "mean": mean_stats,
        "std": std_stats,
        "per_instance": per_instance,
    }
