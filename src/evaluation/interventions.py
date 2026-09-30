"""Explicit undirected edge interventions for models returning scalar logits.

Importance scores choose a support; they never replace message-passing weights.
All interventions preserve the original node features and vertex set. A coarse
representation is a different object and has no necessity score without support.
"""

import torch

PROTOCOL = "undirected-original-support-v1"


def _edge_keys(edges, n):
    return edges.min(dim=0).values * n + edges.max(dim=0).values


def support_mask(data, explanation):
    """Select all original orientations of each requested undirected edge."""
    if getattr(explanation, "is_coarse_graph", False):
        raise ValueError("A coarse graph is not an original-edge support; use coarse_fidelity")
    edges = explanation.edge_index.to(data.edge_index.device)
    mapping = getattr(explanation, "original_node_indices", None)
    if mapping is not None:
        mapping = mapping.to(edges.device)
        if edges.numel() and (edges.min() < 0 or edges.max() >= mapping.numel()):
            raise ValueError("Invalid local explanation edge index")
        edges = mapping[edges]
    n = data.x.size(0)
    if edges.numel() and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("Explanation edge lies outside the original vertex set")
    original = _edge_keys(data.edge_index, n)
    selected = _edge_keys(edges, n)
    if not torch.isin(selected, original).all():
        raise ValueError("Explanation contains an edge absent from the input graph")
    return torch.isin(original, selected)


@torch.no_grad()
def predict_logit(model, x, edges, a, b, weight=None):
    target = torch.tensor([[a], [b]], device=x.device)
    value = model(x, edges, target, edge_weight=weight).reshape(-1)
    if value.numel() != 1 or not torch.isfinite(value).all():
        raise ValueError("Expected one finite raw logit for the queried link")
    return value.item()


def _probability(logit):
    return torch.sigmoid(torch.tensor(logit, dtype=torch.float64)).item()


def evaluate_support(model, data, explanation, a, b, device="cpu"):
    """Three forwards on G, G[E_support], G minus E_support.

    Binary decisions use sigmoid(logit) > 0.5, equivalently logit > 0.
    Confidence drops are relative to the original predicted class, so negative
    and positive predictions have the same interpretation. Drops can be negative.
    """
    model = model.to(device).eval()
    data = data.to(device)
    mask = support_mask(data, explanation)
    weight = getattr(data, "edge_weight", None)
    full = predict_logit(model, data.x, data.edge_index, a, b, weight)
    kept = predict_logit(model, data.x, data.edge_index[:, mask], a, b,
                         None if weight is None else weight[mask])
    removed = predict_logit(model, data.x, data.edge_index[:, ~mask], a, b,
                            None if weight is None else weight[~mask])
    decision = full > 0
    p, pk, pr = map(_probability, (full, kept, removed))
    direction = 1 if decision else -1
    keys = _edge_keys(data.edge_index, data.x.size(0))
    size = int(torch.unique(keys[mask]).numel())
    total = int(torch.unique(keys).numel())
    return {
        "full_logit": full, "retained_logit": kept, "removed_logit": removed,
        "necessity_flip": float((removed > 0) != decision),
        "sufficiency_agreement": float((kept > 0) == decision),
        "necessity_confidence_drop": direction * (p - pr),
        "sufficiency_confidence_drop": direction * (p - pk),
        "support_edges": size, "input_edges": total,
        "evidence_sparsity": 1 - size / total if total else 0.0,
    }


def coarse_fidelity(model, data, explanation, a, b, device="cpu"):
    """Prediction preservation only, never a necessity/sufficiency intervention."""
    if not getattr(explanation, "is_coarse_graph", False):
        raise ValueError("Expected an explicitly marked coarse graph")
    ca, cb = getattr(explanation, "target_a", None), getattr(explanation, "target_b", None)
    if ca is None or cb is None:
        raise ValueError("Coarse endpoint indices are required")
    model = model.to(device).eval()
    data, explanation = data.to(device), explanation.to(device)
    full = predict_logit(model, data.x, data.edge_index, a, b,
                         getattr(data, "edge_weight", None))
    coarse = predict_logit(model, explanation.x, explanation.edge_index, ca, cb,
                           getattr(explanation, "edge_weight", None))
    return {"coarse_agreement": float((full > 0) == (coarse > 0)),
            "coarse_probability_error": abs(_probability(full) - _probability(coarse)),
            "full_logit": full, "coarse_logit": coarse}
