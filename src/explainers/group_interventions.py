"""Exact group deletions, optionally batched as disjoint graph copies.

Batched evaluation requires a graph-separable predictor in evaluation mode,
such as the repository's GCN + endpoint MLP. It is not valid for models that
mix statistics or attention across disconnected components at inference.
"""
import torch


def supportive_edge_scores(edge_index, gradient, num_nodes, sign):
    """Tie directions before clipping to evidence supporting the predicted class.

    Scores are divided over stored entries so later summation per undirected
    edge recovers max(0, sign * sum of gate derivatives), including duplicates.
    """
    keys = edge_index.min(0).values * num_nodes + edge_index.max(0).values
    _, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)
    tied = gradient.new_zeros(counts.numel()).scatter_add_(0, inverse, gradient)
    return (sign * tied).clamp_min(0)[inverse] / counts[inverse]


@torch.no_grad()
def group_deletion_logits(model, data, node_a, node_b, groups, batch_size=1, *, retain=False):
    """Delete each collection of canonical undirected edge keys independently.

    With retain=True, retain each group instead of deleting it.
    Both orientations and duplicate entries are treated together. Original features,
    vertices and surviving edge weights are preserved. Chunking bounds the
    number of full graph copies held on the device simultaneously.
    """
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if model.training:
        raise ValueError('Group interventions require evaluation mode')
    n = data.x.size(0)
    device = data.x.device
    keys = data.edge_index.min(0).values * n + data.edge_index.max(0).values
    weight = getattr(data, 'edge_weight', None)
    outputs = []
    for start in range(0, len(groups), batch_size):
        chunk = groups[start:start + batch_size]
        indices, weights = [], []
        for offset, group in enumerate(chunk):
            keep = ~torch.isin(keys, group.to(device))
            if retain:
                keep = ~keep
            indices.append(data.edge_index[:, keep] + offset * n)
            if weight is not None:
                weights.append(weight[keep])
        targets = torch.tensor([[node_a], [node_b]], device=device)
        targets = targets + torch.arange(len(chunk), device=device)[None, :] * n
        logits = model(data.x.repeat(len(chunk), 1), torch.cat(indices, dim=1),
                       targets, edge_weight=None if weight is None else torch.cat(weights)).reshape(-1)
        if logits.numel() != len(chunk) or not torch.isfinite(logits).all():
            raise ValueError('Expected one finite logit per independent intervention')
        outputs.append(logits)
    return torch.cat(outputs) if outputs else data.x.new_empty(0)
