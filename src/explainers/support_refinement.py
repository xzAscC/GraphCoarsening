"""Fixed-budget support swaps with exact, componentwise acceptance checks.

This is an experimental local search, not a globally optimal explanation.
Coarse groups diversify gradient-proposed additions; an ungrouped control uses
the same initial support, proposal counts, and acceptance rule. No labels from
the data set or held-out results are used to optimize a query's explanation.
"""
import torch
from torch_geometric.data import Data

from src.explainers.group_interventions import group_deletion_logits
from src.evaluation.interventions import predict_logit


def candidate_groups(candidates, partition, n):
    membership = torch.empty(n, dtype=torch.long, device=candidates.device)
    for i, cluster in enumerate(partition):
        membership[torch.tensor(cluster, device=candidates.device)] = i
    left, right = membership[candidates // n], membership[candidates % n]
    return torch.minimum(left, right) * len(partition) + torch.maximum(left, right)


def diversify(order, groups, count):
    """One high-ranked entry per group first, then fill in global rank order."""
    if groups is None:
        return order[:count]
    selected, seen = [], set()
    for idx in order.tolist():
        group = int(groups[idx])
        if group not in seen:
            selected.append(idx)
            seen.add(group)
        if len(selected) == count:
            break
    if len(selected) < count:
        used = set(selected)
        selected.extend(idx for idx in order.tolist() if idx not in used)
    return torch.tensor(selected[:count], device=order.device, dtype=torch.long)


def refine_support(model, data, a, b, candidates, initial, *, groups=None,
                   steps=2, additions=6, removals=3, batch_size=8, min_gain=1e-7):
    """Return a same-size original-edge support and an auditable search trace.

    Accept only exact interventions whose removed-graph class probability does
    not increase and whose retained probability, capped at the full graph's,
    does not decrease. Binary necessity and sufficiency cannot regress either.
    Every accepted candidate is checked again with a single graph per forward.
    This is a finite-search invariant, not a fidelity or global-optimum theorem.
    Batched forwards require graph-separable inference; tested with GCN.
    """
    if steps < 0 or min(additions, removals, batch_size) < 1 or min_gain < 0:
        raise ValueError('Invalid refinement budget')
    if model.training:
        raise ValueError('Frozen evaluation-mode model required')
    n, device = data.x.size(0), data.x.device
    candidates = candidates.to(device)
    if not torch.equal(candidates, torch.unique(candidates, sorted=True)):
        raise ValueError('Candidates must be sorted unique undirected keys')
    if candidates.numel() and ((candidates < 0).any() or (candidates >= n*n).any()
                               or (candidates // n > candidates % n).any()):
        raise ValueError('Invalid canonical edge keys')
    raw = initial.edge_index.to(device)
    selected = torch.unique(raw.min(0).values * n + raw.max(0).values)
    if not torch.isin(selected, candidates).all():
        raise ValueError('Initial support is not a subset of candidates')
    full_keys = data.edge_index.min(0).values * n + data.edge_index.max(0).values
    if not torch.isin(candidates, full_keys).all():
        raise ValueError('Candidates must be original graph edges')
    if groups is not None:
        groups = groups.to(device)
        if groups.shape != candidates.shape:
            raise ValueError('One group per candidate required')
    full = predict_logit(model, data.x, data.edge_index, a, b, getattr(data, 'edge_weight', None))
    sign = 1 if full > 0 else -1
    p_full = torch.sigmoid(torch.tensor(sign * full, dtype=torch.float64, device=device))

    def assess(supports, batch):
        keep = group_deletion_logits(model, data, a, b, supports, batch, retain=True)
        delete = group_deletion_logits(model, data, a, b, supports, batch)
        probabilities = torch.sigmoid(sign * torch.stack((keep, delete), dim=1).double())
        capped = torch.minimum(probabilities[:, 0], p_full)
        quality = capped - probabilities[:, 1]
        sufficiency = (keep > 0) == (full > 0)
        necessity = (delete > 0) != (full > 0)
        return quality, capped, probabilities[:, 1], sufficiency, necessity, keep, delete

    current = tuple(x[0] for x in assess([selected], 1))
    trace = []

    def record(step, proposals, accepted):
        trace.append({'step': step, 'proposals': proposals, 'accepted': accepted,
                      'objective': current[0].item(), 'capped_retained_probability': current[1].item(),
                      'removed_probability': current[2].item(),
                      'retained_logit': current[5].item(), 'removed_logit': current[6].item()})

    record(0, 0, False)
    if not selected.numel() or selected.numel() == candidates.numel():
        return Data(edge_index=torch.stack((selected // n, selected % n))), trace
    positions = torch.searchsorted(candidates, full_keys)
    valid = positions < candidates.numel()
    safe_positions = positions.clamp(max=candidates.numel() - 1)
    valid &= candidates[safe_positions] == full_keys
    original_weight = getattr(data, 'edge_weight', None)
    if original_weight is None:
        original_weight = data.x.new_ones(data.edge_index.size(1))
    target = torch.tensor([[a], [b]], device=device)
    for step in range(1, steps + 1):
        active = torch.isin(candidates, selected)
        gates = active.to(data.x.dtype).requires_grad_()
        stored_gate = torch.where(valid, gates[safe_positions], 0.)
        # Gates propose swaps only. Exact retained/deleted edge lists decide acceptance.
        retained = model(data.x, data.edge_index, target, edge_weight=original_weight * stored_gate).squeeze()
        removed = model(data.x, data.edge_index, target, edge_weight=original_weight * (1 - stored_gate)).squeeze()
        objective = torch.minimum(torch.sigmoid(sign * retained), p_full.to(retained.dtype)) - torch.sigmoid(sign * removed)
        gradient, = torch.autograd.grad(objective, gates)
        if not torch.isfinite(gradient).all():
            raise ValueError('Nonfinite support-gradient proposal')
        inside, outside = active.nonzero().flatten(), (~active).nonzero().flatten()
        out_order = outside[torch.argsort(gradient[outside], descending=True, stable=True)]
        add = diversify(out_order, groups, min(additions, outside.numel()))
        drop = inside[torch.argsort(gradient[inside], stable=True)[:removals]]
        proposals = [torch.cat((selected[selected != candidates[d]], candidates[i:i+1])).sort().values
                     for d in drop.tolist() for i in add.tolist()]
        values = assess(proposals, batch_size)
        eligible = ((values[1] >= current[1]) & (values[2] <= current[2])
                    & (values[3] >= current[3]) & (values[4] >= current[4])
                    & (values[0] > current[0] + min_gain))
        if not eligible.any():
            record(step, len(proposals), False)
            break
        ranking = torch.argsort(values[0].masked_fill(~eligible, -torch.inf), descending=True)
        accepted = False
        for index in ranking[eligible[ranking]].tolist():
            checked = tuple(x[0] for x in assess([proposals[index]], 1))
            if (checked[1] >= current[1] and checked[2] <= current[2]
                    and checked[3] >= current[3] and checked[4] >= current[4]
                    and checked[0] > current[0] + min_gain):
                selected, current, accepted = proposals[index], checked, True
                break
        record(step, len(proposals), accepted)
        if not accepted:
            break
    return Data(edge_index=torch.stack((selected // n, selected % n))), trace
