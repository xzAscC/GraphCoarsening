"""Fixed-budget support swaps with exact, componentwise acceptance checks.

This is an experimental local search, not a globally optimal explanation.
Coarse groups diversify gradient-proposed additions; an ungrouped control uses
the same initial support, proposal counts, and acceptance rule. No labels from
the data set or held-out results are used to optimize a query's explanation.
"""
import itertools
import torch
from torch_geometric.data import Data

from src.explainers.group_interventions import group_deletion_logits
from src.evaluation.interventions import predict_logit
from src.explainers.local_gcn import compact_gcn_query, validate_local_gcn_model


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


def coverage_union(order, groups, count):
    """Union equal-sized global/diverse shortlists, fill to a fixed total cap.

Retains the global top count/2 AND the diverse top count/2 at this state.
Duplicate slots are filled by global rank, then the pool is returned in global
rank order. This covers a smaller ungrouped pool, not a same-size one, and
does not imply dominance of complete adaptive search trajectories.
"""
    if count < 2 or count % 2:
        raise ValueError('Coverage union requires an even total addition count >= 2')
    if groups is None:
        return order[:count]
    half = count // 2
    chosen = set(order[:half].tolist())
    chosen.update(diversify(order, groups, min(half, len(order))).tolist())
    for index in order.tolist():
        if len(chosen) >= min(count, len(order)):
            break
        chosen.add(index)
    return torch.tensor([index for index in order.tolist() if index in chosen],
                        device=order.device, dtype=torch.long)


def ranked_bundles(order, gradient, count, size, *, groups=None, largest=True):
    """Bounded pair proposals; grouping prioritizes two edges in one group.

    For ungrouped pairs, the best ``count`` additive scores occur among the
    first ``count+1`` individually ranked entries (up to ties). A grouped
    proposal pool first takes the best pair per group, then fills from the
    global list. This additive proposal score is not a joint-effect estimate.
    """
    if size not in (1, 2) or count < 1:
        raise ValueError('Positive proposal count and bundle size 1 or 2 required')
    if size == 1:
        return diversify(order, groups, min(count, len(order))).reshape(-1, 1)
    indices = order.tolist()
    scores = gradient.detach().cpu().tolist()
    direction = -1 if largest else 1

    def rank(pairs):
        return sorted(pairs, key=lambda pair: (direction * sum(scores[i] for i in pair), pair))

    global_pairs = rank([tuple(sorted(pair)) for pair in
                         itertools.combinations(indices[:count + 1], 2)])
    preferred = []
    if groups is not None:
        ids = groups.detach().cpu().tolist()
        buckets = {}
        for idx in indices:
            bucket = buckets.setdefault(ids[idx], [])
            if len(bucket) < 2:
                bucket.append(idx)
        preferred = rank([tuple(sorted(bucket)) for bucket in buckets.values() if len(bucket) == 2])
    selected, seen = [], set()
    for pair in itertools.chain(preferred, global_pairs):
        if pair not in seen:
            selected.append(pair)
            seen.add(pair)
        if len(selected) == count:
            break
    return torch.tensor(selected, dtype=torch.long, device=order.device).reshape(-1, 2)


def exchange_proposals(candidates, selected, in_order, out_order, gradient,
                       additions, removals, exchange_size, groups=None, addition_policy='diverse'):
    """Build at most additions*removals supports, including mixed-size mode 0.

    Pure modes preserve the existing Cartesian order. Mixed mode ranks each
    size's pool by additive predicted gain, then alternates single and pair
    proposals under one shared cap. It does not double the model-query budget.
    """
    if exchange_size not in (0, 1, 2) or min(additions, removals) < 1:
        raise ValueError('Invalid proposal settings')
    if addition_policy not in ('diverse', 'coverage-union'):
        raise ValueError('Unknown addition policy')
    if addition_policy == 'coverage-union' and (exchange_size != 1 or additions < 2 or additions % 2):
        raise ValueError('Coverage union requires single exchanges and an even addition count >= 2')
    available = min(len(in_order), len(out_order))
    if available == 0:
        return [], []
    sizes = ([1, 2] if available >= 2 else [1]) if exchange_size == 0 else [min(exchange_size, available)]
    pools = []
    for size in sizes:
        add = (coverage_union(out_order, groups, additions).reshape(-1, 1)
               if addition_policy == 'coverage-union'
               else ranked_bundles(out_order, gradient, additions, size, groups=groups))
        drop = ranked_bundles(in_order, gradient, removals, size, largest=False)
        pool = []
        for d in drop:
            for i in add:
                support = torch.cat((selected[~torch.isin(selected, candidates[d])], candidates[i])).sort().values
                gain = (gradient[i].sum() - gradient[d].sum()).item() if exchange_size == 0 else 0.
                pool.append((support, size, gain))
        if exchange_size == 0:
            pool.sort(key=lambda proposal: -proposal[2])
        pools.append(pool)
    interleaved = itertools.chain.from_iterable(itertools.zip_longest(*pools))
    chosen = list(itertools.islice((p for p in interleaved if p is not None), additions * removals))
    return [p[0] for p in chosen], [p[1] for p in chosen]


def acceptance_mask(values, current, min_gain, policy='componentwise'):
    """Shared batched/serial test; optional policy relaxes continuous Pareto guards.

    Both policies require strict objective gain and binary non-regression.
    binary-monotone may trade retained confidence against deleted confidence;
    it does not inherit componentwise continuous non-regression.
    """
    if policy not in ('componentwise', 'binary-monotone'):
        raise ValueError('Unknown acceptance policy')
    valid = ((values[3] >= current[3]) & (values[4] >= current[4])
             & (values[0] > current[0] + min_gain))
    if policy == 'componentwise':
        valid = valid & (values[1] >= current[1]) & (values[2] <= current[2])
    return valid


def refine_support(model, data, a, b, candidates, initial, *, groups=None,
                   steps=2, additions=6, removals=3, batch_size=8, min_gain=1e-7,
                   exchange_size=1, local_gcn=False, addition_policy='diverse',
                   acceptance_policy='componentwise'):
    """Return a same-size original-edge support and an auditable search trace.

    Accept only exact interventions whose removed-graph class probability does
    not increase and whose retained probability, capped at the full graph's,
    does not decrease. Binary necessity and sufficiency cannot regress either.
    Every accepted candidate is checked again with a single graph per forward.
    This is a finite-search invariant, not a fidelity or global-optimum theorem.
    coverage-union is an optional single-exchange policy retaining both global
    and group-diverse half-sized shortlists under the total additions cap.
    Batched forwards require graph-separable inference; tested with GCN.
    exchange_size=0 mixes single and pair moves under the same proposal cap.
    local_gcn uses a boundary-compacted graph for gradients and batched screening,
    but retains full-graph initialization and serial acceptance checks. Floating
    point differences may change proposals or screening; identical trajectories
    to full-graph search are not promised. Default behavior remains full-graph.
    Optional binary-monotone acceptance drops the two continuous componentwise
    guards, retaining strict objective gain and both binary guards. It is an
    experimental tradeoff policy, not a dominance claim over the default.
    """
    if (steps < 0 or min(additions, removals, batch_size) < 1 or min_gain < 0
            or exchange_size not in (0, 1, 2)):
        raise ValueError('Invalid refinement budget')
    if addition_policy not in ('diverse', 'coverage-union'):
        raise ValueError('Unknown addition policy')
    if acceptance_policy not in ('componentwise', 'binary-monotone'):
        raise ValueError('Unknown acceptance policy')
    if addition_policy == 'coverage-union' and (exchange_size != 1 or additions < 2 or additions % 2):
        raise ValueError('Coverage union requires single exchanges and an even addition count >= 2')
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
    region = None
    if local_gcn:
        layers = validate_local_gcn_model(model)
        region = compact_gcn_query(data, [a, b], layers)
        region.map_keys(candidates)  # Reject any candidate outside the certified region.
    working = region.data if region is not None else data
    local_endpoints = region.targets.flatten().tolist() if region is not None else None
    full = predict_logit(model, data.x, data.edge_index, a, b, getattr(data, 'edge_weight', None))
    sign = 1 if full > 0 else -1
    p_full = torch.sigmoid(torch.tensor(sign * full, dtype=torch.float64, device=device))

    def assess(supports, batch, *, compact=False):
        graph, left, right = data, a, b
        if compact:
            graph = region.data
            left, right = local_endpoints
            supports = region.map_support_batch(supports)
        keep = group_deletion_logits(model, graph, left, right, supports, batch, retain=True)
        delete = group_deletion_logits(model, graph, left, right, supports, batch)
        probabilities = torch.sigmoid(sign * torch.stack((keep, delete), dim=1).double())
        capped = torch.minimum(probabilities[:, 0], p_full)
        quality = capped - probabilities[:, 1]
        sufficiency = (keep > 0) == (full > 0)
        necessity = (delete > 0) != (full > 0)
        return quality, capped, probabilities[:, 1], sufficiency, necessity, keep, delete

    current = tuple(x[0] for x in assess([selected], 1))
    trace = []

    def record(step, proposals, accepted, actual_exchange_size=0, proposal_sizes=(), rechecks=0):
        trace.append({'step': step, 'proposals': proposals, 'accepted': accepted,
                      'acceptance_policy': acceptance_policy,
                      'addition_policy': addition_policy, 'addition_cap': additions,
                      'evaluation_mode': 'local-screen-full-accept' if region is not None else 'full',
                      'gradient_nodes': working.num_nodes,
                      'full_graph_rechecks': rechecks,
                      'exchange_size': actual_exchange_size,
                      'proposals_by_exchange_size': {str(s): proposal_sizes.count(s) for s in sorted(set(proposal_sizes))},
                      'objective': current[0].item(), 'capped_retained_probability': current[1].item(),
                      'removed_probability': current[2].item(),
                      'retained_logit': current[5].item(), 'removed_logit': current[6].item()})

    record(0, 0, False)
    if not selected.numel() or selected.numel() == candidates.numel():
        return Data(edge_index=torch.stack((selected // n, selected % n))), trace
    working_keys = full_keys[region.original_edge_mask] if region is not None else full_keys
    positions = torch.searchsorted(candidates, working_keys)
    valid = positions < candidates.numel()
    safe_positions = positions.clamp(max=candidates.numel() - 1)
    valid &= candidates[safe_positions] == working_keys
    original_weight = getattr(working, 'edge_weight', None)
    if original_weight is None:
        original_weight = working.x.new_ones(working.edge_index.size(1))
    target = region.targets if region is not None else torch.tensor([[a], [b]], device=device)
    for step in range(1, steps + 1):
        active = torch.isin(candidates, selected)
        gates = active.to(data.x.dtype).requires_grad_()
        stored_gate = torch.where(valid, gates[safe_positions], 0.)
        # Gates propose swaps only. Exact retained/deleted edge lists decide acceptance.
        retained = model(working.x, working.edge_index, target, edge_weight=original_weight * stored_gate).squeeze()
        removed = model(working.x, working.edge_index, target, edge_weight=original_weight * (1 - stored_gate)).squeeze()
        objective = torch.minimum(torch.sigmoid(sign * retained), p_full.to(retained.dtype)) - torch.sigmoid(sign * removed)
        gradient, = torch.autograd.grad(objective, gates)
        if not torch.isfinite(gradient).all():
            raise ValueError('Nonfinite support-gradient proposal')
        inside, outside = active.nonzero().flatten(), (~active).nonzero().flatten()
        out_order = outside[torch.argsort(gradient[outside], descending=True, stable=True)]
        in_order = inside[torch.argsort(gradient[inside], stable=True)]
        proposals, proposal_sizes = exchange_proposals(
            candidates, selected, in_order, out_order, gradient,
            additions, removals, exchange_size, groups,
            **({'addition_policy': addition_policy} if addition_policy != 'diverse' else {}))
        size = min(exchange_size, inside.numel(), outside.numel())
        values = assess(proposals, batch_size, compact=region is not None)
        eligible = acceptance_mask(values, current, min_gain, acceptance_policy)
        if not eligible.any():
            record(step, len(proposals), False, size, proposal_sizes)
            break
        ranking = torch.argsort(values[0].masked_fill(~eligible, -torch.inf), descending=True)
        accepted = False
        rechecks = 0
        for index in ranking[eligible[ranking]].tolist():
            rechecks += 1
            checked = tuple(x[0] for x in assess([proposals[index]], 1))
            if acceptance_mask(checked, current, min_gain, acceptance_policy):
                selected, current, accepted = proposals[index], checked, True
                size = proposal_sizes[index]
                break
        record(step, len(proposals), accepted, size, proposal_sizes, rechecks)
        if not accepted:
            break
    return Data(edge_index=torch.stack((selected // n, selected % n))), trace
