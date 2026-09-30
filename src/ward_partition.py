"""Connected, disjoint-round Ward merges for development diagnostics.

For clusters C,D the increase in squared projection loss is
|C||D|/(|C|+|D|) * ||mean(C)-mean(D)||^2. Disjoint merges have additive
increments. Costs are recomputed after every matching round, not every edge;
this is neither sequential Ward linkage nor a globally optimal partition.
GPU tensors handle centroids and distances; CPU chooses a stable matching.
"""
import math

import torch


@torch.no_grad()
def connected_ward_partition(edges, signals, *, alpha=.75, max_relative_loss=None):
    if (not isinstance(signals, torch.Tensor) or signals.ndim != 2
            or min(signals.shape) < 1 or not signals.is_floating_point()
            or not torch.isfinite(signals).all()):
        raise ValueError('Finite nonempty floating-point signal matrix required')
    n, width = signals.shape
    if (edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long
            or edges.device != signals.device or (edges < 0).any() or (edges >= n).any()):
        raise ValueError('Valid long edge indices on signal device required')
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be between zero and one')
    if max_relative_loss is not None and not 0 <= max_relative_loss <= 1:
        raise ValueError('Relative squared-loss cap must be between zero and one')
    # Float64 reduces drift between the telescoping identity and direct SSE.
    values = signals.detach().double()
    norm_squared = float(values.square().sum())
    if not math.isfinite(norm_squared):
        raise ValueError('Signal-norm overflow')
    budget = math.inf if max_relative_loss is None else max_relative_loss * norm_squared
    membership = torch.arange(n, device=signals.device)
    remaining = int(alpha * n)
    total_loss, trace = 0., []
    while remaining:
        m = int(membership.max()) + 1
        sizes = torch.bincount(membership, minlength=m).double()
        means = values.new_zeros((m, width))
        means.index_add_(0, membership, values)
        means /= sizes[:, None]
        current = membership[edges]
        a, b = current.min(0).values, current.max(0).values
        keys = torch.unique(a[a != b] * m + b[a != b], sorted=True)
        if keys.numel() == 0:
            break
        a, b = keys // m, keys % m
        scores = values.new_empty(len(keys))
        for start in range(0, len(keys), 4096):
            x, y = a[start:start + 4096], b[start:start + 4096]
            scores[start:start + 4096] = (sizes[x] * sizes[y] / (sizes[x] + sizes[y])
                                         * (means[x] - means[y]).square().sum(1))
        if not torch.isfinite(scores).all():
            raise ValueError('Merge-cost overflow')
        cpu_scores = scores.cpu()
        order = torch.argsort(cpu_scores, stable=True).tolist()
        pairs = torch.stack((a, b), 1).cpu().tolist()
        used, accepted = set(), []
        round_loss = 0.
        for index in order:
            x, y = pairs[index]
            if x in used or y in used:
                continue
            cost = float(cpu_scores[index])
            if total_loss + round_loss + cost > budget:
                # Costs are sorted; no later cost can fit the remaining cap.
                break
            accepted.append((x, y))
            used.update((x, y))
            round_loss += cost
            if len(accepted) == remaining:
                break
        if not accepted:
            break
        remap = torch.arange(m)
        for x, y in accepted:
            remap[y] = x
        remap = torch.unique(remap, sorted=True, return_inverse=True)[1].to(signals.device)
        membership = remap[membership]
        remaining -= len(accepted)
        total_loss += round_loss
        trace.append({'merges': len(accepted), 'clusters': m - len(accepted),
                      'squared_loss_increment': round_loss, 'cumulative_squared_loss': total_loss})
    m = int(membership.max()) + 1
    means = values.new_zeros((m, width))
    means.index_add_(0, membership, values)
    means /= torch.bincount(membership, minlength=m)[:, None]
    direct_loss = float((values - means[membership]).square().sum())
    if not math.isclose(total_loss, direct_loss, rel_tol=1e-8, abs_tol=1e-10 * max(norm_squared, 1.)):
        raise RuntimeError('Ward increments disagree with direct projection loss')
    partition = [[] for _ in range(m)]
    for node, cluster in enumerate(membership.cpu().tolist()):
        partition[cluster].append(node)
    return partition, {
        'policy': 'connected-disjoint-round-ward', 'alpha': alpha,
        'max_relative_squared_loss': max_relative_loss, 'signal_width': width,
        'clusters': m, 'largest_cluster': max(map(len, partition)),
        'squared_signal_norm': norm_squared, 'cumulative_squared_loss': total_loss,
        'direct_squared_projection_loss': direct_loss,
        'relative_squared_projection_loss': direct_loss / norm_squared if norm_squared else 0.,
        'rounds': trace,
        'warning': 'Disjoint-round greedy heuristic, not sequential Ward or a prediction-fidelity certificate. Loss cap is numerical, not an interval-arithmetic certificate.'}
