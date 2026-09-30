"""Matched connected-partition controls for development, not a new default.

All policies merge the same canonical non-loop positive-weight edges, with
stable score ordering and canonical-edge tie breaks. Random scores use a local
generator. The structural control merges high normalized adjacency weights
first; it uses no eigenvectors, node features, labels, or predictor gradients.
"""
import torch

from src.diffusion_signals import NonstationaryDiffusion
from src.partition import node_partition
from src.spectral import compute_top_k_eigenpairs, pair_projection_scores


class GroupingControls:
    def __init__(self, edge_index, num_nodes, edge_weight=None):
        self.diffusion = NonstationaryDiffusion(edge_index, num_nodes, edge_weight)
        self.n = num_nodes
        operator = self.diffusion.operator.coalesce()
        indices, values = operator.indices(), operator.values()
        keep = (indices[0] < indices[1]) & (values > 0)
        self.edges, self.weights = indices[:, keep], values[keep]

    def partition(self, policy, *, seed=0, alpha=.75, width=100, steps=4, signals=None):
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must be between zero and one')
        diagnostics = {'policy': policy, 'seed': seed, 'alpha': alpha,
                       'edge_tie_break': 'stable score then ascending canonical edge',
                       'components': self.diffusion.num_components}
        if policy == 'random':
            scores = torch.rand(self.edges.shape[1], generator=torch.Generator().manual_seed(seed))
        elif policy == 'normalized-edge':
            scores = 1 - self.weights
        elif policy == 'diffusion':
            signals = self.diffusion.sketch(width=width, steps=steps, seed=seed)
            scores = pair_projection_scores(self.edges, signals)
            diagnostics.update(width=width, steps=steps)
        elif policy == 'signal':
            if (not isinstance(signals, torch.Tensor) or signals.ndim != 2
                    or signals.shape[0] != self.n or signals.shape[1] == 0
                    or not signals.is_floating_point() or not torch.isfinite(signals).all()):
                raise ValueError('Finite floating-point signal matrix with one row per node required')
            # Exact loss for a singleton-pair merge, NOT a dynamic Ward cost
            # after clusters grow. Chunking bounds temporary GPU storage.
            signals = signals.detach().to(self.edges.device)
            scores = signals.new_empty(self.edges.shape[1])
            for start in range(0, self.edges.shape[1], 4096):
                scores[start:start + 4096] = pair_projection_scores(
                    self.edges[:, start:start + 4096], signals)
            if not torch.isfinite(scores).all():
                raise ValueError('Signal-distance overflow')
            diagnostics.update(signal_width=signals.shape[1],
                               score='half squared singleton-pair signal distance',
                               warning='Fixed edge ranking is not dynamic Ward clustering or a multi-merge loss guarantee.')
        elif policy == 'eigen':
            values, _, signals = compute_top_k_eigenpairs(self.diffusion.operator, width)
            scores = pair_projection_scores(self.edges, signals)
            diagnostics.update(width=width, eigenvalues=values.cpu().tolist(),
                               cutoff_inside_stationary=self.diffusion.num_components > width,
                               warning='Existing eigensolver control retains its known cutoff ambiguity.')
        else:
            raise ValueError('Unknown grouping control')
        # Sorting on CPU also avoids one GPU synchronization per union.
        order = torch.argsort(scores.detach().cpu(), stable=True)
        edges = self.edges.cpu()[:, order]
        partition = node_partition(edges, torch.arange(len(order)), self.n, alpha)
        diagnostics.update(clusters=len(partition), largest_cluster=max(map(len, partition)))
        return partition, diagnostics
