"""Experimental nonstationary diffusion sketches, not top-k eigenvectors.

Let S project onto ALL component stationary modes of normalized A+I and
T=(I+A_hat)/2. For independent Z_ij ~ N(0, 1/r), Y=T**t (I-S) Z.
Then E[||Y[a]-Y[b]||^2/2] = ||(e_a-e_b)^T T**t (I-S)||^2/2.
This target weights complete eigenspaces, with no hard rank cutoff. A finite
sketch is random and need not give identical rankings or faithful explanations.
Connectivity is computed on CPU; projection and sparse filtering use the input
device. No dense n-by-n operator or n-by-components basis is materialized.
"""

import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import torch

from src.spectral import compute_normalized_adjacency


class NonstationaryDiffusion:
    def __init__(self, edge_index, num_nodes, edge_weight=None):
        if not isinstance(num_nodes, int) or num_nodes < 1:
            raise ValueError('num_nodes must be a positive integer')
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError('edge_index must be a 2-by-E long tensor')
        if edge_index.numel() and ((edge_index < 0).any() or (edge_index >= num_nodes).any()):
            raise ValueError('edge indices out of range')
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)
        if edge_weight.dtype not in (torch.float32, torch.float64):
            raise ValueError('weights must be float32 or float64')
        self.operator = compute_normalized_adjacency(edge_index, num_nodes, edge_weight)
        raw = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).coalesce()
        indices = raw.indices().cpu().numpy()
        values = raw.values().detach().cpu().numpy()
        adjacency = sp.csr_matrix((values, indices), shape=(num_nodes, num_nodes))
        adjacency.eliminate_zeros()
        difference = adjacency - adjacency.T
        if difference.nnz:
            raise ValueError('coalesced adjacency must be exactly symmetric')
        self.num_components, labels = connected_components(adjacency, directed=False)
        self.labels = torch.as_tensor(labels, dtype=torch.long, device=edge_index.device)
        degree = edge_weight.new_ones(num_nodes)
        degree.index_add_(0, raw.indices()[0], raw.values())
        masses = edge_weight.new_zeros(self.num_components)
        masses.index_add_(0, self.labels, degree)
        self.stationary_entries = (degree / masses[self.labels]).sqrt()

    def remove_stationary(self, signals):
        """Apply I-S, using one nonzero stationary-basis entry per node."""
        if (signals.ndim != 2 or signals.shape[0] != self.operator.shape[0]
                or signals.device != self.operator.device or signals.dtype != self.operator.dtype):
            raise ValueError('signals must match operator size, dtype, and device')
        coefficients = signals.new_zeros((self.num_components, signals.shape[1]))
        coefficients.index_add_(0, self.labels, self.stationary_entries[:, None] * signals)
        return signals - self.stationary_entries[:, None] * coefficients[self.labels]

    def filter(self, signals, steps):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('steps must be a nonnegative integer')
        result = self.remove_stationary(signals)
        for _ in range(steps):
            result = .5 * (result + torch.sparse.mm(self.operator, result))
        # In exact arithmetic the filter commutes with S. Project again to
        # limit accumulation of stationary roundoff, not to alter the target.
        return self.remove_stationary(result)

    def sketch(self, width=100, steps=4, seed=0):
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError('width must be a positive integer')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        probes = torch.randn(self.operator.shape[0], width, generator=generator,
                             dtype=self.operator.dtype).to(self.operator.device) / width ** .5
        return self.filter(probes, steps)
