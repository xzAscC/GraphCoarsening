"""Experimental Galerkin inference for a frozen, local normalized GCN.

For normalized cluster indicators P, use Xc=P^T X, B=P^T A_hat P, and
project each bias as (P^T 1)b = sqrt(cluster_size)*b. Do NOT normalize B
again or add coarse self-loops: fine self-loops are already inside A_hat.
Singleton query endpoints make the coarse endpoint decoder well defined.

Singleton partitions recover the fine model. Nontrivial partitions are only
exact under additional invariant-subspace/feature conditions; in general this
is approximate inference, not an explanation evaluator or fidelity guarantee.
"""
import torch
from torch_geometric.utils import coalesce, is_undirected

from src.spectral import compute_normalized_adjacency
from src.explainers.local_gcn import validate_local_gcn_model


class ProjectedGCN:
    def __init__(self, model, data, membership, targets):
        self.layers = validate_local_gcn_model(model)
        if any(conv.improved for conv in model.encoder.convs):
            raise ValueError('Only unit implicit self-loops are supported')
        if data.x.dtype not in (torch.float32, torch.float64):
            raise ValueError('Expected float32 or float64 features')
        n, device = data.num_nodes, data.x.device
        edges = data.edge_index
        if (edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2
                or edges.device != device or (edges < 0).any() or (edges >= n).any()
                or (edges[0] == edges[1]).any()):
            raise ValueError('Use valid loop-free stored edges and unit implicit loops')
        weights = getattr(data, 'edge_weight', None)
        weights = data.x.new_ones(edges.shape[1]) if weights is None else weights
        if (weights.shape != (edges.shape[1],) or weights.device != device or weights.dtype != data.x.dtype
                or not torch.isfinite(weights).all() or (weights < 0).any()):
            raise ValueError('Invalid edge weights')
        checked_edges, checked_weights = coalesce(edges, weights, num_nodes=n)
        if not is_undirected(checked_edges, edge_attr=checked_weights, num_nodes=n):
            raise ValueError('Undirected weighted adjacency required')
        if (membership.shape != (n,) or membership.dtype != torch.long
                or membership.device != device or (membership < 0).any()):
            raise ValueError('One nonnegative cluster label per node required')
        targets = torch.as_tensor(targets, dtype=torch.long, device=device).reshape(-1)
        if targets.shape != (2,) or (targets < 0).any() or (targets >= n).any():
            raise ValueError('Two valid endpoint indices required')
        _, self.membership = torch.unique(membership, sorted=True, return_inverse=True)
        self.counts = torch.bincount(self.membership)
        if (self.counts[self.membership[targets]] != 1).any():
            raise ValueError('Query endpoints must be singleton clusters')
        self.model, self.data, self.weights = model, data, weights
        self.targets = self.membership[targets].reshape(2, 1)
        self.num_coarse_nodes = len(self.counts)
        self.singleton = self.num_coarse_nodes == n and torch.equal(self.membership, torch.arange(n, device=device))
        self.scale = self.counts.to(data.x.dtype).sqrt()
        self.node_scale = self.scale[self.membership].reciprocal()
        self.keys = edges.min(0).values * n + edges.max(0).values
        self.unique_keys = torch.unique(self.keys)
        if self.singleton:
            self.features = data.x
        else:
            self.features = data.x.new_zeros((self.num_coarse_nodes, data.x.shape[1]))
            self.features.index_add_(0, self.membership, data.x * self.node_scale[:, None])

    def edge_mask(self, support=None, *, retain=False):
        if support is None:
            return torch.ones_like(self.keys, dtype=torch.bool)
        if (support.ndim != 1 or support.dtype != torch.long or support.device != self.keys.device
                or not torch.isin(support, self.unique_keys).all()):
            raise ValueError('Support must contain original canonical edge keys')
        present = torch.isin(self.keys, support)
        return present if retain else ~present

    def fine_operator(self, support=None, *, retain=False):
        mask = self.edge_mask(support, retain=retain)
        return compute_normalized_adjacency(self.data.edge_index[:, mask], self.data.num_nodes, self.weights[mask])

    def operator(self, support=None, *, retain=False):
        fine = self.fine_operator(support, retain=retain)
        if self.singleton:
            return fine
        row, col = fine.indices()
        values = fine.values() * self.node_scale[row] * self.node_scale[col]
        coarse_edges = torch.stack((self.membership[row], self.membership[col]))
        return torch.sparse_coo_tensor(coarse_edges, values,
                                      (self.num_coarse_nodes, self.num_coarse_nodes)).coalesce()

    def __call__(self, support=None, *, retain=False):
        operator = self.operator(support, retain=retain)
        features = self.features
        for index, conv in enumerate(self.model.encoder.convs):
            features = torch.sparse.mm(operator, conv.lin(features))
            if conv.bias is not None:
                features = features + self.scale[:, None] * conv.bias
            if index + 1 < self.layers:
                features = features.relu()
        return self.model.predictor(features, self.targets).reshape(-1)

    def relative_feature_residual(self):
        lifted = self.features[self.membership] * self.node_scale[:, None]
        denominator = torch.linalg.vector_norm(self.data.x)
        residual = torch.linalg.vector_norm(self.data.x - lifted)
        return float(residual / denominator) if denominator > 0 else float(residual)

    def operator_residual_fro(self, support=None, *, retain=False):
        """Numerical full-operator residual, not a selected-eigenvalue statistic.

        For orthonormal P and B=P^T A P, ||AP-PB||_F^2=||AP||_F^2-||B||_F^2.
        Use double precision to reduce cancellation. This is a diagnostic, not
        a floating-point certificate or a prediction-error guarantee.
        """
        fine = self.fine_operator(support, retain=retain)
        row, col = fine.indices()
        scale = self.counts.double().rsqrt()[self.membership]
        values = fine.values().double()
        AP = torch.sparse_coo_tensor(torch.stack((row, self.membership[col])), values * scale[col],
                                     (self.data.num_nodes, self.num_coarse_nodes)).coalesce()
        B = torch.sparse_coo_tensor(torch.stack((self.membership[row], self.membership[col])),
                                    values * scale[row] * scale[col],
                                    (self.num_coarse_nodes, self.num_coarse_nodes)).coalesce()
        return float((AP.values().square().sum() - B.values().square().sum()).clamp_min(0).sqrt())
