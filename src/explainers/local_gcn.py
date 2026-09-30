"""Compact inference regions for undirected, degree-normalized local GCNs.

This is an inference optimization, not a new support definition. For K layers,
keep every edge incident to the original K-hop endpoint region, then relabel
its incident nodes and the query endpoints. The extra boundary preserves the
degrees needed by messages reaching the endpoints. Deletions cannot introduce
new paths. Ordinary K-hop induced subgraphs are insufficient.

Callers must use K equal to the actual number of message-passing layers,
fixed positive implicit self-loops, local pointwise operations, evaluation
mode, uncached normalization and endpoint-only decoding. This utility does
not certify arbitrary models, graph-global normalization, or edge additions.
The original graph remains authoritative for explanation-size metrics.
"""
from dataclasses import dataclass

import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, is_undirected

from .candidates import candidate_edge_mask


@dataclass
class GCNQueryRegion:
    data: Data
    targets: torch.Tensor
    original_num_nodes: int
    original_nodes: torch.Tensor
    original_edge_mask: torch.Tensor
    original_keys: torch.Tensor
    original_to_local: torch.Tensor

    def map_keys(self, keys):
        """Map a support of original canonical edge keys, rejecting outsiders."""
        keys = keys.to(self.original_nodes.device)
        if keys.dtype != torch.long or keys.ndim != 1:
            raise ValueError('Support keys must be a one-dimensional long tensor')
        if not torch.isin(keys, self.original_keys).all():
            raise ValueError('Support contains an edge outside the original query region')
        a, b = keys // self.original_num_nodes, keys % self.original_num_nodes
        mapped = self.original_to_local[a] * self.data.num_nodes + self.original_to_local[b]
        return torch.unique(mapped, sorted=True)

    def restore_keys(self, keys):
        """Restore valid compact support keys to original node identities."""
        keys = keys.to(self.original_nodes.device)
        if keys.dtype != torch.long or keys.ndim != 1:
            raise ValueError('Support keys must be a one-dimensional long tensor')
        n = self.data.num_nodes
        if ((keys < 0) | (keys >= n * n)).any():
            raise ValueError('Invalid compact edge key')
        a, b = keys // n, keys % n
        original = self.original_nodes[a] * self.original_num_nodes + self.original_nodes[b]
        if not torch.isin(original, self.original_keys).all():
            raise ValueError('Compact support is not an original query-region edge')
        return torch.unique(original, sorted=True)


def compact_gcn_query(data, endpoints, hops):
    """Construct a reusable boundary-preserving region before interventions.

    Stored entry order, duplicate entries, weights and feature values are
    preserved. Isolated query endpoints are retained even for empty regions.
    Only x, edge_index, edge_weight and explicit identity maps are copied;
    dataset split labels and unrelated metadata are deliberately not remapped.
    """
    if not isinstance(hops, int) or isinstance(hops, bool) or hops < 1:
        raise ValueError('hops must be the positive integer GCN layer count')
    n = data.x.size(0)
    target = torch.as_tensor(endpoints, dtype=torch.long, device=data.x.device)
    if target.shape != (2,) or ((target < 0) | (target >= n)).any():
        raise ValueError('Expected two valid query endpoints')
    edge_index = data.edge_index
    if (edge_index.dtype != torch.long or edge_index.ndim != 2
            or edge_index.size(0) != 2 or edge_index.device != data.x.device
            or ((edge_index < 0) | (edge_index >= n)).any()):
        raise ValueError('Invalid original edge index')
    if (edge_index[0] == edge_index[1]).any():
        raise ValueError('Use fixed implicit self-loops, not stored self-loops')
    weight = getattr(data, 'edge_weight', None)
    if weight is not None and (weight.shape != (edge_index.size(1),)
                               or weight.device != data.x.device
                               or not torch.isfinite(weight).all() or (weight < 0).any()):
        raise ValueError('Expected finite nonnegative original edge weights')
    # Validate aggregated adjacency symmetry, not the arbitrary ordering of
    # duplicate entries. Preserve the original entries for actual inference.
    checked_edges, checked_weights = coalesce(edge_index, weight, num_nodes=n)
    if not is_undirected(checked_edges, edge_attr=checked_weights, num_nodes=n):
        raise ValueError('Boundary compaction requires an undirected weighted graph')
    mask = candidate_edge_mask(edge_index, n, target, hops, 'gcn-boundary')
    edges = edge_index[:, mask]
    nodes = torch.unique(torch.cat((target, edges.reshape(-1))), sorted=True)
    inverse = torch.full((n,), -1, dtype=torch.long, device=data.x.device)
    inverse[nodes] = torch.arange(nodes.numel(), device=data.x.device)
    local = Data(x=data.x[nodes], edge_index=inverse[edges],
                 original_node_indices=nodes, num_nodes=nodes.numel())
    if weight is not None:
        local.edge_weight = weight[mask]
    keys = torch.unique(edges.min(0).values * n + edges.max(0).values)
    return GCNQueryRegion(local, inverse[target].reshape(2, 1), n, nodes, mask, keys, inverse)
