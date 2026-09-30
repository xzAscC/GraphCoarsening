"""Original-edge candidate regions for local message-passing explanations."""
import torch
from torch_geometric.utils import k_hop_subgraph


def candidate_edge_mask(edge_index, num_nodes, endpoints, hops, mode='induced'):
    """Return stored-edge membership, preserving order and both orientations.

    induced: edges with both endpoints in the K-hop node region.
    gcn-boundary: edges incident to that region, including its degree boundary.
    On undirected graphs, the latter covers symmetric degree-normalized
    K-layer GCN dependencies for edge deletions with fixed features, local
    pointwise nonlinearities, and endpoint-only decoding. This
    is not a locality certificate for arbitrary GNNs or graph-global layers.
    """
    if hops < 0 or mode not in {'induced', 'gcn-boundary'}:
        raise ValueError('Unknown or invalid candidate region')
    subset, _, _, induced = k_hop_subgraph(endpoints, hops, edge_index, num_nodes=num_nodes)
    if mode == 'induced':
        return induced
    inside = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    inside[subset] = True
    return inside[edge_index[0]] | inside[edge_index[1]]
