"""Split-disjoint negatives for static undirected link benchmarks.

Known positive pairs across the fixed benchmark split, and reserved validation
and test negative pairs, are forbidden in training negative supervision. This
is split construction, not permission to use held-out edges for propagation or
positive training labels. It is not a temporal or open-world sampling policy.
"""
import torch
from torch_geometric.utils import negative_sampling


SPLIT_NAMES = ('train_pos_edge_index', 'val_pos_edge_index', 'val_neg_edge_index',
               'test_pos_edge_index', 'test_neg_edge_index')
NEGATIVE_PROTOCOL = 'static-undirected-split-disjoint-negatives-v1'


def undirected_keys(edges, num_nodes):
    if edges.dtype != torch.long or edges.ndim != 2 or edges.size(0) != 2:
        raise ValueError('Expected a long [2, E] edge index')
    if edges.numel() and ((edges < 0).any() or (edges >= num_nodes).any()
                          or (edges[0] == edges[1]).any()):
        raise ValueError('Invalid node index or explicit self-loop')
    return edges.min(0).values * num_nodes + edges.max(0).values


def training_negative_exclusion(data):
    """Validate split disjointness and return both orientations of all exclusions."""
    n = data.num_nodes
    if n is None or n < 2:
        raise ValueError('At least two explicitly counted vertices required')
    keys = {}
    for name in SPLIT_NAMES:
        edges = getattr(data, name, None)
        if edges is None:
            raise ValueError(f'Missing split: {name}')
        keys[name] = torch.unique(undirected_keys(edges.detach().cpu(), n))
    names = list(keys)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if torch.isin(keys[left], keys[right]).any():
                raise ValueError(f'Overlapping undirected pairs: {left}, {right}')
    excluded = torch.unique(torch.cat(list(keys.values())))
    upper = torch.stack((excluded // n, excluded % n))
    return torch.cat((upper, upper.flip(0)), dim=1)


def sample_training_negatives(exclusion, num_nodes, count):
    """Keep the requested stored-edge sample count, failing on a shortfall.

Samples are directed entries for compatibility with the training loss. The
exclusion contains both orientations, so held-out undirected pairs cannot leak.
"""
    if count < 1:
        raise ValueError('A positive number of training negatives is required')
    edges = negative_sampling(exclusion, num_nodes=num_nodes,
                              num_neg_samples=count, method='sparse')
    if edges.size(1) != count:
        raise ValueError(f'Insufficient training negatives: requested {count}, got {edges.size(1)}')
    if torch.isin(undirected_keys(edges, num_nodes),
                  undirected_keys(exclusion, num_nodes)).any():
        raise AssertionError('Negative sampler returned an excluded pair')
    return edges
