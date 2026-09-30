"""Count exact unit-eigenvalue multiplicity of unit-weight training graphs.

Each connected component supplies a stationary mode for normalized A+I.
If k is smaller than the component count, an exact largest-k eigenspace is
not uniquely specified. This diagnoses a cutoff ambiguity, not its measured
effect on explanation quality or a failure of every computed eigenpair.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import torch
from torch_geometric.datasets import Planetoid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.spectral import compute_normalized_adjacency


def audit(path, k):
    state = torch.load(path, map_location='cpu', weights_only=False)
    dataset = state['config']['dataset']
    if dataset not in ('Cora', 'Citeseer', 'PubMed'):
        raise ValueError('This audit is limited to the unit-weight Planetoid training protocol')
    n = Planetoid(root=f'data/{dataset}', name=dataset)[0].num_nodes
    edges = state['edge_splits']['train_pos_edge_index']
    adjacency = sp.csr_matrix((np.ones(edges.size(1)), edges.numpy()), shape=(n, n))
    if (adjacency != adjacency.T).nnz or adjacency.diagonal().any():
        raise ValueError('Expected symmetric training edges without stored self-loops')
    count, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels)
    degree = torch.from_numpy(np.asarray(adjacency.sum(1)).reshape(-1)).double() + 1
    operator = compute_normalized_adjacency(edges, n, torch.ones(edges.size(1), dtype=torch.float64))
    stationary = degree.sqrt().reshape(-1, 1)
    residual = (torch.sparse.mm(operator, stationary) - stationary).abs().max().item()
    return {'checkpoint': str(path), 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'dataset': dataset, 'seed': state['config']['seed'], 'nodes': n,
            'training_split_sha256': hashlib.sha256(edges.numpy().tobytes()).hexdigest(),
            'connected_components': count, 'isolates': int((sizes == 1).sum()),
            'largest_component_nodes': int(sizes.max()), 'requested_k': k,
            'exact_unit_eigenvalue_multiplicity': count,
            'cutoff_inside_stationary_eigenspace': k < count,
            'global_stationary_vector_residual_linf': residual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', type=Path, nargs='+')
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.k < 1 or args.output.exists():
        raise ValueError('Positive k and a new output path are required')
    result = {'audit': 'stationary-multiplicity-v1',
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), Path('src/spectral.py'))},
              'scope': 'Exact component-count obstruction for normalized A+I on unit-weight saved training graphs. A small eigenpair residual alone does not certify that a partial solver recovered the algebraically largest k modes.',
              'runs': [audit(path, args.k) for path in args.checkpoints]}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
    for row in result['runs']:
        print(row['dataset'], row['seed'], 'components', row['connected_components'],
              'cutoff ambiguous', row['cutoff_inside_stationary_eigenspace'])


if __name__ == '__main__':
    main()
