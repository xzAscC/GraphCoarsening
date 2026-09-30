"""Replay the legacy sampler on saved splits, not unrecorded historical draws.

This reconstructs Python-seeded PyG sampling under the installed implementation.
Historical training did not log negative edges or initial RNG states, so these
counts are a conditional replay and must not be called an observed training log.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import random
import sys

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import torch_geometric.utils._negative_sampling as sampling_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_gcn import load_dataset
from src.negative_sampling import SPLIT_NAMES, training_negative_exclusion, undirected_keys


def audit(path, neg_ratio):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config, summary = checkpoint['config'], checkpoint['training_summary']
    if 'negative_sampling_protocol' in config:
        raise ValueError('This audit targets legacy, train-positive-only sampling')
    torch.manual_seed(config['seed'])
    current = load_dataset(config['dataset'])
    splits = checkpoint['edge_splits']
    if any(not torch.equal(getattr(current, name), splits[name]) for name in SPLIT_NAMES):
        raise ValueError('Reconstructed dataset splits differ from the saved checkpoint')
    data = Data(num_nodes=current.num_nodes, **splits)
    training_negative_exclusion(data)  # Verify split labels/disjointness first.
    n = data.num_nodes
    reserved = {name: set(undirected_keys(splits[name], n).tolist())
                for name in SPLIT_NAMES if name != 'train_pos_edge_index'}
    touched = {name: set() for name in reserved}
    epochs = config['epochs']
    selected = summary['selected_epoch']
    if not 1 <= selected <= epochs:
        raise ValueError('Invalid selected epoch')
    random.seed(config['seed'])
    rows, at_selection = [], None
    count = int(splits['train_pos_edge_index'].size(1) * neg_ratio)
    for epoch in range(1, epochs + 1):
        negatives = negative_sampling(splits['train_pos_edge_index'], num_nodes=n,
                                      num_neg_samples=count)
        keys = undirected_keys(negatives, n).tolist()
        row = {'epoch': epoch, 'sampled_stored_entries': len(keys)}
        for name, forbidden in reserved.items():
            hits = [key for key in keys if key in forbidden]
            touched[name].update(hits)
            row[name + '_collision_entries'] = len(hits)
        rows.append(row)
        if epoch == selected:
            at_selection = {name: len(values) for name, values in touched.items()}
    return {'checkpoint': str(path), 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'dataset': config['dataset'], 'seed': config['seed'], 'num_nodes': n,
            'epochs': epochs, 'selected_epoch': selected, 'negative_ratio': neg_ratio,
            'recorded_pyg_version': summary.get('pyg'),
            'reserved_unique_pairs': {name: len(values) for name, values in reserved.items()},
            'unique_reserved_pairs_sampled_by_selected_epoch': at_selection,
            'unique_reserved_pairs_sampled_by_training_end': {name: len(values) for name, values in touched.items()},
            'split_sha256': {name: hashlib.sha256(splits[name].numpy().tobytes()).hexdigest() for name in SPLIT_NAMES},
            'epochs_replayed': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', type=Path, nargs='+')
    parser.add_argument('--neg-ratio', type=float, default=1.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous audit outputs')
    if args.neg_ratio <= 0:
        raise ValueError('Positive negative-sampling ratio required')
    result = {
        'audit': 'legacy-negative-sampling-conditional-replay-v1',
        'scope': 'Replay using saved splits and a fresh Python RNG seeded with the recorded training seed. Original draws were not logged. Exact historical equivalence additionally requires the same sampling implementation and no intervening Python RNG consumption.',
        'python': platform.python_version(), 'torch': torch.__version__, 'pyg': torch_geometric.__version__,
        'sampler_source_sha256': hashlib.sha256(inspect.getsource(sampling_module).encode()).hexdigest(),
        'runs': [audit(path, args.neg_ratio) for path in args.checkpoints],
    }
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
    for run in result['runs']:
        print(run['dataset'], run['seed'], run['unique_reserved_pairs_sampled_by_selected_epoch'])


if __name__ == '__main__':
    main()
