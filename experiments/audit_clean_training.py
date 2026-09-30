"""Verify logged negative-sample hashes and split-disjoint training metadata.

Unlike the legacy sampler audit, this matches every replayed sample to a hash
recorded during training. It does not rerun optimization or prove model quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_gcn import load_dataset
from src.negative_sampling import (NEGATIVE_PROTOCOL, SPLIT_NAMES,
                                   sample_training_negatives, training_negative_exclusion)


def verify_samples(data, records, seed, ratio):
    exclusion = training_negative_exclusion(data)
    count = int(data.train_pos_edge_index.size(1) * ratio)
    previous_rng = random.getstate()
    try:
        random.seed(seed)
        for epoch, record in enumerate(records, 1):
            sampled = sample_training_negatives(exclusion, data.num_nodes, count)
            digest = hashlib.sha256(sampled.numpy().tobytes()).hexdigest()
            if (record['sha256'] != digest or record['stored_negative_entries'] != count
                    or record['excluded_pair_collisions'] != 0):
                raise AssertionError(f'Negative-sample log does not match replay at epoch {epoch}')
    finally:
        random.setstate(previous_rng)
    return hashlib.sha256(exclusion.numpy().tobytes()).hexdigest()


def audit(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config, summary = checkpoint['config'], checkpoint['training_summary']
    if (config.get('negative_sampling_protocol') != NEGATIVE_PROTOCOL
            or summary.get('negative_sampling_protocol') != NEGATIVE_PROTOCOL):
        raise ValueError('Expected corrected static split-disjoint protocol')
    if summary['test_evaluated'] or summary['test_auc'] is not None:
        raise ValueError('This development audit requires skipped test evaluation')
    torch.manual_seed(config['seed'])
    data = load_dataset(config['dataset'])
    for name in SPLIT_NAMES:
        saved = checkpoint['edge_splits'][name]
        if not torch.equal(saved, getattr(data, name)):
            raise AssertionError('Saved split differs from seeded reconstruction')
        if hashlib.sha256(saved.numpy().tobytes()).hexdigest() != summary['split_sha256'][name]:
            raise AssertionError('Split hash mismatch')
    records = summary['negative_sampling_audit']
    if len(records) != config['epochs']:
        raise AssertionError('Missing per-epoch negative-sample records')
    exclusion_hash = verify_samples(data, records, config['seed'], summary['negative_ratio'])
    if exclusion_hash != summary['negative_exclusion_sha256']:
        raise AssertionError('Exclusion set hash mismatch')
    for name, digest in summary['source_sha256_at_start'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f'Training source changed: {name}')
    selected = next(row for row in summary['history'] if row['epoch'] == summary['selected_epoch'])
    if summary['args']['select_best']:
        if selected['validation_auc'] != max(row['validation_auc'] for row in summary['history']):
            raise AssertionError('Selected epoch is not validation-best')
    return {'checkpoint': str(path), 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'dataset': config['dataset'], 'seed': config['seed'],
            'verified_negative_sample_epochs': len(records), 'excluded_pair_collisions': 0,
            'negative_sampling_protocol': NEGATIVE_PROTOCOL, 'test_evaluated': False,
            'selected_epoch': selected['epoch'], 'selected_validation_auc': selected['validation_auc'],
            'split_sha256': summary['split_sha256'], 'source_hashes_match': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous audit outputs')
    result = {'audit': 'logged-negative-sample-replay-v1',
              'scope': 'Matches replayed samples to hashes recorded at training time; checks disjoint pairs, source and split hashes, and validation-only checkpoint selection. Does not rerun model optimization.',
              'runs': [audit(path) for path in args.checkpoints]}
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
    for run in result['runs']:
        print(run['dataset'], run['seed'], 'verified epochs', run['verified_negative_sample_epochs'],
              'validation AUC', run['selected_validation_auc'])


if __name__ == '__main__':
    main()
