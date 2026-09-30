"""Replay saved validation supports on full and boundary-compacted GCN graphs.

This checks numerical agreement, not support-search trajectory equivalence.
Optional paired timings cover forward interventions, not end-to-end search.
Model weights and source manifests are recorded; no optimization occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_gcn import load_dataset, MLPLinkPredictor
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel
from src.explainers.local_gcn import compact_gcn_query
from src.explainers.group_interventions import group_deletion_logits


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def benchmark_pair(full, local, device, repeats, warmups=2):
    """Alternate evaluation order with synchronization and per-process peaks."""
    if repeats < 2 or repeats % 2 or warmups < 0:
        raise ValueError('Use a positive even repeat count and nonnegative warmups')
    functions = {'full': full, 'local': local}
    for _ in range(warmups):
        full()
        local()
    synchronize(device)
    records = []
    for iteration in range(repeats):
        order = ('full', 'local') if iteration % 2 == 0 else ('local', 'full')
        record = {'order': list(order)}
        for name in order:
            synchronize(device)
            cuda = torch.device(device).type == 'cuda'
            baseline = torch.cuda.memory_allocated(device) if cuda else None
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            functions[name]()
            synchronize(device)
            record[name] = {'wall_seconds': time.perf_counter() - start,
                            'peak_extra_allocated_bytes':
                                torch.cuda.max_memory_allocated(device) - baseline if cuda else None}
        records.append(record)
    return records


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', default='Swap-gradient')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--timing-repeats', type=int, default=0,
                        help='Zero disables timing; otherwise use a positive even count')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous replay outputs')
    if args.timing_repeats < 0 or args.timing_repeats % 2:
        raise ValueError('Timing repeat count must be zero or positive even')
    saved = json.loads(args.input.read_bytes())
    if saved['args']['query_split'] != 'val' or saved['args']['candidate_region'] != 'gcn-boundary':
        raise ValueError('Use boundary-protocol validation records')
    checkpoint = Path(saved['args']['checkpoint_dir']) / (saved['args']['dataset'] + '_gcn.pt')
    if sha(checkpoint) != saved['checkpoint_sha256']:
        raise ValueError('Checkpoint hash differs')
    state = torch.load(checkpoint, map_location=args.device, weights_only=False)
    config = state['config']
    torch.manual_seed(config['seed'])
    data = load_dataset(config['dataset'])
    for name, edges in state['edge_splits'].items():
        if hashlib.sha256(edges.cpu().numpy().tobytes()).hexdigest() != saved['split_sha256'][name]:
            raise ValueError('Saved split hash differs')
        setattr(data, name, edges)
    data.edge_index = data.train_pos_edge_index
    data = data.to(args.device)
    encoder = GCN(config['in_channels'], config['hidden_channels'], config['out_channels'], config['num_layers'])
    predictor = MLPLinkPredictor(config['out_channels'], config['hidden_channels'])
    encoder.load_state_dict(state['model_state_dict'])
    predictor.load_state_dict(state['predictor_state_dict'])
    model = LinkPredictionModel(encoder, predictor).to(args.device).eval()
    if len(encoder.convs) != config['num_layers']:
        raise ValueError('Declared and actual GCN layer counts differ')
    sources = [Path(__file__), Path('experiments/train_gcn.py'),
               *sorted(Path('src').rglob('*.py'))]
    source_hashes = {str(path): sha(path) for path in sources}
    rows = [row for row in saved['rows'] if row['method'] == args.method]
    if not rows:
        raise ValueError('No rows for requested method')
    queries = list(dict.fromkeys(tuple(row['query']) for row in rows))
    checks, failures = [], []
    for query in queries:
        selected = [row for row in rows if tuple(row['query']) == query]
        synchronize(args.device)
        construction_start = time.perf_counter()
        region = compact_gcn_query(data, query, len(encoder.convs))
        synchronize(args.device)
        construction_seconds = time.perf_counter() - construction_start
        groups = [torch.tensor([min(a, b) * data.num_nodes + max(a, b) for a, b in row['support']],
                               dtype=torch.long, device=args.device) for row in selected]
        synchronize(args.device)
        mapping_start = time.perf_counter()
        local_groups = [region.map_keys(keys) for keys in groups]
        synchronize(args.device)
        mapping_seconds = time.perf_counter() - mapping_start
        for original, local in zip(groups, local_groups):
            torch.testing.assert_close(region.restore_keys(local), original.unique(sorted=True))
        a, b = query
        la, lb = region.targets.flatten().tolist()
        target = torch.tensor([[a], [b]], device=args.device)
        full_base = model(data.x, data.edge_index, target, edge_weight=getattr(data, 'edge_weight', None)).reshape(-1)
        local_base = model(region.data.x, region.data.edge_index, region.targets,
                           edge_weight=getattr(region.data, 'edge_weight', None)).reshape(-1)
        values = {'full_logit': (full_base.expand(len(groups)), local_base.expand(len(groups)))}
        for retain, name in ((True, 'retained_logit'), (False, 'removed_logit')):
            values[name] = (
                group_deletion_logits(model, data, a, b, groups, args.batch_size, retain=retain),
                group_deletion_logits(model, region.data, la, lb, local_groups, args.batch_size, retain=retain))
        differences = {}
        for name, (full, local) in values.items():
            reference = full.new_tensor([row[name] for row in selected])
            for description, left, right in (('full_vs_local', full, local), ('saved_vs_replayed', reference, full)):
                valid = (torch.isfinite(left) & torch.isfinite(right)
                         & torch.isclose(left, right, rtol=1e-5, atol=2e-5)
                         & ((left > 0) == (right > 0)))
                differences[description + '_' + name] = (left - right).abs().max().item()
                if not valid.all():
                    failures.append({'query': query, 'comparison': description, 'prediction': name,
                                     'budgets': [selected[i]['budget'] for i in (~valid).nonzero().flatten().tolist()]})
        timing = None
        if args.timing_repeats:
            def evaluate(graph, left, right, supports):
                for retain in (True, False):
                    group_deletion_logits(model, graph, left, right, supports,
                                          args.batch_size, retain=retain)
            timing = benchmark_pair(lambda: evaluate(data, a, b, groups),
                                    lambda: evaluate(region.data, la, lb, local_groups),
                                    args.device, args.timing_repeats)
        checks.append({'query': query, 'support_count': len(groups),
                       'region_construction_seconds': construction_seconds,
                       'support_mapping_seconds': mapping_seconds,
                       'paired_timing': timing,
                       'original_nodes': data.num_nodes, 'local_nodes': region.data.num_nodes,
                       'original_stored_edges': data.edge_index.size(1),
                       'local_stored_edges': region.data.edge_index.size(1),
                       'max_absolute_differences': differences})
    result = {'audit': 'boundary-compacted-gcn-replay-v1', 'input': str(args.input),
              'input_sha256': sha(args.input), 'checkpoint_sha256': sha(checkpoint),
              'args': vars(args) | {'input': str(args.input), 'output': str(args.output)},
              'source_sha256': source_hashes, 'torch': torch.__version__,
              'threads': torch.get_num_threads(),
              'gpu': torch.cuda.get_device_name(args.device) if torch.device(args.device).type == 'cuda' else None,
              'tolerance': {'relative': 1e-5, 'absolute': 2e-5, 'binary_decisions_must_match': True},
              'scope': 'Saved-support forward replay only; does not prove adaptive search trajectories remain identical.',
              'timing_scope': 'Optional synchronized paired retain/delete forward timings with two warmups and alternating order. Includes graph-copy construction within interventions; excludes spectral preprocessing, ranking, gradients, search and model loading. Region construction and support remapping are recorded separately. Peak extra allocated memory is process-local, not whole-device usage or allocator reservations. Shared-device timings are diagnostic.',
              'queries': len(queries), 'supports': len(rows), 'failures': failures, 'checks': checks}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
    print(f'Replayed {len(rows)} supports on {len(queries)} queries; failures={len(failures)}')
    if failures:
        raise RuntimeError('Numerical agreement audit failed; inspect saved records')


if __name__ == '__main__':
    main()
