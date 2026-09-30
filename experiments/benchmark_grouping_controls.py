"""Validation-only grouping controls from identical saved initial supports.

Single-edge search is held fixed. Spectral, diffusion, random, structural, and
ungrouped variants use the same candidate region and per-step proposal caps.
Actual proposal counts and times are recorded, not assumed equal. Timing is a
shared-device diagnostic, not a controlled end-to-end speed claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_gcn import load_dataset, MLPLinkPredictor
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel
from src.grouping_controls import GroupingControls
from src.partition import isolate_query_endpoints
from src.explainers.candidates import candidate_edge_mask
from src.explainers.support_refinement import candidate_groups, refine_support
from src.evaluation.interventions import evaluate_support


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier development results')
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Seeds must be unique')
    reference = json.loads(args.input.read_bytes())
    if reference['args']['query_split'] != 'val' or reference['args']['candidate_region'] != 'gcn-boundary':
        raise ValueError('Use normalization-boundary validation records')
    path = Path(reference['args']['checkpoint_dir']) / (reference['args']['dataset'] + '_gcn.pt')
    if sha(path) != reference['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from reference')
    state = torch.load(path, map_location=args.device, weights_only=False)
    config = state['config']
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    data = load_dataset(config['dataset'])
    for name, edges in state['edge_splits'].items():
        if hashlib.sha256(edges.cpu().numpy().tobytes()).hexdigest() != reference['split_sha256'][name]:
            raise ValueError('Split hash differs')
        setattr(data, name, edges)
    data.edge_index = data.train_pos_edge_index
    data = data.to(args.device)
    encoder = GCN(config['in_channels'], config['hidden_channels'], config['out_channels'], config['num_layers'])
    predictor = MLPLinkPredictor(config['out_channels'], config['hidden_channels'])
    encoder.load_state_dict(state['model_state_dict'])
    predictor.load_state_dict(state['predictor_state_dict'])
    model = LinkPredictionModel(encoder, predictor).to(args.device).eval()
    sources = [Path(__file__), Path('experiments/train_gcn.py'), *sorted(Path('src').rglob('*.py'))]
    source_hashes = {str(source): sha(source) for source in sources}

    def sync():
        if torch.device(args.device).type == 'cuda':
            torch.cuda.synchronize(args.device)

    sync()
    start = time.perf_counter()
    builder = GroupingControls(data.edge_index, data.num_nodes, getattr(data, 'edge_weight', None))
    sync()
    shared_setup = time.perf_counter() - start
    partitions, metadata = {'ungrouped': None}, {}
    settings = [('eigen', 0), ('normalized-edge', 0)]
    settings += [(policy, seed) for policy in ('diffusion', 'random') for seed in args.seeds]
    for policy, seed in settings:
        sync()
        start = time.perf_counter()
        partition, details = builder.partition(policy, seed=seed, width=100, steps=4, alpha=.75)
        sync()
        name = f'{policy}-{seed}'
        partitions[name] = partition
        metadata[name] = details | {'partition_seconds': time.perf_counter() - start,
                                   'partition_sha256': hashlib.sha256(json.dumps(partition).encode()).hexdigest()}
        print(f'Built {name}: {len(partition)} clusters', flush=True)
    originals = [r for r in reference['rows'] if r['method'] == 'Saliency-supportive']
    if not originals:
        raise ValueError('No saved initial supports')
    records, failures = [], []
    seen = set()
    for index, saved in enumerate(originals):
        a, b = saved['query']
        key = (a, b, saved['budget'])
        if key in seen:
            raise ValueError('Repeated query/budget')
        seen.add(key)
        pool = getattr(data, 'val_pos_edge_index' if saved['label'] else 'val_neg_edge_index')
        canonical = lambda e: e.min(0).values * data.num_nodes + e.max(0).values
        if not (canonical(pool) == min(a, b) * data.num_nodes + max(a, b)).any():
            raise ValueError('Query absent from validation split')
        mask = candidate_edge_mask(data.edge_index, data.num_nodes, [a, b], len(encoder.convs), 'gcn-boundary')
        candidates = torch.unique(canonical(data.edge_index[:, mask]))
        if len(candidates) != saved['candidate_edges']:
            raise ValueError('Candidate region differs')
        initial = Data(edge_index=torch.tensor(saved['support'], dtype=torch.long, device=args.device).reshape(-1, 2).t())
        before = evaluate_support(model, data, initial, a, b, args.device)
        if (before['support_edges'] != saved['effective_budget']
                or (before['full_logit'] > 0) != (saved['full_logit'] > 0)
                or not math.isclose(before['full_logit'], saved['full_logit'], rel_tol=1e-6, abs_tol=1e-5)):
            raise ValueError('Initial support budget or full prediction differs')
        results = {}
        names = list(partitions)
        names = names[index % len(names):] + names[:index % len(names)]
        for name in names:
            sync()
            start = time.perf_counter()
            partition = partitions[name]
            groups = (candidate_groups(candidates, isolate_query_endpoints(partition, a, b), data.num_nodes)
                      if partition is not None else None)
            support, trace = refine_support(model, data, a, b, candidates, initial,
                                            groups=groups, steps=8, additions=6, removals=3,
                                            batch_size=8, exchange_size=1)
            sync()
            elapsed = time.perf_counter() - start
            metrics = evaluate_support(model, data, support, a, b, args.device)
            valid = (metrics['support_edges'] == before['support_edges']
                     and metrics['necessity_flip'] >= before['necessity_flip']
                     and metrics['sufficiency_agreement'] >= before['sufficiency_agreement']
                     and metrics['necessity_confidence_drop'] >= before['necessity_confidence_drop'] - 1e-6
                     and max(0, metrics['sufficiency_confidence_drop']) <= max(0, before['sufficiency_confidence_drop']) + 1e-6)
            if not valid:
                failures.append({'query': [a, b], 'budget': saved['budget'], 'policy': name})
            sizes = torch.unique(groups, return_counts=True)[1] if groups is not None else None
            results[name] = {'support': support.edge_index.t().cpu().tolist(), 'metrics': metrics,
                             'trace': trace, 'seconds': elapsed,
                             'evaluated_proposals': sum(step['proposals'] for step in trace),
                             'candidate_groups': len(sizes) if sizes is not None else None,
                             'largest_group_fraction': float(sizes.max() / len(candidates)) if sizes is not None and len(sizes) else None}
        records.append({'query': [a, b], 'label': saved['label'], 'budget': saved['budget'],
                        'candidate_edges': len(candidates), 'initial_metrics': before, 'results': results})
        print(f'Completed {index + 1}/{len(originals)}: query=({a},{b}) budget={saved["budget"]}', flush=True)
    summaries = {}
    for name in metadata:
        outcomes = dict(wins=0, losses=0, ties=0, tradeoffs=0)
        for row in records:
            treated, control = row['results'][name]['metrics'], row['results']['ungrouped']['metrics']
            differences = [treated[m] - control[m] for m in ('necessity_flip', 'sufficiency_agreement')]
            outcome = ('ties' if differences == [0, 0] else 'wins' if min(differences) >= 0
                       else 'losses' if max(differences) <= 0 else 'tradeoffs')
            outcomes[outcome] += 1
        summaries[name] = {'binary_pareto_vs_ungrouped': outcomes,
                           'identical_supports': sum(sorted(row['results'][name]['support']) == sorted(row['results']['ungrouped']['support']) for row in records),
                           'mean_proposals': statistics.mean(row['results'][name]['evaluated_proposals'] for row in records),
                           'mean_groups': statistics.mean(row['results'][name]['candidate_groups'] for row in records)}
    report = {'benchmark': 'single-search-grouping-controls-v1', 'input': str(args.input),
              'input_sha256': sha(args.input), 'checkpoint_sha256': sha(path), 'source_sha256': source_hashes,
              'split_sha256': reference['split_sha256'], 'dataset': config['dataset'], 'training_seed': config['seed'],
              'device': args.device, 'torch': torch.__version__,
              'config': {'random_seeds': args.seeds, 'width': 100, 'diffusion_steps': 4, 'alpha': .75,
                         'search_steps': 8, 'additions': 6, 'removals': 3, 'batch_size': 8, 'exchange_size': 1},
              'scope': __doc__, 'shared_setup_seconds': shared_setup, 'partitions': metadata,
              'summaries': summaries, 'failures': failures, 'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
    if failures:
        raise RuntimeError('Recorded nonregression failures; inspect output')


if __name__ == '__main__':
    main()
