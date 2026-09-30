"""Fixed coverage-union study with small-pool and equal-cap controls.

The union combines global top-six and diverse top-six additions, then fills
duplicates by global rank to twelve. Compare to global twelve, not just six.
Additional queries, when requested, exclude every query in the input record
and are sampled from validation only. No outcomes tune the fixed configuration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_gcn import load_dataset, MLPLinkPredictor
from experiments.run_support_benchmark import rank_support
from experiments.audit_support_refinement import paired_policy_comparison
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel
from src.grouping_controls import GroupingControls
from src.partition import isolate_query_endpoints
from src.explainers.baselines import SaliencyExplainer
from src.explainers.candidates import candidate_edge_mask
from src.explainers.support_refinement import candidate_groups, refine_support
from src.evaluation.interventions import PROTOCOL, evaluate_support


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--additional-queries-per-class', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior measurements')
    if args.additional_queries_per_class < 0:
        raise ValueError('Nonnegative query count required')
    reference = json.loads(args.input.read_bytes())
    if reference['args']['query_split'] != 'val' or reference['args']['candidate_region'] != 'gcn-boundary':
        raise ValueError('Use normalization-boundary validation records')
    path = Path(reference['args']['checkpoint_dir']) / (reference['args']['dataset'] + '_gcn.pt')
    if sha(path) != reference['checkpoint_sha256']:
        raise ValueError('Checkpoint mismatch')
    state = torch.load(path, map_location=args.device, weights_only=False)
    config = state['config']
    torch.manual_seed(config['seed'])
    data = load_dataset(config['dataset'])
    for name, edges in state['edge_splits'].items():
        if hashlib.sha256(edges.cpu().numpy().tobytes()).hexdigest() != reference['split_sha256'][name]:
            raise ValueError('Split mismatch')
        setattr(data, name, edges)
    data.edge_index = data.train_pos_edge_index
    data = data.to(args.device)
    encoder = GCN(config['in_channels'], config['hidden_channels'], config['out_channels'], config['num_layers'])
    predictor = MLPLinkPredictor(config['out_channels'], config['hidden_channels'])
    encoder.load_state_dict(state['model_state_dict'])
    predictor.load_state_dict(state['predictor_state_dict'])
    model = LinkPredictionModel(encoder, predictor).to(args.device).eval()
    sources = [Path(__file__), Path('experiments/train_gcn.py'), Path('experiments/run_support_benchmark.py'),
               Path('experiments/audit_support_refinement.py'), *sorted(Path('src').rglob('*.py'))]
    source_hashes = {str(p): sha(p) for p in sources}
    n = data.num_nodes
    canonical = lambda e: e.min(0).values * n + e.max(0).values
    saved = {(tuple(r['query']), r['budget']): r for r in reference['rows'] if r['method'] == 'Saliency-supportive'}
    queries = list(dict.fromkeys((tuple(r['query']), r['label']) for r in reference['rows']))
    excluded = sorted({min(q) * n + max(q) for q, _ in queries})
    if args.additional_queries_per_class:
        queries = []
        rng = np.random.default_rng(20260930)
        for label in (1, 0):
            pool = getattr(data, 'val_pos_edge_index' if label else 'val_neg_edge_index')
            keys = canonical(pool).cpu().numpy()
            available = np.flatnonzero(~np.isin(keys, excluded))
            if len(available) < args.additional_queries_per_class:
                raise ValueError('Insufficient unused validation queries')
            ids = rng.choice(available, args.additional_queries_per_class, replace=False)
            queries.extend((tuple(map(int, pool[:, int(i)].tolist())), label) for i in ids)
    if len({min(q) * n + max(q) for q, _ in queries}) != len(queries):
        raise ValueError('Duplicate validation queries')

    def sync():
        if torch.device(args.device).type == 'cuda':
            torch.cuda.synchronize(args.device)

    sync()
    start = time.perf_counter()
    builder = GroupingControls(data.edge_index, n, getattr(data, 'edge_weight', None))
    partitions, partition_metadata = {}, {}
    for name in ('diffusion', 'random', 'normalized-edge'):
        partition, details = builder.partition(name, seed=0, alpha=.75, width=100, steps=4)
        partitions[name] = partition
        partition_metadata[name] = details | {'partition_sha256': hashlib.sha256(json.dumps(partition).encode()).hexdigest()}
    sync()
    offline_seconds = time.perf_counter() - start
    # Frozen before observing the new validation queries.
    variants = [('Swap-global6', None, 6, 'diverse'), ('Swap-global12', None, 12, 'diverse'),
                ('Swap-diverse6', 'diffusion', 6, 'diverse'), ('Swap-diverse12', 'diffusion', 12, 'diverse'),
                ('Swap-union-diffusion', 'diffusion', 12, 'coverage-union'),
                ('Swap-union-random', 'random', 12, 'coverage-union'),
                ('Swap-union-structural', 'normalized-edge', 12, 'coverage-union')]
    saliency = SaliencyExplainer(model, k_frac=1., evidence_mode='supportive', device=args.device)
    rows = []
    for index, (query, label) in enumerate(queries):
        a, b = query
        pool = getattr(data, 'val_pos_edge_index' if label else 'val_neg_edge_index')
        if not (canonical(pool) == min(query) * n + max(query)).any():
            raise ValueError('Query is not in validation split')
        if (canonical(data.edge_index) == min(query) * n + max(query)).any():
            raise ValueError('Target leakage into message-passing graph')
        mask = candidate_edge_mask(data.edge_index, n, query, len(encoder.convs), 'gcn-boundary')
        candidates = torch.unique(canonical(data.edge_index[:, mask]))
        grouping = {name: candidate_groups(candidates, isolate_query_endpoints(p, a, b), n)
                    for name, p in partitions.items()}
        explanation = saliency.explain_link(data, a, b) if args.additional_queries_per_class else None
        for budget in (5, 10, 20):
            if explanation is None:
                original = saved[(query, budget)]
                initial = Data(edge_index=torch.tensor(original['support'], dtype=torch.long, device=args.device).reshape(-1, 2).t())
            else:
                initial = rank_support(explanation, candidates.cpu(), n, min(budget, len(candidates)))
            before = evaluate_support(model, data, initial, a, b, args.device)
            if before['support_edges'] != min(budget, len(candidates)):
                raise ValueError('Initial support budget differs')
            if explanation is None and not math.isclose(before['full_logit'], original['full_logit'], rel_tol=1e-6, abs_tol=1e-5):
                raise ValueError('Saved full prediction differs')
            common = {'query': list(query), 'label': label, 'budget': budget,
                      'effective_budget': before['support_edges'], 'candidate_edges': len(candidates)}
            rows.append(common | {'method': 'Saliency-supportive', 'support': initial.edge_index.t().cpu().tolist(),
                                   'ranking_seconds': 0., **before})
            ordering = variants[index % len(variants):] + variants[:index % len(variants)]
            for name, partition_name, additions, addition_policy in ordering:
                sync()
                start = time.perf_counter()
                support, trace = refine_support(model, data, a, b, candidates, initial,
                                                groups=grouping.get(partition_name), additions=additions,
                                                removals=3, steps=8, batch_size=8, addition_policy=addition_policy)
                sync()
                elapsed = time.perf_counter() - start
                metrics = evaluate_support(model, data, support, a, b, args.device)
                if any(step['proposals'] > additions * 3 for step in trace):
                    raise ValueError('Per-variant proposal cap exceeded')
                rows.append(common | {'method': name, 'support': support.edge_index.t().cpu().tolist(),
                                       'search_trace': trace, 'ranking_seconds': elapsed,
                                       'timing_scope': 'Search only; excludes shared partition, grouping, initialization, and final metric evaluation.',
                                       'addition_policy': addition_policy, 'additions': additions, **metrics})
        print(f'Completed {index + 1}/{len(queries)} validation queries', flush=True)
    comparison_pairs = [(name, 'Swap-global12') for name, _, additions, _ in variants if additions == 12 and name != 'Swap-global12']
    comparison_pairs += [('Swap-union-diffusion', 'Swap-diverse6'), ('Swap-global12', 'Swap-global6'),
                         ('Swap-union-diffusion', 'Swap-union-random'), ('Swap-union-diffusion', 'Swap-union-structural')]
    comparisons = [item for treatment, control in comparison_pairs
                   for item in paired_policy_comparison(rows, treatment, control)]
    report = {'protocol': PROTOCOL, 'study': 'coverage-union-fixed-controls-v1', 'scope': __doc__,
              'input': str(args.input), 'input_sha256': sha(args.input),
              'args': reference['args'] | {'query_reference': None, 'device': args.device,
                                          'swap_steps': 8, 'swap_additions': 12, 'swap_removals': 3,
                                          'output': str(args.output), 'additional_queries_per_class': args.additional_queries_per_class,
                                          'queries_per_class': len(queries) // 2},
              'variant_configuration': variants, 'query_sampling_seed': 20260930,
              'excluded_reference_query_keys': excluded if args.additional_queries_per_class else [],
              'source_sha256_at_start': source_hashes, 'checkpoint_sha256': sha(path),
              'split_sha256': reference['split_sha256'], 'partition_metadata': partition_metadata,
              'offline_seconds_all_partitions': offline_seconds, 'torch': torch.__version__,
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'rows': rows, 'paired_comparisons': comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)


if __name__ == '__main__':
    main()
