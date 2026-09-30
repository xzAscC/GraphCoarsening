"""Replay saved support methods and add fixed-config validation baselines.

Uses original-graph undirected interventions for every discrete support.
PyG GNNExplainer's native message-mask training differs from physical edge
deletion and is reported explicitly, not silently redefined as our objective.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch_geometric
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_gcn import load_dataset, MLPLinkPredictor
from experiments.run_support_benchmark import rank_support, reference_query_indices
from experiments.audit_local_gcn import synchronize
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel
from src.explainers.candidates import candidate_edge_mask
from src.explainers.pyg_baselines import GNNExplainerWrapper
from src.evaluation.interventions import PROTOCOL, evaluate_support


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def query_seed(seed, query):
    payload = json.dumps([seed, *sorted(query)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')


def summarize(rows):
    metrics = ('necessity_flip', 'sufficiency_agreement',
               'necessity_confidence_drop', 'sufficiency_confidence_drop')
    result = {}
    for method in sorted({r['method'] for r in rows}):
        result[method] = {}
        for budget in sorted({r['budget'] for r in rows}):
            selected = [r for r in rows if r['method'] == method and r['budget'] == budget]
            keys = {tuple(sorted(r['query'])) for r in selected}
            if len(keys) != len(selected) or not selected:
                raise ValueError('Expected one result per query, method and budget')
            result[method][str(budget)] = {'queries': len(selected), **{
                key: statistics.mean(r[key] for r in selected) for key in metrics}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-methods', nargs='+',
                        default=['Saliency-supportive', 'Swap-global12', 'Swap-union-diffusion'])
    parser.add_argument('--baseline', choices=('gnnexplainer', 'cf2-full', 'cf2-local', 'cf2-paired'),
                        default='gnnexplainer')
    parser.add_argument('--epochs', type=int, help='Defaults to 100 for GNNExplainer, 2000 for CF2')
    parser.add_argument('--queries-per-class', type=int, help='Optional first-N development subset, never outcome selected')
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 100 if args.baseline == 'gnnexplainer' else 2000
    if args.output.exists():
        raise FileExistsError('Preserve prior results')
    if args.epochs < 1 or len(set(args.reference_methods)) != len(args.reference_methods):
        raise ValueError('Positive epochs and distinct comparison methods required')
    if args.queries_per_class is not None and args.queries_per_class < 1:
        raise ValueError('Positive query subset count required')
    saved = json.loads(args.input.read_bytes())
    config = saved['args']
    if (config['query_split'] != 'val' or config['candidate_region'] != 'gcn-boundary'
            or saved['protocol'] != PROTOCOL):
        raise ValueError('Validation boundary-protocol reference required')
    checkpoint = Path(config['checkpoint_dir']) / (config['dataset'] + '_gcn.pt')
    if sha(checkpoint) != saved['checkpoint_sha256']:
        raise ValueError('Checkpoint mismatch')
    state = torch.load(checkpoint, map_location=args.device, weights_only=False)
    c = state['config']
    torch.manual_seed(c['seed'])
    data = load_dataset(c['dataset'])
    for name, edges in state['edge_splits'].items():
        if hashlib.sha256(edges.cpu().numpy().tobytes()).hexdigest() != saved['split_sha256'][name]:
            raise ValueError('Split mismatch')
        setattr(data, name, edges)
    data.edge_index = data.train_pos_edge_index
    data = data.to(args.device)
    encoder = GCN(c['in_channels'], c['hidden_channels'], c['out_channels'], c['num_layers'])
    predictor = MLPLinkPredictor(c['out_channels'], c['hidden_channels'])
    encoder.load_state_dict(state['model_state_dict'])
    predictor.load_state_dict(state['predictor_state_dict'])
    model = LinkPredictionModel(encoder, predictor).to(args.device).eval()
    if args.baseline == 'gnnexplainer':
        explainers = {'GNNExplainer': GNNExplainerWrapper(model, epochs=args.epochs, lr=.01, k_frac=1., device=args.device)}
    else:
        from src.explainers.cf2_link import CF2LinkExplainer
        modes = [False, True] if args.baseline == 'cf2-paired' else [args.baseline == 'cf2-local']
        explainers = {('CF2-link-local' if local else 'CF2-link-full'): CF2LinkExplainer(
            model, epochs=args.epochs, hops=len(encoder.convs), device=args.device, local_gcn=local) for local in modes}
    paths = [Path(__file__), Path('experiments/run_support_benchmark.py'), Path('experiments/train_gcn.py'),
             Path('experiments/audit_local_gcn.py'), *sorted(Path('src').rglob('*.py'))]
    sources = {str(p): sha(p) for p in paths}
    rows, failures, mask_comparisons = [], [], []
    queries = []
    for label in (1, 0):
        pool = getattr(data, 'val_pos_edge_index' if label else 'val_neg_edge_index')
        indices = reference_query_indices(saved, pool, label, data.num_nodes)
        if args.queries_per_class is not None:
            if len(indices) < args.queries_per_class:
                raise ValueError('Insufficient reference queries for requested subset')
            indices = indices[:args.queries_per_class]
        for index in indices:
            query = tuple(map(int, pool[:, index].tolist()))
            queries.append((label, query))
    budgets = sorted(config['budgets'])
    for qi, (label, query) in enumerate(queries):
        selected = [r for r in saved['rows'] if tuple(sorted(r['query'])) == tuple(sorted(query))
                    and r['method'] in args.reference_methods]
        if len(selected) != len(args.reference_methods) * len(budgets):
            raise ValueError('Incomplete reference comparison')
        mask = candidate_edge_mask(data.edge_index, data.num_nodes, query, len(encoder.convs), 'gcn-boundary')
        edges = data.edge_index[:, mask]
        candidates = torch.unique(edges.min(0).values * data.num_nodes + edges.max(0).values).cpu()
        qseed = query_seed(args.seed, query)
        order = list(explainers)
        if qi % 2:
            order.reverse()
        explanations = {}
        for name in order:
            explainer = explainers[name]
            torch.manual_seed(qseed)
            synchronize(args.device)
            start = time.perf_counter()
            explanation = explainer.explain_link(data, *query)
            synchronize(args.device)
            duration = time.perf_counter() - start
            explanations[name] = explanation
            for budget in budgets:
                effective = min(budget, len(candidates))
                support = rank_support(explanation, candidates, data.num_nodes, effective)
                metrics = evaluate_support(model, data, support, *query, args.device)
                if metrics['support_edges'] != effective:
                    raise RuntimeError('Baseline support budget mismatch')
                rows.append({'query': query, 'label': label, 'budget': budget, 'effective_budget': effective,
                             'candidate_edges': len(candidates), 'method': name,
                             'mask_seed': qseed, 'ranking_seconds': duration, 'optimization_order': order,
                             'optimization': dict(getattr(explainer, 'last_diagnostics', {})),
                             'support': support.edge_index.t().tolist(), **metrics})
        if args.baseline == 'cf2-paired':
            full, local = explanations['CF2-link-full'], explanations['CF2-link-local']
            if not torch.equal(full.edge_index, local.edge_index):
                raise RuntimeError('CF2 mask parameter order differs')
            difference = (full.edge_weight - local.edge_weight).abs()
            mask_comparisons.append({'query': query, 'max_absolute_mask_difference': float(difference.max()) if difference.numel() else 0.,
                                     'mean_absolute_mask_difference': float(difference.mean()) if difference.numel() else 0.})
        for old in selected:
            if (old['label'] != label or old['candidate_edges'] != len(candidates)
                    or old['effective_budget'] != min(old['budget'], len(candidates))):
                raise ValueError('Saved class, candidates or effective budget differs')
            support = Data(edge_index=torch.tensor(old['support'], dtype=torch.long).reshape(-1, 2).t())
            keys = support.edge_index.min(0).values * data.num_nodes + support.edge_index.max(0).values
            if not torch.isin(keys, candidates).all() or len(torch.unique(keys)) != old['effective_budget']:
                raise ValueError('Invalid saved support')
            metrics = evaluate_support(model, data, support, *query, args.device)
            for key in ('full_logit', 'retained_logit', 'removed_logit'):
                if not torch.isclose(torch.tensor(metrics[key]), torch.tensor(old[key]), rtol=1e-5, atol=2e-5):
                    failures.append({'query': query, 'method': old['method'], 'budget': old['budget'], 'metric': key})
            for key in ('necessity_flip', 'sufficiency_agreement'):
                if metrics[key] != old[key]:
                    failures.append({'query': query, 'method': old['method'], 'budget': old['budget'], 'metric': key})
            rows.append({k: old[k] for k in ('query', 'label', 'budget', 'effective_budget',
                                            'candidate_edges', 'method', 'support')} | metrics |
                        {'reference_ranking_seconds': old.get('ranking_seconds'), 'independently_replayed': True})
        print(f'Completed {qi + 1}/{len(queries)} queries', flush=True)
    report = {'study': 'frozen-baseline-reference-replay-v2', 'protocol': PROTOCOL,
              'input': str(args.input), 'input_sha256': sha(args.input),
              'checkpoint_sha256': sha(checkpoint), 'split_sha256': saved['split_sha256'],
              'source_sha256_at_start': sources, 'dataset': c['dataset'], 'training_seed': c['seed'],
              'device': args.device, 'torch': torch.__version__, 'pyg': torch_geometric.__version__,
              'baseline': {'mode': args.baseline, 'methods': list(explainers), 'epochs': args.epochs, 'lr': .01, 'seed': args.seed,
                           'scope': ('PyG native directed message masks, no feature mask, frozen model, full training graph; rank directional scores by undirected sum, restrict to boundary candidates, external top-B and original-weight interventions.'
                                     if args.baseline == 'gnnexplainer' else
                                     'CF2 binary-link adaptation with paper probability margins, lambda=500, alpha=0.6, gamma=0.5, original-n initialization and tied original-edge gates. Local backend changes only objective evaluation region; FP trajectories may differ. All hard supports evaluated on full original graph.')},
              'sampling': 'All reference validation queries, or requested first-N per class, no outcome filtering.',
              'queries_per_class_requested': args.queries_per_class,
              'timing_scope': 'New baseline mask-learning time recorded; historical search times are not controlled paired timings.',
              'reference_replay_failures': failures, 'mask_comparisons': mask_comparisons,
              'summaries': summarize(rows), 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
    if failures:
        raise RuntimeError('Saved support replay failed; output preserved')


if __name__ == '__main__':
    main()
