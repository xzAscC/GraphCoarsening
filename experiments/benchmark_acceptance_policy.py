"""Matched proposal-cap validation of continuous versus binary acceptance guards.

No baseline support is injected into search. Both policies start from the same
saved saliency support and use the same proposal and gradient rules.
The default step cap is eight for both; --short-search uses four binary-policy
steps versus eight componentwise steps, following the saved development plan.
Trajectories and actual costs may diverge after different acceptance choices.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_gcn import load_dataset, MLPLinkPredictor
from experiments.audit_local_gcn import synchronize
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel
from src.explainers.candidates import candidate_edge_mask
from src.explainers.support_refinement import refine_support
from src.evaluation.interventions import PROTOCOL, evaluate_support


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_steps(short_search):
    return {'componentwise': 8, 'binary-monotone': 4 if short_search else 8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--short-search', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior outcomes')
    saved = json.loads(args.input.read_bytes())
    c = saved['args']
    if c['query_split'] != 'val' or c['candidate_region'] != 'gcn-boundary' or saved['protocol'] != PROTOCOL:
        raise ValueError('Validation boundary-protocol reference required')
    checkpoint = Path(c['checkpoint_dir']) / (c['dataset'] + '_gcn.pt')
    if sha(checkpoint) != saved['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    state = torch.load(checkpoint, map_location=args.device, weights_only=False)
    config = state['config']
    torch.manual_seed(config['seed'])
    data = load_dataset(config['dataset'])
    for name, edges in state['edge_splits'].items():
        if hashlib.sha256(edges.cpu().numpy().tobytes()).hexdigest() != saved['split_sha256'][name]:
            raise ValueError('Split changed')
        setattr(data, name, edges)
    data.edge_index = data.train_pos_edge_index
    data = data.to(args.device)
    encoder = GCN(config['in_channels'], config['hidden_channels'], config['out_channels'], config['num_layers'])
    predictor = MLPLinkPredictor(config['out_channels'], config['hidden_channels'])
    encoder.load_state_dict(state['model_state_dict'])
    predictor.load_state_dict(state['predictor_state_dict'])
    model = LinkPredictionModel(encoder, predictor).to(args.device).eval()
    paths = [Path(__file__), Path('experiments/train_gcn.py'), Path('experiments/audit_local_gcn.py'),
             *sorted(Path('src').rglob('*.py'))]
    sources = {str(p): sha(p) for p in paths}
    initial_rows = [r for r in saved['rows'] if r['method'] == 'Saliency-supportive']
    rows, failures = [], []
    for index, old in enumerate(initial_rows):
        query, budget = old['query'], old['budget']
        mask = candidate_edge_mask(data.edge_index, data.num_nodes, query, len(encoder.convs), 'gcn-boundary')
        edges = data.edge_index[:, mask]
        candidates = torch.unique(edges.min(0).values * data.num_nodes + edges.max(0).values)
        initial = Data(edge_index=torch.tensor(old['support'], dtype=torch.long, device=args.device).reshape(-1, 2).t())
        before = evaluate_support(model, data, initial, *query, args.device)
        if len(candidates) != old['candidate_edges'] or before['support_edges'] != old['effective_budget']:
            raise ValueError('Candidate or initial support budget changed')
        if any(not math.isclose(before[k], old[k], rel_tol=1e-5, abs_tol=2e-5)
               for k in ('full_logit', 'retained_logit', 'removed_logit')):
            failures.append({'query': query, 'budget': budget, 'kind': 'initial-replay'})
        common = {k: old[k] for k in ('query', 'label', 'budget', 'effective_budget', 'candidate_edges')}
        rows.append(common | {'method': 'Saliency-supportive', 'support': old['support'], **before})
        policies = ['componentwise', 'binary-monotone']
        if index % 2:
            policies.reverse()
        for policy in policies:
            synchronize(args.device)
            start = time.perf_counter()
            support, trace = refine_support(model, data, *query, candidates, initial,
                                            steps=policy_steps(args.short_search)[policy], additions=12, removals=3, batch_size=8,
                                            acceptance_policy=policy)
            synchronize(args.device)
            seconds = time.perf_counter() - start
            metrics = evaluate_support(model, data, support, *query, args.device)
            if metrics['support_edges'] != before['support_edges']:
                raise RuntimeError('Search changed support size')
            for metric in ('necessity_flip', 'sufficiency_agreement'):
                if metrics[metric] < before[metric]:
                    raise RuntimeError('Binary non-regression violated')
            rows.append(common | {'method': 'Swap-' + policy, 'support': support.edge_index.t().cpu().tolist(),
                                   'search_trace': trace, 'ranking_seconds': seconds,
                                   'evaluated_proposals': sum(t['proposals'] for t in trace), **metrics})
        if (index + 1) % 3 == 0:
            print(f'Completed {(index + 1)//3}/{len(initial_rows)//3} queries', flush=True)
    result = {'study': 'acceptance-short-search-v1' if args.short_search else 'acceptance-policy-ablation-v1', 'protocol': PROTOCOL,
              'input': str(args.input), 'input_sha256': sha(args.input), 'checkpoint_sha256': sha(checkpoint),
              'split_sha256': saved['split_sha256'], 'source_sha256_at_start': sources,
              'dataset': config['dataset'], 'training_seed': config['seed'], 'device': args.device,
              'settings': {'steps': 8, 'additions': 12, 'removals': 3, 'batch_size': 8, 'exchange_size': 1,
                           'local_gcn': False, 'groups': None},
              'scope': __doc__, 'reference_replay_failures': failures, 'rows': rows}
    if args.short_search:
        result['settings']['steps'] = policy_steps(True)
        plan = Path('results/tnnls-reproduction/acceptance_short_search_plan.json')
        result['plan_sha256'] = sha(plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
    if failures:
        raise RuntimeError('Reference replay failed; results preserved')


if __name__ == '__main__':
    main()
