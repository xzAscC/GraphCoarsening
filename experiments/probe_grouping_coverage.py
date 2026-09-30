"""Replay one validation loss and inspect grouping-induced candidate omissions.

The optional counterfactual appends one proposal observed in ungrouped search.
It is an oracle-assisted diagnostic with extra work, NOT a deployable method or
a matched-budget improvement. No parameter or checkpoint is optimized here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_gcn import load_dataset, MLPLinkPredictor
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel
from src.grouping_controls import GroupingControls
from src.partition import isolate_query_endpoints
from src.explainers.candidates import candidate_edge_mask
from src.explainers import support_refinement as search
from src.evaluation.interventions import evaluate_support


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def support_keys(support, n):
    return sorted(min(a, b) * n + max(a, b) for a, b in support)


def run_captured(model, data, a, b, candidates, initial, groups, config, injection=None):
    snapshots = []
    original = search.exchange_proposals

    def capture(candidates, selected, in_order, out_order, gradient,
                additions, removals, exchange_size, groups=None):
        proposals, sizes = original(candidates, selected, in_order, out_order, gradient,
                                    additions, removals, exchange_size, groups)
        chosen = selected.tolist()
        global_additions = candidates[out_order[:additions]].tolist()
        actual_additions = sorted(set(v for proposal in proposals for v in proposal.tolist()) - set(chosen))
        inserted = False
        if (injection is not None and chosen == injection['state']
                and not any(proposal.tolist() == injection['proposal'] for proposal in proposals)):
            proposals.append(selected.new_tensor(injection['proposal']))
            sizes.append(1)
            inserted = True
        positions = {int(candidates[i]): i for i in out_order.tolist()}
        details = [{'edge_key': key, 'global_rank': out_order.tolist().index(positions[key]) + 1,
                    'gradient': float(gradient[positions[key]]),
                    'group': int(groups[positions[key]]) if groups is not None else None}
                   for key in sorted(set(global_additions) | set(actual_additions))]
        snapshots.append({'selected': chosen, 'global_top_additions': global_additions,
                          'actual_additions_before_injection': actual_additions,
                          'omitted_global_additions': sorted(set(global_additions) - set(actual_additions)),
                          'addition_details': details, 'injected': inserted,
                          'proposals': [p.tolist() for p in proposals]})
        return proposals, sizes

    with patch.object(search, 'exchange_proposals', capture):
        support, trace = search.refine_support(
            model, data, a, b, candidates, initial, groups=groups,
            steps=config['search_steps'], additions=config['additions'], removals=config['removals'],
            batch_size=config['batch_size'], exchange_size=1)
    return {'support': support.edge_index.t().cpu().tolist(), 'trace': trace, 'snapshots': snapshots,
            'metrics': evaluate_support(model, data, support, a, b, str(data.x.device))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--query', type=int, nargs=2, required=True)
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--policy', default='diffusion-0')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior diagnostic evidence')
    report = json.loads(args.input.read_bytes())
    if report['benchmark'] != 'single-search-grouping-controls-v1':
        raise ValueError('Expected grouping-control benchmark')
    if sha(report['input']) != report['input_sha256']:
        raise ValueError('Reference changed')
    reference = json.loads(Path(report['input']).read_bytes())
    if reference['args']['query_split'] != 'val':
        raise ValueError('Validation-only diagnostic')
    rows = [r for r in report['records'] if r['query'] == args.query and r['budget'] == args.budget]
    originals = [r for r in reference['rows'] if r['query'] == args.query and r['budget'] == args.budget
                 and r['method'] == 'Saliency-supportive']
    if len(rows) != 1 or len(originals) != 1:
        raise ValueError('Expected exactly one query/budget and initial support')
    saved, original = rows[0], originals[0]
    path = Path(reference['args']['checkpoint_dir']) / (report['dataset'] + '_gcn.pt')
    if sha(path) != report['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    state = torch.load(path, map_location=args.device, weights_only=False)
    config = state['config']
    torch.manual_seed(config['seed'])
    data = load_dataset(config['dataset'])
    for name, edges in state['edge_splits'].items():
        if hashlib.sha256(edges.cpu().numpy().tobytes()).hexdigest() != report['split_sha256'][name]:
            raise ValueError('Split changed')
        setattr(data, name, edges)
    data.edge_index = data.train_pos_edge_index
    data = data.to(args.device)
    encoder = GCN(config['in_channels'], config['hidden_channels'], config['out_channels'], config['num_layers'])
    predictor = MLPLinkPredictor(config['out_channels'], config['hidden_channels'])
    encoder.load_state_dict(state['model_state_dict'])
    predictor.load_state_dict(state['predictor_state_dict'])
    model = LinkPredictionModel(encoder, predictor).to(args.device).eval()
    settings = report['config']
    policy, seed = args.policy.rsplit('-', 1)
    partition, _ = GroupingControls(data.edge_index, data.num_nodes).partition(
        policy, seed=int(seed), width=settings['width'], steps=settings['diffusion_steps'], alpha=settings['alpha'])
    partition_hash = hashlib.sha256(json.dumps(partition).encode()).hexdigest()
    if partition_hash != report['partitions'][args.policy]['partition_sha256']:
        raise ValueError('Partition was not reproduced; cannot attribute this saved loss')
    a, b = args.query
    mask = candidate_edge_mask(data.edge_index, data.num_nodes, [a, b], len(encoder.convs), 'gcn-boundary')
    edges = data.edge_index[:, mask]
    candidates = torch.unique(edges.min(0).values * data.num_nodes + edges.max(0).values)
    if len(candidates) != saved['candidate_edges']:
        raise ValueError('Candidate region changed')
    groups = search.candidate_groups(candidates, isolate_query_endpoints(partition, a, b), data.num_nodes)
    initial = Data(edge_index=torch.tensor(original['support'], device=args.device, dtype=torch.long).reshape(-1, 2).t())
    replays = {name: run_captured(model, data, a, b, candidates, initial, grouping, settings)
               for name, grouping in [('ungrouped', None), (args.policy, groups)]}
    for name, replay in replays.items():
        expected = saved['results'][name]
        if support_keys(replay['support'], data.num_nodes) != support_keys(expected['support'], data.num_nodes):
            raise ValueError(f'{name} final support differs from saved run')
        for metric in ('full_logit', 'retained_logit', 'removed_logit'):
            if not math.isclose(replay['metrics'][metric], expected['metrics'][metric], rel_tol=1e-6, abs_tol=1e-5):
                raise ValueError('Saved intervention prediction was not reproduced')
    omissions = []
    ungrouped, grouped = replays['ungrouped'], replays[args.policy]
    for index, snapshot in enumerate(ungrouped['snapshots']):
        for other in grouped['snapshots']:
            if snapshot['selected'] != other['selected'] or not ungrouped['trace'][index + 1]['accepted']:
                continue
            chosen = (ungrouped['snapshots'][index + 1]['selected'] if index + 1 < len(ungrouped['snapshots'])
                      else support_keys(ungrouped['support'], data.num_nodes))
            if chosen not in other['proposals']:
                omissions.append({'state': snapshot['selected'], 'proposal': chosen,
                                  'ungrouped_step': index + 1,
                                  'grouped_omitted_global_additions': other['omitted_global_additions']})
    counterfactual = (run_captured(model, data, a, b, candidates, initial, groups, settings, omissions[0])
                      if omissions else None)
    sources = [Path(__file__), Path('experiments/train_gcn.py'), *sorted(Path('src').rglob('*.py'))]
    output = {'diagnostic': 'grouping-proposal-coverage-replay-v1', 'scope': __doc__,
              'input': str(args.input), 'input_sha256': sha(args.input), 'query': args.query,
              'budget': args.budget, 'policy': args.policy, 'device': args.device,
              'checkpoint_sha256': sha(path), 'partition_sha256': partition_hash,
              'source_sha256': {str(p): sha(p) for p in sources}, 'replays': replays,
              'omitted_accepted_ungrouped_proposals': omissions,
              'counterfactual_extra_proposal': counterfactual}
    with args.output.open('x') as file:
        json.dump(output, file, indent=2)
    print('Exact final supports reproduced; omitted accepted proposals:', len(omissions))
    if counterfactual:
        print('Counterfactual metrics:', counterfactual['metrics'])


if __name__ == '__main__':
    main()
