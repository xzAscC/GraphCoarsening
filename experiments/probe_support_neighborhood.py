"""Exhaustively diagnose one-edge exchanges around one saved validation support.

This expensive diagnostic is not the proposed explanation method. It separates
an incomplete gradient shortlist from a genuine one-exchange local optimum.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_gcn import load_dataset, MLPLinkPredictor
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel
from src.explainers.candidates import candidate_edge_mask
from src.explainers.group_interventions import group_deletion_logits
from src.evaluation.interventions import evaluate_support


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--query', nargs=2, type=int, required=True)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--method', default='Swap-coarse')
    p.add_argument('--max-proposals', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous probes')
    report = json.loads(args.input.read_bytes())
    if report['args']['query_split'] != 'val':
        raise ValueError('Diagnostic development probes use validation queries only')
    matches = [r for r in report['rows'] if r['query'] == args.query
               and r['budget'] == args.budget and r['method'] == args.method]
    if len(matches) != 1:
        raise ValueError('Expected exactly one saved support')
    saved = matches[0]
    path = Path(report['args']['checkpoint_dir']) / (report['args']['dataset'] + '_gcn.pt')
    if hashlib.sha256(path.read_bytes()).hexdigest() != report['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    ckpt = torch.load(path, map_location=args.device, weights_only=False)
    c = ckpt['config']
    torch.manual_seed(c['seed'])
    data = load_dataset(c['dataset'])
    for name, edges in ckpt['edge_splits'].items():
        if hashlib.sha256(edges.cpu().numpy().tobytes()).hexdigest() != report['split_sha256'][name]:
            raise ValueError('Split mismatch')
        setattr(data, name, edges)
    data.edge_index = data.train_pos_edge_index
    data = data.to(args.device)
    encoder = GCN(c['in_channels'], c['hidden_channels'], c['out_channels'], c['num_layers'])
    predictor = MLPLinkPredictor(c['out_channels'], c['hidden_channels'])
    encoder.load_state_dict(ckpt['model_state_dict'])
    predictor.load_state_dict(ckpt['predictor_state_dict'])
    model = LinkPredictionModel(encoder, predictor).to(args.device).eval()
    a, b = args.query
    n = data.num_nodes
    region = candidate_edge_mask(data.edge_index, n, args.query, c['num_layers'],
                                 report['args']['candidate_region'])
    edges = data.edge_index[:, region]
    candidates = torch.unique(edges.min(0).values * n + edges.max(0).values)
    if len(candidates) != saved['candidate_edges']:
        raise ValueError('Candidate region changed')
    initial = Data(edge_index=torch.tensor(saved['support'], device=args.device,
                                          dtype=torch.long).reshape(-1, 2).t())
    base = evaluate_support(model, data, initial, a, b, args.device)
    for key in ('full_logit', 'retained_logit', 'removed_logit'):
        if not math.isclose(base[key], saved[key], rel_tol=1e-6, abs_tol=1e-5):
            raise ValueError(f'Saved prediction not reproduced: {key}')
    selected = torch.unique(initial.edge_index.min(0).values * n + initial.edge_index.max(0).values)
    outside = candidates[~torch.isin(candidates, selected)]
    if not torch.isin(selected, candidates).all() or len(selected) != saved['effective_budget']:
        raise ValueError('Saved support is not a valid candidate subset')
    count = selected.numel() * outside.numel()
    if count > args.max_proposals or count == 0:
        raise ValueError(f'Proposal count {count} exceeds limit or is empty')
    moves = [(drop, add) for drop in selected.tolist() for add in outside.tolist()]
    proposals = [torch.cat((selected[selected != drop], selected.new_tensor([add]))).sort().values
                 for drop, add in moves]
    sign = 1 if base['full_logit'] > 0 else -1
    p_full = torch.tensor(sign * base['full_logit'], dtype=torch.float64).sigmoid().item()

    def assess(supports, batch):
        keep = group_deletion_logits(model, data, a, b, supports, batch, retain=True)
        remove = group_deletion_logits(model, data, a, b, supports, batch)
        cap = (sign * keep.double()).sigmoid().clamp(max=p_full)
        deleted = (sign * remove.double()).sigmoid()
        return cap - deleted, cap, deleted, (keep > 0) == (sign > 0), (remove > 0) != (sign > 0)

    current = [v[0] for v in assess([selected], 1)]

    def eligible(values):
        return ((values[0] > current[0] + 1e-7) & (values[1] >= current[1])
                & (values[2] <= current[2]) & (values[3] >= current[3])
                & (values[4] >= current[4]))

    values = assess(proposals, args.batch_size)
    valid = eligible(values)
    best = None
    order = torch.argsort(values[0], descending=True)
    for index in order[valid[order]].tolist():
        checked = assess([proposals[index]], 1)
        if eligible(checked).item():
            support = Data(edge_index=torch.stack((proposals[index] // n, proposals[index] % n)))
            best = {'remove': list(divmod(moves[index][0], n)),
                    'add': list(divmod(moves[index][1], n)),
                    'support': support.edge_index.t().cpu().tolist(),
                    'objective': checked[0].item(),
                    'metrics': evaluate_support(model, data, support, a, b, args.device)}
            break
    result = {'diagnostic': 'exhaustive-one-exchange-validation-probe-v1',
              'input': str(args.input), 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'args': vars(args) | {'input': str(args.input), 'output': str(args.output)},
              'checkpoint_sha256': report['checkpoint_sha256'], 'baseline': base,
              'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'baseline_objective': current[0].item(), 'candidate_edges': len(candidates),
              'evaluated_proposals': count, 'batch_eligible_proposals': valid.sum().item(),
              'best_serially_checked_improvement': best,
              'scope': 'Enumerates every one-edge exchange around one saved support. Only the reported best improvement is serially rechecked; this is not global support optimization.'}
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
    print('Evaluated', count, 'proposals; batch eligible', valid.sum().item(), 'best improvement', best)


if __name__ == '__main__':
    main()
