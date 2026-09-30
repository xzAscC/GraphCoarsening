"""Matched undirected-budget audit; diagnostics, not a replacement for full study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.train_gcn import load_dataset, MLPLinkPredictor
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel
from src.explainers.baselines import SaliencyExplainer
from src.explainers.coarsen_explainer import CoarsenExplainer
from src.evaluation.interventions import PROTOCOL, evaluate_support
from src.spectral import compute_perturbation_scores, pair_projection_scores


def rank_support(explanation, candidates, n, budget):
    """Sum directional importance per undirected edge, deterministic tie breaks."""
    edges = explanation.edge_index.cpu()
    mapping = getattr(explanation, 'original_node_indices', None)
    if mapping is not None:
        edges = mapping.cpu()[edges]
    keys = edges.min(dim=0).values * n + edges.max(dim=0).values
    weights = getattr(explanation, 'edge_weight', None)
    if weights is None and edges.size(1):
        raise ValueError('Nonempty ranked explanations require importance scores')
    values = torch.empty(0) if weights is None else weights.detach().cpu().abs()
    scores = torch.zeros(len(candidates), dtype=values.dtype)
    positions = torch.searchsorted(candidates, keys)
    valid = positions < len(candidates)
    valid[valid.clone()] &= candidates[positions[valid]] == keys[valid]
    scores.scatter_add_(0, positions[valid], values[valid])
    order = torch.argsort(scores, descending=True, stable=True)[:budget]
    chosen = candidates[order]
    return Data(edge_index=torch.stack((chosen // n, chosen % n)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='Cora')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--queries-per-class', type=int, default=10)
    p.add_argument('--budgets', type=int, nargs='+', default=[5, 10, 20])
    p.add_argument('--checkpoint-dir', default='checkpoints/tnnls-reproduction')
    p.add_argument('--output', required=True)
    p.add_argument('--score-ablation', action='store_true',
                   help='Also compare symmetric, projection-loss, and random merge scores')
    p.add_argument('--protection-ablation', action='store_true')
    p.add_argument('--global-refine', action='store_true')
    p.add_argument('--endpoint-isolation', action='store_true')
    p.add_argument('--merge-score', choices=['legacy', 'projection'], default='legacy')
    p.add_argument('--query-split', choices=['val', 'test'], default='test')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.score_ablation and args.merge_score != 'legacy':
        p.error('--score-ablation uses legacy as its reference; do not combine with --merge-score projection')
    source_files = [Path(__file__), Path('experiments/train_gcn.py'), *Path('src').rglob('*.py')]
    source_hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}
    if Path(args.output).exists():
        raise FileExistsError('Use a new output path to preserve prior evidence')
    if args.queries_per_class < 1 or min(args.budgets) < 1:
        raise ValueError('Query counts and budgets must be positive')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_dataset(args.dataset)
    data.edge_index = data.train_pos_edge_index
    path = Path(args.checkpoint_dir) / f'{args.dataset}_gcn.pt'
    ckpt = torch.load(path, map_location=args.device, weights_only=False)
    c = ckpt['config']
    if 'seed' in c and c['seed'] != args.seed:
        raise ValueError('Checkpoint training seed differs from requested split seed')
    if 'edge_splits' in ckpt:
        for name, edges in ckpt['edge_splits'].items():
            setattr(data, name, edges.cpu())
        data.edge_index = data.train_pos_edge_index
    encoder = GCN(c['in_channels'], c['hidden_channels'], c['out_channels'], c['num_layers'])
    encoder.load_state_dict(ckpt['model_state_dict'])
    predictor = MLPLinkPredictor(c['out_channels'], c['hidden_channels'])
    predictor.load_state_dict(ckpt['predictor_state_dict'])
    model = LinkPredictionModel(encoder, predictor).to(args.device).eval()
    n = data.x.size(0)
    hashes = {name: hashlib.sha256(getattr(data, name).numpy().tobytes()).hexdigest()
              for name in ['train_pos_edge_index', 'val_pos_edge_index', 'val_neg_edge_index',
                           'test_pos_edge_index', 'test_neg_edge_index']}
    data = data.to(args.device)
    saliency = SaliencyExplainer(model, k_frac=1., device=args.device)
    ours = CoarsenExplainer(model, k_frac=1., k_hop=c['num_layers'], device=args.device,
                           score_method=args.merge_score)

    def sync():
        if str(args.device).startswith('cuda'):
            torch.cuda.synchronize(args.device)

    sync()
    start = time.perf_counter()
    coarsener = ours._ensure_fitted(data)
    sync()
    offline = time.perf_counter() - start
    variants = {'Pathway': coarsener.scores.clone()}
    if args.score_ablation:
        reverse = compute_perturbation_scores(data.edge_index.flip(0), coarsener.eigenvalues,
                                             coarsener.left_vecs, coarsener.right_vecs)
        variants['Pathway-symmetric'] = (coarsener.scores + reverse) / 2
        variants['Pathway-projection'] = pair_projection_scores(data.edge_index, coarsener.right_vecs)
        keys = data.edge_index.min(dim=0).values * n + data.edge_index.max(dim=0).values
        unique, inverse = torch.unique(keys, return_inverse=True)
        random_scores = torch.rand(len(unique), generator=torch.Generator().manual_seed(args.seed + 123))
        variants['Pathway-random'] = random_scores.to(data.edge_index.device)[inverse]
    methods = [('Saliency', saliency)] + [(name, ours) for name in variants] + [('Random', None)]
    if args.protection_ablation:
        endpoints = CoarsenExplainer(model, k_frac=1., k_hop=c['num_layers'],
                                    device=args.device, protect_hops=0)
        endpoints._coarsener, endpoints._cached_data_id = coarsener, id(data)
        methods.insert(-1, ('Pathway-endpoints', endpoints))
    if args.global_refine:
        global_refine = CoarsenExplainer(model, k_frac=1., k_hop=c['num_layers'],
                                        device=args.device, partition_mode='global-refine')
        global_refine._coarsener, global_refine._cached_data_id = coarsener, id(data)
        methods.insert(-1, ('Global-refine', global_refine))
    if args.endpoint_isolation:
        endpoint_isolation = CoarsenExplainer(model, k_frac=1., k_hop=c['num_layers'],
                                            device=args.device, partition_mode='global-endpoints')
        endpoint_isolation._coarsener, endpoint_isolation._cached_data_id = coarsener, id(data)
        methods.insert(-1, ('Endpoint-isolation', endpoint_isolation))
    rng = np.random.default_rng(args.seed)
    rows = []
    for label, pool in [(1, getattr(data, args.query_split + '_pos_edge_index')),
                        (0, getattr(data, args.query_split + '_neg_edge_index'))]:
        ids = rng.choice(pool.size(1), min(args.queries_per_class, pool.size(1)), replace=False)
        for idx in ids:
            a, b = map(int, pool[:, int(idx)].tolist())
            train_keys = data.edge_index.min(dim=0).values * n + data.edge_index.max(dim=0).values
            assert not (train_keys == min(a, b) * n + max(a, b)).any(), 'Target leakage'
            _, es, _, _ = k_hop_subgraph([a, b], c['num_layers'], data.edge_index, num_nodes=n)
            candidates = torch.unique(es.min(dim=0).values * n + es.max(dim=0).values).cpu()
            random_exp = Data(edge_index=torch.stack((candidates // n, candidates % n)),
                              edge_weight=torch.from_numpy(rng.random(len(candidates))))
            for name, explainer in methods:
                if name in variants:
                    coarsener.scores = variants[name]
                elif name == 'Pathway-endpoints':
                    coarsener.scores = variants['Pathway']
                sync()
                start = time.perf_counter()
                exp = random_exp if explainer is None else explainer.explain_link(data, a, b)
                sync()
                duration = time.perf_counter() - start
                for budget in args.budgets:
                    effective = min(budget, len(candidates))
                    support = rank_support(exp, candidates, n, effective)
                    metrics = evaluate_support(model, data, support, a, b, args.device)
                    assert metrics['support_edges'] == effective
                    rows.append({'query': [a, b], 'label': label, 'method': name,
                                 'budget': budget, 'effective_budget': effective,
                                 'candidate_edges': len(candidates), 'ranking_seconds': duration,
                                 'support': support.edge_index.t().tolist(),
                                 'grouping': getattr(explainer, 'last_diagnostics', {}),
                                 **metrics})
            print(f'Completed label={label} query=({a},{b})', flush=True)
    report = {'protocol': PROTOCOL, 'args': vars(args), 'split_sha256': hashes,
              'source_sha256_at_start': source_hashes,
              'training_summary': ckpt.get('training_summary'),
              'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'working_tree_status': subprocess.check_output(['git', 'status', '--short'], text=True),
              'torch': torch.__version__, 'pyg': torch_geometric.__version__,
              'offline_seconds': offline, 'rows': rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'x') as f:
        json.dump(report, f, indent=2)
    for method, _ in methods:
        for budget in args.budgets:
            group = [r for r in rows if r['method'] == method and r['budget'] == budget]
            print(method, budget, {k: float(np.mean([r[k] for r in group])) for k in
                  ['necessity_flip', 'sufficiency_agreement', 'necessity_confidence_drop',
                   'sufficiency_confidence_drop']})


if __name__ == '__main__':
    main()
