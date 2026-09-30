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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.train_gcn import load_dataset, MLPLinkPredictor
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel
from src.explainers.baselines import SaliencyExplainer
from src.explainers.coarsen_explainer import CoarsenExplainer
from src.evaluation.interventions import PROTOCOL, evaluate_support
from src.spectral import compute_perturbation_scores, pair_projection_scores
from src.explainers.candidates import candidate_edge_mask


def reference_query_indices(reference, pool, label, n):
    """Resolve ordered, unique reference queries against a verified split pool."""
    pool_map = {min(a, b) * n + max(a, b): i
                for i, (a, b) in enumerate(pool.t().cpu().tolist())}
    indices, seen = [], set()
    for row in reference['rows']:
        if row['label'] != label:
            continue
        a, b = row['query']
        key = min(a, b) * n + max(a, b)
        if key in seen:
            continue
        if key not in pool_map:
            raise ValueError('Reference query is absent from the specified split and class')
        indices.append(pool_map[key])
        seen.add(key)
    return indices


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
    p.add_argument('--signed-ablation', action='store_true',
                   help='Add class-supportive saliency and bounded-refinement calibration')
    p.add_argument('--gnnexplainer', action='store_true')
    p.add_argument('--gnnexplainer-epochs', type=int, default=100)
    p.add_argument('--cf2', action='store_true', help='Edge-only binary-link CF2 adaptation')
    p.add_argument('--cf2-epochs', type=int, default=2000)
    p.add_argument('--cf2-lr', type=float, default=.01)
    p.add_argument('--cf2-lambda', type=float, default=500.)
    p.add_argument('--cf2-alpha', type=float, default=.6)
    p.add_argument('--cf2-gamma', type=float, default=.5)
    p.add_argument('--intervention-batch-size', type=int, default=1,
                   help='Independent deletion graphs per GPU forward (GCN only)')
    p.add_argument('--support-swaps', action='store_true',
                   help='Compare ungrouped and coarse-diverse swaps from the same signed saliency support')
    p.add_argument('--swap-steps', type=int, default=2)
    p.add_argument('--pair-swaps', action='store_true',
                   help='Also compare two-edge exchanges with matched proposal counts')
    p.add_argument('--mixed-swaps', action='store_true',
                   help='Mix single and pair exchanges under one shared proposal cap')
    p.add_argument('--swap-additions', type=int, default=6)
    p.add_argument('--swap-removals', type=int, default=3)
    p.add_argument('--merge-score', choices=['legacy', 'projection'], default='legacy')
    p.add_argument('--signal-policy', choices=['eigen', 'diffusion'], default='eigen')
    p.add_argument('--signal-width', type=int, default=100)
    p.add_argument('--diffusion-steps', type=int, default=4)
    p.add_argument('--signal-seed', type=int, default=0)
    p.add_argument('--query-split', choices=['val', 'test'], default='test')
    p.add_argument('--query-reference', help='Reuse queries from an existing benchmark JSON')
    p.add_argument('--candidate-region', choices=['induced', 'gcn-boundary'], default='induced')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.signal_policy == 'diffusion' and (args.merge_score != 'projection' or args.score_ablation):
        p.error('Diffusion requires --merge-score projection without --score-ablation')
    if args.signal_width < 1 or args.diffusion_steps < 0:
        p.error('Signal width must be positive and diffusion steps nonnegative')
    if args.support_swaps and not args.signed_ablation:
        p.error('--support-swaps requires --signed-ablation for a shared initial support')
    if args.pair_swaps and not args.support_swaps:
        p.error('--pair-swaps requires --support-swaps')
    if args.mixed_swaps and not args.support_swaps:
        p.error('--mixed-swaps requires --support-swaps')
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
    reference, reference_hash = None, None
    if args.query_reference:
        reference_bytes = Path(args.query_reference).read_bytes()
        reference_hash = hashlib.sha256(reference_bytes).hexdigest()
        reference = json.loads(reference_bytes)
        if any(reference['args'][key] != getattr(args, key) for key in ('dataset', 'seed', 'query_split')):
            raise ValueError('Reference dataset, seed, and split must match')
        if reference['split_sha256'] != hashes:
            raise ValueError('Reference split hashes differ')
        if reference['checkpoint_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Reference checkpoint differs')
    data = data.to(args.device)
    saliency = SaliencyExplainer(model, k_frac=1., device=args.device)
    ours = CoarsenExplainer(model, k_frac=1., k_hop=c['num_layers'], device=args.device,
                           score_method=args.merge_score,
                           signal_policy=args.signal_policy, signal_width=args.signal_width,
                           diffusion_steps=args.diffusion_steps, signal_seed=args.signal_seed,
                           intervention_batch_size=args.intervention_batch_size)

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
    if args.gnnexplainer:
        from src.explainers.pyg_baselines import GNNExplainerWrapper
        methods.insert(-1, ('GNNExplainer', GNNExplainerWrapper(
            model, epochs=args.gnnexplainer_epochs, k_frac=1., device=args.device)))
    if args.cf2:
        from src.explainers.cf2_link import CF2LinkExplainer
        methods.insert(-1, ('CF2-link-adapted', CF2LinkExplainer(
            model, epochs=args.cf2_epochs, lr=args.cf2_lr, lam=args.cf2_lambda,
            alpha=args.cf2_alpha, gamma=args.cf2_gamma, hops=c['num_layers'],
            candidate_region=args.candidate_region, device=args.device)))
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
    if args.signed_ablation:
        signed_saliency = SaliencyExplainer(model, k_frac=1., device=args.device,
                                           evidence_mode='supportive')
        signed_pathway = CoarsenExplainer(model, k_frac=1., k_hop=c['num_layers'],
                                         device=args.device, partition_mode='global-endpoints',
                                         evidence_mode='supportive', score_method=args.merge_score)
        signed_pathway._coarsener, signed_pathway._cached_data_id = coarsener, id(data)
        methods.insert(-1, ('Saliency-supportive', signed_saliency))
        methods.insert(-1, ('Endpoint-supportive', signed_pathway))
    for _, explainer in methods:
        if isinstance(explainer, CoarsenExplainer):
            explainer.intervention_batch_size = args.intervention_batch_size
            explainer.candidate_region = args.candidate_region
    query_rng = np.random.default_rng(args.seed)
    random_rng = np.random.default_rng(args.seed + 7919)
    rows = []
    for label, pool in [(1, getattr(data, args.query_split + '_pos_edge_index')),
                        (0, getattr(data, args.query_split + '_neg_edge_index'))]:
        count = min(args.queries_per_class, pool.size(1))
        ids = (reference_query_indices(reference, pool, label, n) if reference is not None
               else query_rng.choice(pool.size(1), count, replace=False))
        if len(ids) != count:
            raise ValueError('Reference query count differs; set --queries-per-class to match')
        for idx in ids:
            a, b = map(int, pool[:, int(idx)].tolist())
            train_keys = data.edge_index.min(dim=0).values * n + data.edge_index.max(dim=0).values
            assert not (train_keys == min(a, b) * n + max(a, b)).any(), 'Target leakage'
            region_mask = candidate_edge_mask(data.edge_index, n, [a, b], c['num_layers'], args.candidate_region)
            es = data.edge_index[:, region_mask]
            candidates = torch.unique(es.min(dim=0).values * n + es.max(dim=0).values).cpu()
            random_exp = Data(edge_index=torch.stack((candidates // n, candidates % n)),
                              edge_weight=torch.from_numpy(random_rng.random(len(candidates))))
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
                    if args.support_swaps and name == 'Saliency-supportive':
                        from src.explainers.support_refinement import candidate_groups, refine_support
                        from src.partition import isolate_query_endpoints
                        partition = isolate_query_endpoints(coarsener.partition, a, b)
                        group_ids = candidate_groups(candidates, partition, n)
                        swap_variants = [('Swap-gradient', None, 1), ('Swap-coarse', group_ids, 1)]
                        if args.pair_swaps:
                            swap_variants += [('Swap-pair-gradient', None, 2), ('Swap-pair-coarse', group_ids, 2)]
                        if args.mixed_swaps:
                            swap_variants += [('Swap-mixed-gradient', None, 0), ('Swap-mixed-coarse', group_ids, 0)]
                        for variant, grouping, exchange_size in swap_variants:
                            sync()
                            swap_start = time.perf_counter()
                            refined, trace = refine_support(
                                model, data, a, b, candidates, support, groups=grouping,
                                steps=args.swap_steps, additions=args.swap_additions,
                                removals=args.swap_removals, batch_size=args.intervention_batch_size,
                                exchange_size=exchange_size)
                            sync()
                            swap_seconds = time.perf_counter() - swap_start
                            refined_metrics = evaluate_support(model, data, refined, a, b, args.device)
                            assert refined_metrics['support_edges'] == effective
                            rows.append({'query': [a, b], 'label': label, 'method': variant,
                                         'budget': budget, 'effective_budget': effective,
                                         'candidate_edges': len(candidates),
                                         'ranking_seconds': duration + swap_seconds,
                                         'refinement_seconds': swap_seconds, 'search_trace': trace,
                                         'initial_method': name,
                                         'requested_exchange_size': exchange_size,
                                         'support': refined.edge_index.t().cpu().tolist(),
                                         'grouping': {'clusters': len(partition)} if grouping is not None else {},
                                         **refined_metrics})
            print(f'Completed label={label} query=({a},{b})', flush=True)
    spectral_diagnostics = dict(coarsener.signal_diagnostics)
    if coarsener.eigenvalues is not None:
        spectral_diagnostics.update({
            'selected_eigenvalues': coarsener.eigenvalues.detach().cpu().tolist(),
            'eigenpair_residual_norms': torch.linalg.vector_norm(
                torch.sparse.mm(coarsener.A_hat, coarsener.right_vecs)
                - coarsener.right_vecs * coarsener.eigenvalues[None, :], dim=0).detach().cpu().tolist(),
            'scope': 'Selected normalized-adjacency eigenpairs only; not a full-spectrum eigengap or explanation-fidelity certificate.'})
    report = {'protocol': PROTOCOL, 'args': vars(args), 'split_sha256': hashes,
              'query_reference_sha256': reference_hash,
              'query_sampling': 'reference-v1' if reference is not None else 'independent-streams-v1',
              'source_sha256_at_start': source_hashes,
              'training_summary': ckpt.get('training_summary'),
              'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'working_tree_status': subprocess.check_output(['git', 'status', '--short'], text=True),
              'torch': torch.__version__, 'pyg': torch_geometric.__version__,
              'offline_seconds': offline,
              'spectral_diagnostics': spectral_diagnostics,
              'rows': rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'x') as f:
        json.dump(report, f, indent=2)
    for method in sorted({r['method'] for r in rows}):
        for budget in args.budgets:
            group = [r for r in rows if r['method'] == method and r['budget'] == budget]
            print(method, budget, {k: float(np.mean([r[k] for r in group])) for k in
                  ['necessity_flip', 'sufficiency_agreement', 'necessity_confidence_drop',
                   'sufficiency_confidence_drop']})


if __name__ == '__main__':
    main()
