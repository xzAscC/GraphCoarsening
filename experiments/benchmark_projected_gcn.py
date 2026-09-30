"""Forward-only Galerkin feasibility study against exact local controls.

By default global partitions are restricted to the exact query region and
endpoints isolated. Optional local-matched mode instead rebuilds partitions
within each region, protecting endpoints and matching achieved cluster counts;
this mode is a diagnostic, not a reusable global-coarsening implementation.
Approximate coarse logits are diagnostics, never replacements for stored
full-graph fidelity measurements. Timings include per-intervention masking,
normalization and coarse-operator assembly, but exclude separately reported
partition/region/feature preparation. This is not a support-search speed study.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_gcn import load_dataset, MLPLinkPredictor
from experiments.audit_local_gcn import benchmark_pair, synchronize
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel
from src.explainers.local_gcn import compact_gcn_query
from src.explainers.projected_gcn import ProjectedGCN
from src.explainers.group_interventions import group_deletion_logits
from src.grouping_controls import GroupingControls
from src.ward_partition import connected_ward_partition


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def membership_vector(partition, n, device):
    nodes = torch.tensor([v for cluster in partition for v in cluster])
    labels = torch.tensor([i for i, cluster in enumerate(partition) for _ in cluster])
    if not torch.equal(nodes.sort().values, torch.arange(n)):
        raise ValueError('Not a partition of all nodes')
    membership = torch.empty(n, dtype=torch.long)
    membership[nodes] = labels
    return membership.to(device)


@torch.no_grad()
def prepare_partitions(data, policies, linear_transform, protected_nodes=None):
    """Build connected controls, optionally excluding protected merge edges.

    Protection changes only the graph used to choose clusters, never the
    inference graph. All controls must attain the same feasible cluster count.
    """
    edges = data.edge_index
    weights = getattr(data, 'edge_weight', None)
    if protected_nodes is not None:
        protected_nodes = torch.as_tensor(protected_nodes, device=edges.device).reshape(-1)
        mask = ~torch.isin(edges, protected_nodes).any(0)
        edges = edges[:, mask]
        weights = weights[mask] if weights is not None else None
    builder = GroupingControls(edges, data.num_nodes, weights)
    expected = max(data.num_nodes - int(.75 * data.num_nodes), builder.diffusion.num_components)
    memberships, metadata = {}, {}
    for policy in policies:
        synchronize(data.x.device)
        start = time.perf_counter()
        signals = None
        if policy in ('raw-feature', 'ward-raw-feature'):
            signals = data.x
        elif policy in ('first-linear-feature', 'ward-first-linear-feature'):
            signals = linear_transform(data.x)
        if policy.startswith('ward-'):
            partition, details = connected_ward_partition(builder.edges, signals, alpha=.75)
        else:
            partition, details = builder.partition('signal' if signals is not None else policy,
                                                   seed=0, alpha=.75, width=100, steps=4, signals=signals)
        if len(partition) != expected:
            raise RuntimeError('Unmatched feasible cluster count')
        membership = membership_vector(partition, data.num_nodes, data.x.device)
        if protected_nodes is not None:
            sizes = torch.bincount(membership)
            if not (sizes[membership[protected_nodes]] == 1).all():
                raise RuntimeError('Protected endpoint was merged')
        memberships[policy] = membership
        synchronize(data.x.device)
        metadata[policy] = details | {
            'signal_source': policy, 'expected_clusters': expected,
            'seconds': time.perf_counter() - start,
            'partition_sha256': hashlib.sha256(json.dumps(partition).encode()).hexdigest()}
    return memberships, metadata


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', default='Swap-global12')
    parser.add_argument('--queries-per-class', type=int, default=10)
    parser.add_argument('--timing-repeats', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--feature-controls', action='store_true',
                        help='Add fixed raw-feature and frozen first-linear feature rankings; no label/outcome tuning')
    parser.add_argument('--ward-controls', action='store_true',
                        help='Add dynamic disjoint-round Ward controls on raw and first-linear features')
    parser.add_argument('--partition-scope', choices=('global', 'local-matched'), default='global',
                        help='Local-matched rebuilds partitions per query; it is not reusable global coarsening')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior measurements')
    if args.queries_per_class < 1 or args.timing_repeats < 2 or args.timing_repeats % 2:
        raise ValueError('Positive query count and positive even timing repeats required')
    reference = json.loads(args.input.read_bytes())
    if reference['args']['query_split'] != 'val' or reference['args']['candidate_region'] != 'gcn-boundary':
        raise ValueError('Validation boundary-protocol records required')
    checkpoint = Path(reference['args']['checkpoint_dir']) / (reference['args']['dataset'] + '_gcn.pt')
    if sha(checkpoint) != reference['checkpoint_sha256']:
        raise ValueError('Checkpoint mismatch')
    state = torch.load(checkpoint, map_location=args.device, weights_only=False)
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
    files = [Path(__file__), Path('experiments/train_gcn.py'), Path('experiments/audit_local_gcn.py'),
             *sorted(Path('src').rglob('*.py'))]
    sources = {str(p): sha(p) for p in files}
    rows = [r for r in reference['rows'] if r['method'] == args.method]
    queries = []
    for label in (1, 0):
        available = list(dict.fromkeys(tuple(r['query']) for r in rows if r['label'] == label))
        if len(available) < args.queries_per_class:
            raise ValueError('Insufficient saved queries')
        queries.extend(available[:args.queries_per_class])
    memberships, global_metadata = {}, {}
    policies = ['diffusion', 'random', 'normalized-edge']
    if args.feature_controls:
        policies.extend(['raw-feature', 'first-linear-feature'])
    if args.ward_controls:
        policies.extend(['ward-raw-feature', 'ward-first-linear-feature'])
    offline_seconds = 0.
    if args.partition_scope == 'global':
        synchronize(args.device)
        start = time.perf_counter()
        memberships, global_metadata = prepare_partitions(data, policies, encoder.convs[0].lin)
        synchronize(args.device)
        offline_seconds = time.perf_counter() - start
    records, failures = [], []
    for query in queries:
        selected = sorted([r for r in rows if tuple(r['query']) == query], key=lambda r: r['budget'])
        if len(selected) != 3 or [r['budget'] for r in selected] != [5, 10, 20]:
            raise ValueError('Expected exactly three support budgets per query')
        synchronize(args.device)
        start = time.perf_counter()
        region = compact_gcn_query(data, query, len(encoder.convs))
        synchronize(args.device)
        region_seconds = time.perf_counter() - start
        original_supports = [torch.tensor([min(a, b) * data.num_nodes + max(a, b) for a, b in row['support']],
                                           dtype=torch.long, device=args.device) for row in selected]
        local_supports = [region.map_keys(keys) for keys in original_supports]
        conditions = [('full', None, False)]
        for row, keys in zip(selected, local_supports):
            conditions.extend([(f'retain-{row["budget"]}', keys, True), (f'delete-{row["budget"]}', keys, False)])
        prepared, preparation = {}, {}
        labels = {'local-operator': torch.arange(region.data.num_nodes, device=args.device)}
        local_metadata, grouping_seconds = {}, 0.
        if args.partition_scope == 'local-matched':
            synchronize(args.device)
            start = time.perf_counter()
            local_memberships, local_metadata = prepare_partitions(
                region.data, policies, encoder.convs[0].lin, region.targets.flatten())
            synchronize(args.device)
            grouping_seconds = time.perf_counter() - start
            labels.update(local_memberships)
        else:
            for policy, membership in memberships.items():
                local = membership[region.original_nodes].clone()
                endpoints = region.targets.flatten()
                local[endpoints[0]] = data.num_nodes
                local[endpoints[1]] = data.num_nodes + 1
                labels[policy] = local
        for name, membership in labels.items():
            synchronize(args.device)
            start = time.perf_counter()
            prepared[name] = ProjectedGCN(model, region.data, membership, region.targets)
            synchronize(args.device)
            preparation[name] = {'seconds': time.perf_counter() - start,
                                 'nodes': prepared[name].num_coarse_nodes,
                                 'relative_feature_residual': prepared[name].relative_feature_residual()}
            projected = prepared[name]
            transformed = encoder.convs[0].lin(region.data.x)
            lifted = encoder.convs[0].lin(projected.features)[projected.membership] * projected.node_scale[:, None]
            denominator = torch.linalg.vector_norm(transformed)
            residual = torch.linalg.vector_norm(transformed - lifted)
            preparation[name]['relative_first_linear_residual'] = float(residual / denominator) if denominator > 0 else float(residual)
        exact = prepared['local-operator']

        def evaluate(name):
            output = []
            for _, support, retain in conditions:
                if name in prepared:
                    output.append(prepared[name](support, retain=retain).squeeze())
                    continue
                mask = exact.edge_mask(support, retain=retain)
                current = Data(x=region.data.x, edge_index=region.data.edge_index[:, mask], edge_weight=exact.weights[mask])
                targets = region.targets
                if name == 'dynamic-local':
                    active = compact_gcn_query(current, targets.flatten(), len(encoder.convs))
                    current, targets = active.data, active.targets
                output.append(model(current.x, current.edge_index, targets, edge_weight=current.edge_weight).squeeze())
            return torch.stack(output)

        target = torch.tensor(query, device=args.device).reshape(2, 1)
        base = model(data.x, data.edge_index, target, edge_weight=getattr(data, 'edge_weight', None)).reshape(-1)[0]
        kept = group_deletion_logits(model, data, *query, original_supports, 1, retain=True)
        deleted = group_deletion_logits(model, data, *query, original_supports, 1)
        authoritative = torch.stack([base, *[v for pair in zip(kept, deleted) for v in pair]])
        saved_values = authoritative.new_tensor([selected[0]['full_logit'],
                                                  *[r[k] for r in selected for k in ('retained_logit', 'removed_logit')]])
        if not (torch.isclose(authoritative, saved_values, atol=2e-5, rtol=1e-5)
                & ((authoritative > 0) == (saved_values > 0))).all():
            failures.append({'query': query, 'comparison': 'saved-vs-full-replay'})
        accuracy = {}
        names = ['local-operator', 'native-local', 'dynamic-local', *policies]
        for name in names:
            values = evaluate(name)
            if not torch.isfinite(values).all():
                raise ValueError('Nonfinite inference result')
            disagreement = ((values > 0) != (authoritative > 0))
            accuracy[name] = {'logits': values.cpu().tolist(),
                              'absolute_errors': (values - authoritative).abs().cpu().tolist(),
                              'binary_disagreement': disagreement.cpu().tolist()}
            if name in ('local-operator', 'native-local', 'dynamic-local') and not (
                    torch.isclose(values, authoritative, atol=2e-5, rtol=1e-5) & ~disagreement).all():
                failures.append({'query': query, 'comparison': 'full-vs-' + name})
        residuals = {name: [value.operator_residual_fro(support, retain=retain)
                            for _, support, retain in conditions] for name, value in prepared.items()}
        timing = {name: benchmark_pair(lambda: evaluate('local-operator'), lambda: evaluate(name),
                                       args.device, args.timing_repeats, warmups=1)
                  for name in names if name != 'local-operator'}
        records.append({'query': query, 'conditions': [c[0] for c in conditions],
                        'local_partitions': local_metadata, 'local_grouping_seconds_all_partitions': grouping_seconds,
                        'original_nodes': data.num_nodes, 'local_nodes': region.data.num_nodes,
                        'region_seconds': region_seconds, 'preparation': preparation,
                        'authoritative_logits': authoritative.cpu().tolist(), 'accuracy': accuracy,
                        'operator_residual_fro': residuals, 'timing_against_local_operator': timing})
        print(f'Completed {len(records)}/{len(queries)} queries', flush=True)
    summaries = {}
    for name in records[0]['accuracy']:
        errors = [e for row in records for e in row['accuracy'][name]['absolute_errors']]
        item = {'interventions': len(errors), 'mean_absolute_logit_error': statistics.mean(errors),
                'max_absolute_logit_error': max(errors),
                'binary_disagreements': sum(sum(row['accuracy'][name]['binary_disagreement']) for row in records)}
        if name != 'local-operator':
            ratios = [statistics.median(t['full']['wall_seconds'] for t in row['timing_against_local_operator'][name])
                      / statistics.median(t['local']['wall_seconds'] for t in row['timing_against_local_operator'][name])
                      for row in records]
            item['median_local_operator_over_variant_seconds'] = statistics.median(ratios)
        summaries[name] = item
    report = {'study': 'projected-gcn-forward-feasibility-v3', 'scope': __doc__,
              'input': str(args.input), 'input_sha256': sha(args.input), 'checkpoint_sha256': sha(checkpoint),
              'split_sha256': reference['split_sha256'], 'source_sha256_at_start': sources,
              'dataset': config['dataset'], 'training_seed': config['seed'], 'device': args.device,
              'selection': 'First requested count per class in input order, without outcome selection.',
              'method': args.method, 'timing_repeats': args.timing_repeats, 'torch': torch.__version__,
              'feature_controls': args.feature_controls,
              'ward_controls': args.ward_controls,
              'partition_scope': args.partition_scope,
              'compression_scope': ('Same global merge budget, not matched query-local cluster counts after restriction and endpoint isolation.'
                                    if args.partition_scope == 'global' else
                                    'Equal achieved cluster counts per query; partitioning is query-local with endpoint merge edges excluded, not reusable global coarsening.'),
              'offline_seconds_all_partitions': offline_seconds, 'global_partitions': global_metadata,
              'timing_labels': {'full': 'exact local singleton/operator-reuse baseline', 'local': 'named comparison variant'},
              'memory_scope': 'Peak extra allocation per process, excluding resident model/data/projected features; not total GPU memory.',
              'exact_control_failures': failures, 'summaries': summaries, 'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
    if failures:
        raise RuntimeError('Exact control/replay checks failed; inspect preserved output')


if __name__ == '__main__':
    main()
