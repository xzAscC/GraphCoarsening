"""Audit saved swap-search results and write descriptive, paired JSON summaries.

This checks recorded invariants, not model re-execution or statistical
generalization. Query-level observations on one graph are not independent
training runs, so this utility does not report significance or confidence bands.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


METRICS = ('necessity_flip', 'sufficiency_agreement', 'necessity_confidence_drop',
           'sufficiency_confidence_drop')


def validate_proposal_trace(trace, args):
    """Check new mixed-proposal accounting, retaining legacy trace compatibility."""
    for index, step in enumerate(trace):
        histogram = step.get('proposals_by_exchange_size')
        if histogram is None:
            continue
        if step['step'] != index or index > args['swap_steps']:
            raise AssertionError('Invalid search step count')
        if any(key not in ('1', '2') or not isinstance(value, int) or value < 1
               for key, value in histogram.items()):
            raise AssertionError('Invalid proposal-size histogram')
        if sum(histogram.values()) != step['proposals']:
            raise AssertionError('Proposal histogram does not match total')
        if step['proposals'] > args['swap_additions'] * args['swap_removals']:
            raise AssertionError('Shared proposal cap exceeded')
        if step['accepted'] and str(step['exchange_size']) not in histogram:
            raise AssertionError('Accepted exchange size was not proposed')


def audit(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    if report['protocol'] != 'undirected-original-support-v1':
        raise ValueError('Unsupported intervention protocol')
    rows = report['rows']
    keys = [(tuple(r['query']), r['method'], r['budget']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate query/method/budget row')
    baseline = {(tuple(r['query']), r['budget']): r for r in rows
                if r['method'] == 'Saliency-supportive'}
    paired = []
    for row in rows:
        if not row['method'].startswith('Swap-'):
            continue
        original = baseline[(tuple(row['query']), row['budget'])]
        if row['effective_budget'] != original['effective_budget']:
            raise AssertionError('Unmatched edge budget')
        support = {tuple(sorted(edge)) for edge in row['support']}
        if len(support) != row['support_edges'] or len(support) != row['effective_budget']:
            raise AssertionError('Support size does not match reported budget')
        if row['necessity_flip'] < original['necessity_flip'] or row['sufficiency_agreement'] < original['sufficiency_agreement']:
            raise AssertionError('Binary fidelity regressed')
        if row['necessity_confidence_drop'] < original['necessity_confidence_drop'] - 1e-6:
            raise AssertionError('Deletion effect regressed')
        if max(0, row['sufficiency_confidence_drop']) > max(0, original['sufficiency_confidence_drop']) + 1e-6:
            raise AssertionError('Retention deficit regressed')
        trace = row['search_trace']
        validate_proposal_trace(trace, report['args'])
        for previous, current in zip(trace, trace[1:]):
            if current['objective'] < previous['objective'] - 1e-12:
                raise AssertionError('Search objective regressed')
            if current['capped_retained_probability'] < previous['capped_retained_probability'] - 1e-12:
                raise AssertionError('Search retention regressed')
            if current['removed_probability'] > previous['removed_probability'] + 1e-12:
                raise AssertionError('Search deletion regressed')
        paired.append({**{k: row[k] for k in ('method', 'budget')},
                       'accepted_steps': sum(t['accepted'] for t in trace),
                       'evaluated_proposals': sum(t['proposals'] for t in trace),
                       **{k: row[k] - original[k] for k in METRICS}})
    summaries = []
    for method in sorted({r['method'] for r in rows}):
        for budget in sorted({r['budget'] for r in rows}):
            group = [r for r in rows if r['method'] == method and r['budget'] == budget]
            if not group:
                continue
            summary = {'method': method, 'budget': budget, 'queries': len(group),
                       'mean': {m: statistics.mean(r[m] for r in group) for m in METRICS},
                       'median_ranking_seconds': statistics.median(r['ranking_seconds'] for r in group)}
            comparison = [r for r in paired if r['method'] == method and r['budget'] == budget]
            if comparison:
                summary['paired_mean_difference_from_signed_saliency'] = {
                    m: statistics.mean(r[m] for r in comparison) for m in METRICS}
                summary['mean_accepted_steps'] = statistics.mean(r['accepted_steps'] for r in comparison)
                summary['mean_evaluated_proposals'] = statistics.mean(r['evaluated_proposals'] for r in comparison)
            summaries.append(summary)
    candidate_comparison = None
    reference_path = report['args'].get('query_reference')
    if reference_path:
        reference_raw = Path(reference_path).read_bytes()
        if hashlib.sha256(reference_raw).hexdigest() != report['query_reference_sha256']:
            raise AssertionError('Query reference changed since the run')
        reference = json.loads(reference_raw)
        if reference['checkpoint_sha256'] != report['checkpoint_sha256'] or reference['split_sha256'] != report['split_sha256']:
            raise AssertionError('Reference checkpoint or splits differ')
        reference_rows = {(tuple(r['query']), r['method'], r['budget']): r for r in reference['rows']}
        if not set(reference_rows).issubset(set(keys)):
            raise AssertionError('Reference query/method/budget pairs differ')
        if {(k[0], k[2]) for k in reference_rows} != {(k[0], k[2]) for k in keys}:
            raise AssertionError('Reference query/budget coverage differs')
        comparisons = []
        max_logit_difference = 0.0
        for method in sorted({r['method'] for r in rows}):
            if not any(key[1] == method for key in reference_rows):
                continue
            for budget in sorted({r['budget'] for r in rows}):
                matched, mismatched = [], []
                for row in rows:
                    if row['method'] != method or row['budget'] != budget:
                        continue
                    prior = reference_rows[(tuple(row['query']), method, budget)]
                    current_logit, prior_logit = row['full_logit'], prior['full_logit']
                    max_logit_difference = max(max_logit_difference, abs(current_logit - prior_logit))
                    # Float32 GPU reductions can differ by a few ULPs at large
                    # logit magnitudes. Still reject every changed binary decision.
                    if ((current_logit > 0) != (prior_logit > 0)
                            or not math.isclose(current_logit, prior_logit, rel_tol=1e-6, abs_tol=1e-5)):
                        raise AssertionError('Original predictions differ')
                    if row['effective_budget'] == prior['effective_budget']:
                        matched.append({m: row[m] - prior[m] for m in METRICS})
                    else:
                        mismatched.append({'query': row['query'],
                                           'reference_edges': prior['effective_budget'],
                                           'current_edges': row['effective_budget']})
                comparisons.append({'method': method, 'budget': budget,
                                    'matched_actual_budget_queries': len(matched),
                                    'different_actual_budget_queries': mismatched,
                                    'paired_mean_difference_on_matched_actual_budgets':
                                        {m: statistics.mean(r[m] for r in matched) for m in METRICS}
                                        if matched else None})
        candidate_comparison = {
            'reference': reference_path,
            'reference_candidate_region': reference['args'].get('candidate_region', 'induced'),
            'max_original_logit_difference': max_logit_difference,
            'original_logit_tolerance': {'relative': 1e-6, 'absolute': 1e-5,
                                         'require_identical_binary_decision': True},
            'scope': 'Descriptive differences conditional on equal actual edge counts; unequal counts are listed, not pooled.',
            'comparisons': comparisons}
    return {'input': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
            'dataset': report['args']['dataset'], 'seed': report['args']['seed'],
            'query_split': report['args']['query_split'], 'device': report['args']['device'],
            'candidate_region': report['args'].get('candidate_region', 'induced'),
            'audited_swap_rows': len(paired), 'summaries': summaries,
            'reference_comparison': candidate_comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier audit outputs')
    result = {'audit': 'saved-support-swap-invariants-v1',
              'scope': 'Recorded paired validation checks; not independent model re-execution or significance testing.',
              'runs': [audit(path) for path in args.inputs]}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
    print(f'Audited {len(result["runs"])} runs; saved {args.output}')


if __name__ == '__main__':
    main()
