"""Audit saved swap-search results and write descriptive, paired JSON summaries.

This checks recorded invariants, not model re-execution or statistical
generalization. Query-level observations on one graph are not independent
training runs, so this utility does not report significance or confidence bands.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


METRICS = ('necessity_flip', 'sufficiency_agreement', 'necessity_confidence_drop',
           'sufficiency_confidence_drop')


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
    return {'input': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
            'dataset': report['args']['dataset'], 'seed': report['args']['seed'],
            'query_split': report['args']['query_split'], 'device': report['args']['device'],
            'audited_swap_rows': len(paired), 'summaries': summaries}


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
