"""Join fixed-baseline runs and check paired recorded intervention outcomes.

This is a record audit, not another model replay or a significance test.
All inputs must share the reference artifact, checkpoint, split and queries.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


METRICS = ('necessity_flip', 'sufficiency_agreement',
           'necessity_confidence_drop', 'sufficiency_confidence_drop')


def sigmoid(value):
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def validate_row(row):
    if not all(math.isfinite(row[k]) for k in (*METRICS, 'full_logit', 'retained_logit', 'removed_logit')):
        raise ValueError('Nonfinite metrics')
    support = {tuple(sorted(e)) for e in row['support']}
    if (len(support) != len(row['support']) or len(support) != row['support_edges']
            or len(support) != row['effective_budget']
            or row['effective_budget'] != min(row['budget'], row['candidate_edges'])):
        raise ValueError('Support budget mismatch or duplicate edges')
    full, kept, removed = (row[k] for k in ('full_logit', 'retained_logit', 'removed_logit'))
    sign = 1 if full > 0 else -1
    expected = [float((full > 0) != (removed > 0)), float((full > 0) == (kept > 0)),
                sign * (sigmoid(full) - sigmoid(removed)), sign * (sigmoid(full) - sigmoid(kept))]
    if any(not math.isclose(row[k], v, rel_tol=1e-10, abs_tol=1e-12) for k, v in zip(METRICS, expected)):
        raise ValueError('Recorded metric disagrees with logits')


def audit_reports(reports):
    if not reports:
        raise ValueError('No reports')
    rows, duplicates = {}, 0
    for report in reports:
        if report['protocol'] != 'undirected-original-support-v1' or report['reference_replay_failures']:
            raise ValueError('Wrong protocol or failed model replay')
        for key in ('input_sha256', 'checkpoint_sha256', 'split_sha256', 'dataset', 'training_seed'):
            if report[key] != reports[0][key]:
                raise ValueError('Input, checkpoint or split mismatch')
        seen = set()
        for row in report['rows']:
            validate_row(row)
            key = (row['method'], tuple(sorted(row['query'])), row['budget'])
            if key in seen:
                raise ValueError('Duplicate result within one input')
            seen.add(key)
            if key in rows:
                old = rows[key]
                if ({tuple(sorted(e)) for e in old['support']} != {tuple(sorted(e)) for e in row['support']}
                        or any(old[k] != row[k] for k in ('label', 'candidate_edges', 'effective_budget', *METRICS[:2]))
                        or any(not math.isclose(old[k], row[k], rel_tol=1e-5, abs_tol=2e-5)
                               for k in ('full_logit', 'retained_logit', 'removed_logit'))):
                    raise ValueError('Repeated reference method differs across inputs')
                duplicates += 1
            else:
                rows[key] = row
    methods = sorted({key[0] for key in rows})
    if not methods:
        raise ValueError('No recorded methods')
    groups = {m: {key[1:]: r for key, r in rows.items() if key[0] == m} for m in methods}
    if any(group.keys() != groups[methods[0]].keys() for group in groups.values()):
        raise ValueError('Method query/budget coverage differs')
    budgets = sorted({key[2] for key in rows})
    summaries, comparisons = {}, []
    for method, group in groups.items():
        summaries[method] = {}
        for budget in budgets:
            selected = [r for (_, b), r in group.items() if b == budget]
            summaries[method][str(budget)] = {'queries': len(selected),
                **{k: statistics.mean(r[k] for r in selected) for k in METRICS},
                'both_binary_criteria': statistics.mean(r['necessity_flip'] * r['sufficiency_agreement'] for r in selected)}
    for left, right in itertools.combinations(methods, 2):
        for budget in budgets:
            pairs = [(r, groups[right][key]) for key, r in groups[left].items() if key[1] == budget]
            counts = dict(wins=0, losses=0, ties=0, tradeoffs=0)
            same_support = 0
            for a, b in pairs:
                if (any(a[k] != b[k] for k in ('label', 'candidate_edges', 'effective_budget'))
                        or not math.isclose(a['full_logit'], b['full_logit'], rel_tol=1e-5, abs_tol=2e-5)
                        or (a['full_logit'] > 0) != (b['full_logit'] > 0)):
                    raise ValueError('Paired query context differs')
                delta = [a[k]-b[k] for k in METRICS[:2]]
                category = ('ties' if delta == [0, 0] else 'wins' if min(delta) >= 0 else
                            'losses' if max(delta) <= 0 else 'tradeoffs')
                counts[category] += 1
                same_support += {tuple(sorted(e)) for e in a['support']} == {tuple(sorted(e)) for e in b['support']}
            comparisons.append({'left': left, 'right': right, 'budget': budget,
                                'paired_queries': len(pairs), 'binary_pareto_left_vs_right': counts,
                                'identical_supports': same_support,
                                'mean_difference_left_minus_right': {k: statistics.mean(a[k]-b[k] for a, b in pairs) for k in METRICS}})
    return {'unique_rows': len(rows), 'duplicate_reference_rows_verified': duplicates,
            'summaries': summaries, 'paired_comparisons': comparisons,
            'scope': 'Descriptive recorded-data audit, not independent model replay or significance. Query conditions on one trained graph are not independent training runs. Higher sufficiency confidence drop is worse.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior audits')
    reports, files = [], {}
    for path in args.inputs:
        raw = path.read_bytes()
        report = json.loads(raw)
        if hashlib.sha256(Path(report['input']).read_bytes()).hexdigest() != report['input_sha256']:
            raise ValueError('Reference artifact changed')
        files[str(path)] = hashlib.sha256(raw).hexdigest()
        reports.append(report)
    result = audit_reports(reports) | {'input_sha256': files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result['summaries'], indent=2))


if __name__ == '__main__':
    main()
