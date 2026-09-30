"""Audit acceptance invariants and descriptive paired outcomes per cohort."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.audit_frozen_baselines import audit_reports, sigmoid


def audit_trace(row):
    trace = row['search_trace']
    policy = row['method'].removeprefix('Swap-')
    if policy not in ('componentwise', 'binary-monotone') or not trace or len(trace) > 9:
        raise ValueError('Invalid policy or trace length')
    sign = 1 if row['full_logit'] > 0 else -1
    p0 = sigmoid(sign * row['full_logit'])
    for index, step in enumerate(trace):
        if step['step'] != index or step['acceptance_policy'] != policy or not 0 <= step['proposals'] <= 36:
            raise ValueError('Trace policy, step or proposal count differs')
        retained = min(sigmoid(sign * step['retained_logit']), p0)
        deleted = sigmoid(sign * step['removed_logit'])
        for key, expected in [('capped_retained_probability', retained), ('removed_probability', deleted),
                              ('objective', retained-deleted)]:
            if not math.isclose(step[key], expected, rel_tol=1e-5, abs_tol=2e-5):
                raise ValueError('Trace probability arithmetic disagrees')
        if not index:
            if step['accepted'] or step['proposals']:
                raise ValueError('Invalid initial trace entry')
            continue
        old = trace[index-1]
        if step['accepted']:
            if not step['objective'] > old['objective'] + 1e-7:
                raise ValueError('Accepted objective did not strictly increase')
            for key, agreement in [('retained_logit', True), ('removed_logit', False)]:
                now = ((step[key] > 0) == (sign > 0)) == agreement
                before = ((old[key] > 0) == (sign > 0)) == agreement
                if now < before:
                    raise ValueError('Binary regression in accepted step')
            if policy == 'componentwise' and (step['capped_retained_probability'] < old['capped_retained_probability']
                                               or step['removed_probability'] > old['removed_probability']):
                raise ValueError('Default continuous guard violated')
        elif index != len(trace)-1 or any(step[k] != old[k] for k in
                                         ('objective', 'retained_logit', 'removed_logit')):
            raise ValueError('Rejected proposal changed state or search continued')
    if sum(t['proposals'] for t in trace) != row['evaluated_proposals']:
        raise ValueError('Actual proposal count mismatch')
    if any(not math.isclose(trace[-1][k], row[k], rel_tol=1e-5, abs_tol=2e-5)
           for k in ('retained_logit', 'removed_logit')):
        raise ValueError('Final trace differs from independently evaluated support')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior audit')
    cohorts, seen = [], set()
    aggregate = dict(wins=0, losses=0, ties=0, tradeoffs=0)
    for path in args.inputs:
        raw = path.read_bytes()
        report = json.loads(raw)
        if report['settings'] != {'steps': 8, 'additions': 12, 'removals': 3, 'batch_size': 8,
                                  'exchange_size': 1, 'local_gcn': False, 'groups': None}:
            raise ValueError('Unexpected study settings')
        if hashlib.sha256(Path(report['input']).read_bytes()).hexdigest() != report['input_sha256']:
            raise ValueError('Reference changed')
        key = (report['dataset'], report['training_seed'])
        if key in seen:
            raise ValueError('Duplicate dataset/seed cohort')
        seen.add(key)
        audited = audit_reports([report])
        for row in report['rows']:
            if row['method'].startswith('Swap-'):
                audit_trace(row)
        paired = [c for c in audited['paired_comparisons']
                  if c['left'] == 'Swap-binary-monotone' and c['right'] == 'Swap-componentwise']
        for comparison in paired:
            for name, count in comparison['binary_pareto_left_vs_right'].items():
                aggregate[name] += count
        cost = {}
        for method in ('Swap-componentwise', 'Swap-binary-monotone'):
            selected = [r for r in report['rows'] if r['method'] == method]
            cost[method] = {'mean_evaluated_proposals': statistics.mean(r['evaluated_proposals'] for r in selected),
                            'total_search_seconds': sum(r['ranking_seconds'] for r in selected)}
        cohorts.append({'dataset': key[0], 'training_seed': key[1], 'input': str(path),
                        'input_sha256': hashlib.sha256(raw).hexdigest(),
                        'summaries': audited['summaries'], 'paired': paired, 'cost': cost})
    result = {'cohorts': cohorts, 'descriptive_aggregate_binary_outcomes': aggregate,
              'scope': 'Recorded metric and trace audit, not model rerun or significance. Query/budget conditions share graphs and checkpoints. Costs are actual proposal counts and scoped search timings, not full-pipeline times.'}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
    print(json.dumps(aggregate))
    for cohort in cohorts:
        print(cohort['dataset'], cohort['training_seed'], [c['binary_pareto_left_vs_right'] for c in cohort['paired']])


if __name__ == '__main__':
    main()
