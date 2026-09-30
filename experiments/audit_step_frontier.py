"""Fixed-step prefix analysis of saved acceptance-policy validation trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.audit_acceptance_policy import audit_trace
from experiments.audit_search_cost import search_cost


def prefix_metrics(row, steps, batch_size=8):
    if type(steps) is not int or steps < 0:
        raise ValueError('Nonnegative integer step limit required')
    audit_trace(row)
    trace = row['search_trace'][:steps+1]
    final = trace[-1]
    positive = row['full_logit'] > 0
    necessity = int((final['removed_logit'] > 0) != positive)
    sufficiency = int((final['retained_logit'] > 0) == positive)
    return dict(necessity=necessity, sufficiency=sufficiency,
                joint=necessity*sufficiency, objective=final['objective'],
                **search_cost(trace, batch_size))


def summarize(report):
    if report['settings'] != dict(steps=8, additions=12, removals=3, batch_size=8,
                                   exchange_size=1, local_gcn=False, groups=None):
        raise ValueError('Unexpected search configuration')
    policies = ('Swap-componentwise', 'Swap-binary-monotone')
    indexed = {m: {} for m in policies}
    for row in report['rows']:
        if row['method'] not in indexed:
            continue
        key = (tuple(row['query']), row['budget'])
        target = indexed[row['method']]
        if key in target:
            raise ValueError('Duplicate condition')
        target[key] = row
    if not indexed[policies[0]] or indexed[policies[0]].keys() != indexed[policies[1]].keys():
        raise ValueError('Unpaired conditions')
    output = []
    for limit in (0, 1, 2, 4, 8):
        values = {m: {k: prefix_metrics(r, limit) for k, r in indexed[m].items()} for m in policies}
        counts = dict(wins=0, losses=0, ties=0, tradeoffs=0)
        for key in indexed[policies[0]]:
            old, new = (values[m][key] for m in policies)
            diffs = [new[k]-old[k] for k in ('necessity', 'sufficiency')]
            name = ('ties' if diffs == [0, 0] else 'wins' if min(diffs) >= 0
                    else 'losses' if max(diffs) <= 0 else 'tradeoffs')
            counts[name] += 1
        output.append({'step_limit': limit, 'binary_monotone_vs_componentwise': counts,
                       'methods': {m: {'conditions': len(rows), 'means': {
                           k: statistics.mean(r[k] for r in rows.values()) for k in next(iter(rows.values()))}}
                                   for m, rows in values.items()}})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier analysis')
    cohorts = []
    for path in args.inputs:
        raw = path.read_bytes()
        report = json.loads(raw)
        cohorts.append({'input': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                        'dataset': report['dataset'], 'training_seed': report['training_seed'],
                        'prefixes': summarize(report)})
    result = {'scope': 'Offline fixed-step prefixes of recorded validation trajectories, not new model runs. Includes early-stopped and empty-candidate cases. Means pool query-budget conditions within each cohort.',
              'limitations': 'Step limits were chosen for retrospective development analysis. No per-query outcome selection. Same step limits are not equal compute: rechecks and early stopping vary. Counts are not runtime or FLOPs. Prefix supports were not saved, so this is a logit-trace analysis, not an independent support replay. Conditions share queries and checkpoints; no significance or held-out claims.',
              'cohorts': cohorts}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    for cohort in cohorts:
        print(cohort['dataset'], cohort['training_seed'],
              [(p['step_limit'], p['binary_monotone_vs_componentwise']) for p in cohort['prefixes']])


if __name__ == '__main__':
    main()
