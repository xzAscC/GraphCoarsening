"""Paired saved-support diagnostics, not a proof of local-search reachability.

Compare alternative explainers' final supports with the search's final state.
A better alternative may require several simultaneous exchanges. Recorded
logits can expose acceptance-rule conflicts but do not replace model replay.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def compare(search, alternative, tolerance=1e-6):
    if (search['effective_budget'] != alternative['effective_budget']
            or search['candidate_edges'] != alternative['candidate_edges']):
        raise ValueError('Unmatched candidate or achieved support budget')
    if (not math.isclose(search['full_logit'], alternative['full_logit'], rel_tol=1e-6, abs_tol=1e-5)
            or (search['full_logit'] > 0) != (alternative['full_logit'] > 0)):
        raise ValueError('Original predictions differ')
    sign = 1 if search['full_logit'] > 0 else -1
    p_full = sigmoid(sign * search['full_logit'])

    def values(row):
        kept, removed = row['retained_logit'], row['removed_logit']
        cap = min(sigmoid(sign * kept), p_full)
        deleted_probability = sigmoid(sign * removed)
        return {'necessity': (removed > 0) != (row['full_logit'] > 0),
                'sufficiency': (kept > 0) == (row['full_logit'] > 0),
                'capped_retained_probability': cap, 'removed_probability': deleted_probability,
                'objective': cap - deleted_probability}

    current, other = values(search), values(alternative)
    binary_nonregression = all(other[k] >= current[k] for k in ('necessity', 'sufficiency'))
    binary_dominance = binary_nonregression and any(other[k] > current[k] for k in ('necessity', 'sufficiency'))
    retention_block = other['capped_retained_probability'] < current['capped_retained_probability'] - tolerance
    deletion_block = other['removed_probability'] > current['removed_probability'] + tolerance
    gain = other['objective'] - current['objective']
    edges = [{tuple(sorted(e)) for e in row['support']} for row in (search, alternative)]
    if any(len(s) != search['effective_budget'] for s in edges):
        raise ValueError('Saved support does not match its budget')
    return {'query': search['query'], 'budget': search['budget'],
            'alternative': alternative['method'], 'search_state': current, 'alternative_state': other,
            'binary_dominance': binary_dominance, 'binary_nonregression': binary_nonregression,
            'objective_gain': gain, 'retention_guard_conflict': retention_block,
            'deletion_guard_conflict': deletion_block,
            'passes_final_state_guards_with_tolerance': binary_nonregression and gain > tolerance
                and not retention_block and not deletion_block,
            'required_edge_exchanges': len(edges[0] - edges[1]),
            'accepted_search_steps': sum(row['accepted'] for row in search['search_trace']),
            'last_step_proposals': search['search_trace'][-1]['proposals']}


def diagnose(path, search_method, alternatives):
    raw = path.read_bytes()
    report = json.loads(raw)
    rows = {(tuple(row['query']), row['budget'], row['method']): row for row in report['rows']}
    if len(rows) != len(report['rows']):
        raise ValueError('Duplicate query/budget/method row')
    cases, summaries = [], []
    for method in alternatives:
        comparisons = [compare(row, rows[(query, budget, method)])
                       for (query, budget, name), row in rows.items() if name == search_method]
        if not comparisons:
            raise ValueError('No search rows')
        summaries.append({'alternative': method, 'paired_query_budgets': len(comparisons),
                          'binary_dominance_cases': sum(c['binary_dominance'] for c in comparisons),
                          'binary_dominance_with_continuous_guard_conflict': sum(
                              c['binary_dominance'] and (c['retention_guard_conflict'] or c['deletion_guard_conflict'])
                              for c in comparisons),
                          'passes_final_state_guards_with_tolerance': sum(
                              c['passes_final_state_guards_with_tolerance'] for c in comparisons)})
        cases.extend(c for c in comparisons if c['binary_dominance']
                     or c['passes_final_state_guards_with_tolerance'])
    return {'input': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
            'search_method': search_method, 'summaries': summaries, 'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--search-method', default='Swap-coarse')
    parser.add_argument('--alternatives', nargs='+', default=['CF2-link-adapted', 'GNNExplainer'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior diagnostics')
    result = {'diagnostic': 'paired-final-support-guards-v1',
              'scope': 'Saved-logit comparison with tolerance 1e-6. Does not reexecute the model, establish statistical superiority, or show that one-edge proposals could reach the alternative support.',
              'runs': [diagnose(path, args.search_method, args.alternatives) for path in args.inputs]}
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
    for run in result['runs']:
        print(run['input'], run['summaries'])


if __name__ == '__main__':
    main()
