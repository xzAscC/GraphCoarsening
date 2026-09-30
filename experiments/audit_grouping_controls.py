"""Audit saved grouping controls and make descriptive matched comparisons.

Checks saved records, not independent predictor re-execution or significance.
Random grouping seeds are not independent model-training repetitions.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.audit_support_refinement import METRICS, paired_policy_comparison, validate_proposal_trace


def audit(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    if report['benchmark'] != 'single-search-grouping-controls-v1' or report['failures']:
        raise ValueError('Wrong protocol or recorded failures')
    original_raw = Path(report['input']).read_bytes()
    if hashlib.sha256(original_raw).hexdigest() != report['input_sha256']:
        raise ValueError('Reference changed')
    reference = json.loads(original_raw)
    if (reference['checkpoint_sha256'] != report['checkpoint_sha256']
            or reference['split_sha256'] != report['split_sha256']):
        raise ValueError('Checkpoint or split mismatch')
    initial = {(tuple(row['query']), row['budget']): row for row in reference['rows']
               if row['method'] == 'Saliency-supportive'}
    policies = set(report['partitions']) | {'ungrouped'}
    expected_args = {'swap_steps': report['config']['search_steps'],
                     'swap_additions': report['config']['additions'],
                     'swap_removals': report['config']['removals']}
    seen, rows = set(), []
    for record in report['records']:
        key = (tuple(record['query']), record['budget'])
        if key in seen or key not in initial or set(record['results']) != policies:
            raise ValueError('Duplicate or unmatched query/policy coverage')
        seen.add(key)
        before = record['initial_metrics']
        saved = initial[key]
        if record['candidate_edges'] != saved['candidate_edges'] or record['label'] != saved['label']:
            raise ValueError('Candidate count or query label differs')
        for metric in (*METRICS, 'full_logit'):
            if not math.isclose(before[metric], saved[metric], rel_tol=1e-6, abs_tol=1e-5):
                raise ValueError('Initial metric differs from saved support')
        for policy, result in record['results'].items():
            metrics, trace = result['metrics'], result['trace']
            if not all(math.isfinite(metrics[m]) for m in (*METRICS, 'full_logit', 'retained_logit', 'removed_logit')):
                raise ValueError('Nonfinite metric')
            support = {tuple(sorted(edge)) for edge in result['support']}
            if len(support) != metrics['support_edges'] or len(support) != saved['effective_budget']:
                raise ValueError('Support count differs')
            if (metrics['necessity_flip'] < before['necessity_flip']
                    or metrics['sufficiency_agreement'] < before['sufficiency_agreement']
                    or metrics['necessity_confidence_drop'] < before['necessity_confidence_drop'] - 1e-6
                    or max(0, metrics['sufficiency_confidence_drop']) > max(0, before['sufficiency_confidence_drop']) + 1e-6):
                raise ValueError('Search regressed from shared initial support')
            validate_proposal_trace(trace, expected_args)
            for previous, current in zip(trace, trace[1:]):
                if (current['objective'] < previous['objective'] - 1e-12
                        or current['capped_retained_probability'] < previous['capped_retained_probability'] - 1e-12
                        or current['removed_probability'] > previous['removed_probability'] + 1e-12):
                    raise ValueError('Recorded search trajectory regressed')
            if sum(step['proposals'] for step in trace) != result['evaluated_proposals']:
                raise ValueError('Proposal count mismatch')
            rows.append({'query': record['query'], 'label': record['label'], 'budget': record['budget'],
                         'method': policy, 'support': result['support'], 'candidate_edges': record['candidate_edges'],
                         'effective_budget': metrics['support_edges'], 'search_trace': trace, **metrics})
    if seen != set(initial):
        raise ValueError('Missing query/budget records')
    pairs = [(name, 'ungrouped') for name in sorted(policies - {'ungrouped'})]
    pairs += [(name, control) for name in sorted(policies) if name.startswith(('diffusion-', 'eigen-'))
              for control in sorted(policies) if control.startswith(('random-', 'normalized-edge-'))]
    comparisons = [item for treatment, control in pairs
                   for item in paired_policy_comparison(rows, treatment, control)]
    return {'input': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'dataset': report['dataset'],
            'training_seed': report['training_seed'], 'audited_results': len(rows), 'comparisons': comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous audits')
    report = {'audit': 'grouping-control-record-audit-v1', 'scope': __doc__,
              'runs': [audit(path) for path in args.inputs]}
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
    print(f'Audited {sum(run["audited_results"] for run in report["runs"])} results')


if __name__ == '__main__':
    main()
