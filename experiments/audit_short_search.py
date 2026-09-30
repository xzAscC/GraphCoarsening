"""Audit direct four-versus-eight-step runs without treating them as held-out evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.audit_acceptance_policy import audit_trace
from experiments.audit_frozen_baselines import audit_reports
from experiments.audit_search_cost import search_cost


def audit(report):
    expected = dict(steps={'componentwise': 8, 'binary-monotone': 4}, additions=12,
                    removals=3, batch_size=8, exchange_size=1, local_gcn=False, groups=None)
    if report['study'] != 'acceptance-short-search-v1' or report['settings'] != expected:
        raise ValueError('Wrong experiment settings')
    checked = audit_reports([report])
    grouped = {}
    for row in report['rows']:
        if not row['method'].startswith('Swap-'):
            continue
        audit_trace(row)
        policy = row['method'].removeprefix('Swap-')
        if len(row['search_trace']) > expected['steps'][policy]+1:
            raise ValueError('Step cap exceeded')
        grouped.setdefault(policy, []).append((row, search_cost(row['search_trace'], 8)))
    checked['costs'] = {policy: {
        'mean_forward_calls': statistics.mean(c['total_forward_calls'] for _, c in pairs),
        'mean_graph_evaluations': statistics.mean(c['graph_evaluations'] for _, c in pairs),
        'mean_gradient_steps': statistics.mean(c['gradient_steps'] for _, c in pairs),
        'total_search_seconds': sum(r['ranking_seconds'] for r, _ in pairs),
        'median_search_seconds': statistics.median(r['ranking_seconds'] for r, _ in pairs),
    } for policy, pairs in grouped.items()}
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier audit')
    raw = args.input.read_bytes()
    report = json.loads(raw)
    for path, expected in report['source_sha256_at_start'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Recorded source changed: ' + path)
    if hashlib.sha256(Path(report['input']).read_bytes()).hexdigest() != report['input_sha256']:
        raise ValueError('Reference changed')
    plan = Path('results/tnnls-reproduction/acceptance_short_search_plan.json')
    if hashlib.sha256(plan.read_bytes()).hexdigest() != report['plan_sha256']:
        raise ValueError('Plan changed')
    result = audit(report) | {'input': str(args.input), 'input_sha256': hashlib.sha256(raw).hexdigest(),
        'scope': 'Direct GPU search with independent final original-support evaluation; this tool audits saved arithmetic, coverage and acceptance traces. Previously inspected validation data, not held-out confirmation. Timings include search only on shared hardware; counts are not FLOPs. Four-step selection was retrospective. No coarsening benefit is measured.'}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result['costs'], indent=2))
    print(json.dumps(result['paired_comparisons'], indent=2))


if __name__ == '__main__':
    main()
