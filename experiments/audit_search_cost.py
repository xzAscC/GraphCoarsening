"""Reconstruct search model calls from traces, not FLOPs or end-to-end runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def search_cost(trace, batch_size):
    if type(batch_size) is not int or batch_size < 1 or not trace:
        raise ValueError('Nonempty trace and positive integer batch size required')
    proposals = rechecks = screen_calls = 0
    for index, step in enumerate(trace):
        p, r = step['proposals'], step['full_graph_rechecks']
        if (step['step'] != index or type(p) is not int or type(r) is not int
                or not 0 <= r <= p or (index == 0 and (p or r or step['accepted']))):
            raise ValueError('Invalid search counters')
        if index and ((step['accepted'] and r == 0)
                      or (not step['accepted'] and index != len(trace)-1)):
            raise ValueError('Invalid acceptance or stopping trace')
        proposals += p
        rechecks += r
        screen_calls += 2 * ((p + batch_size - 1) // batch_size)
    steps = len(trace)-1
    return {
        'gradient_steps': steps,
        'screened_proposals': proposals,
        'serial_rechecks': rechecks,
        'initial_forward_calls': 3,
        'gradient_forward_calls': 2 * steps,
        'screen_forward_calls': screen_calls,
        'recheck_forward_calls': 2 * rechecks,
        'total_forward_calls': 3 + 2 * steps + screen_calls + 2 * rechecks,
        'graph_evaluations': 3 + 2 * steps + 2 * proposals + 2 * rechecks,
        'autograd_grad_calls': steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous results')
    cohorts = []
    for path in args.inputs:
        raw = path.read_bytes()
        report = json.loads(raw)
        batch = (report['settings']['batch_size'] if 'settings' in report
                 else report['args']['intervention_batch_size'])
        methods = {}
        for row in report['rows']:
            if not row.get('search_trace'):
                continue
            cost = search_cost(row['search_trace'], batch)
            methods.setdefault(row['method'], []).append(cost)
        cohorts.append({'input': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                        'methods': {m: {'conditions': len(rows),
                                        'means': {k: statistics.mean(r[k] for r in rows) for k in rows[0]}}
                                    for m, rows in methods.items()}})
    result = {
        'scope': 'Search only: one full prediction and two initial interventions, two gradient forwards per attempted step, retained/deleted screening batches, and two serial forwards per recheck. Excludes grouping, saliency initialization, final evaluation, and training.',
        'limitations': 'Calls and graph copies are not FLOPs, latency, or memory. Batched calls contain multiple graphs; retained and deleted graphs differ in edge count. Local screening and full rechecks have different graph sizes. One autograd.grad invocation differentiates both gradient forwards. No equal-cost quality claim follows from these counts.',
        'cohorts': cohorts}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
