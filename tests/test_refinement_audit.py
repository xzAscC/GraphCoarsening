import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.audit_support_refinement import audit


class RefinementAuditTests(unittest.TestCase):
    def report(self, count):
        rows = []
        for method in ('Saliency-supportive', 'Swap-gradient'):
            row = {'query': [0, 3], 'label': 1, 'method': method, 'budget': 2,
                   'effective_budget': count, 'support_edges': count,
                   'support': [[0, 1], [1, 2]][:count], 'full_logit': 2.197224577,
                   'necessity_flip': 0., 'sufficiency_agreement': 1.,
                   'necessity_confidence_drop': .2, 'sufficiency_confidence_drop': .1,
                   'ranking_seconds': .01}
            if method.startswith('Swap-'):
                row['search_trace'] = [{'step': 0, 'proposals': 0, 'accepted': False,
                                        'objective': .1, 'capped_retained_probability': .8,
                                        'removed_probability': .7}]
            rows.append(row)
        return {'protocol': 'undirected-original-support-v1',
                'args': {'dataset': 'fixture', 'seed': 42, 'query_split': 'val', 'device': 'cpu'},
                'checkpoint_sha256': 'fixture-checkpoint', 'split_sha256': {'train': 'fixture'},
                'rows': rows}

    def test_separates_actual_budget_mismatches(self):
        for old_count in (1, 2):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                reference = root / 'reference.json'
                reference.write_text(json.dumps(self.report(old_count)))
                current = self.report(2)
                current['args']['query_reference'] = str(reference)
                current['args']['candidate_region'] = 'gcn-boundary'
                current['query_reference_sha256'] = hashlib.sha256(reference.read_bytes()).hexdigest()
                path = root / 'current.json'
                path.write_text(json.dumps(current))
                result = audit(path)
                for comparison in result['reference_comparison']['comparisons']:
                    self.assertEqual(comparison['matched_actual_budget_queries'], int(old_count == 2))
                    self.assertEqual(len(comparison['different_actual_budget_queries']), int(old_count != 2))

    def test_rejects_recorded_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.report(2)
            trace = report['rows'][1]['search_trace']
            worse = copy.deepcopy(trace[0])
            worse['objective'] = .09
            trace.append(worse)
            path = Path(directory) / 'regression.json'
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(AssertionError, 'objective regressed'):
                audit(path)

    def test_close_logits_must_preserve_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.report(2)
            for row in original['rows']:
                row['full_logit'] = -1e-8
            reference = root / 'reference.json'
            reference.write_text(json.dumps(original))
            current = copy.deepcopy(original)
            for row in current['rows']:
                row['full_logit'] = 1e-8
            current['args']['query_reference'] = str(reference)
            current['query_reference_sha256'] = hashlib.sha256(reference.read_bytes()).hexdigest()
            path = root / 'current.json'
            path.write_text(json.dumps(current))
            with self.assertRaisesRegex(AssertionError, 'Original predictions differ'):
                audit(path)


if __name__ == '__main__':
    unittest.main()
