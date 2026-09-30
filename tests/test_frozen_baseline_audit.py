import copy
import unittest

from experiments.audit_frozen_baselines import audit_reports, sigmoid, validate_row


class FrozenBaselineAuditTests(unittest.TestCase):
    def row(self, method='a', kept=1., removed=-1.):
        return {'query': [0, 2], 'method': method, 'label': 1, 'budget': 1,
                'candidate_edges': 3, 'effective_budget': 1, 'support_edges': 1, 'support': [[0, 1]],
                'full_logit': 1., 'retained_logit': kept, 'removed_logit': removed,
                'necessity_flip': float(removed <= 0), 'sufficiency_agreement': float(kept > 0),
                'necessity_confidence_drop': sigmoid(1.)-sigmoid(removed),
                'sufficiency_confidence_drop': sigmoid(1.)-sigmoid(kept)}

    def report(self, rows):
        return {'protocol': 'undirected-original-support-v1', 'reference_replay_failures': [],
                'input_sha256': 'i', 'checkpoint_sha256': 'c', 'split_sha256': {},
                'dataset': 'toy', 'training_seed': 1, 'rows': rows}

    def test_pairing_and_duplicate_reference_verification(self):
        a, b = self.row(), self.row('b', removed=1.)
        result = audit_reports([self.report([a]), self.report([a, b])])
        self.assertEqual(result['unique_rows'], 2)
        self.assertEqual(result['duplicate_reference_rows_verified'], 1)
        self.assertEqual(result['paired_comparisons'][0]['binary_pareto_left_vs_right']['wins'], 1)
        altered = copy.deepcopy(a)
        altered['support'] = [[1, 2]]
        with self.assertRaises(ValueError):
            audit_reports([self.report([a]), self.report([altered])])

    def test_corrupt_metrics_coverage_and_metadata_rejected(self):
        with self.assertRaises(ValueError):
            validate_row(self.row() | {'necessity_confidence_drop': 0.})
        with self.assertRaises(ValueError):
            audit_reports([self.report([self.row(), self.row()])])
        with self.assertRaises(ValueError):
            audit_reports([self.report([self.row(), self.row('b') | {'query': [1, 3]}])])
        with self.assertRaises(ValueError):
            audit_reports([self.report([self.row()]), self.report([self.row('b')]) | {'checkpoint_sha256': 'different'}])

    def test_stable_extreme_sigmoid(self):
        self.assertEqual(sigmoid(1000), 1.)
        self.assertEqual(sigmoid(-1000), 0.)


if __name__ == '__main__':
    unittest.main()
