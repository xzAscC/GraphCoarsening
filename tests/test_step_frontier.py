import unittest

from experiments.audit_step_frontier import prefix_metrics
from tests import test_acceptance_audit as fixtures


class StepFrontierTests(unittest.TestCase):
    def test_fixed_prefix_and_early_stop(self):
        row = fixtures.AcceptanceAuditTests().fixture()
        for entry in row['search_trace']:
            entry['full_graph_rechecks'] = int(entry['accepted'])
        initial = prefix_metrics(row, 0)
        final = prefix_metrics(row, 1)
        self.assertEqual((initial['necessity'], initial['sufficiency']), (1, 0))
        self.assertEqual((final['necessity'], final['sufficiency']), (1, 1))
        self.assertEqual(initial['total_forward_calls'], 3)
        self.assertEqual(final['total_forward_calls'], 17)
        self.assertEqual(final, prefix_metrics(row, 8))
        with self.assertRaises(ValueError):
            prefix_metrics(row, -1)


if __name__ == '__main__':
    unittest.main()
