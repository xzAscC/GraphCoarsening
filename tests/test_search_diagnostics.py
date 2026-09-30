import unittest

from experiments.diagnose_support_search import compare


def row(keep, remove, edge, method='Swap-coarse'):
    return {'full_logit': 2., 'retained_logit': keep, 'removed_logit': remove,
            'effective_budget': 1, 'candidate_edges': 3, 'budget': 1,
            'query': [0, 4], 'method': method, 'support': [[0, edge]],
            'search_trace': [{'accepted': False, 'proposals': 0}]}


class SearchDiagnosticsTests(unittest.TestCase):
    def test_binary_gain_can_conflict_with_continuous_guard(self):
        result = compare(row(-1., -2., 1), row(1., -1., 2, 'Alternative'))
        self.assertTrue(result['binary_dominance'])
        self.assertGreater(result['objective_gain'], 0.)
        self.assertTrue(result['deletion_guard_conflict'])
        self.assertFalse(result['passes_final_state_guards_with_tolerance'])
        self.assertEqual(result['required_edge_exchanges'], 1)

    def test_both_components_improve(self):
        result = compare(row(-1., 1., 1), row(1., -1., 2, 'Alternative'))
        self.assertTrue(result['binary_dominance'])
        self.assertTrue(result['passes_final_state_guards_with_tolerance'])

    def test_mismatched_budget_rejected(self):
        alternate = row(1., -1., 2, 'Alternative')
        alternate['effective_budget'] = 2
        with self.assertRaises(ValueError):
            compare(row(-1., 1., 1), alternate)
