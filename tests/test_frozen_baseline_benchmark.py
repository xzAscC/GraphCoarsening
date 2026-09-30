import unittest

from experiments.benchmark_frozen_baselines import query_seed, summarize


class FrozenBaselineBenchmarkTests(unittest.TestCase):
    def test_query_seed_is_orientation_and_order_independent(self):
        self.assertEqual(query_seed(4, (2, 9)), query_seed(4, (9, 2)))
        self.assertNotEqual(query_seed(4, (2, 9)), query_seed(5, (2, 9)))

    def test_summary_rejects_duplicate_queries_and_reports_directions(self):
        row = {'query': [2, 9], 'method': 'baseline', 'budget': 5,
               'necessity_flip': 1., 'sufficiency_agreement': 0.,
               'necessity_confidence_drop': .3, 'sufficiency_confidence_drop': .7}
        result = summarize([row])['baseline']['5']
        self.assertEqual(result['queries'], 1)
        self.assertEqual(result['necessity_flip'], 1.)
        self.assertEqual(result['sufficiency_confidence_drop'], .7)
        with self.assertRaises(ValueError):
            summarize([row, row | {'query': [9, 2]}])


if __name__ == '__main__':
    unittest.main()
