import unittest

import torch

from experiments.run_support_benchmark import reference_query_indices


class QueryReferenceTests(unittest.TestCase):
    def test_preserves_first_occurrence_order_and_deduplicates_methods(self):
        reference = {'rows': [{'query': query, 'label': label}
                              for query, label in [([3, 1], 1), ([1, 3], 1), ([0, 2], 0), ([0, 1], 1)]]}
        pool = torch.tensor([[0, 1], [1, 3]])
        self.assertEqual(reference_query_indices(reference, pool, 1, 4), [1, 0])

    def test_rejects_query_outside_split(self):
        reference = {'rows': [{'query': [0, 2], 'label': 1}]}
        with self.assertRaises(ValueError):
            reference_query_indices(reference, torch.tensor([[0], [1]]), 1, 4)


if __name__ == '__main__':
    unittest.main()
