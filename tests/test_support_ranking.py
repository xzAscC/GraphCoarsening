import unittest

import torch
from torch_geometric.data import Data

from experiments.run_support_benchmark import rank_support


class RankingTests(unittest.TestCase):
    def test_directional_scores_sum_and_budget_counts_undirected_edges(self):
        exp = Data(edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                   edge_weight=torch.tensor([3., 3., 5., 0.]))
        kept = rank_support(exp, torch.tensor([1, 5]), 3, 1)
        self.assertEqual(kept.edge_index.tolist(), [[0], [1]])

    def test_outside_candidates_filtered_and_ties_deterministic(self):
        exp = Data(edge_index=torch.tensor([[0, 2], [2, 0]]),
                   edge_weight=torch.tensor([100., 100.], dtype=torch.float64))
        kept = rank_support(exp, torch.tensor([1, 5]), 3, 1)
        self.assertEqual(kept.edge_index.tolist(), [[0], [1]])

    def test_empty_candidates(self):
        exp = Data(edge_index=torch.empty((2, 0), dtype=torch.long), edge_weight=torch.empty(0))
        kept = rank_support(exp, torch.empty(0, dtype=torch.long), 3, 0)
        self.assertEqual(tuple(kept.edge_index.shape), (2, 0))


if __name__ == '__main__':
    unittest.main()
