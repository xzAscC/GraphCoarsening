import unittest

import torch

from src.spectral import compute_normalized_adjacency, pair_projection_scores


class StationaryCutoffTests(unittest.TestCase):
    def test_partial_repeated_eigenspace_changes_merge_scores(self):
        half = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]])
        edges = torch.cat((half, half.flip(0)), dim=1)
        operator = compute_normalized_adjacency(edges, 6, torch.ones(8, dtype=torch.float64))
        basis = torch.zeros(6, 2, dtype=torch.float64)
        mode = torch.tensor([2., 3., 2.], dtype=torch.float64).sqrt() / (7. ** .5)
        basis[:3, 0], basis[3:, 1] = mode, mode
        torch.testing.assert_close(torch.sparse.mm(operator, basis), basis, atol=1e-14, rtol=0)
        torch.testing.assert_close(basis.T @ basis, torch.eye(2, dtype=torch.float64), atol=1e-14, rtol=0)
        # Both columns are legitimate top-1 eigenvectors, but rank the same
        # original edges differently. This is not an eigenpair residual error.
        first = pair_projection_scores(edges, basis[:, :1])
        second = pair_projection_scores(edges, basis[:, 1:])
        self.assertGreater(first[0].item(), first[2].item())
        self.assertLess(second[0].item(), second[2].item())
        rotation = torch.tensor([[1., -1.], [1., 1.]], dtype=torch.float64) / (2. ** .5)
        torch.testing.assert_close(pair_projection_scores(edges, basis),
                                   pair_projection_scores(edges, basis @ rotation), atol=1e-14, rtol=0)


if __name__ == '__main__':
    unittest.main()
