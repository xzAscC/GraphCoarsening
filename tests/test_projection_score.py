import unittest
import torch
from src.spectral import pair_projection_scores
from src.coarsen import GraphCoarsener
from unittest.mock import patch


class ProjectionTests(unittest.TestCase):
    def test_projection_pipeline_does_not_evaluate_legacy_surrogate(self):
        edges = torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]])
        with patch('src.coarsen.compute_perturbation_scores', side_effect=AssertionError('legacy called')):
            c = GraphCoarsener(k=2, alpha=0.5, score_method='projection').fit(edges, 4)
        torch.testing.assert_close(c.scores, pair_projection_scores(edges, c.right_vecs))

    def test_exact_single_merge_identity(self):
        g = torch.Generator().manual_seed(7)
        for n in range(3, 12):
            x = torch.randn(n, 4, generator=g, dtype=torch.float64)
            p = torch.zeros(n, n - 1, dtype=x.dtype)
            p[0, 0] = p[1, 0] = 2 ** -0.5
            p[2:, 1:] = torch.eye(n - 2, dtype=x.dtype)
            loss = torch.linalg.norm(x - p @ p.T @ x).square()
            score = pair_projection_scores(torch.tensor([[0], [1]]), x)[0]
            torch.testing.assert_close(loss, score)

    def test_orientation_and_basis_invariance(self):
        g = torch.Generator().manual_seed(9)
        x = torch.randn(7, 3, generator=g, dtype=torch.float64)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=x.dtype))
        edges = torch.tensor([[0, 2, 4], [1, 3, 5]])
        reference = pair_projection_scores(edges, x)
        torch.testing.assert_close(reference, pair_projection_scores(edges.flip(0), x))
        torch.testing.assert_close(reference, pair_projection_scores(edges, x @ rotation))

    def test_repeated_merge_cannot_use_singleton_formula(self):
        # Exact Ward increment for clusters A,B is |A||B|/(|A|+|B|)||meanA-meanB||^2.
        x = torch.tensor([[0.], [2.], [5.]], dtype=torch.float64)
        before = ((x[:2] - x[:2].mean(0)) ** 2).sum()
        after = ((x - x.mean(0)) ** 2).sum()
        increment = 2 / 3 * (x[:2].mean() - x[2, 0]) ** 2
        torch.testing.assert_close(after - before, increment)
        self.assertNotAlmostEqual(increment.item(), pair_projection_scores(torch.tensor([[1], [2]]), x).item())


if __name__ == '__main__':
    unittest.main()
