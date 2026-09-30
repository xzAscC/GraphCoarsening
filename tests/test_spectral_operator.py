import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.spectral import compute_normalized_adjacency, compute_top_k_eigenpairs


class SpectralOperatorTests(unittest.TestCase):
    def check_operator(self, device):
        edges = torch.tensor([[0, 0, 1, 1, 0], [1, 1, 0, 0, 0]], device=device)
        weights = torch.tensor([2., 3., 2., 3., 4.], dtype=torch.float64, device=device)
        actual = compute_normalized_adjacency(edges, 3, weights).to_dense()
        adjacency = torch.tensor([[5., 5., 0.], [5., 1., 0.], [0., 0., 1.]],
                                 dtype=weights.dtype, device=device)
        inv = adjacency.sum(1).rsqrt()
        expected = inv[:, None] * adjacency * inv[None, :]
        torch.testing.assert_close(actual, expected)
        vals, vecs, _ = compute_top_k_eigenpairs(actual, 2)
        self.assertEqual(vals.device, actual.device)
        torch.testing.assert_close(actual @ vecs, vecs * vals)

    def test_weighted_duplicates_and_self_loops_cpu(self):
        self.check_operator('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_weighted_operator_and_eigensolve_cuda(self):
        self.check_operator('cuda')

    def test_sparse_branch_selection_and_dtype(self):
        n = 10001
        indices = torch.arange(n).repeat(2, 1)
        matrix = torch.sparse_coo_tensor(indices, torch.ones(n, dtype=torch.float64), (n, n))
        with patch('src.spectral.spla.eigsh', return_value=(np.array([1.]), np.ones((n, 1)))) as solve:
            values, vectors, _ = compute_top_k_eigenpairs(matrix, 1)
        self.assertEqual(solve.call_args.kwargs['which'], 'LA')
        self.assertEqual(values.dtype, matrix.dtype)
        self.assertEqual(vectors.device, matrix.device)

    def test_invalid_weights(self):
        edges = torch.tensor([[0], [1]])
        for weight in (torch.tensor([-1.]), torch.tensor([float('nan')])):
            with self.assertRaises(ValueError):
                compute_normalized_adjacency(edges, 2, weight)
