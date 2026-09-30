"""Finite numerical sanity checks, not substitutes for mathematical proofs."""
import unittest
import numpy as np


class TheoryChecks(unittest.TestCase):
    def test_polynomial_bound_with_feature_mismatch(self):
        rng = np.random.default_rng(314)
        for _ in range(50):
            n, m = 8, 4
            a = rng.uniform(size=(n, n))
            a = (a + a.T) / 2
            d = a.sum(1)
            lap = np.eye(n) - a / np.sqrt(d[:, None] * d[None, :])
            p = np.zeros((n, m))
            p[np.arange(n), np.arange(n) // 2] = 1 / np.sqrt(2)
            lc = p.T @ lap @ p
            x, xc = rng.normal(size=(n, 3)), rng.normal(size=(m, 3))
            coefficients = rng.normal(size=4)
            q = sum(v * np.linalg.matrix_power(lap, j) for j, v in enumerate(coefficients))
            qc = sum(v * np.linalg.matrix_power(lc, j) for j, v in enumerate(coefficients))
            delta = np.linalg.norm(lap @ p - p @ lc, 2)
            factor = sum(abs(coefficients[j]) * j * 2 ** (j - 1) for j in range(1, 4))
            bound = np.linalg.norm(q, 2) * np.linalg.norm(x - p @ xc) + factor * delta * np.linalg.norm(xc)
            self.assertLessEqual(np.linalg.norm(q @ x - p @ qc @ xc), bound + 1e-10)

    def test_combinatorial_resistance_contracts(self):
        rng = np.random.default_rng(271)
        for _ in range(50):
            a = rng.uniform(size=(8, 8))
            a = (a + a.T) / 2
            np.fill_diagonal(a, 0)
            lap = np.diag(a.sum(1)) - a
            indicator = np.zeros((8, 4))
            indicator[np.arange(8), np.arange(8) // 2] = 1
            coarse = indicator.T @ lap @ indicator
            fine_inv, coarse_inv = np.linalg.pinv(lap), np.linalg.pinv(coarse)
            for s, t in [(0, 1), (0, 7), (3, 4)]:
                b = np.eye(8)[s] - np.eye(8)[t]
                bc = indicator.T @ b
                fine, reduced = b @ fine_inv @ b, bc @ coarse_inv @ bc
                self.assertLessEqual(reduced, fine + 1e-10)
                self.assertGreaterEqual(reduced, -1e-10)


if __name__ == '__main__':
    unittest.main()
