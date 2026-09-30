import copy
import unittest
import numpy as np
from src.partition import isolate_query_endpoints


class EndpointIsolationTests(unittest.TestCase):
    def test_partition_and_growth_bound_for_all_endpoint_pairs(self):
        for original in [[[0,1,2,3,4]], [[0,1,2],[3,4]], [[0],[1],[2],[3],[4]]]:
            before = copy.deepcopy(original)
            for a in range(5):
                for b in range(5):
                    result = isolate_query_endpoints(original, a, b)
                    self.assertEqual(sorted(v for c in result for v in c), list(range(5)))
                    self.assertIn([a], result)
                    self.assertIn([b], result)
                    self.assertLessEqual(len(result), len(original) + len({a,b}))
                    self.assertEqual(original, before)
                    for c in original:
                        remaining = [v for v in c if v not in {a,b}]
                        if remaining:
                            self.assertIn(remaining, result)

    def test_missing_endpoint_fails(self):
        with self.assertRaises(ValueError):
            isolate_query_endpoints([[0,1]], 0, 2)

    def test_projection_error_cannot_increase(self):
        rng = np.random.default_rng(8)
        for n in range(3, 20):
            original = [list(range(n // 2)), list(range(n // 2, n))]
            refined = isolate_query_endpoints(original, 0, n - 1)
            y = rng.normal(size=(n, 5))
            def residual(partition):
                p = np.zeros((n, len(partition)))
                for j, c in enumerate(partition):
                    p[c, j] = len(c) ** -0.5
                return np.linalg.norm(y - p @ p.T @ y)
            self.assertLessEqual(residual(refined), residual(original) + 1e-10)


if __name__ == '__main__':
    unittest.main()
