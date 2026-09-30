import unittest

import torch

from src.coarsen import GraphCoarsener
from src.diffusion_signals import NonstationaryDiffusion
from src.spectral import pair_projection_scores


class DiffusionSignalsTests(unittest.TestCase):
    def fixture(self, device='cpu'):
        # Two disconnected weighted paths, a zero-weight bridge, a duplicate
        # entry, an explicit loop, and an isolated node.
        edges = torch.tensor([[0, 0, 1, 3, 4, 2, 3], [1, 1, 2, 4, 5, 3, 3]], device=device)
        weights = torch.tensor([.25, .75, 2., .5, 1.5, 0., .4], dtype=torch.float64, device=device)
        edges = torch.cat((edges, edges.flip(0)), 1)
        weights = torch.cat((weights, weights))
        return edges, weights, NonstationaryDiffusion(edges, 7, weights)

    def check_dense_oracle(self, device):
        _, _, diffusion = self.fixture(device)
        self.assertEqual(diffusion.num_components, 3)
        operator = diffusion.operator.to_dense()
        values, vectors = torch.linalg.eigh(operator)
        stationary = vectors[:, values > 1 - 1e-12]
        projector = stationary @ stationary.T
        identity = torch.eye(7, dtype=torch.float64, device=device)
        torch.testing.assert_close(diffusion.remove_stationary(identity), identity - projector,
                                   atol=1e-12, rtol=1e-12)
        for steps in (0, 1, 4, 12):
            expected = torch.linalg.matrix_power((identity + operator) / 2, steps) @ (identity - projector)
            actual = diffusion.filter(identity, steps)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(projector @ actual, torch.zeros_like(actual), atol=1e-12, rtol=0)
        # Negative adjacency eigenmodes get a nonnegative lazy response;
        # they are not silently substituted for algebraically leading modes.
        self.assertLess(values.min().item(), 0)

    def test_dense_oracle_cpu(self):
        self.check_dense_oracle('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_dense_oracle_cuda_and_cpu_agreement(self):
        self.check_dense_oracle('cuda')
        _, _, cpu = self.fixture()
        _, _, gpu = self.fixture('cuda')
        torch.testing.assert_close(cpu.sketch(seed=17), gpu.sketch(seed=17).cpu(), atol=1e-12, rtol=1e-12)

    def test_reproducible_without_global_rng_changes(self):
        _, _, diffusion = self.fixture()
        state = torch.random.get_rng_state().clone()
        first = diffusion.sketch(width=32, steps=4, seed=8)
        torch.testing.assert_close(state, torch.random.get_rng_state())
        torch.testing.assert_close(first, diffusion.sketch(width=32, steps=4, seed=8), rtol=0, atol=0)
        self.assertFalse(torch.equal(first, diffusion.sketch(width=32, steps=4, seed=9)))
        self.assertEqual(torch.count_nonzero(first[6]).item(), 0)

    def test_sketch_scores_approximate_full_kernel_not_top_k(self):
        edges, _, diffusion = self.fixture()
        kernel = diffusion.filter(torch.eye(7, dtype=torch.float64), steps=4)
        expected = pair_projection_scores(edges, kernel)
        actual = pair_projection_scores(edges, diffusion.sketch(width=32768, steps=4, seed=13))
        torch.testing.assert_close(actual, expected, rtol=.04, atol=1e-12)

    def test_complete_stationary_space_is_removed(self):
        # More stationary dimensions than sketch width, including isolates.
        edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        diffusion = NonstationaryDiffusion(edges, 9, torch.ones(4, dtype=torch.float64))
        self.assertEqual(diffusion.num_components, 7)
        signal = diffusion.sketch(width=2, steps=2, seed=0)
        torch.testing.assert_close(signal, diffusion.remove_stationary(signal), atol=1e-14, rtol=0)
        torch.testing.assert_close(signal[4:], torch.zeros_like(signal[4:]), atol=0, rtol=0)

    def test_permutation_equivariance_for_fixed_probes(self):
        edges, weights, original = self.fixture()
        permutation = torch.tensor([5, 2, 6, 0, 3, 1, 4])
        renamed = NonstationaryDiffusion(permutation[edges], 7, weights)
        probes = torch.randn(7, 3, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        renamed_probes = torch.empty_like(probes)
        renamed_probes[permutation] = probes
        torch.testing.assert_close(original.filter(probes, 4), renamed.filter(renamed_probes, 4)[permutation],
                                   atol=1e-12, rtol=1e-12)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            NonstationaryDiffusion(torch.tensor([[0], [1]]), 2)
        with self.assertRaises(ValueError):
            NonstationaryDiffusion(torch.tensor([[0], [2]]), 2)
        with self.assertRaises(ValueError):
            NonstationaryDiffusion(torch.tensor([[0], [0]]), 1, torch.tensor([-1.]))
        _, _, diffusion = self.fixture()
        for kwargs in ({'width': 0}, {'steps': -1}, {'width': True}, {'steps': 1.5}):
            with self.assertRaises(ValueError):
                diffusion.sketch(**kwargs)

    def test_coarsener_integration_keeps_eigen_branch_explicit(self):
        edges, weights, _ = self.fixture()
        model = GraphCoarsener(score_method='projection', signal_policy='diffusion', signal_width=8)
        model.fit(edges, 7, edge_weight=weights)
        self.assertIsNone(model.eigenvalues)
        self.assertIsNone(model.right_vecs)
        self.assertEqual(model.signals.shape, (7, 8))
        torch.testing.assert_close(model.scores, pair_projection_scores(edges, model.signals))
        self.assertEqual(sorted(v for part in model.partition for v in part), list(range(7)))
        legacy = GraphCoarsener(k=2, score_method='projection').fit(edges, 7, edge_weight=weights)
        self.assertEqual(legacy.eigenvalues.shape, (2,))
        torch.testing.assert_close(legacy.signals, legacy.right_vecs)
        with self.assertRaisesRegex(ValueError, 'projection'):
            GraphCoarsener(signal_policy='diffusion')


if __name__ == '__main__':
    unittest.main()
