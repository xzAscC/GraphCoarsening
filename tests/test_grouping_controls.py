import unittest
import torch

from src.grouping_controls import GroupingControls


class GroupingControlsTests(unittest.TestCase):
    def fixture(self, device='cpu'):
        edges = torch.tensor([[0, 1, 2, 3, 0, 4], [1, 2, 3, 0, 2, 5]], device=device)
        return torch.cat((edges, edges.flip(0)), 1)

    def test_controls_cover_nodes_and_match_merge_budget(self):
        control = GroupingControls(self.fixture(), 7)
        for policy in ('random', 'normalized-edge', 'diffusion', 'eigen'):
            partition, diagnostics = control.partition(policy, alpha=.5, width=2)
            self.assertEqual(sorted(v for cluster in partition for v in cluster), list(range(7)))
            self.assertEqual(len(partition), 4)
            self.assertIn([6], partition)
            self.assertEqual(diagnostics['components'], 3)

    def test_random_control_is_reproducible_and_input_order_invariant(self):
        edges = self.fixture()
        state = torch.random.get_rng_state().clone()
        first = GroupingControls(edges, 7).partition('random', seed=3)
        second = GroupingControls(edges.flip(1).flip(0), 7).partition('random', seed=3)
        self.assertEqual(first, second)
        torch.testing.assert_close(state, torch.random.get_rng_state())

    def test_structural_control_prioritizes_high_normalized_weight(self):
        # A normalized edge between the two degree-one nodes exceeds either
        # edge of the path, so it must be merged first.
        edges = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
        partition, _ = GroupingControls(edges, 5).partition('normalized-edge', alpha=.2)
        self.assertIn([3, 4], partition)

    def test_zero_weight_edges_do_not_merge_components(self):
        edges = torch.tensor([[0, 1], [1, 0]])
        for policy in ('random', 'normalized-edge', 'diffusion', 'eigen'):
            partition, _ = GroupingControls(edges, 2, torch.zeros(2)).partition(policy, alpha=1, width=1)
            self.assertEqual(partition, [[0], [1]])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_gpu_structural_and_random_controls_match_cpu(self):
        cpu = GroupingControls(self.fixture(), 7)
        gpu = GroupingControls(self.fixture('cuda'), 7)
        for policy in ('random', 'normalized-edge'):
            self.assertEqual(cpu.partition(policy, seed=4), gpu.partition(policy, seed=4))

    def test_invalid_policy_and_budget(self):
        control = GroupingControls(self.fixture(), 7)
        with self.assertRaises(ValueError):
            control.partition('unknown')
        with self.assertRaises(ValueError):
            control.partition('random', alpha=-.1)

    def test_signal_ranking_matches_singleton_loss_and_preserves_rng(self):
        edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        signals = torch.tensor([[0., 0.], [10., 0.], [10., 1.], [20., 0.]])
        before = torch.random.get_rng_state().clone()
        partition, details = GroupingControls(edges, 4).partition('signal', alpha=.25, signals=signals)
        self.assertEqual(partition, [[0], [1, 2], [3]])
        self.assertEqual(details['signal_width'], 2)
        torch.testing.assert_close(before, torch.random.get_rng_state())
        # Orthogonal transforms and offsets leave pair distances unchanged.
        rotated = signals @ torch.tensor([[0., -1.], [1., 0.]]) + 5
        self.assertEqual(partition, GroupingControls(edges.flip(1), 4).partition(
            'signal', alpha=.25, signals=rotated)[0])
        if torch.cuda.is_available():
            self.assertEqual(partition, GroupingControls(edges.cuda(), 4).partition(
                'signal', alpha=.25, signals=signals.cuda())[0])

    def test_signal_validation_empty_edges_and_cluster_budget(self):
        control = GroupingControls(self.fixture(), 7)
        for signals in (None, torch.ones(6, 2), torch.ones(7, 0), torch.ones(7, 2, dtype=torch.long),
                        torch.full((7, 2), float('nan'))):
            with self.assertRaises(ValueError):
                control.partition('signal', signals=signals)
        partition, _ = control.partition('signal', alpha=.5, signals=torch.arange(7.).reshape(-1, 1))
        self.assertEqual(len(partition), 4)
        self.assertIn([6], partition)
        empty = GroupingControls(torch.empty(2, 0, dtype=torch.long), 3)
        self.assertEqual(empty.partition('signal', signals=torch.ones(3, 1))[0], [[0], [1], [2]])


if __name__ == '__main__':
    unittest.main()
