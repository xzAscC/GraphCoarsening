import unittest

import torch

from src.ward_partition import connected_ward_partition


class WardPartitionTests(unittest.TestCase):
    def test_dynamic_cost_avoids_stale_chain(self):
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
        signals = torch.tensor([[0.], [1.], [2.1], [3.3]], dtype=torch.double)
        partition, stats = connected_ward_partition(edges, signals, alpha=.5)
        # Fixed singleton ranking first merges 0-1 then 1-2. Disjoint
        # matching instead keeps two compact pairs at the same merge budget.
        self.assertEqual(partition, [[0, 1], [2, 3]])
        self.assertAlmostEqual(stats['direct_squared_projection_loss'], .5 + .72)
        _, final = connected_ward_partition(edges, signals, alpha=1)
        self.assertEqual(final['clusters'], 1)
        self.assertEqual(len(final['rounds']), 2)
        self.assertAlmostEqual(final['cumulative_squared_loss'], float((signals - signals.mean()).square().sum()))

    def test_budget_and_zero_signal(self):
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
        signals = torch.arange(4., dtype=torch.double).reshape(-1, 1)
        for cap in (0., .01, .04, .1, 1.):
            _, stats = connected_ward_partition(edges, signals, alpha=1, max_relative_loss=cap)
            self.assertLessEqual(stats['direct_squared_projection_loss'], cap * 14 + 1e-12)
        partition, stats = connected_ward_partition(edges, torch.zeros(4, 2), alpha=1, max_relative_loss=0)
        self.assertEqual(partition, [[0, 1, 2, 3]])
        self.assertEqual(stats['direct_squared_projection_loss'], 0)

    def test_components_duplicates_order_and_gpu(self):
        edges = torch.tensor([[0, 1, 2, 4], [1, 2, 3, 5]])
        signals = torch.tensor([[0.], [1.], [3.], [7.], [20.], [25.], [30.]])
        cpu, stats = connected_ward_partition(edges, signals, alpha=.75)
        self.assertEqual(cpu, [[0, 1, 2, 3], [4, 5], [6]])
        duplicate = torch.cat((edges, edges.flip(0), edges), 1).flip(1)
        self.assertEqual(cpu, connected_ward_partition(duplicate, signals, alpha=.75)[0])
        if torch.cuda.is_available():
            gpu, gpu_stats = connected_ward_partition(edges.cuda(), signals.cuda(), alpha=.75)
            self.assertEqual(cpu, gpu)
            self.assertAlmostEqual(stats['direct_squared_projection_loss'], gpu_stats['direct_squared_projection_loss'])

    def test_no_merges_validation_and_rng(self):
        signals = torch.ones(3, 2)
        edges = torch.empty(2, 0, dtype=torch.long)
        before = torch.random.get_rng_state().clone()
        self.assertEqual(connected_ward_partition(edges, signals)[0], [[0], [1], [2]])
        torch.testing.assert_close(before, torch.random.get_rng_state())
        for kwargs in ({'alpha': -1}, {'max_relative_loss': 1.1}):
            with self.assertRaises(ValueError):
                connected_ward_partition(edges, signals, **kwargs)
        with self.assertRaises(ValueError):
            connected_ward_partition(edges, torch.full((3, 2), float('nan')))
        with self.assertRaises(ValueError):
            connected_ward_partition(torch.tensor([[0], [3]]), signals)


if __name__ == '__main__':
    unittest.main()
