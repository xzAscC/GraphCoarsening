import unittest

import torch
from torch_geometric.data import Data

from experiments.benchmark_projected_gcn import prepare_partitions


class MatchedPartitionTests(unittest.TestCase):
    def test_all_controls_match_and_protect_without_changing_data(self):
        policies = ['diffusion', 'random', 'normalized-edge', 'raw-feature',
                    'first-linear-feature', 'ward-raw-feature', 'ward-first-linear-feature']
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            edges = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 5, 5]], device=device)
            edges = torch.cat((edges, edges.flip(0)), 1)
            x = torch.arange(18., device=device).reshape(6, 3)
            data = Data(x=x, edge_index=edges, edge_weight=torch.ones(12, device=device))
            saved = data.clone()
            memberships, metadata = prepare_partitions(data, policies, lambda a: a[:, :2], [0, 5])
            # Protected endpoints plus one remaining connected component.
            for name, membership in memberships.items():
                self.assertEqual(len(torch.unique(membership)), 3)
                self.assertEqual(metadata[name]['expected_clusters'], 3)
                counts = torch.bincount(membership)
                self.assertEqual(counts[membership[0]], 1)
                self.assertEqual(counts[membership[5]], 1)
            torch.testing.assert_close(data.x, saved.x)
            torch.testing.assert_close(data.edge_index, saved.edge_index)
            torch.testing.assert_close(data.edge_weight, saved.edge_weight)

    def test_zero_weight_edges_and_global_count(self):
        edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        data = Data(x=torch.ones(4, 2), edge_index=edges, edge_weight=torch.zeros(4))
        memberships, _ = prepare_partitions(data, ['random', 'ward-raw-feature'], lambda a: a)
        for labels in memberships.values():
            self.assertEqual(len(torch.unique(labels)), 4)


if __name__ == '__main__':
    unittest.main()
