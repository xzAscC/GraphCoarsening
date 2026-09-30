import random
import unittest

import torch
from torch_geometric.data import Data

from src.negative_sampling import (training_negative_exclusion, sample_training_negatives,
                                   undirected_keys)


def fixture():
    return Data(num_nodes=8, x=torch.randn(8, 3),
                train_pos_edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                val_pos_edge_index=torch.tensor([[2], [3]]),
                test_pos_edge_index=torch.tensor([[3], [4]]),
                val_neg_edge_index=torch.tensor([[4], [5]]),
                test_neg_edge_index=torch.tensor([[5], [6]]))


class TrainingNegativeTests(unittest.TestCase):
    def check_samples(self, device):
        random.seed(12)
        data = fixture()
        original = data.train_pos_edge_index.clone()
        exclusion = training_negative_exclusion(data).to(device)
        self.assertEqual(exclusion.size(1), 12)
        forbidden = undirected_keys(exclusion, data.num_nodes)
        for _ in range(20):
            edges = sample_training_negatives(exclusion, data.num_nodes, 12)
            self.assertEqual(edges.device.type, device)
            self.assertEqual(edges.size(1), 12)
            self.assertFalse(torch.isin(undirected_keys(edges, data.num_nodes), forbidden).any())
            self.assertFalse((edges[0] == edges[1]).any())
        torch.testing.assert_close(data.train_pos_edge_index, original)

    def test_cpu(self):
        self.check_samples('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self):
        self.check_samples('cuda')

    def test_reversed_cross_split_overlap(self):
        data = fixture()
        data.test_neg_edge_index = data.val_pos_edge_index.flip(0)
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            training_negative_exclusion(data)

    def test_missing_split(self):
        data = fixture()
        del data.test_neg_edge_index
        with self.assertRaisesRegex(ValueError, 'Missing split'):
            training_negative_exclusion(data)

    def test_invalid_node_and_self_loop(self):
        for edges in (torch.tensor([[8], [1]]), torch.tensor([[1], [1]])):
            data = fixture()
            data.test_neg_edge_index = edges
            with self.assertRaises(ValueError):
                training_negative_exclusion(data)

    def test_shortfall_and_zero_request(self):
        exclusion = torch.tensor([[0, 1], [1, 0]])
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            sample_training_negatives(exclusion, 2, 1)
        with self.assertRaises(ValueError):
            sample_training_negatives(exclusion, 2, 0)

    def test_train_epoch_uses_only_training_edges_for_propagation(self):
        from experiments.train_gcn import train_epoch, MLPLinkPredictor
        from src.models.gcn import GCN
        data = fixture()
        model, predictor = GCN(3, 4, 4, 2), MLPLinkPredictor(4, 4)
        seen = []
        hook = model.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[1].clone()))
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()))
        audit = []
        loss = train_epoch(model, predictor, data, optimizer, torch.device('cpu'), negative_audit=audit)
        hook.remove()
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0], data.train_pos_edge_index)
        self.assertEqual(audit[0]['excluded_pair_collisions'], 0)
        self.assertEqual(audit[0]['stored_negative_entries'], 4)
        self.assertEqual(len(audit[0]['sha256']), 64)


if __name__ == '__main__':
    unittest.main()
