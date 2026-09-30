import unittest

import torch
from torch_geometric.data import Data

from experiments.train_gcn import MLPLinkPredictor
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel
from src.explainers.local_gcn import compact_gcn_query
from src.explainers.group_interventions import group_deletion_logits


class LocalGCNTests(unittest.TestCase):
    def check_equivalence(self, device):
        torch.manual_seed(118)
        n = 32
        half = torch.stack((torch.arange(n - 3), torch.arange(1, n - 2))).to(device)
        # Duplicate entries remain separate and have symmetric original weights.
        half = torch.cat((half, half[:, :4]), dim=1)
        edges = torch.cat((half, half.flip(0)), dim=1)
        weights = .2 + torch.rand(half.size(1), dtype=torch.float64, device=device)
        data = Data(x=torch.randn(n, 4, dtype=torch.float64, device=device),
                    edge_index=edges, edge_weight=weights.repeat(2))
        for layers in (2, 3):
            model = LinkPredictionModel(GCN(4, 7, 6, layers), MLPLinkPredictor(6, 7))
            model = model.to(device).double().eval()
            with torch.no_grad():
                for conv in model.encoder.convs:
                    conv.bias.uniform_(.1, .4)
            # Endpoint 31 is isolated; it must survive compaction.
            region = compact_gcn_query(data, [1, 31], layers)
            self.assertLess(region.data.num_nodes, n)
            self.assertIn(31, region.original_nodes.tolist())
            torch.testing.assert_close(region.data.edge_weight, data.edge_weight[region.original_edge_mask])
            observed_gradient = 0.
            for _ in range(8):
                tied = torch.rand(half.size(1), device=device) > .3
                keep = tied.repeat(2)
                gate = data.edge_weight.detach().clone().requires_grad_()
                original = model(data.x, edges[:, keep],
                                 torch.tensor([[1], [31]], device=device), edge_weight=gate[keep])
                local_keep = keep[region.original_edge_mask]
                local = model(region.data.x, region.data.edge_index[:, local_keep], region.targets,
                              edge_weight=gate[region.original_edge_mask][local_keep])
                full_grad, = torch.autograd.grad(original.sum(), gate)
                observed_gradient = max(observed_gradient, full_grad.abs().max().item())
                local_grad, = torch.autograd.grad(local.sum(), gate)
                torch.testing.assert_close(original, local, atol=1e-12, rtol=1e-10)
                torch.testing.assert_close(full_grad, local_grad, atol=1e-12, rtol=1e-10)
            self.assertGreater(observed_gradient, 1e-8, 'Avoid a constant-output equivalence fixture')
            groups = [region.original_keys[:size] for size in (0, 1, 3, 5)]
            local_groups = [region.map_keys(keys) for keys in groups]
            for keys, mapped in zip(groups, local_groups):
                torch.testing.assert_close(region.restore_keys(mapped), keys)
            for retain in (False, True):
                expected = group_deletion_logits(model, data, 1, 31, groups, 3, retain=retain)
                a, b = region.targets.flatten().tolist()
                actual = group_deletion_logits(model, region.data, a, b, local_groups, 3, retain=retain)
                torch.testing.assert_close(expected, actual, atol=1e-12, rtol=1e-10)

    def test_cpu_predictions_gradients_and_interventions(self):
        self.check_equivalence('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_predictions_gradients_and_interventions(self):
        self.check_equivalence('cuda')

    def test_empty_region_and_invalid_graphs(self):
        data = Data(x=torch.ones(5, 2), edge_index=torch.empty((2, 0), dtype=torch.long))
        region = compact_gcn_query(data, [1, 4], 2)
        self.assertEqual(region.original_nodes.tolist(), [1, 4])
        self.assertEqual(region.targets.flatten().tolist(), [0, 1])
        with self.assertRaisesRegex(ValueError, 'outside'):
            region.map_keys(torch.tensor([1]))
        with self.assertRaisesRegex(ValueError, 'original query-region'):
            region.restore_keys(torch.tensor([1]))
        data.edge_index = torch.tensor([[0], [1]])
        with self.assertRaisesRegex(ValueError, 'undirected'):
            compact_gcn_query(data, [1, 4], 2)
        data.edge_index = torch.tensor([[0], [0]])
        with self.assertRaisesRegex(ValueError, 'self-loops'):
            compact_gcn_query(data, [1, 4], 2)


if __name__ == '__main__':
    unittest.main()
