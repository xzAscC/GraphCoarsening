"""A K-hop induced candidate graph need not cover a normalized GCN's edge dependence."""
import unittest

import torch
from torch_geometric.utils import k_hop_subgraph

from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel, LinkPredictor


class NormalizationBoundaryTests(unittest.TestCase):
    def test_outside_edge_changes_two_layer_endpoint_through_degree(self):
        model = LinkPredictionModel(GCN(1, 1, 1, 2), LinkPredictor()).double().eval()
        with torch.no_grad():
            for conv in model.encoder.convs:
                conv.lin.weight.fill_(1)
                conv.bias.zero_()
        # Query (0,5) is absent. Edge (2,3) is outside its two-hop induced graph,
        # but changes the source normalization of messages along 2 -> 1 -> 0.
        edges = torch.tensor([[0,1,1,2,2,3], [1,0,2,1,3,2]])
        features = torch.ones(6, 1, dtype=torch.float64)
        target = torch.tensor([[0], [5]])
        gate = torch.ones(6, dtype=torch.float64, requires_grad=True)
        value = model(features, edges, target, edge_weight=gate).sum()
        derivative, = torch.autograd.grad(value, gate)
        _, _, _, two_hop = k_hop_subgraph([0,5], 2, edges, num_nodes=6)
        _, _, _, three_hop = k_hop_subgraph([0,5], 3, edges, num_nodes=6)
        self.assertFalse(two_hop[4:].any())
        self.assertTrue(three_hop.all())
        tied = derivative[4:].sum()
        self.assertGreater(tied.abs().item(), 1e-3)
        perturbed = gate.detach().clone()
        step = 1e-5
        perturbed[4:] -= step
        with torch.no_grad():
            changed = model(features, edges, target, edge_weight=perturbed).sum()
        torch.testing.assert_close((value.detach() - changed) / step, tied, atol=1e-6, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
