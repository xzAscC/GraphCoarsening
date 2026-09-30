"""A K-hop induced candidate graph need not cover a normalized GCN's edge dependence."""
import unittest

import torch
from torch_geometric.utils import k_hop_subgraph

from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel, LinkPredictor
from src.explainers.candidates import candidate_edge_mask


class NormalizationBoundaryTests(unittest.TestCase):
    def check_boundary(self, device):
        torch.manual_seed(91)
        for layers in (2, 3):
            n = 2 * layers + 4
            forward = torch.stack((torch.arange(n-1), torch.arange(1, n)))
            edges = torch.cat((forward, forward.flip(0)), dim=1).to(device)
            x = torch.randn(n, 3, dtype=torch.float64, device=device)
            model = LinkPredictionModel(GCN(3, 5, 4, layers), LinkPredictor()).to(device).double().eval()
            boundary = candidate_edge_mask(edges, n, [0, n-1], layers, 'gcn-boundary')
            larger = candidate_edge_mask(edges, n, [0, n-1], layers + 1, 'induced')
            self.assertTrue((~boundary | larger).all())
            self.assertTrue((larger & ~boundary).any())
            gate = torch.ones(edges.size(1), dtype=x.dtype, device=device, requires_grad=True)
            target = torch.tensor([[0], [n-1]], device=device)
            original = model(x, edges, target, edge_weight=gate)
            derivative, = torch.autograd.grad(original.sum(), gate)
            torch.testing.assert_close(derivative[~boundary], torch.zeros_like(derivative[~boundary]), atol=1e-12, rtol=0)
            for _ in range(20):
                keep_half = torch.rand(n-1, device=device) > .25
                keep = torch.cat((keep_half, keep_half))
                with torch.no_grad():
                    arbitrary = model(x, edges[:, keep], target)
                    pruned = model(x, edges[:, keep & boundary], target)
                torch.testing.assert_close(arbitrary, pruned, atol=1e-12, rtol=1e-10)

    def test_boundary_covers_deletion_dependencies_cpu(self):
        self.check_boundary('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_boundary_covers_deletion_dependencies_cuda(self):
        self.check_boundary('cuda')

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
        boundary = candidate_edge_mask(edges, 6, [0,5], 2, 'gcn-boundary')
        self.assertTrue(boundary[4:].all())
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
