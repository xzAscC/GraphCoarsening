import unittest

import torch
from torch_geometric.data import Data

from src.explainers.projected_gcn import ProjectedGCN
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel, LinkPredictor
from src.spectral import compute_normalized_adjacency


class ProjectedGCNTests(unittest.TestCase):
    def fixture(self, device='cpu'):
        forward = torch.tensor([[0, 0, 1, 1, 2], [2, 3, 2, 3, 3]], device=device)
        data = Data(x=torch.tensor([[1., 2.], [3., 1.], [2., 4.], [2., 4.]], dtype=torch.float64, device=device),
                    edge_index=torch.cat((forward, forward.flip(0)), 1))
        model = LinkPredictionModel(GCN(2, 3, 3, 2), LinkPredictor()).to(device).double().eval()
        with torch.no_grad():
            for i, conv in enumerate(model.encoder.convs):
                conv.lin.weight.fill_(.1 + i * .05)
                conv.bias.fill_(.2)
        return model, data

    def check_singletons(self, device):
        model, data = self.fixture(device)
        # Add duplicate weighted entries and isolate one extra node.
        data.x = torch.cat((data.x, data.x.new_tensor([[1., -1.]])))
        data.edge_index = data.edge_index.repeat(1, 2)
        data.edge_weight = data.x.new_full((data.edge_index.shape[1],), .5)
        exact = ProjectedGCN(model, data, torch.arange(5, device=device), [0, 4])
        for support, retain in ((None, False), (torch.tensor([2, 3], device=device), True),
                                 (torch.tensor([2, 3], device=device), False),
                                 (torch.empty(0, dtype=torch.long, device=device), True)):
            mask = exact.edge_mask(support, retain=retain)
            expected = model(data.x, data.edge_index[:, mask], torch.tensor([[0], [4]], device=device),
                             edge_weight=data.edge_weight[mask])
            torch.testing.assert_close(exact(support, retain=retain), expected, atol=1e-12, rtol=1e-12)

    def test_singleton_cpu(self):
        self.check_singletons('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_singleton_cuda(self):
        self.check_singletons('cuda')

    def test_dense_projection_identity_and_exact_invariant_case(self):
        model, data = self.fixture()
        projected = ProjectedGCN(model, data, torch.tensor([0, 1, 2, 2]), [0, 1])
        P = torch.zeros(4, 3, dtype=torch.float64)
        P[torch.arange(4), projected.membership] = projected.node_scale
        A = compute_normalized_adjacency(data.edge_index, 4, data.x.new_ones(10)).to_dense()
        B = projected.operator().to_dense()
        torch.testing.assert_close(P.T @ P, torch.eye(3, dtype=torch.float64))
        torch.testing.assert_close(B, P.T @ A @ P)
        torch.testing.assert_close(A @ P, P @ B, atol=1e-12, rtol=0)
        torch.testing.assert_close(P @ projected.features, data.x)
        self.assertLess(projected.relative_feature_residual(), 1e-12)
        self.assertLess(projected.operator_residual_fro(), 1e-7)
        expected = model(data.x, data.edge_index, torch.tensor([[0], [1]]))
        torch.testing.assert_close(projected(), expected, atol=1e-12, rtol=1e-12)
        # Bias projection is necessary even on an invariant feature subspace.
        h = projected.features
        for i, conv in enumerate(model.encoder.convs):
            h = B @ conv.lin(h) + conv.bias
            if i == 0:
                h = h.relu()
        wrong = model.predictor(h, projected.targets)
        self.assertGreater((wrong - expected).abs().item(), .01)

    def test_projection_after_intervention_is_not_reused_from_full_graph(self):
        model, data = self.fixture()
        projected = ProjectedGCN(model, data, torch.tensor([0, 1, 2, 2]), [0, 1])
        P = torch.zeros(4, 3, dtype=torch.float64)
        P[torch.arange(4), projected.membership] = projected.node_scale
        support = torch.tensor([2])
        for retain in (True, False):
            mask = projected.edge_mask(support, retain=retain)
            A = compute_normalized_adjacency(data.edge_index[:, mask], 4, data.x.new_ones(int(mask.sum()))).to_dense()
            torch.testing.assert_close(projected.operator(support, retain=retain).to_dense(), P.T @ A @ P)
            residual = torch.linalg.vector_norm(A @ P - P @ (P.T @ A @ P)).item()
            self.assertAlmostEqual(projected.operator_residual_fro(support, retain=retain), residual, places=10)
            self.assertFalse(torch.allclose(projected.operator().to_dense(), projected.operator(support, retain=retain).to_dense()))

    def test_reject_non_singleton_endpoints_and_unsupported_models(self):
        model, data = self.fixture()
        with self.assertRaises(ValueError):
            ProjectedGCN(model, data, torch.tensor([0, 0, 1, 2]), [0, 1])
        model.encoder.convs[0].improved = True
        with self.assertRaises(ValueError):
            ProjectedGCN(model, data, torch.arange(4), [0, 1])
        model.encoder.convs[0].improved = False
        data.edge_index = data.edge_index[:, :5]
        with self.assertRaises(ValueError):
            ProjectedGCN(model, data, torch.arange(4), [0, 1])


if __name__ == '__main__':
    unittest.main()
