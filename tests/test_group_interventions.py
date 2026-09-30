import unittest

import torch
from torch_geometric.data import Data

from src.explainers.group_interventions import group_deletion_logits, supportive_edge_scores
from src.explainers.baselines import SaliencyExplainer
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel, LinkPredictor


class GroupInterventionTests(unittest.TestCase):
    def test_zero_logit_uses_negative_class(self):
        class ZeroAtFull(torch.nn.Module):
            def forward(self, x, edge_index, target, edge_weight=None):
                return (edge_weight[0] - 2 * edge_weight[1] + 1).reshape(1)
        data = Data(x=torch.ones(2, 1), edge_index=torch.tensor([[0, 1], [1, 0]]))
        explanation = SaliencyExplainer(ZeroAtFull(), k_frac=1.,
                                       evidence_mode='supportive').explain_link(data, 0, 1)
        torch.testing.assert_close(explanation.edge_weight, torch.tensor([.5, .5]))

    def check_batch(self, device):
        torch.manual_seed(11)
        model = LinkPredictionModel(GCN(3, 5, 4, 2), LinkPredictor()).to(device).eval()
        # Include duplicates, an existing loop, unequal weights, and a full deletion.
        edges = torch.tensor([[0, 1, 0, 1, 1, 2, 2], [1, 0, 1, 0, 2, 1, 2]], device=device)
        data = Data(x=torch.randn(4, 3, device=device), edge_index=edges,
                    edge_weight=torch.tensor([1., 1., 2., 2., 3., 3., 4.], device=device))
        groups = [torch.tensor(values, device=device) for values in ([1], [6], [10], [1, 6, 10])]
        reference = []
        for group in groups:
            keys = edges.min(0).values * 4 + edges.max(0).values
            keep = ~torch.isin(keys, group)
            with torch.no_grad():
                reference.append(model(data.x, edges[:, keep], torch.tensor([[0], [2]], device=device),
                                       edge_weight=data.edge_weight[keep]).reshape(()))
        expected = torch.stack(reference)
        for batch_size in (1, 2, 3, 8):
            actual = group_deletion_logits(model, data, 0, 2, groups, batch_size)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            self.assertEqual(actual.device, data.x.device)
        self.assertEqual(group_deletion_logits(model, data, 0, 2, [], 2).numel(), 0)

    def test_batch_cpu(self):
        self.check_batch('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_batch_cuda(self):
        self.check_batch('cuda')

    def test_tied_direction_sign_and_duplicates(self):
        edges = torch.tensor([[0, 1, 0, 1, 2], [1, 0, 1, 2, 1]])
        derivatives = torch.tensor([3., -4., 2., -3., 1.])
        positive = supportive_edge_scores(edges, derivatives, 3, 1)
        negative = supportive_edge_scores(edges, derivatives, 3, -1)
        torch.testing.assert_close(positive, torch.tensor([1/3, 1/3, 1/3, 0., 0.]))
        torch.testing.assert_close(negative, torch.tensor([0., 0., 0., 1., 1.]))

    def test_explanation_does_not_modify_parameter_gradients(self):
        model = LinkPredictionModel(GCN(2, 4, 3, 2), LinkPredictor()).eval()
        data = Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        SaliencyExplainer(model, k_frac=1., evidence_mode='supportive').explain_link(data, 0, 2)
        for parameter in model.parameters():
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))


if __name__ == '__main__':
    unittest.main()
