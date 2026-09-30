import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel, LinkPredictor
from src.explainers.candidates import candidate_edge_mask
from src.explainers.support_refinement import refine_support
from src.explainers.local_gcn import validate_local_gcn_model
from src.evaluation.interventions import evaluate_support


class LocalSearchTests(unittest.TestCase):
    def fixture(self, device):
        torch.manual_seed(17)
        n = 30
        half = torch.stack((torch.arange(n - 1), torch.arange(1, n))).to(device)
        edges = torch.cat((half, half.flip(0)), dim=1)
        weight = .2 + torch.rand(n - 1, device=device)
        data = Data(x=torch.randn(n, 4, device=device), edge_index=edges, edge_weight=weight.repeat(2))
        model = LinkPredictionModel(GCN(4, 8, 5, 2), LinkPredictor()).to(device).eval()
        mask = candidate_edge_mask(edges, n, [0, 5], 2, 'gcn-boundary')
        candidates = torch.unique(edges[:, mask].min(0).values * n + edges[:, mask].max(0).values)
        initial = Data(edge_index=torch.stack((candidates[:3] // n, candidates[:3] % n)))
        return model, data, candidates, initial

    def check_invariant(self, device):
        model, data, candidates, initial = self.fixture(device)
        before = evaluate_support(model, data, initial, 0, 5, device)
        parameters = [p.detach().clone() for p in model.parameters()]
        gradients = []
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter)
            gradients.append(parameter.grad.clone())
        for size in (0, 1, 2):
            for groups in (None, torch.arange(len(candidates), device=device) % 3):
                support, trace = refine_support(model, data, 0, 5, candidates, initial,
                                                 steps=4, batch_size=3, exchange_size=size,
                                                 groups=groups, local_gcn=True)
                after = evaluate_support(model, data, support, 0, 5, device)
                self.assertEqual(after['support_edges'], before['support_edges'])
                for metric in ('necessity_flip', 'sufficiency_agreement'):
                    self.assertGreaterEqual(after[metric], before[metric])
                self.assertGreaterEqual(after['necessity_confidence_drop'] + 1e-6,
                                        before['necessity_confidence_drop'])
                self.assertLessEqual(max(0, after['sufficiency_confidence_drop']),
                                     max(0, before['sufficiency_confidence_drop']) + 1e-6)
                for step in trace:
                    self.assertEqual(step['evaluation_mode'], 'local-screen-full-accept')
                    self.assertLess(step['gradient_nodes'], data.num_nodes)
                    self.assertGreaterEqual(step['full_graph_rechecks'], int(step['accepted']))
                self.assertAlmostEqual(after['retained_logit'], trace[-1]['retained_logit'], places=5)
        for parameter, original, gradient in zip(model.parameters(), parameters, gradients):
            torch.testing.assert_close(parameter, original, rtol=0, atol=0)
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)

    def test_full_graph_nonregression_cpu(self):
        self.check_invariant('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_full_graph_nonregression_cuda(self):
        self.check_invariant('cuda')

    def test_optimistic_local_screen_cannot_bypass_full_acceptance(self):
        model, data, candidates, initial = self.fixture('cpu')

        def optimistic(model, graph, a, b, supports, batch, *, retain=False):
            magnitude = 1. if graph is data else 100.
            return graph.x.new_full((len(supports),), magnitude if retain else -magnitude)

        with patch('src.explainers.support_refinement.group_deletion_logits', side_effect=optimistic), \
                patch('src.explainers.support_refinement.predict_logit', return_value=2.):
            support, trace = refine_support(model, data, 0, 5, candidates, initial, local_gcn=True)
        torch.testing.assert_close(support.edge_index, initial.edge_index)
        self.assertGreater(trace[1]['full_graph_rechecks'], 0)
        self.assertFalse(trace[1]['accepted'])
        self.assertEqual(trace[0]['objective'], trace[1]['objective'])

    def test_unsupported_or_cached_models_rejected(self):
        with self.assertRaisesRegex(ValueError, 'known'):
            validate_local_gcn_model(torch.nn.Linear(2, 1).eval())
        model, _, _, _ = self.fixture('cpu')
        model.encoder.convs[0].cached = True
        with self.assertRaisesRegex(ValueError, 'uncached'):
            validate_local_gcn_model(model)


if __name__ == '__main__':
    unittest.main()
