import unittest

import torch
from torch_geometric.data import Data

from src.explainers.cf2_link import CF2LinkExplainer, cf2_loss
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel, LinkPredictor


class CF2LinkTests(unittest.TestCase):
    def test_paper_binary_margin_and_class_orientation(self):
        keep, remove = torch.tensor(.8).logit(), torch.tensor(.4).logit()
        # factual hinge = 0, counterfactual hinge = .3
        loss = cf2_loss(torch.tensor(2.), keep, remove, 1, lam=10., alpha=.6)
        torch.testing.assert_close(loss, torch.tensor(3.2))
        torch.testing.assert_close(loss, cf2_loss(torch.tensor(2.), -keep, -remove,
                                                -1, lam=10., alpha=.6))
        keep.requires_grad_()
        remove.requires_grad_()
        cf2_loss(torch.tensor(0.), keep, remove, 1).backward()
        self.assertEqual(keep.grad.item(), 0.)
        self.assertGreater(remove.grad.item(), 0.)

    def check_frozen_and_hard_masks(self, device):
        torch.manual_seed(72)
        model = LinkPredictionModel(GCN(3, 5, 4, 2), LinkPredictor()).to(device).eval()
        # Duplicate stored edges and nonunit propagation weights.
        data = Data(x=torch.randn(6, 3, device=device),
                    edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 0, 1],
                                             [1, 0, 2, 1, 3, 2, 1, 0]], device=device),
                    edge_weight=torch.tensor([.7, .7, 1.2, 1.2, .3, .3, .2, .2], device=device))
        parameters = list(model.parameters())
        parameters[0].requires_grad_(False)
        flags = [p.requires_grad for p in parameters]
        before = [p.detach().clone() for p in parameters]
        for p in parameters:
            p.grad = torch.ones_like(p)
        explainer = CF2LinkExplainer(model, epochs=3, device=device)
        explanation = explainer.explain_link(data, 0, 3)
        self.assertEqual(explanation.edge_index.size(1), 3)
        self.assertTrue(torch.isfinite(explanation.edge_weight).all())
        self.assertTrue(((explanation.edge_weight > 0) & (explanation.edge_weight < 1)).all())
        self.assertEqual(explainer.last_diagnostics['optimization_steps'], 3)
        for p, old, flag in zip(parameters, before, flags):
            torch.testing.assert_close(p, old, rtol=0, atol=0)
            torch.testing.assert_close(p.grad, torch.ones_like(p), rtol=0, atol=0)
            self.assertEqual(p.requires_grad, flag)
        self.assertTrue(all(not m.training for m in model.modules()))
        chosen = (data.edge_index.min(0).values == 0)
        target = torch.tensor([[0], [3]], device=device)
        for active in (chosen, ~chosen):
            hard = model(data.x, data.edge_index[:, active], target,
                         edge_weight=data.edge_weight[active])
            relaxed = model(data.x, data.edge_index, target,
                            edge_weight=data.edge_weight * active)
            torch.testing.assert_close(hard, relaxed, atol=1e-6, rtol=1e-5)
        empty = explainer.explain_link(data, 4, 5)
        self.assertEqual(empty.edge_index.size(1), 0)
        self.assertEqual(explainer.last_diagnostics['optimization_steps'], 0)

    def test_cpu(self):
        self.check_frozen_and_hard_masks('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self):
        self.check_frozen_and_hard_masks('cuda')

    def test_reject_invalid_settings_and_cached_normalization(self):
        model = LinkPredictionModel(GCN(3, 5, 4, 2), LinkPredictor())
        for kwargs in ({'epochs': 0}, {'alpha': 2.}, {'lr': float('nan')}):
            with self.assertRaises(ValueError):
                CF2LinkExplainer(model, **kwargs)
        model.encoder.convs[0].cached = True
        with self.assertRaisesRegex(ValueError, 'uncached'):
            CF2LinkExplainer(model)


if __name__ == '__main__':
    unittest.main()
