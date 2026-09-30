import unittest

import torch
from torch_geometric.data import Data

from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel, LinkPredictor
from src.explainers.pyg_baselines import GNNExplainerWrapper, _frozen_parameters


class FrozenExplainerTests(unittest.TestCase):
    def check_explanation(self, device):
        torch.manual_seed(12)
        model = LinkPredictionModel(GCN(3, 5, 4, 2), LinkPredictor()).to(device).eval()
        data = Data(x=torch.ones(4, 3, device=device),
                    edge_index=torch.tensor([[0,1,1,2,2,3], [1,0,2,1,3,2]], device=device))
        parameters = list(model.parameters())
        parameters[0].requires_grad_(False)
        flags = [p.requires_grad for p in parameters]
        original = [p.detach().clone() for p in parameters]
        for p in parameters:
            p.grad = torch.ones_like(p)
        explainer = GNNExplainerWrapper(model, epochs=2, k_frac=1., device=device)
        result = explainer.explain_link(data, 0, 3)
        self.assertEqual(result.edge_index.size(1), data.edge_index.size(1))
        self.assertTrue(torch.isfinite(result.edge_weight).all())
        self.assertFalse(model.training)
        self.assertTrue(all(not m.training for m in model.modules()))
        for p, old, flag in zip(parameters, original, flags):
            torch.testing.assert_close(p, old)
            torch.testing.assert_close(p.grad, torch.ones_like(p))
            self.assertEqual(p.requires_grad, flag)
        for conv in model.encoder.convs:
            self.assertIsNone(conv._edge_mask)

    def test_cpu(self):
        self.check_explanation('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self):
        self.check_explanation('cuda')

    def test_flags_restored_on_exception(self):
        model = torch.nn.Linear(2, 1)
        with self.assertRaises(RuntimeError):
            with _frozen_parameters(model):
                raise RuntimeError('fixture')
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
