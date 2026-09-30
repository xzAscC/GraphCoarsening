import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from src.explainers.coarsen_explainer import CoarsenExplainer


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, edge_index, target_edge_index, edge_weight=None):
        self.calls.append((edge_index.clone(), None if edge_weight is None else edge_weight.clone()))
        return edge_weight.sum().reshape(1)


class PathwayTests(unittest.TestCase):
    def test_occlusion_removes_both_directions_and_preserves_weights(self):
        edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
        weights = torch.tensor([2., 2., 3., 3., 4., 4., 5., 5.])
        data = Data(x=torch.ones(4, 1), edge_index=edges, edge_weight=weights)
        model = RecordingModel()
        explainer = CoarsenExplainer(model, device='cpu')
        with patch.object(explainer, '_ensure_fitted', return_value=SimpleNamespace(scores=torch.ones(8))), \
             patch.object(explainer, '_get_protected_nodes', return_value=[]), \
             patch.object(explainer, '_gradient_scores', return_value=torch.ones(8)), \
             patch('src.partition.prediction_guided_partition', return_value=[[0, 1], [2, 3]]):
            explainer.explain_link(data, 0, 2)
        self.assertGreater(len(model.calls), 1)
        originals = {tuple(e): w.item() for e, w in zip(edges.t().tolist(), weights)}
        for es, ws in model.calls:
            pairs = {tuple(e) for e in es.t().tolist()}
            self.assertTrue(all((b, a) in pairs for a, b in pairs))
            self.assertIsNotNone(ws)
            self.assertEqual(ws.tolist(), [originals[tuple(e)] for e in es.t().tolist()])
        self.assertTrue(any(es.size(1) < edges.size(1) for es, _ in model.calls))


if __name__ == '__main__':
    unittest.main()
