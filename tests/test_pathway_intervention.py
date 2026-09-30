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
    def test_global_refinement_reuses_partition_and_splits_endpoint_cluster(self):
        edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
        data = Data(x=torch.ones(4, 1), edge_index=edges, edge_weight=torch.ones(8))
        cached = SimpleNamespace(scores=torch.ones(8), partition=[[0, 1], [2], [3]])
        explainer = CoarsenExplainer(RecordingModel(), partition_mode='global-refine')
        with patch.object(explainer, '_ensure_fitted', return_value=cached), \
             patch.object(explainer, '_gradient_scores', return_value=torch.ones(8)), \
             patch('src.partition.prediction_guided_partition') as repartition:
            explainer.explain_link(data, 0, 2)
        repartition.assert_not_called()
        self.assertEqual(explainer.last_diagnostics['clusters'], 4)
        self.assertEqual(cached.partition, [[0, 1], [2], [3]])

    def test_directional_gradients_not_replaced_by_reverse_edge(self):
        edges = torch.tensor([[0, 1], [1, 0]])
        data = Data(x=torch.ones(2, 1), edge_index=edges, edge_weight=torch.ones(2))
        model = RecordingModel()
        explainer = CoarsenExplainer(model, k_frac=1., device='cpu')
        with patch.object(explainer, '_ensure_fitted', return_value=SimpleNamespace(scores=torch.ones(2))), \
             patch.object(explainer, '_get_protected_nodes', return_value={0, 1}), \
             patch.object(explainer, '_gradient_scores', return_value=torch.tensor([1., 3.])), \
             patch('src.partition.prediction_guided_partition', return_value=[[0], [1]]):
            exp = explainer.explain_link(data, 0, 1)
        ordered = exp.edge_weight.sort().values
        # A shared pathway factor preserves the directional gradient ratio.
        self.assertAlmostEqual((ordered[1] / ordered[0]).item(), 3.)

    def test_protection_radius(self):
        data = Data(x=torch.ones(4, 1), edge_index=torch.tensor([[0,1,1,2,2,3], [1,0,2,1,3,2]]))
        endpoint = CoarsenExplainer(RecordingModel(), protect_hops=0)
        neighbor = CoarsenExplainer(RecordingModel(), protect_hops=1)
        self.assertEqual(endpoint._get_protected_nodes(data, 0, 3), {0, 3})
        self.assertEqual(neighbor._get_protected_nodes(data, 0, 3), {0, 1, 2, 3})

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
