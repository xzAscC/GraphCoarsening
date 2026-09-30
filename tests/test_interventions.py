import unittest

import torch
from torch_geometric.data import Data

from src.evaluation.interventions import evaluate_support, coarse_fidelity, support_mask


class SumLogit(torch.nn.Module):
    def __init__(self, bias=-1.8):
        super().__init__()
        self.bias = bias

    def forward(self, x, edge_index, target_edge_index, edge_weight=None):
        weight = torch.ones(edge_index.size(1)) if edge_weight is None else edge_weight
        return (weight.sum() + self.bias).reshape(1)


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.data = Data(x=torch.ones(3, 1), edge_index=torch.tensor([[0, 1], [1, 0]]))
        self.support = Data(edge_index=torch.tensor([[0], [1]]), edge_weight=torch.tensor([100.]))

    def test_logit_threshold_and_importance_not_used_as_weight(self):
        r = evaluate_support(SumLogit(), self.data, self.support, 0, 2)
        self.assertAlmostEqual(r['full_logit'], 0.2, places=5)
        self.assertEqual(r['retained_logit'], r['full_logit'])
        self.assertEqual(r['necessity_flip'], 1.)
        self.assertEqual(r['sufficiency_agreement'], 1.)
        self.assertEqual(r['support_edges'], 1)

    def test_original_weights_and_empty_support(self):
        self.data.edge_weight = torch.tensor([2., 2.])
        r = evaluate_support(SumLogit(), self.data, self.support, 0, 2)
        self.assertAlmostEqual(r['retained_logit'], 2.2, places=5)
        empty = Data(edge_index=torch.empty((2, 0), dtype=torch.long))
        r = evaluate_support(SumLogit(), self.data, empty, 0, 2)
        self.assertEqual(r['necessity_flip'], 0.)
        self.assertEqual(r['sufficiency_agreement'], 0.)
        self.assertEqual(r['evidence_sparsity'], 1.)

    def test_local_remapping_and_duplicate_orientation(self):
        e = Data(edge_index=torch.tensor([[0, 1, 0], [1, 0, 1]]),
                 original_node_indices=torch.tensor([1, 0]))
        self.assertTrue(support_mask(self.data, e).all())
        r = evaluate_support(SumLogit(), self.data, e, 0, 2)
        self.assertEqual(r['support_edges'], 1)

    def test_coarse_rejected_by_support_metrics(self):
        self.support.is_coarse_graph = True
        with self.assertRaisesRegex(ValueError, 'coarse graph'):
            evaluate_support(SumLogit(), self.data, self.support, 0, 2)

    def test_nonexistent_edge_rejected(self):
        self.support.edge_index = torch.tensor([[0], [2]])
        with self.assertRaisesRegex(ValueError, 'absent'):
            support_mask(self.data, self.support)

    def test_negative_class_confidence_direction(self):
        r = evaluate_support(SumLogit(-3.), self.data, self.support, 0, 2)
        self.assertLess(r['necessity_confidence_drop'], 0.)
        self.assertEqual(r['necessity_flip'], 0.)

    def test_coarse_requires_endpoint_mapping(self):
        self.support.is_coarse_graph = True
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            coarse_fidelity(SumLogit(), self.data, self.support, 0, 2)

    def test_coarse_agreement_separate_from_support(self):
        e = self.data.clone()
        e.is_coarse_graph, e.target_a, e.target_b = True, 0, 2
        r = coarse_fidelity(SumLogit(), self.data, e, 0, 2)
        self.assertEqual(r['coarse_agreement'], 1.)
        self.assertNotIn('necessity_flip', r)


if __name__ == '__main__':
    unittest.main()
