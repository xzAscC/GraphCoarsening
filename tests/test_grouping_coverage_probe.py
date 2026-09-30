import unittest

import torch
from torch_geometric.data import Data

from experiments.probe_grouping_coverage import run_captured
from src.explainers import support_refinement as search


class WeightedStar(torch.nn.Module):
    def forward(self, x, edge_index, target, edge_weight=None):
        weights = x.new_ones(edge_index.shape[1]) if edge_weight is None else edge_weight
        values = x.new_zeros(x.shape[0]).index_add(0, edge_index[0], weights * x[edge_index[1], 0])
        return values[target[0]]


class CoverageProbeTests(unittest.TestCase):
    def fixture(self):
        forward = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]])
        data = Data(x=torch.tensor([[0.], [.1], [.2], [.3], [.4]], dtype=torch.float64),
                    edge_index=torch.cat((forward, forward.flip(0)), 1))
        initial = Data(edge_index=forward[:, :1])
        return WeightedStar().eval(), data, torch.arange(1, 5), initial

    def test_recording_does_not_change_search_or_leave_patch_installed(self):
        model, data, candidates, initial = self.fixture()
        original_function = search.exchange_proposals
        plain, trace = search.refine_support(model, data, 0, 4, candidates, initial,
                                             steps=2, additions=2, removals=1, batch_size=1)
        captured = run_captured(model, data, 0, 4, candidates, initial, None,
                                dict(search_steps=2, additions=2, removals=1, batch_size=1))
        self.assertEqual(captured['support'], plain.edge_index.t().tolist())
        self.assertEqual(captured['trace'], trace)
        self.assertIs(search.exchange_proposals, original_function)
        self.assertEqual(len(captured['snapshots']), len(trace) - 1)
        self.assertTrue(all(not step['injected'] for step in captured['snapshots']))

    def test_counterfactual_explicitly_records_extra_proposal(self):
        model, data, candidates, initial = self.fixture()
        captured = run_captured(model, data, 0, 4, candidates, initial, None,
                                dict(search_steps=1, additions=1, removals=1, batch_size=1),
                                injection={'state': [1], 'proposal': [2]})
        snapshot = captured['snapshots'][0]
        self.assertTrue(snapshot['injected'])
        self.assertEqual(len(snapshot['proposals']), 2)
        self.assertEqual(snapshot['proposals'][-1], [2])
        self.assertEqual(captured['trace'][1]['proposals'], 2)


if __name__ == '__main__':
    unittest.main()
