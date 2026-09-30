import unittest

import torch
from torch_geometric.data import Data

from experiments.audit_search_cost import search_cost
from src.explainers.support_refinement import refine_support
from tests.test_support_refinement import PairSynergy


class SearchCostTests(unittest.TestCase):
    def check_hooks(self, device):
        for steps in (0, 2):
            for batch in (1, 3):
                for size in (1, 2):
                    x = torch.zeros(6, 4, device=device)
                    x[1:5] = torch.eye(4, device=device)
                    edges = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]], device=device)
                    data = Data(x=x, edge_index=torch.cat((edges, edges.flip(0)), 1))
                    model = PairSynergy().to(device).eval()
                    calls = []
                    handle = model.register_forward_hook(lambda m, a, out: calls.append(out.numel()))
                    try:
                        _, trace = refine_support(model, data, 0, 5, torch.arange(1, 5, device=device),
                                                  Data(edge_index=edges[:, :2]), steps=steps,
                                                  batch_size=batch, exchange_size=size)
                    finally:
                        handle.remove()
                    counts = search_cost(trace, batch)
                    self.assertEqual(counts['total_forward_calls'], len(calls))
                    self.assertEqual(counts['graph_evaluations'], sum(calls))

    def test_cpu_actual_forward_counts(self):
        self.check_hooks('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_actual_forward_counts(self):
        self.check_hooks('cuda')

    def test_invalid_counters(self):
        initial = {'step': 0, 'proposals': 0, 'full_graph_rechecks': 0, 'accepted': False}
        self.assertEqual(search_cost([initial], 8)['total_forward_calls'], 3)
        for trace in ([], [initial | {'proposals': 1}],
                      [initial, {'step': 1, 'proposals': 2, 'full_graph_rechecks': 3, 'accepted': True}]):
            with self.assertRaises(ValueError):
                search_cost(trace, 8)


if __name__ == '__main__':
    unittest.main()
