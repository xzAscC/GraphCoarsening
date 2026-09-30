import unittest
import torch
from torch_geometric.data import Data

from src.explainers.support_refinement import coverage_union, diversify, exchange_proposals, refine_support
from src.evaluation.interventions import evaluate_support


class WeightedStar(torch.nn.Module):
    def forward(self, x, edge_index, target, edge_weight=None):
        weights = x.new_ones(edge_index.shape[1]) if edge_weight is None else edge_weight
        values = x.new_zeros(x.shape[0]).index_add(0, edge_index[0], weights * x[edge_index[1], 0])
        return values[target[0]]


class CoverageUnionTests(unittest.TestCase):
    def test_both_half_shortlists_included_and_total_cap_filled(self):
        generator = torch.Generator().manual_seed(4)
        for size in range(30):
            order = torch.randperm(size, generator=generator)
            groups = torch.randint(0, 5, (size,), generator=generator)
            for count in (2, 4, 6, 12):
                result = coverage_union(order, groups, count).tolist()
                self.assertEqual(len(result), min(size, count))
                self.assertEqual(len(result), len(set(result)))
                self.assertTrue(set(order[:count // 2].tolist()).issubset(result))
                self.assertTrue(set(diversify(order, groups, min(count // 2, size)).tolist()).issubset(result))
                self.assertEqual(result, [i for i in order.tolist() if i in result])

    def test_no_groups_is_identical_to_global_pool(self):
        order = torch.tensor([7, 3, 5, 1, 0])
        torch.testing.assert_close(coverage_union(order, None, 4), order[:4])

    def test_proposal_inclusion_at_same_state(self):
        candidates = torch.arange(20)
        selected = candidates[:5]
        in_order = torch.arange(5)
        out_order = torch.arange(19, 4, -1)
        gradient = torch.arange(20).float()
        groups = torch.arange(20) % 4
        global_pool, _ = exchange_proposals(candidates, selected, in_order, out_order, gradient, 6, 3, 1)
        diverse_pool, _ = exchange_proposals(candidates, selected, in_order, out_order, gradient, 6, 3, 1, groups)
        union_pool, _ = exchange_proposals(candidates, selected, in_order, out_order, gradient,
                                           12, 3, 1, groups, addition_policy='coverage-union')
        keys = lambda pool: {tuple(s.tolist()) for s in pool}
        self.assertTrue(keys(global_pool) <= keys(union_pool))
        self.assertTrue(keys(diverse_pool) <= keys(union_pool))
        self.assertEqual(len(union_pool), 36)
        for kwargs in ({'additions': 5, 'exchange_size': 1}, {'additions': 12, 'exchange_size': 2}):
            with self.assertRaises(ValueError):
                exchange_proposals(candidates, selected, in_order, out_order, gradient,
                                   removals=3, groups=groups, addition_policy='coverage-union', **kwargs)

    def check_search(self, device):
        edges = torch.stack((torch.zeros(8, dtype=torch.long), torch.arange(1, 9))).to(device)
        data = Data(x=torch.arange(9, device=device).float()[:, None] / 10,
                    edge_index=torch.cat((edges, edges.flip(0)), 1))
        initial = Data(edge_index=edges[:, :2])
        model = WeightedStar().to(device).eval()
        before = evaluate_support(model, data, initial, 0, 8, device)
        result, trace = refine_support(model, data, 0, 8, torch.arange(1, 9, device=device), initial,
                                       groups=torch.arange(8, device=device) % 3, additions=4,
                                       removals=2, addition_policy='coverage-union')
        after = evaluate_support(model, data, result, 0, 8, device)
        self.assertEqual(before['support_edges'], after['support_edges'])
        self.assertGreaterEqual(after['necessity_flip'], before['necessity_flip'])
        self.assertGreaterEqual(after['sufficiency_agreement'], before['sufficiency_agreement'])
        self.assertTrue(all(t['proposals'] <= 8 for t in trace))

    def test_search_cpu(self):
        self.check_search('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_search_cuda(self):
        self.check_search('cuda')


if __name__ == '__main__':
    unittest.main()
