import unittest

import torch
from torch_geometric.data import Data

from src.evaluation.interventions import evaluate_support
from src.explainers.support_refinement import candidate_groups, diversify, refine_support, ranked_bundles
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel, LinkPredictor


class WeightedNeighborhood(torch.nn.Module):
    def forward(self, x, edge_index, target, edge_weight=None):
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))
        score = x.new_zeros(x.size(0)).scatter_add_(0, edge_index[0], edge_weight * x[edge_index[1], 0])
        return score[target[0]]


class PairSynergy(torch.nn.Module):
    def forward(self, x, edge_index, target, edge_weight=None):
        weight = x.new_ones(edge_index.size(1)) if edge_weight is None else edge_weight
        gates = x.new_zeros(x.shape).index_add(0, edge_index[0], x[edge_index[1]] * weight[:, None])
        g = gates[target[0]]
        return g[:, 0] * g[:, 1] + 3 * g[:, 2] * g[:, 3] - .5


class RefinementTests(unittest.TestCase):
    def check_pair_escape(self, device):
        x = torch.zeros(6, 4, device=device)
        x[1:5] = torch.eye(4, device=device)
        forward = torch.tensor([[0,0,0,0], [1,2,3,4]], device=device)
        data = Data(x=x, edge_index=torch.cat((forward, forward.flip(0)), dim=1))
        model = PairSynergy().to(device).eval()
        initial = Data(edge_index=forward[:, :2])
        candidates = torch.arange(1, 5, device=device)
        single, _ = refine_support(model, data, 0, 5, candidates, initial, exchange_size=1)
        self.assertEqual(single.edge_index.tolist(), initial.edge_index.tolist())
        for groups in (None, torch.tensor([0,0,1,1], device=device)):
            pair, trace = refine_support(model, data, 0, 5, candidates, initial,
                                         groups=groups, exchange_size=2, batch_size=2)
            self.assertEqual(pair.edge_index.tolist(), [[0,0], [3,4]])
            self.assertTrue(trace[1]['accepted'])
            self.assertEqual(trace[1]['exchange_size'], 2)
            self.assertGreater(trace[-1]['objective'], trace[0]['objective'])

    def test_pair_escape_cpu(self):
        self.check_pair_escape('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_pair_escape_cuda(self):
        self.check_pair_escape('cuda')

    def test_pair_ranking_and_group_priority(self):
        gradient = torch.tensor([4.,3.,2.,1.])
        order = torch.arange(4)
        global_pairs = ranked_bundles(order, gradient, 2, 2)
        grouped_pairs = ranked_bundles(order, gradient, 2, 2, groups=torch.tensor([0,1,0,1]))
        self.assertEqual(global_pairs.tolist(), [[0,1],[0,2]])
        self.assertEqual(grouped_pairs.tolist(), [[0,2],[1,3]])
        self.assertEqual(ranked_bundles(order[:1], gradient, 2, 2).shape, (0,2))

    def test_exact_exchange_improves_both_interventions(self):
        model = WeightedNeighborhood().eval()
        data = Data(x=torch.tensor([[0.], [1.], [3.], [2.]]),
                    edge_index=torch.tensor([[0,1,0,2,0,3], [1,0,2,0,3,0]]))
        initial = Data(edge_index=torch.tensor([[0], [1]]))
        result, trace = refine_support(model, data, 0, 3, torch.tensor([1, 2, 3]), initial)
        self.assertEqual(result.edge_index.tolist(), [[0], [2]])
        self.assertTrue(any(t['accepted'] for t in trace))
        self.assertGreater(trace[-1]['capped_retained_probability'], trace[0]['capped_retained_probability'])
        self.assertLess(trace[-1]['removed_probability'], trace[0]['removed_probability'])

    def check_gcn(self, device, exchange_size=1):
        torch.manual_seed(123)
        model = LinkPredictionModel(GCN(3, 6, 4, 2), LinkPredictor()).to(device).eval()
        edges = torch.tensor([[0,1,1,2,2,3,3,0,0,2], [1,0,2,1,3,2,0,3,2,0]], device=device)
        data = Data(x=torch.randn(4, 3, device=device), edge_index=edges,
                    edge_weight=torch.tensor([1.,1.,2.,2.,3.,3.,4.,4.,2.,2.], device=device))
        candidates = torch.unique(edges.min(0).values * 4 + edges.max(0).values)
        initial = Data(edge_index=torch.tensor([[0, 1], [1, 2]], device=device))
        groups = candidate_groups(candidates, [[0], [1,2], [3]], 4)
        before = evaluate_support(model, data, initial, 0, 2, device)
        for grouping in (None, groups):
            for batch in (1, 3):
                result, trace = refine_support(model, data, 0, 2, candidates, initial,
                                               groups=grouping, batch_size=batch,
                                               exchange_size=exchange_size)
                after = evaluate_support(model, data, result, 0, 2, device)
                self.assertEqual(after['support_edges'], before['support_edges'])
                self.assertGreaterEqual(after['necessity_flip'], before['necessity_flip'])
                self.assertGreaterEqual(after['sufficiency_agreement'], before['sufficiency_agreement'])
                self.assertGreaterEqual(after['necessity_confidence_drop'] + 1e-8, before['necessity_confidence_drop'])
                self.assertLessEqual(max(0, after['sufficiency_confidence_drop']), max(0, before['sufficiency_confidence_drop']) + 1e-8)
                for old, new in zip(trace, trace[1:]):
                    self.assertGreaterEqual(new['objective'], old['objective'])
                self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_gcn_cpu(self):
        self.check_gcn('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gcn_cuda(self):
        self.check_gcn('cuda')

    def test_pair_gcn_cpu(self):
        self.check_gcn('cpu', exchange_size=2)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_pair_gcn_cuda(self):
        self.check_gcn('cuda', exchange_size=2)

    def test_group_diversity_and_fill(self):
        order = torch.tensor([0, 1, 2, 3])
        groups = torch.tensor([7, 7, 8, 9])
        self.assertEqual(diversify(order, groups, 3).tolist(), [0, 2, 3])
        self.assertEqual(diversify(order, groups, 4).tolist(), [0, 2, 3, 1])
        self.assertEqual(diversify(order, None, 3).tolist(), [0, 1, 2])

    def test_empty_and_full_budget(self):
        model = WeightedNeighborhood().eval()
        data = Data(x=torch.ones(2, 1), edge_index=torch.tensor([[0,1], [1,0]]))
        for count in (0, 1):
            initial = Data(edge_index=data.edge_index[:, :count])
            result, trace = refine_support(model, data, 0, 1, torch.tensor([1]), initial)
            self.assertEqual(result.edge_index.size(1), count)
            self.assertEqual(len(trace), 1)


if __name__ == '__main__':
    unittest.main()
