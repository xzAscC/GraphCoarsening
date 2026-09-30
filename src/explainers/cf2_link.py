"""Edge-only binary-link adaptation of CF², not an exact original reproduction.

Objective: Tan et al., WWW 2022, Eqs. (10)--(12), DOI 10.1145/3485447.3511948.
Implementation reference: https://github.com/chrisjtan/gnn_cff,
commit 9020e7485fc2ff579529b9f3abc3f580d5b06e00, ExplainModelNodeMulti.

We use the paper's competing-class probability margins, including for binary
links (the upstream binary graph variant instead uses a 0.5 threshold margin).
Node-explainer defaults: Adam, 2000 steps, lr=.01, lambda=500, alpha=.6,
gamma=.5. These defaults are starting points, not tuned link-task settings.
Adaptations: fixed features, one sigmoid gate per undirected candidate edge,
fixed candidate region, original weighted GCN normalization recomputed at each
forward, and external top-B discretization instead of threshold .5. The L1
term sums retained adjacency weights over stored entries, as in upstream.
Noncandidate edges are absent in the factual graph and kept in the complement.
Only mask parameters are optimized; predictor parameters and gradients stay
unchanged. Requires uncached weighted message passing and no explicit loops.
"""
import math

import torch
from torch_geometric.data import Data

from src.explainers.base import BaseExplainer
from src.explainers.candidates import candidate_edge_mask
from src.explainers.pyg_baselines import _frozen_parameters


def cf2_loss(mask_l1, retained_logit, removed_logit, sign, *, lam=500.,
             alpha=.6, gamma=.5):
    """Paper probability-margin objective for the fixed original binary class."""
    factual_probability = torch.sigmoid(sign * retained_logit)
    counterfactual_probability = torch.sigmoid(sign * removed_logit)
    factual = torch.relu(gamma + 1 - 2 * factual_probability)
    counterfactual = torch.relu(gamma + 2 * counterfactual_probability - 1)
    return mask_l1 + lam * (alpha * factual + (1 - alpha) * counterfactual)


class CF2LinkExplainer(BaseExplainer):
    def __init__(self, model, *, epochs=2000, lr=.01, lam=500., alpha=.6,
                 gamma=.5, hops=2, candidate_region='gcn-boundary', device='cpu', local_gcn=False):
        if (epochs < 1 or hops < 0 or not all(math.isfinite(v) for v in
                (lr, lam, alpha, gamma)) or lr <= 0 or lam < 0
                or not 0 <= alpha <= 1 or not 0 <= gamma <= 1):
            raise ValueError('Invalid CF2 optimization settings')
        if any(getattr(module, 'cached', False) for module in model.modules()):
            raise ValueError('CF2 requires uncached adjacency normalization')
        super().__init__(model, device)
        self.epochs, self.lr, self.lam = epochs, lr, lam
        self.alpha, self.gamma, self.hops = alpha, gamma, hops
        self.candidate_region = candidate_region
        self.local_gcn = local_gcn
        if local_gcn and candidate_region != 'gcn-boundary':
            raise ValueError('Local CF2 requires the normalization-boundary candidate region')
        self.last_diagnostics = {}

    def explain_link(self, data, node_a, node_b):
        if any(module.training for module in self.model.modules()):
            raise ValueError('Frozen evaluation-mode predictor required')
        data = self._to_device(data)
        if self.local_gcn:
            from src.explainers.local_gcn import validate_local_gcn_model
            if validate_local_gcn_model(self.model) != self.hops:
                raise ValueError('Local CF2 requires the actual GCN layer count')
        n, edges = data.x.size(0), data.edge_index
        if (edges[0] == edges[1]).any():
            raise ValueError('Explicit self-loops are unsupported; GCN adds fixed loops')
        weight = getattr(data, 'edge_weight', None)
        weight = data.x.new_ones(edges.size(1)) if weight is None else weight.detach()
        if not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError('Finite nonnegative original edge weights required')
        region = candidate_edge_mask(edges, n, [node_a, node_b], self.hops,
                                     self.candidate_region)
        keys = edges[:, region].min(0).values * n + edges[:, region].max(0).values
        candidates, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        self.last_diagnostics = {
            'adaptation': 'CF2-edge-only-binary-link-paper-margin-v1',
            'upstream_commit': '9020e7485fc2ff579529b9f3abc3f580d5b06e00',
            'epochs': self.epochs, 'lr': self.lr, 'lambda': self.lam,
            'alpha': self.alpha, 'gamma': self.gamma,
            'candidate_region': self.candidate_region,
            'candidate_edges': candidates.numel(), 'discretization': 'external-top-B',
            'evaluation_mode': 'exact-boundary-local' if self.local_gcn else 'full',
            'initialization_num_nodes': n,
        }
        if not candidates.numel():
            self.last_diagnostics['optimization_steps'] = 0
            return Data(edge_index=edges[:, :0], edge_weight=weight[:0])
        target = torch.tensor([[node_a], [node_b]], device=self.device)
        with torch.no_grad():
            full = self.model(data.x, edges, target, edge_weight=weight).squeeze()
            if not torch.isfinite(full):
                raise ValueError('Nonfinite original prediction')
            sign = 1 if full > 0 else -1
        evaluation_data, evaluation_target = data, target
        evaluation_region, evaluation_weight = region, weight
        if self.local_gcn:
            from src.explainers.local_gcn import compact_gcn_query
            compact = compact_gcn_query(data, [node_a, node_b], self.hops)
            if not torch.equal(compact.original_edge_mask, region):
                raise RuntimeError('Local region must preserve candidate entry order')
            evaluation_data, evaluation_target = compact.data, compact.targets
            evaluation_weight = weight[region]
            evaluation_region = torch.ones_like(evaluation_weight, dtype=torch.bool)
            self.last_diagnostics['local_nodes'] = compact.data.num_nodes
        # Keep original n, parameter order and RNG draws in both backends.
        # Exact-arithmetic objectives agree; FP optimization paths may differ.
        mask_logits = torch.nn.Parameter(data.x.new_empty(candidates.numel()))
        torch.nn.init.normal_(mask_logits, mean=1., std=math.sqrt(2. / n))
        optimizer = torch.optim.Adam([mask_logits], lr=self.lr)
        with _frozen_parameters(self.model), torch.enable_grad():
            def objective():
                stored_mask = evaluation_weight.new_zeros(evaluation_weight.shape)
                stored_mask[evaluation_region] = mask_logits.sigmoid()[inverse]
                retained_weight = evaluation_weight * stored_mask
                retained = self.model(evaluation_data.x, evaluation_data.edge_index, evaluation_target,
                                      edge_weight=retained_weight).squeeze()
                removed = self.model(evaluation_data.x, evaluation_data.edge_index, evaluation_target,
                                     edge_weight=evaluation_weight - retained_weight).squeeze()
                return cf2_loss(retained_weight.sum(), retained, removed, sign,
                                lam=self.lam, alpha=self.alpha, gamma=self.gamma)

            for step in range(self.epochs):
                optimizer.zero_grad(set_to_none=True)
                loss = objective()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite CF2 objective')
                if step == 0:
                    self.last_diagnostics['initial_loss'] = loss.item()
                loss.backward()
                if mask_logits.grad is None or not torch.isfinite(mask_logits.grad).all():
                    raise ValueError('Nonfinite or missing CF2 mask gradient')
                optimizer.step()
            with torch.no_grad():
                final_loss = objective()
                if not torch.isfinite(final_loss):
                    raise ValueError('Nonfinite final CF2 objective')
                self.last_diagnostics.update(final_loss=final_loss.item(),
                                             optimization_steps=self.epochs,
                                             original_logit=full.item())
        return Data(edge_index=torch.stack((candidates // n, candidates % n)),
                    edge_weight=mask_logits.detach().sigmoid())

    def explain_batch(self, data, edges):
        return [self.explain_link(data, a, b) for a, b in edges.t().tolist()]
