"""Partition-guided group-deletion calibration of original-edge gradients.

The edge mode groups candidate edges by partition endpoints, measures group
deletion effects, and rescales gradients heuristically. Absolute sensitivity
and predicted-class-supportive evidence are separate experimental variants.
The coarse mode instead returns a quotient graph and is a distinct object.
"""

from typing import List, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from src.coarsen import GraphCoarsener
from src.explainers.base import BaseExplainer


class CoarsenExplainer(BaseExplainer):
    """Explainer based on Laplacian-guided graph coarsening.

    The coarsener's spectral decomposition is computed once and cached.
    For each target link:
    Partitions may be query-dependent or cached with endpoint refinement.
    Group deletion measures a finite effect on the frozen model; smoothed,
    clipped ratios calibrate edge gradients, without a fidelity guarantee.

    Args:
        model: Trained link-prediction model.
        k: Number of eigenpairs used in spectral scoring.
        alpha: Fraction of original nodes used as the merge budget.
        mode: ``'edge'`` for Protect-and-Project (default),
              ``'coarse'`` for direct coarse-graph output.
        k_hop: Number of hops for neighbourhood extraction.
        k_frac: Fraction of candidate edges to keep.
        device: ``'cpu'`` or ``'cuda'``.
        protect_hops: Protected radius for prediction-guided partitions.
        partition_mode: Query-dependent partition, or a cached global partition
            with full endpoint-cluster splitting (global-refine) or only
            endpoint isolation (global-endpoints).
        evidence_mode: Absolute influence or class-supportive tied-edge scoring.
        intervention_batch_size: Graph copies per deletion forward. Values
            above one require graph-separable inference, as in the tested GCN.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k: int = 100,
        alpha: float = 0.75,
        mode: str = "edge",
        k_hop: int = 2,
        k_frac: float = 0.5,
        device: str = "cpu",
        lambda_pred: float = 1.0,
        fidelity_threshold: float = 0.8,
        protect_hops: int = 1,
        partition_mode: str = "prediction",
        score_method: str = "legacy",
        evidence_mode: str = "absolute",
        intervention_batch_size: int = 1,
        candidate_region: str = 'induced',
    ):
        super().__init__(model, device)
        self.k = k
        self.alpha = alpha
        self.mode = mode
        self.k_hop = k_hop
        self.k_frac = k_frac
        self.lambda_pred = lambda_pred
        self.fidelity_threshold = fidelity_threshold
        if protect_hops < 0:
            raise ValueError("protect_hops must be nonnegative")
        self.protect_hops = protect_hops
        if partition_mode not in {"prediction", "global-refine", "global-endpoints"}:
            raise ValueError("Unknown partition_mode")
        self.partition_mode = partition_mode
        self.score_method = score_method
        if evidence_mode not in {"absolute", "supportive"}:
            raise ValueError('Unknown evidence_mode')
        if intervention_batch_size < 1:
            raise ValueError('intervention_batch_size must be positive')
        self.evidence_mode = evidence_mode
        self.intervention_batch_size = intervention_batch_size
        if candidate_region not in {'induced', 'gcn-boundary'}:
            raise ValueError('Unknown candidate_region')
        self.candidate_region = candidate_region
        self.last_diagnostics = {}
        self._coarsener: Optional[GraphCoarsener] = None
        self._cached_data_id: Optional[int] = None

    def _ensure_fitted(self, data: Data) -> GraphCoarsener:
        data_id = id(data)
        if self._coarsener is not None and self._cached_data_id == data_id:
            return self._coarsener

        device_data = self._to_device(data)
        coarsener = GraphCoarsener(k=self.k, alpha=self.alpha, score_method=self.score_method)
        coarsener.fit(
            edge_index=device_data.edge_index,
            num_nodes=device_data.x.size(0),
            x=device_data.x,
            edge_weight=getattr(device_data, "edge_weight", None),
        )
        self._coarsener = coarsener
        self._cached_data_id = data_id
        return coarsener

    def explain_link(self, data: Data, node_a: int, node_b: int) -> Data:
        if self.mode == "edge":
            return self._explain_link_edge(data, node_a, node_b)
        return self._explain_link_coarse(data, node_a, node_b)

    def _explain_link_edge(self, data: Data, node_a: int, node_b: int) -> Data:
        """Pathway-Calibrated Gradient Selection.

        1. Compute gradient saliency on all edges.
        2. Build prediction-aware partition, map subgraph edges to pathways
           (supernode pairs).
        3. For each pathway, compute group occlusion (remove all pathway edges)
           to measure actual group effect on prediction.
        4. Calibrate individual gradient scores by the pathway's
           redundancy/synergy ratio: score(e) = |g(e)| × CF(pathway(e)).
        5. Select top-k calibrated edges.

        The calibration is a heuristic. Its benefit relative to uncalibrated
        gradients must be measured at matched original-edge budgets.
        """
        self.last_diagnostics = {}
        coarsener = self._ensure_fitted(data)
        data = self._to_device(data)

        protected = self._get_protected_nodes(data, node_a, node_b)

        gradient_all = self._gradient_scores(data, node_a, node_b, data.edge_index)

        from src.partition import prediction_guided_partition, isolate_query_endpoints
        if self.partition_mode == "global-refine":
            partition = [part for cluster in coarsener.partition
                         for part in ([[v] for v in cluster]
                                      if node_a in cluster or node_b in cluster else [cluster])]
        elif self.partition_mode == "global-endpoints":
            partition = isolate_query_endpoints(coarsener.partition, node_a, node_b)
        else:
            partition = prediction_guided_partition(
                edge_index=data.edge_index,
                spectral_scores=coarsener.scores,
                gradient_scores=gradient_all.abs(),
                num_nodes=data.x.size(0),
                alpha=self.alpha,
                protected_nodes=protected,
                lambda_pred=self.lambda_pred,
                fidelity_threshold=self.fidelity_threshold,
            )

        from src.explainers.candidates import candidate_edge_mask
        sub_mask = candidate_edge_mask(data.edge_index, data.x.size(0), [node_a, node_b],
                                       self.k_hop, self.candidate_region)
        sub_ei = data.edge_index[:, sub_mask]
        num_sub = sub_ei.size(1)

        if num_sub == 0:
            nodes = torch.tensor([node_a, node_b], device=self.device)
            return Data(
                x=data.x[nodes],
                edge_index=torch.zeros(2, 0, dtype=torch.long, device=self.device),
                original_node_indices=nodes,
            )

        node_to_super = {nd: si for si, members in enumerate(partition) for nd in members}

        pathway_edges = {}
        edge_pathway = {}
        for j in range(num_sub):
            u, v = int(sub_ei[0, j].item()), int(sub_ei[1, j].item())
            su, sv = node_to_super.get(u, u), node_to_super.get(v, v)
            key = (min(su, sv), max(su, sv))
            pathway_edges.setdefault(key, []).append(j)
            edge_pathway[j] = key

        # Preserve the actual directional derivatives. Matching either direction
        # and taking the first hit silently overwrote one orientation's gradient.
        gradient_scores = gradient_all[sub_mask]
        unique_sizes = []
        multi_gradient = 0.0
        for indices in pathway_edges.values():
            group = sub_ei[:, indices]
            keys = group.min(dim=0).values * data.x.size(0) + group.max(dim=0).values
            size = int(torch.unique(keys).numel())
            unique_sizes.append(size)
            if size > 1:
                multi_gradient += float(gradient_scores[indices].abs().sum())
        total_gradient = float(gradient_scores.abs().sum())
        self.last_diagnostics = {
            "protected_nodes": len(protected), "clusters": len(partition),
            "pathways": len(pathway_edges),
            "multi_edge_pathways": sum(s > 1 for s in unique_sizes),
            "multi_edge_gradient_fraction": multi_gradient / total_gradient if total_gradient else 0.0,
        }

        target = torch.tensor([[node_a], [node_b]], device=self.device)
        with torch.no_grad():
            baseline = self.model(
                data.x, data.edge_index, target,
                edge_weight=getattr(data, "edge_weight", None),
            ).squeeze().item()

        pathway_cf = {}
        from src.explainers.group_interventions import group_deletion_logits
        tested_keys, groups = [], []
        for key, edge_indices in pathway_edges.items():
            if len(edge_indices) < 2:
                pathway_cf[key] = 1.0
                continue
            group = sub_ei[:, edge_indices]
            group_keys = group.min(dim=0).values * data.x.size(0) + group.max(dim=0).values
            tested_keys.append(key)
            groups.append(group_keys)
        modified_logits = group_deletion_logits(
            self.model, data, node_a, node_b, groups, self.intervention_batch_size)
        sign = 1.0 if baseline > 0 else -1.0
        if self.evidence_mode == 'supportive':
            from src.explainers.group_interventions import supportive_edge_scores
            importance_scores = supportive_edge_scores(sub_ei, gradient_scores, data.x.size(0), sign)
        else:
            importance_scores = gradient_scores.abs()
        for key, modified in zip(tested_keys, modified_logits.tolist()):
            edge_indices = pathway_edges[key]
            group_effect = (abs(baseline - modified) if self.evidence_mode == 'absolute'
                            else max(0.0, sign * (baseline - modified)))
            sum_gradient = importance_scores[edge_indices].sum()

            if sum_gradient > 1e-10:
                raw_cf = group_effect / sum_gradient.item()
                # Smooth towards 1.0 for small pathways
                n = len(edge_indices)
                cf = raw_cf * n / (n + 2) + 1.0 * 2 / (n + 2)
                cf = max(0.2, min(5.0, cf))
            else:
                cf = 1.0
            pathway_cf[key] = cf

        calibrated = torch.zeros(num_sub, device=self.device)
        for j in range(num_sub):
            calibrated[j] = importance_scores[j] * pathway_cf.get(edge_pathway[j], 1.0)

        keep_count = max(1, int(num_sub * self.k_frac))
        _, top_idx = calibrated.topk(keep_count)

        kept_ei = sub_ei[:, top_idx]
        kept_weights = calibrated[top_idx]

        involved_nodes = torch.unique(kept_ei)
        node_map = torch.empty(data.x.size(0), dtype=torch.long, device=self.device)
        node_map[involved_nodes] = torch.arange(involved_nodes.size(0), device=self.device)
        relabeled_edges = node_map[kept_ei]

        return Data(
            x=data.x[involved_nodes],
            edge_index=relabeled_edges,
            edge_weight=kept_weights,
            original_node_indices=involved_nodes,
        )

    def _gradient_scores(self, data, node_a, node_b, edge_index):
        edge_mask = torch.ones(
            edge_index.size(1), requires_grad=True, device=self.device,
        )
        weights = edge_mask
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            weights = edge_mask * data.edge_weight
        target = torch.tensor([[node_a], [node_b]], device=self.device)
        out = self.model(data.x, edge_index, target, edge_weight=weights)
        gradient, = torch.autograd.grad(out.squeeze(), edge_mask)
        return gradient.detach()

    def _get_protected_nodes(self, data, node_a, node_b):
        subset, _, _, _ = k_hop_subgraph(
            node_idx=torch.tensor([node_a, node_b], device=self.device),
            num_hops=self.protect_hops,
            edge_index=data.edge_index,
            relabel_nodes=False,
            num_nodes=data.x.size(0),
        )
        return set(int(n.item()) for n in subset)

    def _explain_link_coarse(self, data: Data, node_a: int, node_b: int) -> Data:
        """Legacy mode: return coarse graph directly."""
        coarsener = self._ensure_fitted(data)
        data = self._to_device(data)
        (
            edge_index,
            edge_weight,
            x,
            num_nodes,
            supernode_a,
            supernode_b,
            original_node_indices,
        ) = coarsener.explain_link(node_a, node_b)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            is_coarse_graph=True,
            target_a=supernode_a if supernode_a is not None else 0,
            target_b=supernode_b if supernode_b is not None else 1,
        )

    def explain_batch(self, data: Data, edges: torch.Tensor) -> List[Data]:
        self._ensure_fitted(data)
        results: List[Data] = []
        for i in range(edges.size(1)):
            a = int(edges[0, i].item())
            b = int(edges[1, i].item())
            results.append(self.explain_link(data, a, b))
        return results
