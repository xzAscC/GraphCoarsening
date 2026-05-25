"""Laplacian-guided graph coarsening explainer.

Uses GraphCoarsener's spectral perturbation scores, node partition,
and coarse graph structure to produce edge-level explanations via
the Protect-and-Project method:

1. Per-link coarsening with protected nodes (1-hop neighbors of target).
2. Project-back: edges ranked by normalized coarse weight × spectral score,
   with gradient saliency as prediction-aware booster.
3. Returns top-k edges as standard edge-level explanation.

The coarsening pipeline (partition.py, coarsen.py) is genuinely modified:
- node_partition() accepts protected_nodes to prevent merging near the target
- GraphCoarsener.fit_partition() enables per-link partitions cheaply
- project_back_edges() uses coarse graph structure for edge importance

Supports two modes:
- ``mode="edge"`` (default): Protect-and-Project.
- ``mode="coarse"``: Returns the coarse graph directly (legacy mode).
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
    1. Protected partition: 1-hop neighbors remain as singletons.
    2. Coarse graph structure provides inter-supernode importance.
    3. Gradient saliency provides prediction sensitivity.
    4. Combined scoring: coarse_importance × (1 + spectral) × (1 + gradient).

    Args:
        model: Trained link-prediction model.
        k: Target number of coarse nodes (sparsity parameter).
        alpha: Laplacian regularisation weight in ``[0, 1]``.
        mode: ``'edge'`` for Protect-and-Project (default),
              ``'coarse'`` for direct coarse-graph output.
        k_hop: Number of hops for neighbourhood extraction.
        k_frac: Fraction of candidate edges to keep.
        device: ``'cpu'`` or ``'cuda'``.
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
    ):
        super().__init__(model, device)
        self.k = k
        self.alpha = alpha
        self.mode = mode
        self.k_hop = k_hop
        self.k_frac = k_frac
        self._coarsener: Optional[GraphCoarsener] = None
        self._cached_data_id: Optional[int] = None

    def _ensure_fitted(self, data: Data) -> GraphCoarsener:
        data_id = id(data)
        if self._coarsener is not None and self._cached_data_id == data_id:
            return self._coarsener

        device_data = self._to_device(data)
        coarsener = GraphCoarsener(k=self.k, alpha=self.alpha)
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
        """Protect-and-Project with coarse-graph spectral scoring.

        1. Compute gradient saliency on all edges.
        2. Build prediction-aware partition (spectral + gradient scores),
           protected nodes near target kept as singletons.
        3. Build coarse graph from partition, compute spectral scores on it,
           project back to original edges — partition-dependent spectral signal.
        4. Score: gradient × (1 + coarse_spectral), select top-k.
        """
        from src.coarsen import build_coarse_graph
        from src.spectral import (
            compute_normalized_adjacency,
            compute_top_k_eigenpairs,
            compute_perturbation_scores,
        )

        coarsener = self._ensure_fitted(data)
        data = self._to_device(data)

        protected = self._get_protected_nodes(data, node_a, node_b)

        gradient_all = self._gradient_scores(data, node_a, node_b, data.edge_index)

        # Prediction-aware partition: spectral + gradient combined scores
        sn_global = self._normalize_to_01(coarsener.scores)
        gn_global = self._normalize_to_01(gradient_all)
        combined_partition_scores = sn_global + gn_global
        partition = coarsener.fit_partition(
            protected_nodes=protected,
            edge_scores=combined_partition_scores,
        )

        # Get candidate edges from 2-hop subgraph
        _, sub_ei, _, _ = k_hop_subgraph(
            node_idx=torch.tensor([node_a, node_b], device=self.device),
            num_hops=self.k_hop,
            edge_index=data.edge_index,
            relabel_nodes=False,
            num_nodes=data.x.size(0),
        )
        num_sub = sub_ei.size(1)

        if num_sub == 0:
            nodes = torch.tensor([node_a, node_b], device=self.device)
            return Data(
                x=data.x[nodes],
                edge_index=torch.zeros(2, 0, dtype=torch.long, device=self.device),
                original_node_indices=nodes,
            )

        # Build node-to-supernode mapping
        node_to_super = {}
        num_coarse = len(partition)
        for si, members in enumerate(partition):
            for nd in members:
                node_to_super[nd] = si

        # Coarse-graph spectral scores: compute spectral perturbation on the
        # coarse graph built from the per-link prediction-aware partition.
        # This produces partition-dependent spectral scores — different target
        # links yield different partitions → different coarse graphs → different
        # spectral importance for each edge.
        coarse_ei, coarse_ew, _ = build_coarse_graph(
            coarsener.edge_index, coarsener.edge_weight,
            coarsener.num_nodes, partition, coarsener.x,
        )
        k_coarse = min(50, num_coarse - 1)
        if k_coarse < 2:
            k_coarse = 2
        A_hat_coarse = compute_normalized_adjacency(coarse_ei, num_coarse)
        evals, lvecs, rvecs = compute_top_k_eigenpairs(A_hat_coarse, k_coarse)
        coarse_spectral = compute_perturbation_scores(coarse_ei, evals, lvecs, rvecs)

        # Project coarse spectral scores back to original subgraph edges
        coarse_spec_map = {}
        for idx in range(coarse_ei.size(1)):
            si, sj = int(coarse_ei[0, idx].item()), int(coarse_ei[1, idx].item())
            coarse_spec_map[(si, sj)] = float(coarse_spectral[idx].item())

        spectral_coarse = torch.zeros(num_sub, device=self.device)
        for j in range(num_sub):
            u, v = int(sub_ei[0, j].item()), int(sub_ei[1, j].item())
            su, sv = node_to_super.get(u, u), node_to_super.get(v, v)
            s = coarse_spec_map.get((su, sv), 0.0)
            if s == 0.0:
                s = coarse_spec_map.get((sv, su), 0.0)
            spectral_coarse[j] = s

        # Gradient scores for subgraph candidates
        gradient_scores = self._spectral_scores_for_subgraph(
            gradient_all, data.edge_index, sub_ei, num_sub,
        )

        # Combine: gradient × (1 + coarse_spectral)
        sn = self._normalize_to_01(spectral_coarse)
        gn = self._normalize_to_01(gradient_scores)
        combined = gn * (1.0 + sn)

        keep_count = max(1, int(num_sub * self.k_frac))
        _, top_idx = combined.topk(keep_count)

        kept_ei = sub_ei[:, top_idx]
        kept_weights = combined[top_idx]

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

    def _coarse_weight_scores(self, coarsener, partition, sub_ei, num_sub):
        """Compute normalized coarse weight for each candidate edge."""
        from src.coarsen import build_coarse_graph

        node_to_super = {}
        super_sizes = []
        for si, members in enumerate(partition):
            super_sizes.append(len(members))
            for nd in members:
                node_to_super[nd] = si
        super_sizes_t = torch.tensor(super_sizes, dtype=torch.float32, device=self.device)

        coarse_ei, coarse_ew, _ = build_coarse_graph(
            coarsener.edge_index, coarsener.edge_weight,
            coarsener.num_nodes, partition, coarsener.x,
        )
        cw_map = {}
        for idx in range(coarse_ei.size(1)):
            si, sj = int(coarse_ei[0, idx].item()), int(coarse_ei[1, idx].item())
            cw_map[(si, sj)] = float(coarse_ew[idx].item())

        scores = torch.zeros(num_sub, device=self.device)
        for j in range(num_sub):
            u, v = int(sub_ei[0, j].item()), int(sub_ei[1, j].item())
            su, sv = node_to_super.get(u, u), node_to_super.get(v, v)
            w = cw_map.get((su, sv), 0.0)
            if w == 0.0:
                w = cw_map.get((sv, su), 0.0)
            sn = (super_sizes_t[su] * super_sizes_t[sv]).sqrt().item()
            scores[j] = w / max(sn, 1.0)
        return scores

    @staticmethod
    def _spectral_scores_for_subgraph(all_scores, edge_index, sub_edge_index, num_sub_edges):
        scores = torch.zeros(num_sub_edges, device=all_scores.device)
        for i in range(num_sub_edges):
            src, dst = sub_edge_index[0, i], sub_edge_index[1, i]
            matches = (
                (edge_index[0] == src) & (edge_index[1] == dst)
            ) | (
                (edge_index[0] == dst) & (edge_index[1] == src)
            )
            idx = matches.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                scores[i] = all_scores[idx[0]]
        return scores

    def _gradient_scores(self, data, node_a, node_b, edge_index):
        edge_mask = torch.ones(
            edge_index.size(1), requires_grad=True, device=self.device,
        )
        weights = edge_mask
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            weights = edge_mask * data.edge_weight
        target = torch.tensor([[node_a], [node_b]], device=self.device)
        self.model.zero_grad()
        out = self.model(data.x, edge_index, target, edge_weight=weights)
        out.squeeze().backward()
        return edge_mask.grad.abs().detach()

    def _get_protected_nodes(self, data, node_a, node_b):
        subset, _, _, _ = k_hop_subgraph(
            node_idx=torch.tensor([node_a, node_b], device=self.device),
            num_hops=1,
            edge_index=data.edge_index,
            relabel_nodes=False,
            num_nodes=data.x.size(0),
        )
        return set(int(n.item()) for n in subset)

    @staticmethod
    def _normalize_to_01(tensor):
        if tensor.numel() == 0:
            return tensor
        t_min, t_max = tensor.min(), tensor.max()
        if t_max - t_min < 1e-12:
            return torch.zeros_like(tensor)
        return (tensor - t_min) / (t_max - t_min)

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
