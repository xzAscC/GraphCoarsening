"""Laplacian-guided graph coarsening explainer.

Uses GraphCoarsener to compute spectral perturbation scores, combined with
gradient saliency for spectral-predictive (SP) edge importance scoring.

Supports two modes:
- ``mode="edge"`` (default): SP scoring = |∂f/∂w_e| × (1 + ρ(e)), where ρ(e)
  is the spectral perturbation score. Selects top-k edges from the k-hop
  neighbourhood. Output is a standard edge-level explanation.
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

    The coarsener is fit once on the full graph and cached so that
    subsequent ``explain_link`` calls reuse the partition structure.

    Args:
        model: Trained link-prediction model.
        k: Target number of coarse nodes (sparsity parameter).
        alpha: Laplacian regularisation weight in ``[0, 1]``.
        mode: ``'edge'`` for project-back edge selection (default),
              ``'coarse'`` for direct coarse-graph output.
        k_hop: Number of hops for neighbourhood extraction (edge mode only).
        k_frac: Fraction of k-hop edges to keep (edge mode only).
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
        """Spectral-predictive edge selection.

        Combines spectral perturbation scores (structural importance) with
        gradient saliency (prediction sensitivity) to rank edges, then
        returns the top-k as a standard edge-level explanation.
        """
        coarsener = self._ensure_fitted(data)
        data = self._to_device(data)

        _, sub_edge_index, _, _ = k_hop_subgraph(
            node_idx=torch.tensor([node_a, node_b], device=self.device),
            num_hops=self.k_hop,
            edge_index=data.edge_index,
            relabel_nodes=False,
            num_nodes=data.x.size(0),
        )
        num_sub_edges = sub_edge_index.size(1)
        if num_sub_edges == 0:
            nodes = torch.tensor([node_a, node_b], device=self.device)
            return Data(
                x=data.x[nodes],
                edge_index=torch.zeros(2, 0, dtype=torch.long, device=self.device),
                original_node_indices=nodes,
            )

        spectral_scores = self._spectral_scores_for_subgraph(
            coarsener.scores, data.edge_index, sub_edge_index, num_sub_edges,
        )

        gradient_scores = self._gradient_scores(
            data, node_a, node_b, data.edge_index,
        )

        sub_gradient = self._spectral_scores_for_subgraph(
            gradient_scores, data.edge_index, sub_edge_index, num_sub_edges,
        )

        grad_norm = self._normalize_to_01(sub_gradient)
        spec_norm = self._normalize_to_01(spectral_scores)
        hybrid_scores = grad_norm * (1.0 + spec_norm)

        keep_count = max(1, int(num_sub_edges * self.k_frac))
        _, top_idx = hybrid_scores.topk(keep_count)

        kept_edge_index = sub_edge_index[:, top_idx]
        kept_weights = hybrid_scores[top_idx]

        involved_nodes = torch.unique(kept_edge_index)
        node_map = torch.empty(data.x.size(0), dtype=torch.long, device=self.device)
        node_map[involved_nodes] = torch.arange(involved_nodes.size(0), device=self.device)
        relabeled_edges = node_map[kept_edge_index]

        return Data(
            x=data.x[involved_nodes],
            edge_index=relabeled_edges,
            edge_weight=kept_weights,
            original_node_indices=involved_nodes,
        )

    @staticmethod
    def _spectral_scores_for_subgraph(all_scores, edge_index, sub_edge_index, num_sub_edges):
        device, scores = all_scores.device, torch.zeros(num_sub_edges, device=all_scores.device)
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
