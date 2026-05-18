"""Occlusion and Saliency baseline explainers for GNN link prediction.

These are simple, model-agnostic baselines that do not require any
special PyG explainer module.
"""

from typing import List

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from src.explainers.base import BaseExplainer


class OcclusionExplainer(BaseExplainer):
    """Edge-occlusion explanation by measuring prediction change.

    For each edge in the k-hop neighbourhood of a target link, we
    temporarily remove it and record the absolute change in the
    model's prediction score.  The top ``k_frac`` edges by importance
    form the explanation subgraph.

    Args:
        model: Trained link-prediction model.
        k_hop: Number of hops for the neighbourhood extraction.
        k_frac: Fraction of edges to keep (sparsity control).
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k_hop: int = 2,
        k_frac: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        self.k_hop = k_hop
        self.k_frac = k_frac

    def _get_score(self, data: Data, node_a: int, node_b: int) -> float:
        target = torch.tensor([[node_a], [node_b]], device=self.device)
        return self._predict(data, target).item()

    def explain_link(self, data: Data, node_a: int, node_b: int) -> Data:
        data = self._to_device(data)
        baseline_score = self._get_score(data, node_a, node_b)

        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=torch.tensor([node_a, node_b], device=self.device),
            num_hops=self.k_hop,
            edge_index=data.edge_index,
            relabel_nodes=False,
            num_nodes=data.x.size(0),
        )

        num_edges = sub_edge_index.size(1)
        importances = torch.zeros(num_edges, device=self.device)

        for i in range(num_edges):
            mask = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=self.device)
            global_idx = self._find_global_edge_idx(data.edge_index, sub_edge_index[:, i])
            if global_idx is None:
                continue
            mask[global_idx] = False

            modified_data = Data(
                x=data.x,
                edge_index=data.edge_index[:, mask],
            )
            if hasattr(data, "edge_weight") and data.edge_weight is not None:
                modified_data.edge_weight = data.edge_weight[mask]

            new_score = self._get_score(modified_data, node_a, node_b)
            importances[i] = abs(baseline_score - new_score)

        keep_count = max(1, int(num_edges * self.k_frac))
        _, top_idx = importances.topk(keep_count)

        kept_edge_index = sub_edge_index[:, top_idx]
        kept_weights = importances[top_idx]

        involved_nodes = torch.unique(kept_edge_index)
        node_map = torch.empty(data.x.size(0), dtype=torch.long, device=self.device)
        node_map[involved_nodes] = torch.arange(involved_nodes.size(0), device=self.device)

        relabeled_edges = node_map[kept_edge_index]
        result = Data(
            x=data.x[involved_nodes],
            edge_index=relabeled_edges,
            edge_weight=kept_weights,
            original_node_indices=involved_nodes,
        )
        return result

    @staticmethod
    def _find_global_edge_idx(
        global_edge_index: torch.Tensor,
        edge: torch.Tensor,
    ) -> int | None:
        matches = (
            (global_edge_index[0] == edge[0]) & (global_edge_index[1] == edge[1])
        )
        idxs = matches.nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            return None
        return int(idxs[0].item())

    def explain_batch(self, data: Data, edges: torch.Tensor) -> List[Data]:
        results: List[Data] = []
        for i in range(edges.size(1)):
            a = int(edges[0, i].item())
            b = int(edges[1, i].item())
            results.append(self.explain_link(data, a, b))
        return results


class SaliencyExplainer(BaseExplainer):
    """Gradient-based saliency explanation for GNN link prediction.

    Computes the gradient of the prediction score with respect to each
    edge's presence (represented via a differentiable edge-weight mask)
    and keeps the top ``k_frac`` edges by absolute gradient magnitude.

    Args:
        model: Trained link-prediction model.
        k_frac: Fraction of edges to keep (sparsity control).
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k_frac: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        self.k_frac = k_frac

    def explain_link(self, data: Data, node_a: int, node_b: int) -> Data:
        data = self._to_device(data)

        edge_mask = torch.ones(
            data.edge_index.size(1), requires_grad=True, device=self.device,
        )

        weighted_edges = data.edge_index
        weights = edge_mask
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            weights = edge_mask * data.edge_weight

        target = torch.tensor([[node_a], [node_b]], device=self.device)

        self.model.zero_grad()
        out = self.model(data.x, weighted_edges, target, edge_weight=weights)
        score = out.squeeze()
        score.backward()

        saliency = edge_mask.grad.abs()
        saliency = saliency.detach()

        keep_count = max(1, int(saliency.size(0) * self.k_frac))
        _, top_idx = saliency.topk(keep_count)

        kept_edge_index = data.edge_index[:, top_idx]
        kept_weights = saliency[top_idx]

        involved_nodes = torch.unique(kept_edge_index)
        node_map = torch.empty(data.x.size(0), dtype=torch.long, device=self.device)
        node_map[involved_nodes] = torch.arange(involved_nodes.size(0), device=self.device)

        relabeled_edges = node_map[kept_edge_index]
        result = Data(
            x=data.x[involved_nodes],
            edge_index=relabeled_edges,
            edge_weight=kept_weights,
            original_node_indices=involved_nodes,
        )
        return result

    def explain_batch(self, data: Data, edges: torch.Tensor) -> List[Data]:
        results: List[Data] = []
        for i in range(edges.size(1)):
            a = int(edges[0, i].item())
            b = int(edges[1, i].item())
            results.append(self.explain_link(data, a, b))
        return results
