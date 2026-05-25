"""Laplacian-guided graph coarsening explainer.

Uses GraphCoarsener to produce compact explanatory subgraphs by
identifying the most relevant partition of the graph around a
target edge, guided by the graph Laplacian spectrum.
"""

from typing import List, Optional

import torch
from torch_geometric.data import Data

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
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k: int = 100,
        alpha: float = 0.75,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        self.k = k
        self.alpha = alpha
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
