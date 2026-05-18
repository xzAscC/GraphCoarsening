"""Base explainer interface for GNN link prediction explanations.

All explainers inherit from BaseExplainer and implement explain_link
and explain_batch for a unified comparison framework.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from torch_geometric.data import Data


class BaseExplainer(ABC):
    """Abstract base class for all GNN explainers.

    Every explainer takes a trained link-prediction model and produces
    an explanatory subgraph for a given target edge. The returned Data
    object contains the node features, edge indices, and optional edge
    weights that form the explanation.

    Args:
        model: A trained link-prediction model with the interface
            ``model.forward(x, edge_index, target_edge_index, edge_weight=None)``.
        device: Device string ('cpu' or 'cuda').
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def explain_link(
        self,
        data: Data,
        node_a: int,
        node_b: int,
    ) -> Data:
        """Produce an explanatory subgraph for the link ``(node_a, node_b)``.

        Args:
            data: Full graph as a PyG ``Data`` object (must have ``x`` and
                ``edge_index``; ``edge_weight`` is optional).
            node_a: Source node index.
            node_b: Target node index.

        Returns:
            A ``Data`` object representing the explanation with fields:
            - ``x``: node features for the explanation subgraph.
            - ``edge_index``: edge indices of the explanation subgraph.
            - ``edge_weight`` (optional): importance weights for each edge.
        """
        ...

    @abstractmethod
    def explain_batch(
        self,
        data: Data,
        edges: torch.Tensor,
    ) -> List[Data]:
        """Explain multiple edges at once.

        Args:
            data: Full graph as a PyG ``Data`` object.
            edges: Tensor of shape ``(2, E)`` where each column is an
                edge ``(node_a, node_b)`` to explain.

        Returns:
            List of ``Data`` objects, one per edge.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_device(self, data: Data) -> Data:
        """Move a PyG Data object to ``self.device``."""
        return data.to(self.device)

    def _predict(
        self,
        data: Data,
        target_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Run a forward pass and return raw prediction scores.

        The model is expected to be in eval mode and the returned tensor
        has shape ``(num_target_edges,)`` with values in [0, 1].
        """
        data = self._to_device(data)
        target_edge_index = target_edge_index.to(self.device)
        with torch.no_grad():
            out = self.model(
                data.x,
                data.edge_index,
                target_edge_index,
                edge_weight=getattr(data, "edge_weight", None),
            )
        return out.squeeze()

    def _binary_predict(
        self,
        data: Data,
        target_edge_index: torch.Tensor,
    ) -> int:
        """Return binary prediction (0 or 1) for a single target edge."""
        score = self._predict(data, target_edge_index)
        return int((score > 0.5).item())
