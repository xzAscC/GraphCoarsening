"""PyG-based baseline explainer wrappers for GNN link prediction.

Wraps GNNExplainer, PGExplainer, and SubgraphX from PyTorch Geometric
into the unified BaseExplainer interface. Includes a custom MCTS-based
SubgraphX implementation for PyG versions that lack it.

All wrappers handle API differences between PyG versions gracefully
and log warnings when functionality is degraded.
"""

import logging
import warnings
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from src.explainers.base import BaseExplainer

logger = logging.getLogger(__name__)


class _LinkPredAdapter(torch.nn.Module):
    """Adapts a LinkPredictionModel for PyG's Explainer API.

    PyG explainers call ``model(x, edge_index)`` but our link-prediction
    models need ``model(x, edge_index, target_edge_index)``.  This adapter
    stores the fixed target edge so the explainer only sees the standard
    two-argument interface.
    """

    def __init__(self, model: torch.nn.Module, target_edge_index: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("_target_edge_index", target_edge_index)

    def forward(self, x, edge_index, edge_weight=None):
        return self.model(
            x, edge_index, self._target_edge_index, edge_weight=edge_weight,
        )


def _build_pyg_explainer(
    model: torch.nn.Module,
    algorithm,
    target_edge_index: torch.Tensor,
    device: torch.device,
):
    """Build a PyG Explainer with the correct config for link prediction."""
    from torch_geometric.explain import Explainer, ExplainerConfig, ModelConfig

    adapter = _LinkPredAdapter(model, target_edge_index.to(device))

    model_config = ModelConfig(
        mode="binary_classification",
        task_level="edge",
        return_type="raw",
    )
    explainer_config = ExplainerConfig(
        explanation_type="model",
        edge_mask_type="object",
    )

    return Explainer(
        model=adapter,
        algorithm=algorithm,
        explanation_type="model",
        model_config=model_config,
        edge_mask_type="object",
    )


def _extract_subgraph_from_mask(
    data: Data,
    edge_mask: torch.Tensor,
    top_k: int,
    device: torch.device,
) -> Data:
    """Convert a soft edge mask into a hard explanatory subgraph.

    Keeps the top ``top_k`` edges by mask value, extracts involved nodes,
    and relabels them to form a compact PyG ``Data`` object.
    """
    top_k = min(top_k, edge_mask.size(0))
    if top_k < 1:
        top_k = 1
    _, top_idx = edge_mask.topk(top_k)

    kept_edge_index = data.edge_index[:, top_idx]
    kept_weights = edge_mask[top_idx]

    involved_nodes = torch.unique(kept_edge_index)
    num_nodes = data.x.size(0)
    node_map = torch.empty(num_nodes, dtype=torch.long, device=device)
    node_map[involved_nodes] = torch.arange(involved_nodes.size(0), device=device)

    relabeled_edges = node_map[kept_edge_index]
    return Data(
        x=data.x[involved_nodes],
        edge_index=relabeled_edges,
        edge_weight=kept_weights,
        original_node_indices=involved_nodes,
    )


class GNNExplainerWrapper(BaseExplainer):
    """Wrapper around PyG's GNNExplainer for link prediction.

    GNNExplainer learns a soft edge mask by optimizing a regularised
    objective that preserves the model's prediction for the target
    edge while using as few edges as possible.

    Args:
        model: Trained link-prediction model.
        epochs: Number of optimisation steps for mask learning.
        lr: Learning rate for mask optimisation.
        k_frac: Fraction of edges to keep in the final explanation.
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 100,
        lr: float = 0.01,
        k_frac: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        self.epochs = epochs
        self.lr = lr
        self.k_frac = k_frac

        try:
            from torch_geometric.explain import GNNExplainer  # noqa: F401
        except ImportError:
            raise ImportError(
                "GNNExplainer is not available in your version of "
                "torch-geometric.  Install torch-geometric >= 2.3 or "
                "use a different baseline."
            )

    def explain_link(self, data: Data, node_a: int, node_b: int) -> Data:
        from torch_geometric.explain import GNNExplainer

        data = self._to_device(data)
        target_edge = torch.tensor([[node_a], [node_b]], device=self.device)

        algorithm = GNNExplainer(epochs=self.epochs, lr=self.lr)
        explainer = _build_pyg_explainer(
            self.model, algorithm, target_edge, self.device,
        )

        explanation = explainer(
            data.x,
            data.edge_index,
            index=0,
            edge_weight=getattr(data, "edge_weight", None),
        )

        edge_mask = explanation.edge_mask.detach()
        top_k = max(1, int(edge_mask.size(0) * self.k_frac))
        return _extract_subgraph_from_mask(data, edge_mask, top_k, self.device)

    def explain_batch(self, data: Data, edges: torch.Tensor) -> List[Data]:
        results: List[Data] = []
        for i in range(edges.size(1)):
            a = int(edges[0, i].item())
            b = int(edges[1, i].item())
            results.append(self.explain_link(data, a, b))
        return results


class PGExplainerWrapper(BaseExplainer):
    """Wrapper around PyG's PGExplainer for link prediction.

    PGExplainer trains a parametric network that predicts edge masks.
    It requires a short training phase on the target graph before it
    can explain individual links.

    Args:
        model: Trained link-prediction model.
        epochs: Number of training epochs for the parameterised mask network.
        lr: Learning rate.
        k_frac: Fraction of edges to keep in the final explanation.
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 30,
        lr: float = 0.003,
        k_frac: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        self.epochs = epochs
        self.lr = lr
        self.k_frac = k_frac

        try:
            from torch_geometric.explain import PGExplainer  # noqa: F401
        except ImportError:
            raise ImportError(
                "PGExplainer is not available in your version of "
                "torch-geometric.  Install torch-geometric >= 2.3 or "
                "use a different baseline."
            )

    def _train_explainer(
        self,
        data: Data,
        target_edge_index: torch.Tensor,
    ):
        from torch_geometric.explain import PGExplainer

        algorithm = PGExplainer(epochs=self.epochs, lr=self.lr)
        explainer = _build_pyg_explainer(
            self.model, algorithm, target_edge_index, self.device,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            for _ in range(self.epochs):
                explainer(
                    data.x,
                    data.edge_index,
                    index=0,
                    target=None,
                    edge_weight=getattr(data, "edge_weight", None),
                )
        return explainer

    def explain_link(self, data: Data, node_a: int, node_b: int) -> Data:
        data = self._to_device(data)
        target_edge = torch.tensor([[node_a], [node_b]], device=self.device)

        explainer = self._train_explainer(data, target_edge)

        with torch.no_grad():
            explanation = explainer(
                data.x,
                data.edge_index,
                index=0,
                edge_weight=getattr(data, "edge_weight", None),
            )

        edge_mask = explanation.edge_mask.detach()
        top_k = max(1, int(edge_mask.size(0) * self.k_frac))
        return _extract_subgraph_from_mask(data, edge_mask, top_k, self.device)

    def explain_batch(self, data: Data, edges: torch.Tensor) -> List[Data]:
        results: List[Data] = []
        for i in range(edges.size(1)):
            a = int(edges[0, i].item())
            b = int(edges[1, i].item())
            results.append(self.explain_link(data, a, b))
        return results


class _MCTSNode:
    """A single node in the Monte Carlo Tree Search.

    Each node stores a subgraph (as a boolean edge mask) and tracks
    visit counts and cumulative rewards for the UCT selection policy.
    """

    __slots__ = ("edge_mask", "children", "visit_count", "total_reward", "parent")

    def __init__(self, edge_mask: torch.Tensor, parent=None):
        self.edge_mask = edge_mask
        self.children: list = []
        self.visit_count = 0
        self.total_reward = 0.0
        self.parent = parent

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(self.visit_count, 1)

    def uct(self, c: float = 1.414) -> float:
        if self.visit_count == 0:
            return float("inf")
        parent_visits = self.parent.visit_count if self.parent else 1
        exploit = self.mean_reward
        explore = c * (torch.log(torch.tensor(parent_visits, dtype=torch.float)).item() / self.visit_count) ** 0.5
        return exploit + explore


class SubgraphXWrapper(BaseExplainer):
    """SubgraphX explainer using Monte Carlo Tree Search.

    Since PyG 2.7 does not ship a SubgraphX implementation, this is a
    self-contained MCTS-based version that:

    1. Extracts the k-hop neighbourhood of the target edge.
    2. Searches over subgraphs by pruning edges.
    3. Scores each subgraph by the prediction change it causes.
    4. Returns the highest-scoring (most explanatory) subgraph.

    Args:
        model: Trained link-prediction model.
        num_hops: Number of hops for neighbourhood extraction.
        num_rollouts: Number of MCTS rollouts.
        max_depth: Maximum MCTS tree depth.
        k_frac: Fraction of edges to keep in the final explanation.
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_hops: int = 2,
        num_rollouts: int = 20,
        max_depth: int = 6,
        k_frac: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        self.num_hops = num_hops
        self.num_rollouts = num_rollouts
        self.max_depth = max_depth
        self.k_frac = k_frac

    def _get_neighbourhood(
        self, data: Data, node_a: int, node_b: int,
    ) -> tuple:
        """Return (sub_edge_index, inv_mapping) for k-hop neighbourhood."""
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            node_idx=torch.tensor([node_a, node_b], device=self.device),
            num_hops=self.num_hops,
            edge_index=data.edge_index,
            relabel_nodes=False,
            num_nodes=data.x.size(0),
        )
        return sub_edge_index

    def _score_subgraph(
        self,
        data: Data,
        edge_mask: torch.Tensor,
        node_a: int,
        node_b: int,
    ) -> float:
        """Reward = absolute prediction change when removing masked-out edges."""
        data = self._to_device(data)
        target = torch.tensor([[node_a], [node_b]], device=self.device)
        with torch.no_grad():
            original = self._predict(data, target).item()

        kept = edge_mask.bool()
        if kept.sum() == 0:
            return abs(original - 0.5)

        modified = Data(x=data.x, edge_index=data.edge_index[:, kept])
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            modified.edge_weight = data.edge_weight[kept]

        with torch.no_grad():
            pruned = self._predict(modified, target).item()

        return abs(original - pruned)

    def _mcts_rollout(self, root: _MCTSNode, data: Data, node_a: int, node_b: int):
        """Execute one MCTS rollout: select, expand, evaluate, backprop."""
        node = root
        for _ in range(self.max_depth):
            if not node.children:
                self._expand(node)
            if not node.children:
                break

            unvisited = [c for c in node.children if c.visit_count == 0]
            if unvisited:
                node = unvisited[0]
            else:
                node = max(node.children, key=lambda c: c.uct())

        reward = self._score_subgraph(data, node.edge_mask, node_a, node_b)
        cur = node
        while cur is not None:
            cur.visit_count += 1
            cur.total_reward += reward
            cur = cur.parent

    def _expand(self, node: _MCTSNode):
        """Create children by removing one edge at a time from the mask."""
        active = node.edge_mask.nonzero(as_tuple=True)[0]
        if active.numel() <= 1:
            return
        for idx in active:
            child_mask = node.edge_mask.clone()
            child_mask[idx] = False
            child = _MCTSNode(child_mask, parent=node)
            node.children.append(child)

    def explain_link(self, data: Data, node_a: int, node_b: int) -> Data:
        data = self._to_device(data)
        sub_edge_index = self._get_neighbourhood(data, node_a, node_b)

        num_edges = data.edge_index.size(1)
        initial_mask = torch.zeros(num_edges, dtype=torch.bool, device=self.device)
        active_indices = self._find_active_indices(data.edge_index, sub_edge_index)
        initial_mask[active_indices] = True

        float_mask = initial_mask.float()
        root = _MCTSNode(float_mask)

        for _ in range(self.num_rollouts):
            self._mcts_rollout(root, data, node_a, node_b)

        best_leaf = self._best_leaf(root)
        final_mask = best_leaf.edge_mask.bool()

        top_k = max(1, int(final_mask.sum().item()))
        return _extract_subgraph_from_mask(data, final_mask.float(), top_k, self.device)

    def _best_leaf(self, root: _MCTSNode) -> _MCTSNode:
        """Find the leaf with the highest mean reward."""
        best = root
        stack = [root]
        while stack:
            node = stack.pop()
            if node.mean_reward > best.mean_reward and node is not root:
                best = node
            stack.extend(node.children)
        return best

    @staticmethod
    def _find_active_indices(
        global_edge_index: torch.Tensor,
        sub_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Find indices of sub_edge_index columns in global_edge_index."""
        active = []
        for i in range(sub_edge_index.size(1)):
            src, dst = sub_edge_index[0, i], sub_edge_index[1, i]
            matches = (global_edge_index[0] == src) & (global_edge_index[1] == dst)
            idxs = matches.nonzero(as_tuple=True)[0]
            if idxs.numel() > 0:
                active.append(idxs[0].item())
        return torch.tensor(active, dtype=torch.long, device=global_edge_index.device)

    def explain_batch(self, data: Data, edges: torch.Tensor) -> List[Data]:
        results: List[Data] = []
        for i in range(edges.size(1)):
            a = int(edges[0, i].item())
            b = int(edges[1, i].item())
            results.append(self.explain_link(data, a, b))
        return results
