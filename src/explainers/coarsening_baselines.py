"""Baseline explainers for GNN link prediction.

These baselines directly address reviewer feedback that trivial baselines
(full graph, greedy deletion) can achieve perfect fidelity, by providing a
spectrum of explanation quality from trivial (full graph) to sophisticated
(greedy deletion, coarsening variants).

Non-coarsening baselines (subgraph-based):
    FullGraphBaseline        -- entire graph (trivial upper bound)
    KHopSubgraphBaseline     -- k-hop enclosing subgraph
    RandomSubgraphBaseline   -- random edges from k-hop subgraph
    DegreeBasedBaseline      -- top edges by endpoint degree sum
    PageRankBasedBaseline    -- top edges by PageRank score sum
    GreedyDeletionBaseline   -- iterative least-important edge removal

Coarsening baselines (use GraphCoarsener pipeline with custom scoring):
    RandomCoarseningExplainer           -- random edge scores
    HeavyEdgeCoarseningExplainer        -- merge by weight / degree product
    EffectiveResistanceCoarseningExplainer -- effective resistance scores
    NoRefinementExplainer               -- spectral coarsening without linkwise refinement
"""

from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from src.coarsen import (
    build_coarse_graph,
    linkwise_coarse_graph,
    logsumexp_features,
)
from src.explainers.base import BaseExplainer
from src.partition import node_partition
from src.spectral import (
    compute_normalized_adjacency,
    compute_perturbation_scores,
    compute_top_k_eigenpairs,
)

__all__ = [
    "FullGraphBaseline",
    "KHopSubgraphBaseline",
    "RandomSubgraphBaseline",
    "DegreeBasedBaseline",
    "PageRankBasedBaseline",
    "GreedyDeletionBaseline",
    "RandomCoarseningExplainer",
    "HeavyEdgeCoarseningExplainer",
    "EffectiveResistanceCoarseningExplainer",
    "NoRefinementExplainer",
]


# =====================================================================
# Helper utilities
# =====================================================================


def _relabel_subgraph(
    data: Data,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    device: torch.device,
) -> Data:
    """Relabel involved nodes to a contiguous range and build a Data object.

    Args:
        data: Original full-graph Data (used for ``x``).
        edge_index: Edge indices with *original* node ids, shape ``(2, E')``.
        edge_weight: Optional per-edge weights, shape ``(E',)``.
        device: Target device.

    Returns:
        A ``Data`` object with relabelled ``edge_index``, ``x`` subset,
        ``original_node_indices``, and optional ``edge_weight``.
    """
    if edge_index.numel() == 0:
        assert data.x is not None
        return Data(
            x=torch.zeros(0, data.x.size(1), device=device),
            edge_index=torch.zeros(2, 0, dtype=torch.long, device=device),
            original_node_indices=torch.zeros(0, dtype=torch.long, device=device),
        )

    assert data.x is not None
    involved = torch.unique(edge_index)
    num_original = data.x.size(0)
    node_map = torch.empty(num_original, dtype=torch.long, device=device)
    node_map[involved] = torch.arange(involved.size(0), device=device)

    result = Data(
        x=data.x[involved],
        edge_index=node_map[edge_index],
        original_node_indices=involved,
    )
    if edge_weight is not None:
        result.edge_weight = edge_weight
    return result


def _resolve_keep_count(
    num_edges: int,
    budget: Optional[int],
    k_frac: Optional[float],
) -> int:
    """Determine how many edges to keep from *budget* or *k_frac*.

    ``budget`` takes priority.  If neither is given, return ``num_edges``.
    """
    if budget is not None:
        return max(1, min(budget, num_edges))
    if k_frac is not None:
        return max(1, int(num_edges * k_frac))
    return num_edges


def _khop_mask(
    data: Data,
    node_a: int,
    node_b: int,
    k_hop: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a boolean mask over ``data.edge_index`` for the k-hop subgraph."""
    assert data.edge_index is not None and data.x is not None
    subset, _, _, _ = k_hop_subgraph(
        node_idx=torch.tensor([node_a, node_b], device=device),
        num_hops=k_hop,
        edge_index=data.edge_index,
        relabel_nodes=False,
        num_nodes=data.x.size(0),
    )
    in_sub = torch.zeros(data.x.size(0), dtype=torch.bool, device=device)
    in_sub[subset] = True
    return in_sub[data.edge_index[0]] & in_sub[data.edge_index[1]]


def _khop_edges(
    data: Data,
    node_a: int,
    node_b: int,
    k_hop: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return ``(sub_edge_index, sub_edge_weight)`` for the k-hop subgraph."""
    assert data.edge_index is not None
    mask = _khop_mask(data, node_a, node_b, k_hop, device)
    sub_ei = data.edge_index[:, mask]
    sub_ew: Optional[torch.Tensor] = None
    if hasattr(data, "edge_weight") and data.edge_weight is not None:
        sub_ew = data.edge_weight[mask]
    return sub_ei, sub_ew


def _build_coarse_data(
    coarse_edge_index: torch.Tensor,
    coarse_edge_weight: torch.Tensor,
    coarse_features: torch.Tensor,
    supernode_a: int,
    supernode_b: int,
    device: torch.device,
) -> Data:
    """Build a ``Data`` object from coarse-graph components.

    Follows the same relabelling convention as ``CoarsenExplainer``.
    """
    if coarse_edge_index.numel() == 0:
        return Data(
            x=torch.zeros(0, coarse_features.size(1), device=device),
            edge_index=torch.zeros(2, 0, dtype=torch.long, device=device),
            edge_weight=torch.zeros(0, device=device),
            is_coarse_graph=True,
            target_a=0,
            target_b=0,
        )

    involved = torch.unique(coarse_edge_index)
    max_idx = int(involved.max().item()) + 1
    node_map = torch.empty(max_idx, dtype=torch.long, device=device)
    node_map[involved] = torch.arange(involved.size(0), device=device)

    return Data(
        x=coarse_features[involved],
        edge_index=node_map[coarse_edge_index],
        edge_weight=coarse_edge_weight,
        is_coarse_graph=True,
        target_a=int(node_map[supernode_a].item()) if supernode_a < max_idx else 0,
        target_b=int(node_map[supernode_b].item()) if supernode_b < max_idx else 0,
    )


def _compute_pagerank(
    edge_index: torch.Tensor,
    num_nodes: int,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Power-iteration PageRank on a graph with contiguous node indices.

    Treats the graph as undirected for transition probabilities.

    Returns:
        PageRank scores, shape ``(num_nodes,)``.
    """
    device = edge_index.device

    # Make edges bidirectional for undirected PageRank.
    rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    bi = torch.cat([edge_index, rev], dim=1)
    # Unique columns to avoid double-counting.
    bi = torch.unique(bi, dim=1)

    row, col = bi[0], bi[1]
    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=device))
    deg = torch.clamp(deg, min=1)

    # Sparse column-stochastic transition T[dst, src] = 1/deg(src).
    weights = 1.0 / deg[row]
    T = torch.sparse_coo_tensor(
        torch.stack([col, row]), weights, size=(num_nodes, num_nodes),
    ).coalesce()

    r = torch.full((num_nodes,), 1.0 / num_nodes, device=device)
    for _ in range(max_iter):
        r_new = damping * torch.sparse.mm(T, r.unsqueeze(1)).squeeze(1)
        r_new = r_new + (1 - damping) / num_nodes
        if (r_new - r).abs().max().item() < tol:
            break
        r = r_new
    return r


def _find_supernode_indices(
    partition: List[List[int]],
    node_a: int,
    node_b: int,
) -> Tuple[int, int]:
    """Return the supernode indices containing *node_a* and *node_b*."""
    sa: Optional[int] = None
    sb: Optional[int] = None
    for i, members in enumerate(partition):
        if node_a in members:
            sa = i
        if node_b in members:
            sb = i
    if sa is None or sb is None:
        raise ValueError(
            f"Nodes {node_a}/{node_b} not found in partition "
            f"(sa={sa}, sb={sb})."
        )
    return sa, sb


# =====================================================================
# Non-coarsening baselines
# =====================================================================


class FullGraphBaseline(BaseExplainer):
    """Returns the entire input graph as the explanation.

    This is the trivial upper bound -- no filtering at all.  Useful for
    measuring the maximum possible fidelity.

    Args:
        model: Trained link-prediction model.
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        super().__init__(model, device)

    def explain_link(self, data: Data, node_a: int, node_b: int) -> Data:
        data = self._to_device(data)
        assert data.x is not None and data.edge_index is not None
        n = data.x.size(0)
        return Data(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
            edge_weight=(
                data.edge_weight.clone()
                if hasattr(data, "edge_weight") and data.edge_weight is not None
                else None
            ),
            original_node_indices=torch.arange(n, device=self.device),
        )

    def explain_batch(self, data: Data, edges: torch.Tensor) -> List[Data]:
        data = self._to_device(data)
        result = self.explain_link(data, int(edges[0, 0].item()), int(edges[1, 0].item()))
        return [result] * edges.size(1)


class KHopSubgraphBaseline(BaseExplainer):
    """Returns the k-hop enclosing subgraph around ``(node_a, node_b)``.

    Args:
        model: Trained link-prediction model.
        k_hop: Number of hops. Default 2.
        budget: Exact number of edges to keep. Overrides ``k_frac``.
        k_frac: Fraction of k-hop edges to keep. Default ``None`` (keep all).
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k_hop: int = 2,
        budget: Optional[int] = None,
        k_frac: Optional[float] = None,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        self.k_hop = k_hop
        self.budget = budget
        self.k_frac = k_frac

    def explain_link(self, data: Data, node_a: int, node_b: int) -> Data:
        data = self._to_device(data)
        sub_ei, sub_ew = _khop_edges(data, node_a, node_b, self.k_hop, self.device)
        keep = _resolve_keep_count(sub_ei.size(1), self.budget, self.k_frac)
        if keep < sub_ei.size(1):
            perm = torch.randperm(sub_ei.size(1), device=self.device)[:keep]
            sub_ei = sub_ei[:, perm]
            if sub_ew is not None:
                sub_ew = sub_ew[perm]
        return _relabel_subgraph(data, sub_ei, sub_ew, self.device)

    def explain_batch(self, data: Data, edges: torch.Tensor) -> List[Data]:
        results: List[Data] = []
        for i in range(edges.size(1)):
            results.append(
                self.explain_link(
                    data, int(edges[0, i].item()), int(edges[1, i].item())
                )
            )
        return results


class RandomSubgraphBaseline(BaseExplainer):
    """Returns a random subgraph with exactly ``budget`` edges.

    Edges are sampled uniformly from the k-hop enclosing subgraph.

    Args:
        model: Trained link-prediction model.
        k_hop: Number of hops for the enclosing subgraph. Default 2.
        budget: Exact number of edges to keep.
        k_frac: Fraction of k-hop edges to keep (used when ``budget`` is None).
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k_hop: int = 2,
        budget: Optional[int] = None,
        k_frac: Optional[float] = 0.5,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        self.k_hop = k_hop
        self.budget = budget
        self.k_frac = k_frac

    def explain_link(self, data: Data, node_a: int, node_b: int) -> Data:
        data = self._to_device(data)
        sub_ei, sub_ew = _khop_edges(data, node_a, node_b, self.k_hop, self.device)
        keep = _resolve_keep_count(sub_ei.size(1), self.budget, self.k_frac)
        perm = torch.randperm(sub_ei.size(1), device=self.device)[:keep]
        sub_ei = sub_ei[:, perm]
        if sub_ew is not None:
            sub_ew = sub_ew[perm]
        return _relabel_subgraph(data, sub_ei, sub_ew, self.device)

    def explain_batch(self, data: Data, edges: torch.Tensor) -> List[Data]:
        results: List[Data] = []
        for i in range(edges.size(1)):
            results.append(
                self.explain_link(
                    data, int(edges[0, i].item()), int(edges[1, i].item())
                )
            )
        return results


class DegreeBasedBaseline(BaseExplainer):
    """Ranks edges by the sum of endpoint degrees and keeps the top ones.

    Uses the k-hop enclosing subgraph as the candidate set.

    Args:
        model: Trained link-prediction model.
        k_hop: Number of hops for the enclosing subgraph. Default 2.
        budget: Exact number of edges to keep.
        k_frac: Fraction of edges to keep (used when ``budget`` is None).
            Default 0.5.
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k_hop: int = 2,
        budget: Optional[int] = None,
        k_frac: Optional[float] = 0.5,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        self.k_hop = k_hop
        self.budget = budget
        self.k_frac = k_frac

    def explain_link(self, data: Data, node_a: int, node_b: int) -> Data:
        data = self._to_device(data)
        sub_ei, sub_ew = _khop_edges(data, node_a, node_b, self.k_hop, self.device)
        if sub_ei.size(1) == 0:
            return _relabel_subgraph(data, sub_ei, sub_ew, self.device)

        # Full-graph degrees (global importance).
        assert data.x is not None and data.edge_index is not None
        deg = torch.zeros(data.x.size(0), device=self.device)
        deg.scatter_add_(
            0, data.edge_index[0], torch.ones(data.edge_index.size(1), device=self.device)
        )

        # Score each edge by sum of endpoint degrees.
        edge_scores = deg[sub_ei[0]] + deg[sub_ei[1]]

        keep = _resolve_keep_count(sub_ei.size(1), self.budget, self.k_frac)
        _, top_idx = edge_scores.topk(keep)

        kept_ei = sub_ei[:, top_idx]
        kept_ew = edge_scores[top_idx] if sub_ew is None else sub_ew[top_idx]
        return _relabel_subgraph(data, kept_ei, kept_ew, self.device)

    def explain_batch(self, data: Data, edges: torch.Tensor) -> List[Data]:
        results: List[Data] = []
        for i in range(edges.size(1)):
            results.append(
                self.explain_link(
                    data, int(edges[0, i].item()), int(edges[1, i].item())
                )
            )
        return results


class PageRankBasedBaseline(BaseExplainer):
    """Ranks edges by PageRank scores of their endpoints.

    Runs PageRank on the k-hop subgraph and keeps the top ``budget`` edges
    by the sum of endpoint PageRank values.

    Args:
        model: Trained link-prediction model.
        k_hop: Number of hops for the enclosing subgraph. Default 2.
        budget: Exact number of edges to keep.
        k_frac: Fraction of edges to keep. Default 0.5.
        damping: PageRank damping factor. Default 0.85.
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k_hop: int = 2,
        budget: Optional[int] = None,
        k_frac: Optional[float] = 0.5,
        damping: float = 0.85,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        self.k_hop = k_hop
        self.budget = budget
        self.k_frac = k_frac
        self.damping = damping

    def explain_link(self, data: Data, node_a: int, node_b: int) -> Data:
        data = self._to_device(data)
        sub_ei, sub_ew = _khop_edges(data, node_a, node_b, self.k_hop, self.device)
        if sub_ei.size(1) == 0:
            return _relabel_subgraph(data, sub_ei, sub_ew, self.device)

        # Relabel to contiguous range for PageRank.
        assert data.x is not None
        involved = torch.unique(sub_ei)
        n_sub = involved.size(0)
        node_map = torch.empty(data.x.size(0), dtype=torch.long, device=self.device)
        node_map[involved] = torch.arange(n_sub, device=self.device)
        relabeled_ei = node_map[sub_ei]

        pr = _compute_pagerank(relabeled_ei, n_sub, damping=self.damping)

        # Score each edge by sum of endpoint PR scores.
        pr_full = torch.zeros(data.x.size(0), device=self.device)
        pr_full[involved] = pr
        edge_scores = pr_full[sub_ei[0]] + pr_full[sub_ei[1]]

        keep = _resolve_keep_count(sub_ei.size(1), self.budget, self.k_frac)
        _, top_idx = edge_scores.topk(keep)

        kept_ei = sub_ei[:, top_idx]
        kept_ew = edge_scores[top_idx] if sub_ew is None else sub_ew[top_idx]
        return _relabel_subgraph(data, kept_ei, kept_ew, self.device)

    def explain_batch(self, data: Data, edges: torch.Tensor) -> List[Data]:
        results: List[Data] = []
        for i in range(edges.size(1)):
            results.append(
                self.explain_link(
                    data, int(edges[0, i].item()), int(edges[1, i].item())
                )
            )
        return results


class GreedyDeletionBaseline(BaseExplainer):
    """Iteratively removes the edge whose removal causes the least prediction
    change, stopping when removal would flip the prediction.

    Works on the k-hop subgraph to remain tractable.  This is the hard
    baseline that reviewers warned can achieve near-perfect fidelity.

    Args:
        model: Trained link-prediction model.
        k_hop: Number of hops for the enclosing subgraph. Default 2.
        max_steps: Maximum number of edge-removal iterations. Default 100.
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k_hop: int = 2,
        max_steps: int = 100,
        device: str = "cpu",
    ):
        super().__init__(model, device)
        self.k_hop = k_hop
        self.max_steps = max_steps

    def explain_link(self, data: Data, node_a: int, node_b: int) -> Data:
        data = self._to_device(data)
        target = torch.tensor([[node_a], [node_b]], device=self.device)

        # Identify candidate edges (k-hop subgraph in global edge_index).
        cand_mask = _khop_mask(data, node_a, node_b, self.k_hop, self.device)
        candidate_indices = cand_mask.nonzero(as_tuple=True)[0].tolist()
        if not candidate_indices:
            return _relabel_subgraph(
                data,
                torch.zeros(2, 0, dtype=torch.long, device=self.device),
                None,
                self.device,
            )

        # Original prediction on full graph.
        assert data.edge_index is not None and data.x is not None
        original_score = self._predict(data, target).item()
        original_pred = int(original_score > 0.5)

        # Running mask: edges still present in the graph.
        keep_mask = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=self.device)
        remaining = set(candidate_indices)

        for _ in range(self.max_steps):
            if not remaining:
                break

            best_change = float("inf")
            best_idx: Optional[int] = None

            for idx in remaining:
                keep_mask[idx] = False
                modified = Data(x=data.x, edge_index=data.edge_index[:, keep_mask])
                if hasattr(data, "edge_weight") and data.edge_weight is not None:
                    modified.edge_weight = data.edge_weight[keep_mask]

                new_score = self._predict(modified, target).item()
                keep_mask[idx] = True

                new_pred = int(new_score > 0.5)
                if new_pred != original_pred:
                    continue

                change = abs(original_score - new_score)
                if change < best_change:
                    best_change = change
                    best_idx = idx

            if best_idx is None:
                break

            keep_mask[best_idx] = False
            remaining.discard(best_idx)

        result_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool, device=self.device)
        for idx in remaining:
            result_mask[idx] = True

        kept_ei = data.edge_index[:, result_mask]
        kept_ew: Optional[torch.Tensor] = None
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            kept_ew = data.edge_weight[result_mask]

        return _relabel_subgraph(data, kept_ei, kept_ew, self.device)

    def explain_batch(self, data: Data, edges: torch.Tensor) -> List[Data]:
        results: List[Data] = []
        for i in range(edges.size(1)):
            results.append(
                self.explain_link(
                    data, int(edges[0, i].item()), int(edges[1, i].item())
                )
            )
        return results


# =====================================================================
# Coarsening baselines
# =====================================================================


class _CoarseningBaseExplainer(BaseExplainer):
    """Abstract base for coarsening-based explainers with custom edge scoring.

    Subclasses override :meth:`_compute_scores` to provide different edge
    importance measures.  The pipeline is::

        compute_scores  ->  node_partition  ->  build_coarse_graph
                                            ->  (optional) linkwise refinement

    The partition, coarse graph, and coarse features are cached per ``Data``
    object so that repeated ``explain_link`` calls on the same graph reuse
    the coarsening structure.

    Args:
        model: Trained link-prediction model.
        k: Number of eigenpairs (for spectral methods). Default 100.
        alpha: Coarsening ratio in ``(0, 1]``. Default 0.75.
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

        # Cached state.
        self._cached_data_id: Optional[int] = None
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_weight: Optional[torch.Tensor] = None
        self._x: Optional[torch.Tensor] = None
        self._num_nodes: int = 0
        self._partition: Optional[List[List[int]]] = None

        self._coarse_ei: Optional[torch.Tensor] = None
        self._coarse_ew: Optional[torch.Tensor] = None
        self._coarse_x: Optional[torch.Tensor] = None
        self._num_coarse: int = 0

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    def _compute_scores(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor],
        x: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute edge importance scores.  Lower scores are merged first.

        The default implementation uses spectral perturbation scores
        (identical to the main ``CoarsenExplainer``).

        Override in subclasses for alternative scoring strategies.
        """
        A_hat = compute_normalized_adjacency(edge_index, num_nodes)
        eigenvalues, left_vecs, right_vecs = compute_top_k_eigenpairs(A_hat, self.k)
        return compute_perturbation_scores(edge_index, eigenvalues, left_vecs, right_vecs)

    # ------------------------------------------------------------------
    # Fitting / caching
    # ------------------------------------------------------------------

    def _ensure_fitted(self, data: Data) -> None:
        """Build partition and coarse graph (cached per ``Data`` object)."""
        data_id = id(data)
        if self._cached_data_id == data_id:
            return

        d = self._to_device(data)
        assert d.edge_index is not None and d.x is not None
        self._edge_index = d.edge_index
        self._edge_weight = getattr(d, "edge_weight", None)
        self._x = d.x
        self._num_nodes = d.x.size(0)

        scores = self._compute_scores(
            self._edge_index, self._num_nodes, self._edge_weight, self._x
        )

        self._partition = node_partition(
            self._edge_index, scores, self._num_nodes, self.alpha
        )
        assert self._partition is not None

        self._coarse_ei, self._coarse_ew, self._num_coarse = build_coarse_graph(
            self._edge_index, self._edge_weight, self._num_nodes,
            self._partition, self._x,
        )
        self._coarse_x = logsumexp_features(self._x, self._partition)

        self._cached_data_id = data_id

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def explain_link(self, data: Data, node_a: int, node_b: int) -> Data:
        """Build a linkwise-refined coarse-graph explanation."""
        self._ensure_fitted(data)
        assert self._edge_index is not None
        assert self._x is not None
        assert self._partition is not None

        (
            lw_ei,
            lw_ew,
            lw_x,
            _num_lw,
            sn_a,
            sn_b,
            _original_nodes,
        ) = linkwise_coarse_graph(
            self._edge_index,
            self._edge_weight,
            self._num_nodes,
            self._x,
            self._partition,
            node_a,
            node_b,
        )

        return _build_coarse_data(lw_ei, lw_ew, lw_x, sn_a, sn_b, self.device)

    def explain_batch(self, data: Data, edges: torch.Tensor) -> List[Data]:
        self._ensure_fitted(data)
        results: List[Data] = []
        for i in range(edges.size(1)):
            results.append(
                self.explain_link(
                    data, int(edges[0, i].item()), int(edges[1, i].item())
                )
            )
        return results


class RandomCoarseningExplainer(_CoarseningBaseExplainer):
    """Coarsening baseline using random edge scores.

    Same ``GraphCoarsener`` pipeline but with uniform random scores,
    producing a random partition of the graph.

    Args:
        model: Trained link-prediction model.
        k: Unused (kept for API consistency).
        alpha: Coarsening ratio. Default 0.75.
        seed: Random seed for reproducibility. Default None.
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k: int = 100,
        alpha: float = 0.75,
        seed: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__(model, k=k, alpha=alpha, device=device)
        self.seed = seed

    def _compute_scores(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor],
        x: Optional[torch.Tensor],
    ) -> torch.Tensor:
        n_edges = edge_index.size(1)
        generator: Optional[torch.Generator] = None
        if self.seed is not None:
            generator = torch.Generator(device=edge_index.device)
            generator.manual_seed(self.seed)
        return torch.rand(n_edges, generator=generator, device=edge_index.device)


class HeavyEdgeCoarseningExplainer(_CoarseningBaseExplainer):
    """Coarsening baseline using heavy-edge matching.

    For weighted graphs, merges the heaviest edges first (by ``edge_weight``).
    For unweighted graphs, merges by degree product ``deg(u) * deg(v)``,
    mimicking the heavy-edge matching heuristic from the graph sparsification
    literature.

    ``node_partition`` merges lowest-scored edges first, so this explainer
    negates the weight (or degree product) to give heavy edges the lowest
    scores.

    Args:
        model: Trained link-prediction model.
        k: Unused (kept for API consistency).
        alpha: Coarsening ratio. Default 0.75.
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k: int = 100,
        alpha: float = 0.75,
        device: str = "cpu",
    ):
        super().__init__(model, k=k, alpha=alpha, device=device)

    def _compute_scores(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor],
        x: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if edge_weight is not None:
            # Negate so that heaviest edges get the lowest (most negative)
            # scores and are merged first.
            return -edge_weight

        # Unweighted graph: use degree product.
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(
            0,
            edge_index[0],
            torch.ones(edge_index.size(1), device=edge_index.device),
        )
        return -(deg[edge_index[0]] * deg[edge_index[1]])


class EffectiveResistanceCoarseningExplainer(_CoarseningBaseExplainer):
    """Coarsening baseline using approximate effective resistance.

    Effective resistance measures the electrical-equivalent distance between
    two nodes in a graph::

        R_ab = (L^+)_{aa} + (L^+)_{bb} - 2 * (L^+)_{ab}

    where ``L^+`` is the pseudoinverse of the graph Laplacian.  For
    efficiency, we approximate using the ``k`` smallest non-zero eigenvectors
    of the Laplacian::

        R_ab ≈ Σ_{i: λ_i>0} (1/λ_i)(φ_ia - φ_ib)²

    Edges with *lower* effective resistance are merged first (they are more
    "redundant" in terms of spectral connectivity).

    Args:
        model: Trained link-prediction model.
        k: Number of Laplacian eigenvectors for the approximation. Default 100.
        alpha: Coarsening ratio. Default 0.75.
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k: int = 100,
        alpha: float = 0.75,
        device: str = "cpu",
    ):
        super().__init__(model, k=k, alpha=alpha, device=device)

    def _compute_scores(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor],
        x: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return _effective_resistance_scores(edge_index, num_nodes, self.k)


def _effective_resistance_scores(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int,
) -> torch.Tensor:
    """Compute approximate effective resistance for every edge.

    Uses the bottom-``k`` non-zero eigenvectors of the Laplacian for
    the spectral approximation.  Falls back to dense decomposition for
    small graphs or when ARPACK fails.
    """
    n = num_nodes
    device = edge_index.device
    row_np = edge_index[0].cpu().numpy().astype(np.int64)
    col_np = edge_index[1].cpu().numpy().astype(np.int64)

    # Build symmetric adjacency.
    vals = np.ones(len(row_np), dtype=np.float64)
    adj = sp.coo_matrix((vals, (row_np, col_np)), shape=(n, n))
    adj = adj.maximum(adj.T).tocsr()

    # Laplacian L = D - A.
    deg = np.asarray(adj.sum(axis=1)).ravel()
    L = sp.diags(deg) - adj

    # Compute bottom-(k+1) eigenpairs (smallest magnitude).
    k_eff = min(k + 1, max(n - 2, 1))

    if n <= 500:
        # Dense path -- more robust for small graphs.
        L_dense = L.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        eigenvalues = eigenvalues[: k_eff + 1]
        eigenvectors = eigenvectors[:, : k_eff + 1]
    else:
        try:
            eigenvalues, eigenvectors = spla.eigsh(L, k=k_eff, which="SM")
        except spla.ArpackNoConvergence:
            # Fallback: dense.
            L_dense = L.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            eigenvalues = eigenvalues[: k_eff + 1]
            eigenvectors = eigenvectors[:, : k_eff + 1]

    # Sort ascending.
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Skip zero eigenvalues, keep at most k non-zero.
    eps = 1e-8
    nonzero = eigenvalues > eps
    eigenvalues = eigenvalues[nonzero]
    eigenvectors = eigenvectors[:, nonzero]
    if len(eigenvalues) > k:
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]

    # If no non-zero eigenvalues, fall back to uniform scores.
    if len(eigenvalues) == 0:
        return torch.ones(edge_index.size(1), device=device)

    # Convert to torch (keep on CPU for large gather, then move).
    eigvals_t = torch.from_numpy(eigenvalues.copy()).float()
    eigvecs_t = torch.from_numpy(eigenvectors.copy()).float()

    # Move to the edge_index device.
    eigvals_t = eigvals_t.to(device)
    eigvecs_t = eigvecs_t.to(device)

    row_t = edge_index[0]
    col_t = edge_index[1]

    phi_a = eigvecs_t[row_t]  # (E, k')
    phi_b = eigvecs_t[col_t]  # (E, k')

    diff = phi_a - phi_b
    inv_lambda = 1.0 / torch.clamp(eigvals_t, min=eps)  # (k',)

    R = torch.sum(inv_lambda.unsqueeze(0) * diff.pow(2), dim=1)  # (E,)
    return R


class NoRefinementExplainer(_CoarseningBaseExplainer):
    """Spectral coarsening *without* linkwise refinement (Algorithm 3 skipped).

    Uses the full ``GraphCoarsener`` pipeline but returns the global coarse
    graph directly, without splitting the supernodes containing ``node_a``
    and ``node_b`` into singletons.  This measures how much linkwise
    refinement contributes to explanation quality.

    Args:
        model: Trained link-prediction model.
        k: Number of eigenpairs for spectral scoring. Default 100.
        alpha: Coarsening ratio. Default 0.75.
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k: int = 100,
        alpha: float = 0.75,
        device: str = "cpu",
    ):
        super().__init__(model, k=k, alpha=alpha, device=device)

    def explain_link(self, data: Data, node_a: int, node_b: int) -> Data:
        self._ensure_fitted(data)
        data = self._to_device(data)
        assert self._partition is not None
        assert self._coarse_ei is not None
        assert self._coarse_ew is not None
        assert self._coarse_x is not None

        sn_a, sn_b = _find_supernode_indices(self._partition, node_a, node_b)

        return _build_coarse_data(
            self._coarse_ei,
            self._coarse_ew,
            self._coarse_x,
            sn_a,
            sn_b,
            self.device,
        )
