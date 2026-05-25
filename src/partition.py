"""Greedy node partition based on spectral perturbation scores.

Implements the Union-Find data structure and Algorithm 2 (Node Partition Rule)
for grouping graph nodes into supernodes.

Proposition (Protected Partition Correctness):
    The protected_nodes parameter constrains the greedy merge process:
    if v ∈ protected_nodes, then v is never merged with any other node.
    This is implemented by skipping union(a,b) when either endpoint is
    protected. The resulting partition P' satisfies:
    - v ∈ protected_nodes ⇒ {v} ∈ P' (singleton)
    - v ∉ protected_nodes ⇒ merged per original spectral criteria
    The O(E·α(N)) complexity is preserved since the protection check is O(1).
"""

from typing import List, Tuple

import torch


class UnionFind:
    """Union-Find (Disjoint Set Union) with path compression and union by rank.

    Each element is an integer node index. Supports efficient ``find`` and
    ``union`` operations, both nearly O(α(n)) amortized where α is the
    inverse Ackermann function.
    """

    def __init__(self, n: int) -> None:
        """Initialize with ``n`` singleton sets.

        Args:
            n: Number of elements (nodes).
        """
        self.parent: List[int] = list(range(n))
        self.rank: List[int] = [0] * n
        self.size: int = n

    def find(self, x: int) -> int:
        """Find the root representative of the set containing ``x``.

        Uses path compression: all visited nodes point directly to the root.

        Args:
            x: Element to look up.

        Returns:
            Root representative index.
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Merge the sets containing ``x`` and ``y``.

        Uses union by rank to keep the tree shallow.

        Args:
            x: First element.
            y: Second element.

        Returns:
            True if the sets were merged (were different), False if already
            in the same set.
        """
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return False

        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """Check whether ``x`` and ``y`` are in the same set.

        Args:
            x: First element.
            y: Second element.

        Returns:
            True if both elements share the same root.
        """
        return self.find(x) == self.find(y)


def prediction_guided_partition(
    edge_index: torch.Tensor,
    spectral_scores: torch.Tensor,
    gradient_scores: torch.Tensor,
    num_nodes: int,
    alpha: float = 0.75,
    protected_nodes: set | None = None,
    lambda_pred: float = 1.0,
    fidelity_threshold: float = 0.8,
) -> List[List[int]]:
    """Prediction-guided partition minimizing spectral + prediction cost.

    Optimization objective: min Σ [ρ̂(e) + λ · Φ(a,b)]
    where ρ̂(e) is normalized spectral score and
    Φ(a,b) = ĝ̂(a) · ĝ̂(b) is the product of endpoint gradient importances.

    Hard constraint: skip merge if Φ(a,b) > fidelity_threshold.

    Proposition (Prediction-Guided Merge):
        The merge cost C(e) = ρ̂(e) + λ·ĝ̂(a)·ĝ̂(b) captures both structural
        and predictive importance. Per-edge additive scoring (ρ̂+ĝ) misses the
        interaction: edge e=(a,b) with low individual gradient but both endpoints
        highly important (high Φ) should NOT be merged. The product Φ captures
        this "don't merge two prediction-critical nodes" constraint.

    Args:
        edge_index: Edge list in COO format, shape (2, E).
        spectral_scores: Spectral perturbation score per edge, shape (E,).
        gradient_scores: Gradient saliency per edge, shape (E,).
        num_nodes: Total number of nodes in the graph.
        alpha: Coarsening ratio. Default 0.75.
        protected_nodes: Nodes that remain singletons.
        lambda_pred: Weight for prediction cost term. Default 1.0.
        fidelity_threshold: Hard reject threshold for prediction cost. Default 0.8.

    Returns:
        List of partitions (supernode classes).
    """
    device = edge_index.device

    node_imp = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    abs_grad = gradient_scores.abs()
    node_imp.scatter_add_(0, edge_index[0], abs_grad)
    node_imp.scatter_add_(0, edge_index[1], abs_grad)
    imp_max = node_imp.max()
    if imp_max > 1e-10:
        node_imp = node_imp / imp_max

    imp_a = node_imp[edge_index[0]]
    imp_b = node_imp[edge_index[1]]
    pred_cost = imp_a * imp_b

    spec_norm = _normalize_01(spectral_scores)
    combined = spec_norm + lambda_pred * pred_cost

    max_merges = int(alpha * num_nodes)
    sorted_indices = torch.argsort(combined)

    uf = UnionFind(num_nodes)
    num_merges = 0

    for idx in range(sorted_indices.size(0)):
        if num_merges >= max_merges:
            break
        eidx = int(sorted_indices[idx].item())
        a = int(edge_index[0, eidx].item())
        b = int(edge_index[1, eidx].item())

        if protected_nodes and (a in protected_nodes or b in protected_nodes):
            continue

        if pred_cost[eidx].item() > fidelity_threshold:
            continue

        if uf.union(a, b):
            num_merges += 1

    root_to_nodes: dict[int, List[int]] = {}
    for node in range(num_nodes):
        root = uf.find(node)
        root_to_nodes.setdefault(root, []).append(node)

    return list(root_to_nodes.values())


def _normalize_01(t: torch.Tensor) -> torch.Tensor:
    if t.numel() == 0:
        return t
    mn, mx = t.min(), t.max()
    if mx - mn < 1e-12:
        return torch.zeros_like(t)
    return (t - mn) / (mx - mn)


def node_partition(
    edge_index: torch.Tensor,
    scores: torch.Tensor,
    num_nodes: int,
    alpha: float = 0.75,
    protected_nodes: set | None = None,
) -> List[List[int]]:
    """Greedy node partition using perturbation scores (Algorithm 2).

    Edges are sorted by perturbation score in ascending order (lowest scores
    merged first). Nodes are merged via Union-Find until ``alpha * num_nodes``
    merge operations have been performed. Remaining singleton nodes form
    their own classes.

    Args:
        edge_index: Edge list in COO format, shape (2, E).
        scores: Perturbation score per edge, shape (E,).
        num_nodes: Total number of nodes in the graph.
        alpha: Coarsening ratio in (0, 1]. Controls how many merges to
            perform. ``alpha = 1.0`` merges all possible edges;
            ``alpha = 0.5`` stops after half the budget. Default 0.75.
        protected_nodes: Optional set of node indices that should never be
            merged with other nodes. They remain as singletons in the
            partition.

    Returns:
        List of partitions (supernode classes). Each class is a list of
        original node indices belonging to that supernode.
    """
    max_merges = int(alpha * num_nodes)

    sorted_indices = torch.argsort(scores)
    sorted_edges = edge_index[:, sorted_indices]

    uf = UnionFind(num_nodes)
    num_merges = 0

    row = sorted_edges[0]
    col = sorted_edges[1]

    for idx in range(row.size(0)):
        if num_merges >= max_merges:
            break
        a = int(row[idx].item())
        b = int(col[idx].item())
        if protected_nodes and (a in protected_nodes or b in protected_nodes):
            continue
        if uf.union(a, b):
            num_merges += 1

    root_to_nodes: dict[int, List[int]] = {}
    for node in range(num_nodes):
        root = uf.find(node)
        if root not in root_to_nodes:
            root_to_nodes[root] = []
        root_to_nodes[root].append(node)

    return list(root_to_nodes.values())


def build_partition_matrix(
    partition: List[List[int]],
    num_nodes: int,
) -> torch.Tensor:
    """Build the binary partition indicator matrix P_hat.

    P_hat ∈ {0,1}^{N × N'} where ``P_hat[i, j] = 1`` iff node ``i``
    belongs to supernode (class) ``j``.

    Args:
        partition: List of classes, each a list of node indices.
        num_nodes: Total number of original nodes N.

    Returns:
        Sparse COO tensor of shape (N, N') where N' = len(partition).
    """
    num_classes = len(partition)
    rows: List[int] = []
    cols: List[int] = []

    for class_idx, members in enumerate(partition):
        for node in members:
            rows.append(node)
            cols.append(class_idx)

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    P_hat = torch.sparse_coo_tensor(
        indices, values, size=(num_nodes, num_classes)
    ).coalesce()

    return P_hat


def normalize_partition_matrix(P_hat: torch.Tensor) -> torch.Tensor:
    """Normalize the partition matrix to have orthonormal columns.

    Computes ``P = P_hat * M^{-1/2}`` where
    ``M = diag(|C_1|, ..., |C_{N'}|)`` is the diagonal matrix of class sizes.
    The resulting P satisfies ``P^T P = I``.

    Args:
        P_hat: Binary partition matrix, sparse or dense, shape (N, N').

    Returns:
        Normalized partition matrix of the same shape and format.
    """
    P_hat = P_hat.coalesce()
    num_classes = P_hat.size(1)

    # Compute class sizes: |C_j| = sum_i P_hat[i,j]
    col_idx = P_hat.indices()[1]
    class_sizes = torch.zeros(num_classes, dtype=torch.float32)
    class_sizes.scatter_add_(0, col_idx, P_hat.values())

    # M^{-1/2}
    inv_sqrt_sizes = torch.where(
        class_sizes > 0, class_sizes.pow(-0.5), torch.zeros_like(class_sizes)
    )

    # Scale each nonzero by the inverse sqrt of its class size
    scaled_values = P_hat.values() * inv_sqrt_sizes[col_idx]

    P_sparse = torch.sparse_coo_tensor(
        P_hat.indices(), scaled_values, size=P_hat.size()
    ).coalesce()

    return P_sparse.to_dense()
