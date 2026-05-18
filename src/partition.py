"""Greedy node partition based on spectral perturbation scores.

Implements the Union-Find data structure and Algorithm 2 (Node Partition Rule)
for grouping graph nodes into supernodes.
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


def node_partition(
    edge_index: torch.Tensor,
    scores: torch.Tensor,
    num_nodes: int,
    alpha: float = 0.75,
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

    Returns:
        List of partitions (supernode classes). Each class is a list of
        original node indices belonging to that supernode.
    """
    max_merges = int(alpha * num_nodes)

    # Sort edges by ascending perturbation score
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
        if uf.union(a, b):
            num_merges += 1

    # Collect classes from Union-Find
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
