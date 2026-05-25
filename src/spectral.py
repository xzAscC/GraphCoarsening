"""Spectral graph analysis: eigenvalue computation and perturbation scores.

Implements the spectral decomposition and perturbation analysis from
Theorem 3 / Proposition 2 for computing edge coarsening scores.
"""

from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


def compute_normalized_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Compute symmetric normalized adjacency with self-loops.

    Builds Â = D_+^{-1/2} A_+ D_+^{-1/2} where A_+ is the adjacency
    matrix augmented with self-loops and D_+ is its degree matrix.

    Args:
        edge_index: Edge list in COO format, shape (2, E).
        num_nodes: Number of nodes in the graph.

    Returns:
        Sparse tensor of shape (num_nodes, num_nodes) representing Â.
    """
    # Add self-loops
    loop_index = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).expand(2, -1)
    edge_index_plus = torch.cat([edge_index, loop_index], dim=1)

    # Build sparse adjacency A_+ (with self-loops)
    num_edges_plus = edge_index_plus.size(1)
    values = torch.ones(num_edges_plus, dtype=torch.float32, device=edge_index.device)
    A_plus = torch.sparse_coo_tensor(
        edge_index_plus, values, size=(num_nodes, num_nodes)
    ).coalesce()

    # Compute degree D_+
    row = A_plus.indices()[0]
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    deg.scatter_add_(0, row, A_plus.values())

    # D_+^{-1/2}
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))

    # Normalized values: D_+^{-1/2}(i) * D_+^{-1/2}(j) for each edge (i,j)
    row_idx = A_plus.indices()[0]
    col_idx = A_plus.indices()[1]
    norm_values = deg_inv_sqrt[row_idx] * deg_inv_sqrt[col_idx]

    A_hat = torch.sparse_coo_tensor(
        torch.stack([row_idx, col_idx]),
        norm_values,
        size=(num_nodes, num_nodes),
    ).coalesce()

    return A_hat


def compute_top_k_eigenpairs(
    A_hat: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute top-k eigenvalues and eigenvectors of the normalized adjacency.

    For symmetric normalized adjacency, left and right eigenvectors are
    identical. Eigenvalues are returned in descending order (largest first).

    For small graphs (< 10k nodes), uses dense decomposition via
    ``torch.linalg.eigh``. For larger graphs, uses ARPACK via
    ``scipy.sparse.linalg.eigsh``.

    Args:
        A_hat: Normalized adjacency matrix, shape (N, N), sparse or dense.
        k: Number of top eigenpairs to compute.

    Returns:
        Tuple of:
            - eigenvalues: shape (k,), sorted in descending order.
            - left_eigenvectors: shape (N, k).
            - right_eigenvectors: shape (N, k).
              For symmetric matrices, left == right.
    """
    N = A_hat.size(0)

    # Clamp k to matrix dimension
    k = min(k, N)

    if N <= 10_000:
        # Dense path: convert to dense, use torch.linalg.eigh
        if A_hat.is_sparse:
            A_dense = A_hat.to_dense()
        else:
            A_dense = A_hat

        eigenvalues, eigenvectors = torch.linalg.eigh(A_dense)
        # eigh returns ascending order; take the last k and reverse
        eigenvalues = eigenvalues[-k:].flip(0)
        eigenvectors = eigenvectors[:, -k:].flip(1)
    else:
        # Sparse path: use scipy ARPACK
        if A_hat.is_sparse:
            A_np = _sparse_tensor_to_scipy(A_hat)
        else:
            A_np = A_hat.detach().cpu().numpy()
            if not sp.issparse(A_np):
                A_np = sp.csr_matrix(A_np)

        # eigsh with which='LM' gives largest magnitude eigenvalues
        eigenvalues_np, eigenvectors_np = spla.eigsh(A_np, k=k, which="LM")

        # eigsh returns eigenvalues in ascending order; reverse to descending
        sort_idx = np.argsort(-eigenvalues_np)
        eigenvalues_np = eigenvalues_np[sort_idx]
        eigenvectors_np = eigenvectors_np[:, sort_idx]

        eigenvalues = torch.from_numpy(eigenvalues_np.copy()).float()
        eigenvectors = torch.from_numpy(eigenvectors_np.copy()).float()

    # For symmetric matrices, left == right eigenvectors
    left_vecs = eigenvectors
    right_vecs = eigenvectors.clone()

    return eigenvalues, left_vecs, right_vecs


def compute_perturbation_scores(
    edge_index: torch.Tensor,
    eigenvalues: torch.Tensor,
    left_vecs: torch.Tensor,
    right_vecs: torch.Tensor,
) -> torch.Tensor:
    """Compute perturbation scores for each edge using Theorem 3.

    For each edge (a, b), the perturbation score approximates the change
    in the spectral objective when the edge's endpoints are merged:

        Δ_{(a,b)} ≈ Σ_{i=1}^{k} |ν_i / η_i|

    For undirected graphs (symmetric Â), left_vecs == right_vecs (u = v):

        ν_i = -u_ia² + 3*u_ia*u_ib + (3-λ_i)*u_ib*u_ia + (λ_i-1)*u_ib²
        η_i = 1 - (u_ia² + u_ib²)

    Args:
        edge_index: Edge list in COO format, shape (2, E).
        eigenvalues: Top-k eigenvalues, shape (k,).
        left_vecs: Left eigenvectors, shape (N, k).
        right_vecs: Right eigenvectors, shape (N, k).

    Returns:
        Perturbation scores of shape (E,), one score per edge.
    """
    right_vecs = right_vecs.to(edge_index.device)
    left_vecs = left_vecs.to(edge_index.device)
    eigenvalues = eigenvalues.to(edge_index.device)

    row = edge_index[0]
    col = edge_index[1]
    num_edges = edge_index.size(1)
    k = eigenvalues.size(0)

    u_a = right_vecs[row]  # (E, k)
    u_b = right_vecs[col]  # (E, k)
    v_a = left_vecs[row]  # (E, k)
    v_b = left_vecs[col]  # (E, k)

    # Broadcast eigenvalues: (1, k) for elementwise ops with (E, k)
    lam = eigenvalues.unsqueeze(0)  # (1, k)

    # Check if undirected (left == right within tolerance)
    is_symmetric = torch.allclose(left_vecs, right_vecs, atol=1e-6)

    if is_symmetric:
        # Simplified formula for symmetric matrices where v = u
        # ν_i = -u_ia² + 3*u_ia*u_ib + (3-λ_i)*u_ib*u_ia + (λ_i-1)*u_ib²
        nu = (
            -u_a * u_a
            + 3.0 * u_a * u_b
            + (3.0 - lam) * u_b * u_a
            + (lam - 1.0) * u_b * u_b
        )
        # η_i = 1 - (u_ia² + u_ib²)
        # Since v^T u = 1 for normalized eigenvectors
        eta = 1.0 - (u_a * u_a + u_b * u_b)
    else:
        # General (directed) case
        # ν_i = -v_ia*u_ia + 3*v_ia*u_ib + (3-λ_i)*v_ib*u_ia + (λ_i-1)*v_ib*u_ib
        nu = (
            -v_a * u_a
            + 3.0 * v_a * u_b
            + (3.0 - lam) * v_b * u_a
            + (lam - 1.0) * v_b * u_b
        )
        # η_i = v_i^T u_i - (u_ia*v_ia + u_ib*v_ib)
        # For normalized eigenvectors, v_i^T u_i = 1
        vt_u = torch.sum(left_vecs * right_vecs, dim=0)  # (k,)
        eta = vt_u.unsqueeze(0) - (u_a * v_a + u_b * v_b)  # (E, k)

    # Numerical stability: clamp eta away from zero
    eps = 1e-8
    eta = torch.where(eta.abs() < eps, eta + eps, eta)

    # Score = sum over i of |ν_i / η_i|
    scores = torch.sum(nu.abs() / eta.abs(), dim=1)  # (E,)

    return scores


def _sparse_tensor_to_scipy(sparse_tensor: torch.Tensor) -> sp.csr_matrix:
    """Convert a PyTorch sparse COO tensor to a scipy sparse CSR matrix.

    Args:
        sparse_tensor: PyTorch sparse COO tensor.

    Returns:
        Scipy sparse CSR matrix with same values.
    """
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices().cpu()
    values = sparse_tensor.values().cpu()
    N, M = sparse_tensor.size()
    return sp.csr_matrix(
        (values.numpy(), (indices[0].numpy(), indices[1].numpy())),
        shape=(N, M),
    )
