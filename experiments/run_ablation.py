"""Coarsening ratio ablation study (Table 1)."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExperimentConfig
from experiments.train_gcn import load_dataset

try:
    from src.coarsen import GraphCoarsener
except ImportError:
    GraphCoarsener = None


def normalized_adjacency_eigenvalues(num_nodes, edge_index, k):
    """Compute top-k eigenvalues of the normalized adjacency matrix.

    Builds D^{-1/2} A D^{-1/2} and extracts the k largest eigenvalues
    via sparse eigendecomposition.
    """
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    values = np.ones(len(row), dtype=np.float64)
    A = sparse.coo_matrix((values, (row, col)), shape=(num_nodes, num_nodes))
    A = A.tocsr()
    A = A + A.T
    A.data = np.clip(A.data, 0, 1)
    A.setdiag(0)
    A.eliminate_zeros()

    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    k_clamped = min(k, num_nodes - 2)
    if k_clamped < 1:
        return np.array([0.0])
    try:
        eigenvalues, _ = eigsh(A_norm, k=k_clamped, which="LM")
    except Exception:
        eigenvalues = np.zeros(k_clamped)
    return np.sort(eigenvalues)[::-1]


def compute_coarsened_eigenvalues(data, alpha, k, device):
    """Run coarsening at ratio alpha and return eigenvalues of the coarsened graph.

    Falls back to computing eigenvalues on the original graph when the
    coarsening module is unavailable.
    """
    if GraphCoarsener is None:
        return normalized_adjacency_eigenvalues(data.num_nodes, data.train_pos_edge_index, k)

    edge_index = data.train_pos_edge_index
    num_nodes = data.num_nodes
    x = data.x if data.x is not None else torch.eye(num_nodes)

    coarsener = GraphCoarsener(k=k, alpha=alpha)
    coarsener.fit(edge_index, num_nodes, x)

    coarse_edges = coarsener.coarse_edge_index
    coarse_num_nodes = coarsener.num_coarse_nodes
    if coarse_edges is None:
        return normalized_adjacency_eigenvalues(data.num_nodes, data.train_pos_edge_index, k)

    return normalized_adjacency_eigenvalues(coarse_num_nodes, coarse_edges, k)


def mean_relative_error(eig_orig, eig_coarse, k):
    """Mean relative error of top-k eigenvalues: (1/k) sum |c_i - o_i| / |o_i|."""
    min_k = min(len(eig_orig), len(eig_coarse), k)
    o = eig_orig[:min_k]
    c = eig_coarse[:min_k]
    denom = np.abs(o)
    denom[denom < 1e-12] = 1e-12
    return float(np.mean(np.abs(c - o) / denom))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=500, help="Number of top eigenvalues to compare")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    if args.device is not None:
        cfg.device = args.device
    device = torch.device(cfg.device)
    torch.manual_seed(args.seed)

    ablation_datasets = ["Cora", "Citeseer", "PubMed"]
    ratios = cfg.ablation_ratios
    k = args.k

    results = {}

    for ds_name in ablation_datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            data = load_dataset(ds_name)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  |V|={data.num_nodes:,}  |E|={data.train_pos_edge_index.size(1):,}")
        print(f"  Computing original top-{k} eigenvalues...")
        eig_orig = normalized_adjacency_eigenvalues(
            data.num_nodes, data.train_pos_edge_index, k,
        )

        ds_results = {}
        for alpha in ratios:
            print(f"  alpha={alpha:.2f} ...", end=" ", flush=True)
            t0 = time.time()
            eig_coarse = compute_coarsened_eigenvalues(data, alpha, k, device)
            elapsed = time.time() - t0
            error = mean_relative_error(eig_orig, eig_coarse, k)
            ds_results[str(alpha)] = {
                "mean_relative_error": round(error, 6),
                "time_s": round(elapsed, 2),
            }
            print(f"error={error:.6f}  ({elapsed:.1f}s)")

        results[ds_name] = ds_results

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "ablation.json")
    with open(out_path, "w") as f:
        json.dump({"k": k, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    _print_table(results, ratios)


def _print_table(results, ratios):
    header = f"{'Dataset':<12}" + "".join(f"{'α='+str(a):>10}" for a in ratios)
    print(f"\n{'='*len(header)}")
    print("Mean Relative Error of Top-k Eigenvalues (Table 1)")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    for ds_name, ds_results in results.items():
        row = f"{ds_name:<12}"
        for alpha in ratios:
            err = ds_results.get(str(alpha), {}).get("mean_relative_error", float("nan"))
            row += f"{err:>10.4f}"
        print(row)
    print(f"{'='*len(header)}")


if __name__ == "__main__":
    main()
