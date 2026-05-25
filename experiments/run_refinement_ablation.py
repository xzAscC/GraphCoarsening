"""Link-wise refinement ablation experiment (Priority 7).

Compares different refinement strategies to justify the global coarsening +
endpoint cluster splitting design. Strategies tested:
- No refinement
- Split only endpoint nodes
- Split endpoint clusters
- Split 1-hop neighborhoods around endpoints
- Split 2-hop neighborhoods around endpoints
- Full local k-hop subgraph
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExperimentConfig
from experiments.train_gcn import load_dataset, MLPLinkPredictor
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel
from src.coarsen import GraphCoarsener, build_coarse_graph
from src.partition import node_partition
from src.spectral import compute_normalized_adjacency, compute_top_k_eigenpairs, compute_perturbation_scores

try:
    from src.evaluation.fidelity import fidelity_plus, fidelity_minus
except ImportError:
    fidelity_plus = None
    fidelity_minus = None


REFINEMENT_STRATEGIES = [
    "none",
    "split_endpoints",
    "split_clusters",
    "split_1hop",
    "split_2hop",
    "full_khop",
]


def _build_refined_partition(
    partition: List[List[int]],
    num_nodes: int,
    edge_index: torch.Tensor,
    node_a: int,
    node_b: int,
    strategy: str,
) -> List[List[int]]:
    """Apply a refinement strategy to the partition around target link endpoints."""
    if strategy == "none":
        return partition

    cluster_a_idx = None
    cluster_b_idx = None
    for i, members in enumerate(partition):
        if node_a in members:
            cluster_a_idx = i
        if node_b in members:
            cluster_b_idx = i

    if cluster_a_idx is None or cluster_b_idx is None:
        return partition

    if strategy == "split_endpoints":
        split_indices = {cluster_a_idx}
        if cluster_b_idx != cluster_a_idx:
            split_indices.add(cluster_b_idx)
        refined = []
        for i, members in enumerate(partition):
            if i in split_indices:
                for v in members:
                    if v in (node_a, node_b):
                        refined.append([v])
                    else:
                        refined.append([v])  # Each becomes singleton
            else:
                refined.append(members)
        return refined

    if strategy == "split_clusters":
        split_indices = {cluster_a_idx}
        if cluster_b_idx != cluster_a_idx:
            split_indices.add(cluster_b_idx)
        refined = []
        for i, members in enumerate(partition):
            if i in split_indices:
                for v in members:
                    refined.append([v])
            else:
                refined.append(members)
        return refined

    if strategy.startswith("split_") and "hop" in strategy:
        num_hops = int(strategy.replace("split_", "").replace("hop", ""))
        subset, _, _, _ = k_hop_subgraph(
            node_idx=torch.tensor([node_a, node_b]),
            num_hops=num_hops,
            edge_index=edge_index,
            relabel_nodes=False,
            num_nodes=num_nodes,
        )
        neighborhood = set(subset.tolist())

        refined = []
        for i, members in enumerate(partition):
            has_overlap = any(v in neighborhood for v in members)
            if has_overlap:
                for v in members:
                    if v in neighborhood:
                        refined.append([v])
                    else:
                        refined.append([v])
            else:
                refined.append(members)
        return refined

    if strategy == "full_khop":
        subset, _, _, _ = k_hop_subgraph(
            node_idx=torch.tensor([node_a, node_b]),
            num_hops=2,
            edge_index=edge_index,
            relabel_nodes=False,
            num_nodes=num_nodes,
        )
        neighborhood = set(subset.tolist())
        refined = []
        for i, members in enumerate(partition):
            has_overlap = any(v in neighborhood for v in members)
            if has_overlap:
                for v in members:
                    refined.append([v])
            else:
                refined.append(members)
        return refined

    return partition


def evaluate_refinement(model, data, test_edges, device, k, alpha, strategy, num_edges=30):
    """Run coarsening with a specific refinement strategy and evaluate."""
    x = data.x if data.x is not None else torch.eye(data.num_nodes)

    coarsener = GraphCoarsener(k=k, alpha=alpha)
    coarsener.fit(data.train_pos_edge_index, data.num_nodes, x)

    fid_p_list = []
    fid_m_list = []
    sizes = []
    times = []

    n = min(num_edges, test_edges.size(1))
    for i in range(n):
        a = int(test_edges[0, i].item())
        b = int(test_edges[1, i].item())

        t0 = time.time()
        refined_partition = _build_refined_partition(
            coarsener.partition, data.num_nodes, data.train_pos_edge_index, a, b, strategy,
        )

        coarse_ei, coarse_ew, num_coarse = build_coarse_graph(
            data.train_pos_edge_index, None, data.num_nodes, refined_partition, x,
        )

        sa_idx = None
        sb_idx = None
        for ci, members in enumerate(refined_partition):
            if a in members:
                sa_idx = ci
            if b in members:
                sb_idx = ci

        if sa_idx is None or sb_idx is None or sa_idx == sb_idx:
            continue

        d = x.size(1)
        coarse_feat = torch.zeros(num_coarse, d, dtype=x.dtype, device=x.device)
        for ci, members in enumerate(refined_partition):
            if len(members) == 1:
                coarse_feat[ci] = x[members[0]]
            else:
                coarse_feat[ci] = torch.logsumexp(x[members], dim=0)

        explanation = Data(
            x=coarse_feat, edge_index=coarse_ei, edge_weight=coarse_ew,
            is_coarse_graph=True,
            target_a=sa_idx, target_b=sb_idx,
        )
        times.append(time.time() - t0)
        sizes.append(coarse_ei.size(1))

        if fidelity_plus is not None:
            fp = fidelity_plus(model, data, explanation, a, b, device)
            fm = fidelity_minus(model, data, explanation, a, b, device)
            fid_p_list.append(fp)
            fid_m_list.append(fm)

    if not fid_p_list:
        return None

    return {
        "mean_fidelity_plus": float(np.mean(fid_p_list)),
        "std_fidelity_plus": float(np.std(fid_p_list)),
        "mean_fidelity_minus": float(np.mean(fid_m_list)),
        "std_fidelity_minus": float(np.std(fid_m_list)),
        "mean_size": float(np.mean(sizes)),
        "std_size": float(np.std(sizes)),
        "mean_time": float(np.mean(times)),
        "num_samples": len(fid_p_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--num_edges", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategies", type=str, default=",".join(REFINEMENT_STRATEGIES))
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    if args.device:
        cfg.device = args.device
    device = torch.device(cfg.device)
    torch.manual_seed(args.seed)

    strategies = [s.strip() for s in args.strategies.split(",")]

    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    # train_test_split_edges sets edge_index=None; restore for fidelity functions
    if data.edge_index is None:
        data.edge_index = data.train_pos_edge_index

    checkpoint_path = os.path.join("checkpoints", f"{args.dataset}_gcn.pt")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No checkpoint at {checkpoint_path}. Run train_gcn.py first.")
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mc = ckpt["config"]
    gcn = GCN(mc["in_channels"], mc["hidden_channels"], mc["out_channels"], mc["num_layers"]).to(device)
    gcn.load_state_dict(ckpt["model_state_dict"])
    predictor = MLPLinkPredictor(mc["out_channels"], mc["hidden_channels"]).to(device)
    if "predictor_state_dict" in ckpt:
        predictor.load_state_dict(ckpt["predictor_state_dict"])
    model = LinkPredictionModel(gcn, predictor).to(device)
    model.eval()

    if hasattr(data, "test_pos_edge_index") and data.test_pos_edge_index is not None:
        pos = data.test_pos_edge_index
    else:
        pos = data.train_pos_edge_index
    rng = np.random.RandomState(args.seed)
    n = min(args.num_edges, pos.size(1))
    indices = rng.choice(pos.size(1), size=n, replace=False)
    test_edges = pos[:, indices]

    results = {}
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        result = evaluate_refinement(
            model, data, test_edges, device,
            cfg.spectral.k, cfg.spectral.alpha, strategy, args.num_edges,
        )
        if result:
            results[strategy] = result
            print(f"  Fid+: {result['mean_fidelity_plus']:.4f} ± {result['std_fidelity_plus']:.4f}")
            print(f"  Fid-: {result['mean_fidelity_minus']:.4f} ± {result['std_fidelity_minus']:.4f}")
            print(f"  Size: {result['mean_size']:.1f} ± {result['std_size']:.1f}")
            print(f"  Time: {result['mean_time']:.4f}s")
        else:
            print("  No results")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"refinement_ablation_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    try:
        _plot_refinement(args.dataset, results)
    except Exception as e:
        print(f"Plotting skipped: {e}")


def _plot_refinement(dataset, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        return

    strategies = list(results.keys())
    fid_p = [results[s]["mean_fidelity_plus"] for s in strategies]
    fid_p_err = [results[s].get("std_fidelity_plus", 0) for s in strategies]
    fid_m = [results[s]["mean_fidelity_minus"] for s in strategies]
    fid_m_err = [results[s].get("std_fidelity_minus", 0) for s in strategies]
    sizes = [results[s]["mean_size"] for s in strategies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(strategies))
    width = 0.35
    ax1.bar(x - width/2, fid_p, width, yerr=fid_p_err, label="Necessity (Fid+)", capsize=3)
    ax1.bar(x + width/2, fid_m, width, yerr=fid_m_err, label="Sufficiency (Fid-)", capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=30, ha="right")
    ax1.set_ylabel("Fidelity")
    ax1.set_title(f"Fidelity by Refinement Strategy ({dataset})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(strategies, sizes, color="steelblue")
    ax2.set_xticklabels(strategies, rotation=30, ha="right")
    ax2.set_ylabel("Mean Explanation Size (edges)")
    ax2.set_title(f"Explanation Size by Strategy ({dataset})")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig_path = os.path.join("figures", f"refinement_ablation_{dataset}.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
