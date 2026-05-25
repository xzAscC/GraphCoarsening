"""Oversmoothing depth sweep experiment.

Verifies the paper's core claim that spectral coarsening mitigates oversmoothing
by measuring how node embedding quality degrades with GCN depth (2..32 layers)
under different coarsening strategies.

Metrics (all computed per depth per method):
  - Mean pairwise cosine similarity of node embeddings (higher = more smoothing)
  - Embedding variance: mean of per-dimension variance (lower = more smoothing)
  - Dirichlet energy: trace(Z^T L Z) / N  (lower = more smoothing)
  - Link prediction AUC on the test set (lower = degraded by smoothing)

Coarsening methods compared:
  - none        : full graph, no coarsening
  - random      : random edge-merge partition
  - heavy_edge  : merge high-degree-product edges first
  - spectral    : Laplacian-guided spectral coarsening (ours)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExperimentConfig
from experiments.train_gcn import (
    load_dataset,
    train_epoch,
    evaluate,
    MLPLinkPredictor,
)
from src.models.gcn import GCN
from src.coarsen import build_coarse_graph, logsumexp_features, GraphCoarsener
from src.partition import node_partition

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEPTHS = [2, 4, 8, 16, 32]
DATASETS = ["Cora", "Citeseer", "PubMed"]
COARSENING_METHODS = ["random", "heavy_edge", "spectral"]


# ---------------------------------------------------------------------------
# Oversmoothing metrics
# ---------------------------------------------------------------------------

def compute_dirichlet_energy(z, edge_index, edge_weight=None):
    """Dirichlet energy = trace(Z^T L Z) / N  where L = D - A.

    Uses the identity  trace(Z^T L Z) = sum_i deg_i ||z_i||^2
                                        - sum_{(i,j)} A_ij z_i^T z_j
    which works for directed edge_index with both directions stored.
    """
    num_nodes = z.size(0)
    num_edges = edge_index.size(1)
    if num_edges == 0:
        return 0.0

    row, col = edge_index[0], edge_index[1]
    if edge_weight is None:
        edge_weight = torch.ones(num_edges, dtype=z.dtype, device=z.device)

    deg = torch.zeros(num_nodes, dtype=z.dtype, device=z.device)
    deg.scatter_add_(0, row, edge_weight)

    term1 = (deg.unsqueeze(1) * z * z).sum()
    term2 = (edge_weight * (z[row] * z[col]).sum(dim=1)).sum()

    energy = (term1 - term2).item()
    return energy / num_nodes


def compute_mean_cosine_similarity(z, max_pairs=200000):
    """Mean pairwise cosine similarity (sampled for large graphs)."""
    z_norm = F.normalize(z, dim=1)
    n = z.size(0)
    total_pairs = n * (n - 1) // 2

    if total_pairs <= max_pairs:
        sim = z_norm @ z_norm.T
        mask = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=z.device), diagonal=1
        )
        return sim[mask].mean().item()

    # Sampled estimate for large graphs
    idx_a = torch.randint(0, n, (max_pairs,), device=z.device)
    idx_b = torch.randint(0, n, (max_pairs,), device=z.device)
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % n
    return (z_norm[idx_a] * z_norm[idx_b]).sum(dim=1).mean().item()


def compute_embedding_variance(z):
    """Mean variance across embedding dimensions."""
    return z.var(dim=0).mean().item()


def compute_metrics(z, edge_index, edge_weight=None):
    """Return dict of the three oversmoothing metrics."""
    return {
        "cosine_sim": round(compute_mean_cosine_similarity(z), 6),
        "variance": round(compute_embedding_variance(z), 6),
        "dirichlet_energy": round(
            compute_dirichlet_energy(z, edge_index, edge_weight), 6
        ),
    }


# ---------------------------------------------------------------------------
# Coarsening helpers
# ---------------------------------------------------------------------------

def build_coarsened_graph(edge_index, num_nodes, x, method, alpha, k):
    """Build a coarsened graph via the requested method.

    Returns
    -------
    partition : list[list[int]]
    coarse_edge_index : Tensor (2, E')
    coarse_edge_weight : Tensor (E',) or None
    num_coarse : int
    """
    if method == "spectral":
        coarsener = GraphCoarsener(k=k, alpha=alpha)
        coarsener.fit(edge_index.cpu(), num_nodes, x.cpu())
        return (
            coarsener.partition,
            coarsener.coarse_edge_index,
            coarsener.coarse_edge_weight,
            coarsener.num_coarse_nodes,
        )

    num_edges = edge_index.size(1)

    if method == "random":
        # Random scores  ->  random merge order
        scores = torch.rand(num_edges, dtype=torch.float)
    elif method == "heavy_edge":
        # Negative degree product  ->  heavy edges get LOW scores, merged first
        row, col = edge_index
        deg = torch.zeros(num_nodes, dtype=torch.float)
        deg.scatter_add_(0, row, torch.ones(num_edges))
        scores = -(deg[row] * deg[col])
    else:
        raise ValueError("Unknown coarsening method: " + method)

    partition = node_partition(edge_index.cpu(), scores, num_nodes, alpha)
    coarse_ei, coarse_ew, num_coarse = build_coarse_graph(
        edge_index.cpu(), None, num_nodes, partition, x.cpu()
    )
    return partition, coarse_ei, coarse_ew, num_coarse


def compute_auc_with_projection(
    predictor, z_coarse, partition, data, num_nodes, device
):
    """Project coarsened embeddings back to original-node space and evaluate AUC."""
    node_to_super = torch.zeros(num_nodes, dtype=torch.long)
    for s_idx, members in enumerate(partition):
        for node_id in members:
            node_to_super[node_id] = s_idx

    z_proj = z_coarse[node_to_super].to(device)

    test_pos = data.test_pos_edge_index
    test_neg = data.test_neg_edge_index
    if test_pos is None or test_neg is None or test_pos.numel() == 0:
        return 0.0

    with torch.no_grad():
        pos_score = (
            torch.sigmoid(predictor(z_proj, test_pos.to(device))).cpu().numpy()
        )
        neg_score = (
            torch.sigmoid(predictor(z_proj, test_neg.to(device))).cpu().numpy()
        )

    labels = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
    scores = np.concatenate([pos_score, neg_score])
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_gcn_linkpred(model, predictor, data, device, epochs):
    """Train model+predictor for *epochs* iterations. Returns final loss."""
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=0.01,
        weight_decay=5e-4,
    )
    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        final_loss = train_epoch(model, predictor, data, optimizer, device)
        if epoch % 25 == 0 or epoch == 1:
            print(
                "      Epoch {:3d}/{}: loss={:.4f}".format(epoch, epochs, final_loss),
                flush=True,
            )
    return final_loss


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, datasets):
    """Generate the oversmoothing depth-sweep figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_keys = ["cosine_sim", "variance", "dirichlet_energy", "auc"]
    metric_titles = [
        "Mean Cosine Similarity",
        "Embedding Variance",
        "Dirichlet Energy",
        "Link Prediction AUC",
    ]

    method_cfg = {
        "none":       ("#888888", "--", "o", "Full graph (no coarsening)"),
        "random":     ("#4A90D9", ":",  "s", "Random coarsening"),
        "heavy_edge": ("#E8820C", "-.", "^", "Heavy-edge coarsening"),
        "spectral":   ("#D62728", "-",  "D", "Spectral (ours)"),
    }

    n_ds = len(datasets)
    n_met = len(metric_keys)
    fig, axes = plt.subplots(n_ds, n_met, figsize=(5 * n_met, 4 * n_ds), squeeze=False)

    for i, ds_name in enumerate(datasets):
        ds_results = results.get(ds_name, {})
        for j, (mkey, mtitle) in enumerate(zip(metric_keys, metric_titles)):
            ax = axes[i][j]
            for method, (color, ls, mk, label) in method_cfg.items():
                xs, ys = [], []
                for depth in DEPTHS:
                    entry = ds_results.get(str(depth), {}).get(method)
                    if entry and isinstance(entry, dict) and mkey in entry:
                        val = entry[mkey]
                        if val is not None and not (val != val):  # not NaN
                            xs.append(depth)
                            ys.append(val)
                if xs:
                    ax.plot(
                        xs, ys,
                        color=color, linestyle=ls, marker=mk,
                        label=label, markersize=5,
                    )

            ax.set_xlabel("GCN Depth (layers)")
            if i == 0:
                ax.set_title(mtitle)
            ax.set_ylabel(ds_name if j == 0 else "")
            ax.set_xscale("log", base=2)
            ax.set_xticks(DEPTHS)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.grid(True, alpha=0.3)

    # Single shared legend below all subplots
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(
        "Oversmoothing Depth Sweep: Spectral Coarsening vs Baselines",
        fontsize=14, y=1.01,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.98))

    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", "oversmoothing_depth_sweep.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Plot saved to " + out_path)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Oversmoothing depth sweep: spectral coarsening mitigates oversmoothing"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS, help="Datasets to evaluate"
    )
    parser.add_argument("--alpha", type=float, default=0.75, help="Coarsening ratio")
    parser.add_argument(
        "--k", type=int, default=100, help="Spectral: top-k eigenpairs"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dim")
    args = parser.parse_args()

    # Graceful CPU fallback
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device: {}".format(device))

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = {}

    for ds_name in args.datasets:
        print("\n" + "=" * 60)
        print("Dataset: {}".format(ds_name))
        print("=" * 60)

        try:
            data = load_dataset(ds_name)
        except Exception as exc:
            print("  SKIP: could not load dataset: {}".format(exc))
            continue

        in_channels = data.num_features
        if data.num_nodes is not None:
            num_nodes = int(data.num_nodes)
        elif data.x is not None:
            num_nodes = int(data.x.size(0))
        else:
            num_nodes = int(data.train_pos_edge_index.max().item()) + 1
        train_edges = data.train_pos_edge_index.size(1)
        print(
            "  |V|={:,}  |E_train|={:,}  d={}".format(num_nodes, train_edges, in_channels)
        )

        ds_results = {}

        for depth in DEPTHS:
            depth_key = str(depth)
            print("\n  --- Depth {} ---".format(depth))

            # ---- Train GCN at this depth --------------------------------
            try:
                torch.manual_seed(args.seed)
                model = GCN(
                    in_channels=in_channels,
                    hidden_channels=args.hidden,
                    out_channels=args.hidden,
                    num_layers=depth,
                    dropout=0.5,
                ).to(device)
                predictor = MLPLinkPredictor(args.hidden, args.hidden).to(device)

                print("    Training GCN (L={})...".format(depth), flush=True)
                train_gcn_linkpred(model, predictor, data, device, args.epochs)

                # Save checkpoint for reuse
                ckpt_dir = "checkpoints"
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, "{}_gcn_L{}.pt".format(ds_name, depth))
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "predictor_state_dict": predictor.state_dict(),
                        "config": {
                            "in_channels": in_channels,
                            "hidden_channels": args.hidden,
                            "out_channels": args.hidden,
                            "num_layers": depth,
                            "dataset": ds_name,
                        },
                    },
                    ckpt_path,
                )
                print("    Checkpoint: {}".format(ckpt_path))

            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or "oom" in msg:
                    print("    OOM at depth {}, skipping.".format(depth))
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    ds_results[depth_key] = {"error": "OOM"}
                    continue
                raise

            # ---- Full-graph embeddings ----------------------------------
            model.eval()
            with torch.no_grad():
                x = (
                    data.x.to(device)
                    if data.x is not None
                    else torch.eye(num_nodes, device=device)
                )
                ei = data.train_pos_edge_index.to(device)
                z_full = model(x, ei)

            if torch.isnan(z_full).any():
                print("    NaN in embeddings at depth {}, skipping.".format(depth))
                ds_results[depth_key] = {"error": "NaN"}
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            test_auc = float(evaluate(model, predictor, data, device, "test"))
            print("    Test AUC: {:.4f}".format(test_auc))

            depth_results = {}

            # -- (1) No coarsening (full graph) ---------------------------
            z_cpu = z_full.cpu()
            ei_cpu = data.train_pos_edge_index
            metrics = compute_metrics(z_cpu, ei_cpu)
            metrics["auc"] = round(float(test_auc), 6)
            depth_results["none"] = metrics
            print(
                "    [none        ] cos={:.4f}  var={:.4f}  E={:.4f}  auc={:.4f}".format(
                    metrics["cosine_sim"],
                    metrics["variance"],
                    metrics["dirichlet_energy"],
                    metrics["auc"],
                )
            )

            # -- (2)-(4) Coarsened methods --------------------------------
            for method in COARSENING_METHODS:
                tag = method.replace("_", "-")
                try:
                    t0 = time.time()
                    partition, c_ei, c_ew, num_coarse = build_coarsened_graph(
                        data.train_pos_edge_index, num_nodes, x.cpu(),
                        method, args.alpha, args.k,
                    )
                    elapsed = time.time() - t0

                    if num_coarse < 2:
                        print(
                            "    [{}] Too few coarse nodes ({}), skipping.".format(
                                tag, num_coarse
                            )
                        )
                        depth_results[method] = {"error": "too_few_nodes"}
                        continue

                    assert partition is not None
                    assert c_ei is not None

                    x_coarse = logsumexp_features(x.cpu(), partition).to(device)
                    c_ei_d = c_ei.to(device)
                    c_ew_d = c_ew.to(device) if c_ew is not None else None

                    with torch.no_grad():
                        z_coarse = model(x_coarse, c_ei_d, edge_weight=c_ew_d)

                    if torch.isnan(z_coarse).any():
                        print(
                            "    [{}] NaN in coarsened embeddings, skipping.".format(tag)
                        )
                        depth_results[method] = {"error": "NaN_coarse"}
                        continue

                    z_c_cpu = z_coarse.cpu()
                    c_ei_cpu = c_ei.cpu()
                    c_ew_cpu = c_ew

                    metrics = compute_metrics(z_c_cpu, c_ei_cpu, c_ew_cpu)

                    metrics["auc"] = round(
                        float(compute_auc_with_projection(
                            predictor, z_c_cpu, partition, data, num_nodes, device
                        )),
                        6,
                    )
                    metrics["num_coarse_nodes"] = num_coarse
                    metrics["coarsening_time_s"] = round(float(elapsed), 2)

                    depth_results[method] = metrics
                    print(
                        "    [{:11s}] cos={:.4f}  var={:.4f}  E={:.4f}  auc={:.4f}  |V'|={}".format(
                            tag,
                            metrics["cosine_sim"],
                            metrics["variance"],
                            metrics["dirichlet_energy"],
                            metrics["auc"],
                            num_coarse,
                        )
                    )

                except Exception as exc:
                    print("    [{}] FAILED: {}".format(tag, exc))
                    depth_results[method] = {"error": str(exc)}

            ds_results[depth_key] = depth_results

            del model, predictor, z_full, z_cpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

        results[ds_name] = ds_results

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "oversmoothing.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to {}".format(results_path))

    # ------------------------------------------------------------------
    # Generate figure
    # ------------------------------------------------------------------
    try:
        plot_results(results, args.datasets)
    except Exception as exc:
        print("Plotting failed: {}".format(exc))


if __name__ == "__main__":
    main()
