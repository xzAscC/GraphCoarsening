"""Runtime scaling analysis across datasets (Figure 5)."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExperimentConfig
from experiments.train_gcn import load_dataset
from src.models.gcn import GCN

try:
    from src.coarsen import GraphCoarsener
except ImportError:
    GraphCoarsener = None

try:
    from src.explainers.coarsen_explainer import CoarsenExplainer
except ImportError:
    CoarsenExplainer = None


def measure_coarsening_time(data, alpha: float, k: int, device):
    """Time the one-time coarsening step (spectral decomposition + partition).

    Returns (coarsener_or_None, elapsed_seconds).
    """
    if GraphCoarsener is None:
        return None, 0.0

    edge_index = data.train_pos_edge_index
    num_nodes = data.num_nodes
    x = data.x if data.x is not None else torch.eye(num_nodes)

    t0 = time.time()
    coarsener = GraphCoarsener(k=k, alpha=alpha)
    coarsener.fit(edge_index, num_nodes, x)
    elapsed = time.time() - t0
    return coarsener, elapsed


def measure_per_link_time(coarsener, data, device, num_edges=50, seed=42):
    if coarsener is None:
        return 0.0

    rng = np.random.RandomState(seed)
    if hasattr(data, "test_pos_edge_index") and data.test_pos_edge_index is not None:
        pos = data.test_pos_edge_index
    else:
        pos = data.train_pos_edge_index

    n = min(num_edges, pos.size(1))
    indices = rng.choice(pos.size(1), size=n, replace=False)
    times = []

    for idx in indices:
        a = int(pos[0, idx].item())
        b = int(pos[1, idx].item())
        t0 = time.time()
        coarsener.explain_link(a, b)
        times.append(time.time() - t0)

    return float(np.mean(times)) if times else 0.0


def load_model_for_dataset(dataset: str, device):
    checkpoint_path = os.path.join("checkpoints", f"{dataset}_gcn.pt")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        mc = ckpt["config"]
        model = GCN(mc["in_channels"], mc["hidden_channels"], mc["out_channels"], mc["num_layers"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        from experiments.train_gcn import MLPLinkPredictor
        predictor = MLPLinkPredictor(mc["out_channels"], mc["hidden_channels"]).to(device)
        if "predictor_state_dict" in ckpt:
            predictor.load_state_dict(ckpt["predictor_state_dict"])
        predictor.eval()
        return model, predictor

    print(f"  WARNING: No checkpoint for {dataset}, using untrained model")
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_edges", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    if args.device is not None:
        cfg.device = args.device
    device = torch.device(cfg.device)
    torch.manual_seed(args.seed)

    datasets = cfg.scalability_datasets
    results = []

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            data = load_dataset(ds_name)
        except Exception as e:
            print(f"  SKIP: failed to load ({e})")
            continue

        num_nodes = data.num_nodes
        num_edges = data.train_pos_edge_index.size(1)
        print(f"  |V|={num_nodes:,}  |E|={num_edges:,}")

        coarsener, coarse_time = measure_coarsening_time(
            data, cfg.spectral.alpha, cfg.spectral.k, device,
        )
        print(f"  Coarsening time: {coarse_time:.2f}s")

        avg_explain_time = measure_per_link_time(
            coarsener, data, device, args.num_edges, args.seed,
        )
        print(f"  Avg per-link explain time: {avg_explain_time:.4f}s")

        results.append({
            "dataset": ds_name,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "coarsening_time_s": round(coarse_time, 4),
            "avg_explain_time_s": round(avg_explain_time, 4),
        })

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "runtime.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    try:
        _plot_runtime(results)
    except Exception as e:
        print(f"Plotting skipped: {e}")

    _print_table(results)


def _print_table(results):
    print(f"\n{'Dataset':<20} {'|V|':>10} {'|E|':>12} {'Coarsen(s)':>12} {'AvgExpl(s)':>12}")
    print("-" * 68)
    for r in results:
        print(
            f"{r['dataset']:<20} {r['num_nodes']:>10,} {r['num_edges']:>12,} "
            f"{r['coarsening_time_s']:>12.2f} {r['avg_explain_time_s']:>12.4f}"
        )


def _plot_runtime(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        return

    names = [r["dataset"] for r in results]
    num_v = np.array([r["num_nodes"] for r in results])
    num_e = np.array([r["num_edges"] for r in results])
    times = np.array([r["avg_explain_time_s"] for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(num_e, times, s=80, c="steelblue", edgecolors="k", zorder=3)
    for i, name in enumerate(names):
        ax1.annotate(name, (num_e[i], times[i]), fontsize=7, ha="left", va="bottom")
    ax1.set_xlabel("|E| (num edges)")
    ax1.set_ylabel("Avg per-link runtime (s)")
    ax1.set_title("Runtime vs |E|")
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    ax2.scatter(num_v, times, s=80, c="coral", edgecolors="k", zorder=3)
    for i, name in enumerate(names):
        ax2.annotate(name, (num_v[i], times[i]), fontsize=7, ha="left", va="bottom")
    ax2.set_xlabel("|V| (num nodes)")
    ax2.set_ylabel("Avg per-link runtime (s)")
    ax2.set_title("Runtime vs |V|")
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig_path = os.path.join("figures", "runtime_scaling.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
