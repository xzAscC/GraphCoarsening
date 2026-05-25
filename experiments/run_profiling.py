"""Detailed runtime/memory profiling experiment (Priority 9).

Breaks down runtime into: preprocessing time, preprocessing memory,
per-query explanation time, per-query forward time, total time for
batched queries, peak memory. Compares across all methods.
"""

import argparse
import json
import os
import sys
import time
import tracemalloc

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExperimentConfig
from experiments.train_gcn import load_dataset, MLPLinkPredictor
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel


ALL_METHODS = ["Occlusion", "Saliency", "GNNExplainer", "Ours"]
BATCH_SIZES = [1, 10, 100, 1000]


def _get_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return tracemalloc.get_traced_memory()[1] / 1024 / 1024


def profile_method(method_name, model, data, test_edges, device, num_edges=50):
    """Profile a single method's runtime and memory usage."""
    if method_name == "Occlusion":
        from src.explainers.baselines import OcclusionExplainer
        ExplainerClass = OcclusionExplainer
        kwargs = {"k_hop": 2, "k_frac": 0.5}
    elif method_name == "Saliency":
        from src.explainers.baselines import SaliencyExplainer
        ExplainerClass = SaliencyExplainer
        kwargs = {"k_frac": 0.5}
    elif method_name == "GNNExplainer":
        try:
            from src.explainers.pyg_baselines import GNNExplainerWrapper
            ExplainerClass = GNNExplainerWrapper
        except (ImportError, Exception):
            return None
        kwargs = {"epochs": 100, "k_frac": 0.5}
    elif method_name == "Ours":
        from src.explainers.coarsen_explainer import CoarsenExplainer
        ExplainerClass = CoarsenExplainer
        kwargs = {"k": 100, "alpha": 0.75}
    else:
        return None

    # Preprocessing time
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()

    t0 = time.time()
    explainer = ExplainerClass(model, device=device, **kwargs)
    preprocess_time = time.time() - t0
    preprocess_mem = _get_memory_mb()

    # Per-query times
    n = min(num_edges, test_edges.size(1))
    explain_times = []
    forward_times = []

    for i in range(n):
        a = int(test_edges[0, i].item())
        b = int(test_edges[1, i].item())

        t1 = time.time()
        try:
            explanation = explainer.explain_link(data, a, b)
        except Exception:
            continue
        explain_times.append(time.time() - t1)

        # Forward time on explanation
        t2 = time.time()
        try:
            target = torch.tensor([[a], [b]], device=device)
            with torch.no_grad():
                model(data.x.to(device), data.edge_index.to(device), target)
        except Exception:
            pass
        forward_times.append(time.time() - t2)

    peak_mem = _get_memory_mb()
    tracemalloc.stop()

    if not explain_times:
        return None

    # Batch timings
    batch_results = {}
    for batch_size in BATCH_SIZES:
        actual_size = min(batch_size, n)
        t3 = time.time()
        for i in range(actual_size):
            a = int(test_edges[0, i].item())
            b = int(test_edges[1, i].item())
            try:
                explainer.explain_link(data, a, b)
            except Exception:
                continue
        batch_time = time.time() - t3
        batch_results[str(batch_size)] = round(batch_time, 4)

    return {
        "preprocess_time_s": round(preprocess_time, 4),
        "preprocess_memory_mb": round(preprocess_mem, 2),
        "mean_explain_time_s": round(float(np.mean(explain_times)), 6),
        "std_explain_time_s": round(float(np.std(explain_times)), 6),
        "median_explain_time_s": round(float(np.median(explain_times)), 6),
        "mean_forward_time_s": round(float(np.mean(forward_times)), 6) if forward_times else 0.0,
        "peak_memory_mb": round(peak_mem, 2),
        "batch_times": batch_results,
        "num_samples": len(explain_times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--methods", type=str, default=",".join(ALL_METHODS))
    parser.add_argument("--num_edges", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    if args.device:
        cfg.device = args.device
    device = torch.device(cfg.device)
    torch.manual_seed(args.seed)

    methods = [m.strip() for m in args.methods.split(",")]

    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    if data.edge_index is None:
        data.edge_index = data.train_pos_edge_index
    print(f"  Nodes: {data.num_nodes}, Edges: {data.train_pos_edge_index.size(1)}")

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
    for method in methods:
        print(f"\n--- Profiling {method} ---")
        result = profile_method(method, model, data, test_edges, device, args.num_edges)
        if result:
            results[method] = result
            print(f"  Preprocess: {result['preprocess_time_s']:.4f}s, {result['preprocess_memory_mb']:.1f}MB")
            print(f"  Per-query: {result['mean_explain_time_s']:.6f}s (median: {result['median_explain_time_s']:.6f}s)")
            print(f"  Forward: {result['mean_forward_time_s']:.6f}s")
            print(f"  Peak memory: {result['peak_memory_mb']:.1f}MB")
        else:
            print(f"  SKIP: method unavailable")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"profiling_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    try:
        _plot_profiling(args.dataset, results)
    except Exception as e:
        print(f"Plotting skipped: {e}")


def _plot_profiling(dataset, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        return

    methods = list(results.keys())
    preprocess_times = [results[m]["preprocess_time_s"] for m in methods]
    query_times = [results[m]["mean_explain_time_s"] for m in methods]
    peak_mems = [results[m]["peak_memory_mb"] for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].barh(methods, preprocess_times)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_title("Preprocessing Time")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].barh(methods, query_times)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_title("Per-Query Explanation Time")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].barh(methods, peak_mems)
    axes[1, 0].set_xlabel("Memory (MB)")
    axes[1, 0].set_title("Peak Memory")
    axes[1, 0].grid(True, alpha=0.3)

    for m in methods:
        batch = results[m].get("batch_times", {})
        if batch:
            sizes = sorted([int(k) for k in batch.keys()])
            times_list = [batch[str(s)] for s in sizes]
            axes[1, 1].plot(sizes, times_list, marker="o", label=m)
    axes[1, 1].set_xlabel("Number of Queries")
    axes[1, 1].set_ylabel("Total Time (s)")
    axes[1, 1].set_title("Batch Query Scaling")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Runtime & Memory Profiling on {dataset}")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig_path = os.path.join("figures", f"profiling_{dataset}.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
