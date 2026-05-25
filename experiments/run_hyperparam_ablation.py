"""Hyperparameter robustness 2D ablation experiment (Priority 6).

Sweeps k and alpha in a grid, reports sufficiency, necessity, sparsity,
runtime, and spectral error as heatmaps. Directly addresses Reviewer 1's
concern that robustness is not fully characterized.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExperimentConfig
from experiments.train_gcn import load_dataset, MLPLinkPredictor
from experiments.run_ablation import normalized_adjacency_eigenvalues, mean_relative_error
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel

try:
    from src.coarsen import GraphCoarsener
except ImportError:
    GraphCoarsener = None

try:
    from src.evaluation.fidelity import fidelity_plus, fidelity_minus
except ImportError:
    fidelity_plus = None
    fidelity_minus = None


K_VALUES = [20, 50, 100, 200, 500]
ALPHA_VALUES = [0.3, 0.5, 0.7, 0.75, 0.9, 0.95]


def evaluate_at_params(model, data, test_edges, device, k, alpha, num_edges=30):
    """Run coarsening with given (k, alpha) and evaluate explanation quality."""
    if GraphCoarsener is None:
        return None

    x = data.x if data.x is not None else torch.eye(data.num_nodes)

    t0 = time.time()
    try:
        coarsener = GraphCoarsener(k=k, alpha=alpha)
        coarsener.fit(data.train_pos_edge_index, data.num_nodes, x)
    except Exception:
        return None
    coarsening_time = time.time() - t0

    fid_p_list = []
    fid_m_list = []
    explain_times = []

    n = min(num_edges, test_edges.size(1))
    for i in range(n):
        a = int(test_edges[0, i].item())
        b = int(test_edges[1, i].item())

        t1 = time.time()
        try:
            edge_index, edge_weight, feat, num_coarse, sa, sb, _orig_nodes = coarsener.explain_link(a, b)
            from torch_geometric.data import Data
            explanation = Data(
                x=feat, edge_index=edge_index, edge_weight=edge_weight,
                is_coarse_graph=True, target_a=sa, target_b=sb,
            )
        except Exception:
            continue
        explain_times.append(time.time() - t1)

        if fidelity_plus is not None:
            fp = fidelity_plus(model, data, explanation, a, b, device)
            fm = fidelity_minus(model, data, explanation, a, b, device)
            fid_p_list.append(fp)
            fid_m_list.append(fm)

    if not fid_p_list:
        return {
            "coarsening_time": coarsening_time,
            "mean_explain_time": float(np.mean(explain_times)) if explain_times else 0.0,
            "num_samples": 0,
        }

    # Spectral error
    eig_orig = normalized_adjacency_eigenvalues(
        data.num_nodes, data.train_pos_edge_index, min(k, 500)
    )
    eig_coarse = normalized_adjacency_eigenvalues(
        coarsener.num_coarse_nodes, coarsener.coarse_edge_index, min(k, 500)
    )
    spectral_error = mean_relative_error(eig_orig, eig_coarse, min(k, 500))

    return {
        "mean_fidelity_plus": float(np.mean(fid_p_list)),
        "std_fidelity_plus": float(np.std(fid_p_list)),
        "mean_fidelity_minus": float(np.mean(fid_m_list)),
        "std_fidelity_minus": float(np.std(fid_m_list)),
        "coarsening_time": coarsening_time,
        "mean_explain_time": float(np.mean(explain_times)) if explain_times else 0.0,
        "spectral_error": spectral_error,
        "num_coarse_nodes": coarsener.num_coarse_nodes,
        "num_samples": len(fid_p_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--num_edges", type=int, default=30)
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

    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    # train_test_split_edges sets edge_index=None; restore for fidelity functions
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

    import signal

    class TimeoutError(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise TimeoutError("Config timed out")

    results = {}
    total = len(K_VALUES) * len(ALPHA_VALUES)
    count = 0
    out_path = os.path.join("results", f"hyperparam_ablation_{args.dataset}.json")

    def _save_results():
        os.makedirs("results", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "dataset": args.dataset,
                "k_values": K_VALUES,
                "alpha_values": ALPHA_VALUES,
                "results": results,
            }, f, indent=2)

    for k in K_VALUES:
        for alpha in ALPHA_VALUES:
            count += 1
            print(f"\n[{count}/{total}] k={k}, alpha={alpha:.2f}", flush=True)
            try:
                # Per-config timeout: 120 seconds
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(120)
                result = evaluate_at_params(model, data, test_edges, device, k, alpha, args.num_edges)
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                if result:
                    key = f"k={k}_alpha={alpha}"
                    results[key] = result
                    fp = result.get('mean_fidelity_plus')
                    fm = result.get('mean_fidelity_minus')
                    se = result.get('spectral_error')
                    nc = result.get('num_coarse_nodes', 'N/A')
                    fp_str = f"{fp:.4f}" if fp is not None else "N/A"
                    fm_str = f"{fm:.4f}" if fm is not None else "N/A"
                    se_str = f"{se:.4f}" if se is not None else "N/A"
                    print(f"  Fid+: {fp_str}  Fid-: {fm_str}  SpecErr: {se_str}  CoarseNodes: {nc}", flush=True)
                    # Incremental save after each config
                    _save_results()
            except Exception as e:
                signal.alarm(0)
                print(f"  SKIPPED (error: {e})", flush=True)
                key = f"k={k}_alpha={alpha}"
                results[key] = {"error": str(e)}
                _save_results()

    _save_results()
    print(f"\nResults saved to {out_path}", flush=True)

    try:
        _plot_heatmaps(args.dataset, results)
    except Exception as e:
        print(f"Plotting skipped: {e}")


def _plot_heatmaps(dataset, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_to_plot = ["mean_fidelity_plus", "mean_fidelity_minus", "spectral_error", "coarsening_time"]
    metric_labels = ["Necessity (Fid+)", "Sufficiency (Fid-)", "Spectral Error", "Coarsening Time (s)"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        matrix = np.full((len(K_VALUES), len(ALPHA_VALUES)), np.nan)
        for ki, k in enumerate(K_VALUES):
            for ai, alpha in enumerate(ALPHA_VALUES):
                key = f"k={k}_alpha={alpha}"
                if key in results and metric in results[key]:
                    matrix[ki, ai] = results[key][metric]

        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(ALPHA_VALUES)))
        ax.set_xticklabels([str(a) for a in ALPHA_VALUES])
        ax.set_yticks(range(len(K_VALUES)))
        ax.set_yticklabels([str(k) for k in K_VALUES])
        ax.set_xlabel("alpha")
        ax.set_ylabel("k")
        ax.set_title(label)
        fig.colorbar(im, ax=ax)

        for ki in range(len(K_VALUES)):
            for ai in range(len(ALPHA_VALUES)):
                if not np.isnan(matrix[ki, ai]):
                    ax.text(ai, ki, f"{matrix[ki, ai]:.3f}", ha="center", va="center", fontsize=7, color="white")

    fig.suptitle(f"Hyperparameter Robustness on {dataset}")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig_path = os.path.join("figures", f"hyperparam_heatmap_{dataset}.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
