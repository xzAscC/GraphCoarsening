"""Run explanation fidelity experiments (Figure 4)."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExperimentConfig
from experiments.train_gcn import load_dataset, MLPLinkPredictor as TrainMLPLinkPredictor
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel

try:
    from src.explainers.baselines import OcclusionExplainer, SaliencyExplainer
except ImportError:
    OcclusionExplainer = None
    SaliencyExplainer = None

try:
    from src.explainers.coarsen_explainer import CoarsenExplainer
except ImportError:
    CoarsenExplainer = None

try:
    from src.explainers.pyg_baselines import GNNExplainerWrapper, PGExplainerWrapper, SubgraphXWrapper
except ImportError:
    GNNExplainerWrapper = None
    PGExplainerWrapper = None
    SubgraphXWrapper = None

try:
    from src.evaluation.fidelity import fidelity_plus, fidelity_minus
except ImportError:
    fidelity_plus = None
    fidelity_minus = None

ALL_METHODS = ["Occlusion", "Saliency", "GNNExplainer", "PGExplainer", "SubgraphX", "Ours"]


def get_explainer(method: str, model, data, device):
    if method == "Occlusion" and OcclusionExplainer is not None:
        return OcclusionExplainer(model, device=device)
    if method == "Saliency" and SaliencyExplainer is not None:
        return SaliencyExplainer(model, device=device)
    if method == "GNNExplainer":
        if GNNExplainerWrapper is not None:
            try:
                return GNNExplainerWrapper(model, device=device)
            except Exception:
                return None
        return None
    if method == "PGExplainer":
        if PGExplainerWrapper is not None:
            try:
                return PGExplainerWrapper(model, device=device)
            except Exception:
                return None
        return None
    if method == "SubgraphX":
        if SubgraphXWrapper is not None:
            try:
                return SubgraphXWrapper(model, device=device)
            except Exception:
                return None
        return None
    if method == "Ours" and CoarsenExplainer is not None:
        return CoarsenExplainer(model, device=device)
    return None


def sample_test_edges(data, num_edges: int, seed: int = 42):
    if hasattr(data, "test_pos_edge_index") and data.test_pos_edge_index is not None:
        pos = data.test_pos_edge_index
    elif hasattr(data, "val_pos_edge_index") and data.val_pos_edge_index is not None:
        pos = data.val_pos_edge_index
    else:
        pos = data.train_pos_edge_index[:, :num_edges]

    n = min(num_edges, pos.size(1))
    rng = np.random.RandomState(seed)
    indices = rng.choice(pos.size(1), size=n, replace=False)
    return pos[:, indices]


def run_explanation_method(method, explainer, model, data, test_edges, device):
    fid_p_list = []
    fid_m_list = []
    times = []

    for i in range(test_edges.size(1)):
        node_a = int(test_edges[0, i].item())
        node_b = int(test_edges[1, i].item())

        t0 = time.time()
        try:
            explanation = explainer.explain_link(data, node_a, node_b)
        except Exception:
            explanation = None
        elapsed = time.time() - t0
        times.append(elapsed)

        if explanation is None:
            continue

        if fidelity_plus is not None:
            fp = fidelity_plus(model, data, explanation, node_a, node_b, device)
            fm = fidelity_minus(model, data, explanation, node_a, node_b, device)
        else:
            fp = 0.0
            fm = 0.0

        fid_p_list.append(fp)
        fid_m_list.append(fm)

    if not fid_p_list:
        return {"status": "no_results"}

    return {
        "fidelity_plus": fid_p_list,
        "fidelity_minus": fid_m_list,
        "mean_fidelity_plus": float(np.mean(fid_p_list)),
        "mean_fidelity_minus": float(np.mean(fid_m_list)),
        "std_fidelity_plus": float(np.std(fid_p_list)),
        "std_fidelity_minus": float(np.std(fid_m_list)),
        "mean_time": float(np.mean(times)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--methods", type=str, default="all", help="Comma-separated methods or 'all'")
    parser.add_argument("--num_edges", type=int, default=100)
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

    methods = ALL_METHODS if args.methods == "all" else [m.strip() for m in args.methods.split(",")]

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
    model_config = ckpt["config"]
    gcn = GCN(
        in_channels=model_config["in_channels"],
        hidden_channels=model_config["hidden_channels"],
        out_channels=model_config["out_channels"],
        num_layers=model_config["num_layers"],
    ).to(device)
    gcn.load_state_dict(ckpt["model_state_dict"])

    predictor = TrainMLPLinkPredictor(
        in_channels=model_config["out_channels"],
        hidden_channels=model_config["hidden_channels"],
    ).to(device)
    if "predictor_state_dict" in ckpt:
        predictor.load_state_dict(ckpt["predictor_state_dict"])

    model = LinkPredictionModel(gcn, predictor).to(device)
    model.eval()

    test_edges = sample_test_edges(data, args.num_edges, args.seed)

    results = {}
    for method in methods:
        print(f"\n--- {method} ---")
        explainer = get_explainer(method, model, data, device)
        if explainer is None:
            print(f"  SKIP: {method} not available (missing dependency)")
            results[method] = {"status": "unavailable"}
            continue

        method_result = run_explanation_method(
            method, explainer, model, data, test_edges, device,
        )
        results[method] = method_result
        if "mean_fidelity_plus" in method_result:
            print(
                f"  Fid+: {method_result['mean_fidelity_plus']:.4f} +/- {method_result['std_fidelity_plus']:.4f}"
            )
            print(
                f"  Fid-: {method_result['mean_fidelity_minus']:.4f} +/- {method_result['std_fidelity_minus']:.4f}"
            )
            print(f"  Avg time: {method_result['mean_time']:.4f}s")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"explanation_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "num_edges": args.num_edges, "methods": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    try:
        _plot_results(args.dataset, results)
    except Exception as e:
        print(f"Plotting skipped: {e}")


def _plot_results(dataset: str, results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = {k: v for k, v in results.items() if "mean_fidelity_plus" in v}
    if not available:
        return

    methods = list(available.keys())
    fid_p = [available[m]["mean_fidelity_plus"] for m in methods]
    fid_p_err = [available[m].get("std_fidelity_plus", 0) for m in methods]
    fid_m = [available[m]["mean_fidelity_minus"] for m in methods]
    fid_m_err = [available[m].get("std_fidelity_minus", 0) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, fid_p, width, yerr=fid_p_err, label="Fidelity+", capsize=3)
    ax.bar(x + width / 2, fid_m, width, yerr=fid_m_err, label="Fidelity-", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Fidelity")
    ax.set_title(f"Explanation Fidelity on {dataset}")
    ax.legend()
    fig.tight_layout()

    os.makedirs("figures", exist_ok=True)
    fig_path = os.path.join("figures", f"explanation_{dataset}.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
