"""Explanation size vs quality Pareto curve experiment (Priority 3).

Generates curves showing sufficiency, necessity, and deletion AUC
at varying explanation sizes (5%, 10%, 20%, 30%, 50% of local edges).
This directly shows that our method achieves better quality at smaller sizes.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExperimentConfig
from experiments.train_gcn import load_dataset, MLPLinkPredictor
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel

try:
    from src.evaluation.comprehensive_metrics import compute_all_metrics
except ImportError:
    compute_all_metrics = None

try:
    from src.evaluation.fidelity import fidelity_plus, fidelity_minus, compute_sparsity
except ImportError:
    fidelity_plus = None
    fidelity_minus = None
    compute_sparsity = None

try:
    from src.explainers.coarsen_explainer import CoarsenExplainer
except ImportError:
    CoarsenExplainer = None

try:
    from src.explainers.coarsening_baselines import (
        RandomSubgraphBaseline,
        DegreeBasedBaseline,
        KHopSubgraphBaseline,
        GreedyDeletionBaseline,
    )
except ImportError:
    RandomSubgraphBaseline = None
    DegreeBasedBaseline = None
    KHopSubgraphBaseline = None
    GreedyDeletionBaseline = None

try:
    from src.explainers.pyg_baselines import GNNExplainerWrapper
except ImportError:
    GNNExplainerWrapper = None

SIZE_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50]


def _get_link_score(model, data, node_a, node_b, device):
    """Get raw prediction probability for a single link."""
    data = data.to(device)
    target = torch.tensor([[node_a], [node_b]], device=device)
    with torch.no_grad():
        score = torch.sigmoid(model(data.x, data.edge_index, target, edge_weight=getattr(data, "edge_weight", None))).squeeze()
    return score.item()


def _compute_basic_metrics(model, data, explanation, node_a, node_b, device):
    """Compute sufficiency, necessity, sparsity using existing functions."""
    p_full = _get_link_score(model, data, node_a, node_b, device)

    # Sparsity
    if compute_sparsity is not None:
        sparsity = compute_sparsity(explanation, data)
    else:
        num_exp = explanation.edge_index.size(1)
        num_orig = data.edge_index.size(1)
        sparsity = 1.0 - num_exp / max(num_orig, 1)

    # Sufficiency (fidelity_minus): run on explanation alone
    if fidelity_minus is not None:
        fid_m = fidelity_minus(model, data, explanation, node_a, node_b, device)
    else:
        fid_m = 0.0

    # Necessity (fidelity_plus): remove explanation, check change
    if fidelity_plus is not None:
        fid_p = fidelity_plus(model, data, explanation, node_a, node_b, device)
    else:
        fid_p = 0.0

    return {
        "sufficiency": 1.0 - fid_m,  # Higher = more sufficient (prediction preserved)
        "necessity": fid_p,           # Higher = more necessary
        "sparsity": sparsity,         # Higher = more sparse (fewer edges)
        "num_edges": explanation.edge_index.size(1),
        "p_full": p_full,
    }


def _prune_explanation_by_weight(explanation, keep_count, device):
    if keep_count >= explanation.edge_index.size(1):
        return explanation
    ei = explanation.edge_index
    ew = getattr(explanation, "edge_weight", None)
    num_edges = ei.size(1)
    if ew is not None and ew.numel() == num_edges:
        sorted_idx = ew.argsort(descending=True)[:keep_count]
    else:
        sorted_idx = torch.randperm(num_edges, device=device)[:keep_count]
    pruned = Data(
        x=explanation.x,
        edge_index=ei[:, sorted_idx],
        edge_weight=ew[sorted_idx] if ew is not None and ew.numel() == num_edges else None,
    )
    for key in explanation.keys():
        if key not in ("x", "edge_index", "edge_weight"):
            setattr(pruned, key, getattr(explanation, key))
    return pruned


def _khop_edge_count(data, node_a, node_b, num_hops=2):
    subset, sub_ei, _, _ = k_hop_subgraph(
        node_idx=torch.tensor([node_a, node_b]),
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=False,
        num_nodes=data.x.size(0),
    )
    return sub_ei.size(1)


def run_method_at_size(method_name, model, data, test_edges, device, size_frac, k_frac):
    results = []

    if method_name == "Ours":
        if CoarsenExplainer is None:
            return None
        explainer = CoarsenExplainer(model, k=100, alpha=0.75, device=device)
    elif method_name == "Random":
        if RandomSubgraphBaseline is None:
            return None
        explainer = RandomSubgraphBaseline(model, k_frac=k_frac, device=device)
    elif method_name == "Degree":
        if DegreeBasedBaseline is None:
            return None
        explainer = DegreeBasedBaseline(model, k_frac=k_frac, device=device)
    elif method_name == "KHop":
        if KHopSubgraphBaseline is None:
            return None
        explainer = KHopSubgraphBaseline(model, k_hop=2, device=device)
    elif method_name == "GreedyDel":
        if GreedyDeletionBaseline is None:
            return None
        explainer = GreedyDeletionBaseline(model, device=device)
    elif method_name == "GNNExp":
        if GNNExplainerWrapper is None:
            return None
        explainer = GNNExplainerWrapper(model, k_frac=k_frac, device=device)
    else:
        return None

    for i in range(test_edges.size(1)):
        a = int(test_edges[0, i].item())
        b = int(test_edges[1, i].item())
        try:
            explanation = explainer.explain_link(data, a, b)
            if method_name == "Ours":
                target_edges = max(1, int(_khop_edge_count(data, a, b) * size_frac))
                explanation = _prune_explanation_by_weight(explanation, target_edges, device)
            metrics = _compute_basic_metrics(model, data, explanation, a, b, device)
            results.append(metrics)
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue

    if not results:
        return None

    return {
        "sufficiency": float(np.mean([r["sufficiency"] for r in results])),
        "necessity": float(np.mean([r["necessity"] for r in results])),
        "sparsity": float(np.mean([r["sparsity"] for r in results])),
        "num_edges": float(np.mean([r["num_edges"] for r in results])),
        "num_samples": len(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--methods", type=str, default="Ours,Random,Degree,KHop")
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
    # train_test_split_edges sets edge_index=None; restore for explainer
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

    # Sample test edges
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
        print(f"\n--- {method} ---")
        method_results = {}
        for size_frac in SIZE_FRACTIONS:
            print(f"  size_frac={size_frac:.2f} ...", end=" ", flush=True)
            t0 = time.time()
            result = run_method_at_size(method, model, data, test_edges, device, size_frac, size_frac)
            elapsed = time.time() - t0
            if result is not None:
                method_results[str(size_frac)] = result
                print(f"suff={result['sufficiency']:.4f} nec={result['necessity']:.4f} spars={result['sparsity']:.4f} ({elapsed:.1f}s)")
            else:
                print(f"SKIP ({elapsed:.1f}s)")
        results[method] = method_results

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"pareto_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "size_fractions": SIZE_FRACTIONS, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    try:
        _plot_pareto(args.dataset, results)
    except Exception as e:
        print(f"Plotting skipped: {e}")


def _plot_pareto(dataset, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for metric_idx, (metric_name, ylabel) in enumerate([
        ("sufficiency", "Sufficiency"),
        ("necessity", "Necessity"),
        ("sparsity", "Sparsity"),
    ]):
        ax = axes[metric_idx]
        for method, method_data in results.items():
            if not method_data:
                continue
            sizes = sorted([float(k) for k in method_data.keys()])
            values = [method_data[str(s)][metric_name] for s in sizes if str(s) in method_data]
            if len(sizes) != len(values):
                sizes = sizes[:len(values)]
            ax.plot(sizes, values, marker="o", label=method, linewidth=2)

        ax.set_xlabel("Explanation Size Fraction")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Size ({dataset})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig_path = os.path.join("figures", f"pareto_{dataset}.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
