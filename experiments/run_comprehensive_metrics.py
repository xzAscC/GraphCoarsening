"""Comprehensive metrics experiment using all 7 explanation quality metrics.

Reports: sufficiency, necessity, comprehensiveness, sparsity,
sparsity_abs_edges, deletion_auc, insertion_auc.
Directly addresses reviewer concern that fidelity alone is too weak.
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
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel

from src.evaluation.comprehensive_metrics import compute_all_metrics

METRIC_KEYS = [
    "sufficiency", "necessity", "comprehensiveness",
    "sparsity", "sparsity_abs_edges", "deletion_auc", "insertion_auc",
]

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
    from src.explainers.pyg_baselines import GNNExplainerWrapper
except ImportError:
    GNNExplainerWrapper = None

ALL_METHODS = ["Occlusion", "Saliency", "GNNExplainer", "Ours"]


def get_explainer(method, model, device):
    if method == "Occlusion" and OcclusionExplainer is not None:
        return OcclusionExplainer(model, device=device)
    if method == "Saliency" and SaliencyExplainer is not None:
        return SaliencyExplainer(model, device=device)
    if method == "GNNExplainer" and GNNExplainerWrapper is not None:
        try:
            return GNNExplainerWrapper(model, device=device)
        except Exception:
            return None
    if method == "Ours" and CoarsenExplainer is not None:
        return CoarsenExplainer(model, device=device)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--methods", type=str, default="all")
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
        print(f"\n--- {method} ---")
        explainer = get_explainer(method, model, device)
        if explainer is None:
            print(f"  SKIP: unavailable")
            results[method] = {"status": "unavailable"}
            continue

        per_instance = []
        times = []
        for i in range(test_edges.size(1)):
            a = int(test_edges[0, i].item())
            b = int(test_edges[1, i].item())

            t0 = time.time()
            try:
                explanation = explainer.explain_link(data, a, b)
            except Exception:
                continue
            elapsed = time.time() - t0
            times.append(elapsed)

            if explanation is None:
                continue

            try:
                metrics = compute_all_metrics(
                    model, data, explanation, a, b,
                    device=str(device), num_steps=20, num_hops=2,
                )
                per_instance.append(metrics)
            except Exception as e:
                print(f"    Edge ({a},{b}) metrics failed: {e}")
                continue

        if not per_instance:
            results[method] = {"status": "no_results"}
            print(f"  No valid results")
            continue

        aggregated = {"num_samples": len(per_instance), "mean_time": float(np.mean(times))}
        for key in METRIC_KEYS:
            values = [m[key] for m in per_instance]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        results[method] = aggregated
        print(f"  Samples: {len(per_instance)}, Time: {aggregated['mean_time']:.4f}s")
        for key in METRIC_KEYS:
            print(f"    {key}: {aggregated[key]['mean']:.4f} +/- {aggregated[key]['std']:.4f}")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"comprehensive_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "metrics": METRIC_KEYS, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
