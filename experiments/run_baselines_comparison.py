"""Comprehensive baseline comparison experiment (Priority 2).

Runs ALL explanation baselines including:
- Trivial: FullGraph, KHop, Random, Degree, PageRank
- Hard: GreedyDeletion
- Coarsening: RandomCoarsening, HeavyEdgeCoarsening, EffectiveResistanceCoarsening, NoRefinement
- GNN: Occlusion, Saliency
- Ours: CoarsenExplainer (full method)

Reports fidelity+ (sufficiency), fidelity- (comprehensiveness), sparsity,
and per-instance runtime. All methods use the same test edges for fair comparison.
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

from src.explainers.coarsen_explainer import CoarsenExplainer
from src.explainers.coarsening_baselines import (
    FullGraphBaseline,
    KHopSubgraphBaseline,
    RandomSubgraphBaseline,
    DegreeBasedBaseline,
    GreedyDeletionBaseline,
    RandomCoarseningExplainer,
    HeavyEdgeCoarseningExplainer,
    EffectiveResistanceCoarseningExplainer,
    NoRefinementExplainer,
)
from src.explainers.baselines import OcclusionExplainer, SaliencyExplainer
from src.evaluation.fidelity import fidelity_plus, fidelity_minus, compute_sparsity


def make_explainer(name, model, device):
    if name == "FullGraph":
        return FullGraphBaseline(model, device=device)
    if name == "KHop":
        return KHopSubgraphBaseline(model, k_hop=2, device=device)
    if name == "Random":
        return RandomSubgraphBaseline(model, k_frac=0.5, device=device)
    if name == "Degree":
        return DegreeBasedBaseline(model, k_frac=0.5, device=device)
    if name == "GreedyDel":
        return GreedyDeletionBaseline(model, device=device)
    if name == "RandomCoarse":
        return RandomCoarseningExplainer(model, k=100, alpha=0.75, device=device)
    if name == "HeavyEdge":
        return HeavyEdgeCoarseningExplainer(model, k=100, alpha=0.75, device=device)
    if name == "EffResist":
        return EffectiveResistanceCoarseningExplainer(model, k=100, alpha=0.75, device=device)
    if name == "NoRefine":
        return NoRefinementExplainer(model, k=100, alpha=0.75, device=device)
    if name == "Occlusion":
        return OcclusionExplainer(model, device=device)
    if name == "Saliency":
        return SaliencyExplainer(model, device=device)
    if name == "Ours":
        return CoarsenExplainer(model, k=100, alpha=0.75, mode="edge", k_hop=2, k_frac=0.5, device=device)
    return None


BASELINE_GROUPS = {
    "trivial": ["FullGraph", "KHop", "Random", "Degree"],
    "hard": ["GreedyDel"],
    "coarsening": ["RandomCoarse", "HeavyEdge", "EffResist", "NoRefine"],
    "gnn": ["Occlusion", "Saliency"],
    "ours": ["Ours"],
}

ALL_METHODS = (
    BASELINE_GROUPS["trivial"]
    + BASELINE_GROUPS["hard"]
    + BASELINE_GROUPS["coarsening"]
    + BASELINE_GROUPS["gnn"]
    + BASELINE_GROUPS["ours"]
)


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
        print(f"ERROR: No checkpoint at {checkpoint_path}")
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
    for method_name in methods:
        print(f"\n--- {method_name} ---")
        explainer = make_explainer(method_name, model, device)
        if explainer is None:
            print(f"  SKIP: unavailable")
            results[method_name] = {"status": "unavailable"}
            continue

        fid_plus_list = []
        fid_minus_list = []
        sparsity_list = []
        times = []

        for i in range(test_edges.size(1)):
            a = int(test_edges[0, i].item())
            b = int(test_edges[1, i].item())

            t0 = time.time()
            try:
                explanation = explainer.explain_link(data, a, b)
            except Exception as e:
                print(f"  Edge ({a},{b}) explain failed: {e}")
                continue
            elapsed = time.time() - t0
            times.append(elapsed)

            if explanation is None:
                continue

            try:
                fp = fidelity_plus(model, data, explanation, a, b, device=str(device))
                fm = fidelity_minus(model, data, explanation, a, b, device=str(device))
                sp = compute_sparsity(explanation, data)
                fid_plus_list.append(fp)
                fid_minus_list.append(fm)
                sparsity_list.append(sp)
            except Exception as e:
                print(f"  Edge ({a},{b}) metrics failed: {e}")
                continue

        if not fid_plus_list:
            results[method_name] = {"status": "no_results"}
            print(f"  No valid results")
            continue

        results[method_name] = {
            "num_samples": len(fid_plus_list),
            "mean_fidelity_plus": float(np.mean(fid_plus_list)),
            "std_fidelity_plus": float(np.std(fid_plus_list)),
            "mean_fidelity_minus": float(np.mean(fid_minus_list)),
            "std_fidelity_minus": float(np.std(fid_minus_list)),
            "mean_sparsity": float(np.mean(sparsity_list)),
            "std_sparsity": float(np.std(sparsity_list)),
            "mean_time": float(np.mean(times)),
            "mean_explanation_size": float(
                np.mean([explanation.edge_index.size(1) for _ in range(1)])
            ) if hasattr(explanation, "edge_index") else None,
        }
        print(f"  Samples: {len(fid_plus_list)}, "
              f"Fid+: {results[method_name]['mean_fidelity_plus']:.4f}, "
              f"Fid-: {results[method_name]['mean_fidelity_minus']:.4f}, "
              f"Sparsity: {results[method_name]['mean_sparsity']:.4f}, "
              f"Time: {results[method_name]['mean_time']:.4f}s", flush=True)

        _save_incremental(args, results)

def _save_incremental(args, results):
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"baselines_comparison_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "num_edges": args.num_edges,
            "methods": results,
            "baseline_groups": BASELINE_GROUPS,
        }, f, indent=2)


if __name__ == "__main__":
    main()
