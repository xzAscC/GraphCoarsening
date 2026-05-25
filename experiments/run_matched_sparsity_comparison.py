"""Matched-sparsity comparison: Ours vs Saliency with paired statistical tests.

For each test edge, both methods generate explanations, then prune to the
SAME edge budget (k edges).  Continuous sufficiency (Fidelity-), necessity
(Fidelity+), and connectivity metrics are computed per-instance, enabling
paired t-tests and Wilcoxon signed-rank tests.

Usage:
    python -m experiments.run_matched_sparsity_comparison --dataset Cora
    python -m experiments.run_matched_sparsity_comparison --dataset Cora --num_edges 100 --budgets 5,10,20,50
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.train_gcn import load_dataset, MLPLinkPredictor
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel
from src.evaluation.comprehensive_metrics import sufficiency, necessity
from src.evaluation.fidelity import fidelity_plus_continuous


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


def _count_components(explanation):
    ei = explanation.edge_index.cpu().numpy()
    n = explanation.x.size(0)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(ei.shape[1]):
        union(int(ei[0, i]), int(ei[1, i]))

    return len(set(find(i) for i in range(n)))


def _target_connected(explanation, node_a, node_b):
    oi = getattr(explanation, "original_node_indices", None)
    if oi is None:
        return False
    oi = oi.cpu().numpy()
    if node_a not in oi or node_b not in oi:
        return False
    local_a = int(np.where(oi == node_a)[0][0])
    local_b = int(np.where(oi == node_b)[0][0])

    ei = explanation.edge_index.cpu().numpy()
    n = explanation.x.size(0)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(ei.shape[1]):
        union(int(ei[0, i]), int(ei[1, i]))

    return find(local_a) == find(local_b)


def run_comparison(dataset_name, num_edges, budgets, seed=42,
                   lambda_pred=1.0, fidelity_threshold=0.8):
    from src.explainers.coarsen_explainer import CoarsenExplainer
    from src.explainers.baselines import SaliencyExplainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_dataset(dataset_name)
    if data.edge_index is None:
        data.edge_index = data.train_pos_edge_index
    data = data.to(device)

    ckpt = torch.load(
        f"checkpoints/{dataset_name}_gcn.pt",
        map_location=device,
        weights_only=False,
    )
    mc = ckpt["config"]
    gcn = GCN(
        mc["in_channels"], mc["hidden_channels"],
        mc["out_channels"], mc["num_layers"],
    ).to(device)
    gcn.load_state_dict(ckpt["model_state_dict"])
    predictor = MLPLinkPredictor(mc["out_channels"], mc["hidden_channels"]).to(device)
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    model = LinkPredictionModel(gcn, predictor).to(device)
    model.eval()

    pos = data.test_pos_edge_index
    rng = np.random.RandomState(seed)
    n = min(num_edges, pos.size(1))
    indices = rng.choice(pos.size(1), size=n, replace=False)
    test_edges = pos[:, indices]

    ours_method = CoarsenExplainer(
        model, k=100, alpha=0.75, mode="edge", k_hop=2, k_frac=0.5, device="cuda",
        lambda_pred=lambda_pred, fidelity_threshold=fidelity_threshold,
    )
    sali_method = SaliencyExplainer(model, k_frac=0.5, device="cuda")

    all_results = {}

    for budget in budgets:
        print(f"\n=== Budget k={budget} ===")
        ours_suff, sali_suff = [], []
        ours_nec, sali_nec = [], []
        ours_cont, sali_cont = [], []
        ours_comp, sali_comp = [], []
        ours_tconn, sali_tconn = 0, 0

        for i in range(n):
            a = int(test_edges[0, i].item())
            b = int(test_edges[1, i].item())

            try:
                o_exp = ours_method.explain_link(data, a, b)
                s_exp = sali_method.explain_link(data, a, b)

                o_pruned = _prune_explanation_by_weight(o_exp, budget, device)
                s_pruned = _prune_explanation_by_weight(s_exp, budget, device)

                o_s = sufficiency(model, data, o_pruned, a, b, device=device)
                s_s = sufficiency(model, data, s_pruned, a, b, device=device)
                ours_suff.append(o_s)
                sali_suff.append(s_s)

                o_n = necessity(model, data, o_pruned, a, b, device=device)
                s_n = necessity(model, data, s_pruned, a, b, device=device)
                ours_nec.append(o_n)
                sali_nec.append(s_n)

                o_c = fidelity_plus_continuous(model, data, o_pruned, a, b, device=device)
                s_c = fidelity_plus_continuous(model, data, s_pruned, a, b, device=device)
                ours_cont.append(o_c)
                sali_cont.append(s_c)

                ours_comp.append(_count_components(o_pruned))
                sali_comp.append(_count_components(s_pruned))

                if _target_connected(o_pruned, a, b):
                    ours_tconn += 1
                if _target_connected(s_pruned, a, b):
                    sali_tconn += 1

                if (i + 1) % 20 == 0:
                    print(f"  {i+1}/{n} done")

            except Exception as e:
                print(f"  Edge {i} ({a},{b}) failed: {e}")

        valid = len(ours_suff)
        if valid < 5:
            print(f"  Too few valid edges ({valid}), skipping budget {budget}")
            continue

        oa, sa = np.array(ours_suff), np.array(sali_suff)
        on, sn = np.array(ours_nec), np.array(sali_nec)
        oc, sc = np.array(ours_cont), np.array(sali_cont)
        ocomp, scomp = np.array(ours_comp), np.array(sali_comp)

        def paired_tests(a_vals, b_vals, name, lower_is_better=True):
            diff = a_vals - b_vals
            t_stat, t_p = stats.ttest_rel(a_vals, b_vals)
            try:
                if np.all(diff == 0):
                    w_stat, w_p = 0.0, 1.0
                else:
                    w_stat, w_p = stats.wilcoxon(diff)
            except ValueError:
                w_stat, w_p = 0.0, 1.0

            d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
            result = {
                "ours_mean": float(np.mean(a_vals)),
                "saliency_mean": float(np.mean(b_vals)),
                "ttest_p": float(t_p),
                "wilcoxon_p": float(w_p),
                "cohens_d": float(d),
                "ours_wins": bool(
                    (np.mean(a_vals) < np.mean(b_vals)) if lower_is_better
                    else (np.mean(a_vals) > np.mean(b_vals))
                ),
            }
            sig = "***" if min(t_p, w_p) < 0.001 else "**" if min(t_p, w_p) < 0.01 else "*" if min(t_p, w_p) < 0.05 else ""
            direction = "WIN" if result["ours_wins"] else "LOSS"
            print(f"  {name}: Ours={np.mean(a_vals):.4f} Saliency={np.mean(b_vals):.4f} "
                  f"t_p={t_p:.4f} w_p={w_p:.4f} d={d:.3f} [{direction}{sig}]")
            return result

        print(f"\n  Results (n={valid}, k={budget}):")
        suff_r = paired_tests(oa, sa, "Sufficiency (lower=better)", lower_is_better=True)
        nec_r = paired_tests(on, sn, "Necessity (higher=better)", lower_is_better=False)
        cont_r = paired_tests(oc, sc, "Fidelity+ cont (higher=better)", lower_is_better=False)
        comp_r = paired_tests(ocomp, scomp, "Num components (lower=better)", lower_is_better=True)

        conn_r = {
            "ours_target_connected": ours_tconn,
            "saliency_target_connected": sali_tconn,
            "ours_pct": float(ours_tconn / valid),
            "saliency_pct": float(sali_tconn / valid),
        }
        print(f"  Target connectivity: Ours={ours_tconn}/{valid} ({ours_tconn/valid:.1%}) "
              f"Saliency={sali_tconn}/{valid} ({sali_tconn/valid:.1%})")

        all_results[str(budget)] = {
            "n": valid,
            "budget": budget,
            "sufficiency": suff_r,
            "necessity": nec_r,
            "fidelity_plus_continuous": cont_r,
            "num_components": comp_r,
            "target_connectivity": conn_r,
        }

    output = {
        "dataset": dataset_name,
        "num_test_edges": num_edges,
        "seed": seed,
        "budgets": budgets,
        "methods": ["CoarsenExplainer", "SaliencyExplainer"],
        "results": all_results,
    }

    os.makedirs("results", exist_ok=True)
    out_path = f"results/matched_sparsity_{dataset_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Matched-sparsity comparison")
    parser.add_argument("--dataset", type=str, default="Cora",
                        choices=["Cora", "Citeseer", "PubMed"])
    parser.add_argument("--num_edges", type=int, default=100)
    parser.add_argument("--budgets", type=str, default="5,10,20,50")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_pred", type=float, default=1.0,
                        help="Weight for prediction cost in partition")
    parser.add_argument("--fidelity_threshold", type=float, default=0.8,
                        help="Hard reject threshold for prediction cost")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    run_comparison(args.dataset, args.num_edges, budgets, args.seed,
                   lambda_pred=args.lambda_pred,
                   fidelity_threshold=args.fidelity_threshold)


if __name__ == "__main__":
    main()
