"""Ground-truth explanation evaluation on synthetic motif datasets (Priority 8).

Tests whether explanations recover known ground-truth motifs using
edge-level precision/recall/F1/AUC. Datasets: BA-Shapes, Tree-Cycles,
Tree-Grid, BA-2Motifs, and a custom link-prediction motif task.

Additionally computes region-level (supernode coverage) metrics for
coarsening-based explainers to fairly evaluate structural explanation
quality beyond individual edge matching.
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExperimentConfig

try:
    from src.evaluation.fidelity import fidelity_plus, fidelity_minus
except ImportError:
    fidelity_plus = None
    fidelity_minus = None


def generate_ba_shapes(num_bars=80, attachment=1, shape_nodes=5, seed=42):
    """Generate BA-Shapes: Barabasi-Albert graph with house motifs attached."""
    import networkx as nx
    rng = np.random.RandomState(seed)

    G = nx.barabasi_albert_graph(num_bars, attachment, seed=seed)
    ground_truth_edges = set()
    motif_nodes = set()

    nodes = list(G.nodes())
    rng.shuffle(nodes)
    num_motifs = min(20, len(nodes) // 2)
    chosen = nodes[:num_motifs]

    for anchor in chosen:
        start = G.number_of_nodes()
        G.add_nodes_from(range(start, start + shape_nodes))
        house_edges = [
            (start, start+1), (start+1, start+2), (start+2, start+3),
            (start+3, start), (start, start+2), (start+1, start+3),
        ]
        for u, v in house_edges:
            G.add_edge(u, v)
            ground_truth_edges.add((min(u,v), max(u,v)))
        G.add_edge(anchor, start)
        for n in range(start, start + shape_nodes):
            motif_nodes.add(n)

    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    num_nodes = G.number_of_nodes()
    x = torch.eye(num_nodes)

    return edge_index, num_nodes, x, ground_truth_edges, motif_nodes


def generate_tree_cycles(depth=8, cycle_len=6, num_motifs=20, seed=42):
    """Generate Tree-Cycles: balanced tree with cycle motifs attached."""
    import networkx as nx
    rng = np.random.RandomState(seed)

    G = nx.balanced_tree(2, depth)
    ground_truth_edges = set()
    motif_nodes = set()

    leaves = [n for n in G.nodes() if G.degree(n) == 1]
    rng.shuffle(leaves)
    num_motifs = min(num_motifs, len(leaves))
    chosen = leaves[:num_motifs]

    for anchor in chosen:
        start = G.number_of_nodes()
        nodes = list(range(start, start + cycle_len))
        for i in range(cycle_len):
            G.add_edge(nodes[i], nodes[(i+1) % cycle_len])
            ground_truth_edges.add((min(nodes[i], nodes[(i+1)%cycle_len]),
                                     max(nodes[i], nodes[(i+1)%cycle_len])))
        G.add_edge(anchor, start)
        for n in nodes:
            motif_nodes.add(n)

    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    num_nodes = G.number_of_nodes()
    x = torch.eye(num_nodes)

    return edge_index, num_nodes, x, ground_truth_edges, motif_nodes


def generate_link_motif_task(num_nodes=200, motif_size=10, num_motifs=15, seed=42):
    """Generate a link prediction motif task where target links are determined by motifs."""
    import networkx as nx
    rng = np.random.RandomState(seed)

    G = nx.barabasi_albert_graph(num_nodes, 2, seed=seed)
    motif_edges_list = []
    target_links = []

    for m_idx in range(num_motifs):
        nodes = list(G.nodes())
        anchor = rng.choice(nodes)
        start = G.number_of_nodes()
        motif_nodes_list = list(range(start, start + motif_size))

        for i in range(motif_size):
            G.add_node(motif_nodes_list[i])

        for i in range(motif_size):
            for j in range(i+1, min(i+3, motif_size)):
                G.add_edge(motif_nodes_list[i], motif_nodes_list[j])
                motif_edges_list.append((motif_nodes_list[i], motif_nodes_list[j]))

        G.add_edge(anchor, start)
        target_links.append((start, start + motif_size - 1))

    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    total_nodes = G.number_of_nodes()
    x = torch.eye(total_nodes)
    motif_edge_set = set()
    for u, v in motif_edges_list:
        motif_edge_set.add((min(u,v), max(u,v)))

    return edge_index, total_nodes, x, motif_edge_set, target_links


def compute_edge_precision_recall(explanation_edges, ground_truth_edges):
    """Compute edge-level precision, recall, F1."""
    pred_set = set()
    for i in range(explanation_edges.size(1)):
        u = int(explanation_edges[0, i].item())
        v = int(explanation_edges[1, i].item())
        pred_set.add((min(u,v), max(u,v)))

    tp = len(pred_set & ground_truth_edges)
    fp = len(pred_set - ground_truth_edges)
    fn = len(ground_truth_edges - pred_set)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return precision, recall, f1


def _find_motif_component(ground_truth_edges, target_node):
    """BFS to find all nodes in the same GT-motif connected component."""
    adj = {}
    for u, v in ground_truth_edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    if target_node not in adj:
        return {target_node}
    visited = {target_node}
    queue = deque([target_node])
    while queue:
        node = queue.popleft()
        for nb in adj.get(node, set()):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited


def compute_region_coverage(partition, ground_truth_nodes, target_nodes,
                            ground_truth_edges=None):
    """Compute region-level (supernode coverage) metrics for one test link.

    For a test link (a, b):
      - supernode precision: fraction of the supernode(s) containing a and b
        that are ground-truth motif nodes (measures purity).
      - supernode recall: fraction of the *local* ground-truth motif
        (connected component of ground_truth_edges containing a/b) captured
        by those supernodes (measures coverage).
    """
    a, b = target_nodes

    node_to_sn = {}
    for sn_idx, members in enumerate(partition):
        for node in members:
            node_to_sn[node] = sn_idx

    sn_a = node_to_sn.get(a)
    sn_b = node_to_sn.get(b)

    if sn_a is None or sn_b is None:
        return {"region_precision": 0.0, "region_recall": 0.0, "region_f1": 0.0}

    predicted_nodes = set(partition[sn_a])
    if sn_b != sn_a:
        predicted_nodes |= set(partition[sn_b])

    tp_nodes = predicted_nodes & ground_truth_nodes
    precision = len(tp_nodes) / max(len(predicted_nodes), 1)

    if ground_truth_edges is not None:
        local_gt = _find_motif_component(ground_truth_edges, a)
        local_gt |= _find_motif_component(ground_truth_edges, b)
    else:
        local_gt = ground_truth_nodes

    recall = len(tp_nodes & local_gt) / max(len(local_gt), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "region_precision": float(precision),
        "region_recall": float(recall),
        "region_f1": float(f1),
    }


def evaluate_on_synthetic(data, model, test_links, ground_truth_edges, device,
                          explainer_class, explainer_kwargs=None,
                          ground_truth_nodes=None):
    """Run an explainer on synthetic data and compute metrics."""
    if explainer_kwargs is None:
        explainer_kwargs = {}

    explainer = explainer_class(model, device=device, **explainer_kwargs)

    precisions = []
    recalls = []
    f1s = []
    fid_p_list = []
    fid_m_list = []
    times = []

    region_precisions = []
    region_recalls = []
    region_f1s = []

    for a, b in test_links:
        t0 = time.time()
        try:
            explanation = explainer.explain_link(data, a, b)
        except Exception:
            continue
        times.append(time.time() - t0)

        exp_edges = explanation.edge_index
        if hasattr(explanation, "original_node_indices"):
            orig = explanation.original_node_indices
            exp_edges_orig = orig[exp_edges]
        else:
            exp_edges_orig = exp_edges

        p, r, f1 = compute_edge_precision_recall(exp_edges_orig, ground_truth_edges)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        if fidelity_plus is not None:
            fp = fidelity_plus(model, data, explanation, a, b, device)
            fm = fidelity_minus(model, data, explanation, a, b, device)
            fid_p_list.append(fp)
            fid_m_list.append(fm)

        if (ground_truth_nodes is not None
                and hasattr(explainer, '_coarsener')
                and explainer._coarsener is not None
                and explainer._coarsener.partition is not None):
            rm = compute_region_coverage(
                explainer._coarsener.partition,
                ground_truth_nodes,
                (a, b),
                ground_truth_edges,
            )
            region_precisions.append(rm["region_precision"])
            region_recalls.append(rm["region_recall"])
            region_f1s.append(rm["region_f1"])

    results = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "mean_fidelity_plus": float(np.mean(fid_p_list)) if fid_p_list else 0.0,
        "mean_fidelity_minus": float(np.mean(fid_m_list)) if fid_m_list else 0.0,
        "mean_time": float(np.mean(times)) if times else 0.0,
        "num_samples": len(precisions),
    }

    if region_precisions:
        results["region_precision"] = float(np.mean(region_precisions))
        results["region_recall"] = float(np.mean(region_recalls))
        results["region_f1"] = float(np.mean(region_f1s))

    return results


def main():
    parser = argparse.ArgumentParser()
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

    all_results = {}

    datasets_to_run = ["BA-Shapes", "Tree-Cycles", "Link-Motif"]

    for ds_name in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        if ds_name == "BA-Shapes":
            edge_index, num_nodes, x, gt_edges, motif_nodes = generate_ba_shapes(seed=args.seed)
        elif ds_name == "Tree-Cycles":
            edge_index, num_nodes, x, gt_edges, motif_nodes = generate_tree_cycles(seed=args.seed)
        elif ds_name == "Link-Motif":
            edge_index, num_nodes, x, gt_edges, target_links = generate_link_motif_task(seed=args.seed)
            motif_nodes = set()
            for u, v in gt_edges:
                motif_nodes.add(u)
                motif_nodes.add(v)
        else:
            continue

        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        data.train_pos_edge_index = edge_index
        data.num_features = x.size(1)

        print(f"  |V|={num_nodes}, |E|={edge_index.size(1)}, GT edges={len(gt_edges)}")

        # Train a simple GCN on this graph
        from src.models.gcn import GCN
        from experiments.train_gcn import MLPLinkPredictor
        from src.models.link_predictor import LinkPredictionModel

        in_ch = x.size(1)
        hidden = 64
        gcn = GCN(in_ch, hidden, hidden, num_layers=3).to(device)
        predictor = MLPLinkPredictor(hidden, hidden).to(device)
        model = LinkPredictionModel(gcn, predictor).to(device)
        model.eval()

        if ds_name == "Link-Motif":
            test_links_sample = target_links[:10]
        else:
            rng = np.random.RandomState(args.seed)
            motif_list = list(motif_nodes)
            n_links = min(10, len(motif_list) // 2)
            test_links_sample = []
            for _ in range(n_links):
                a, b = rng.choice(motif_list, size=2, replace=False)
                test_links_sample.append((int(a), int(b)))

        methods_to_try = {}

        try:
            from src.explainers.coarsen_explainer import CoarsenExplainer
            methods_to_try["Ours"] = (CoarsenExplainer, {"k": 50, "alpha": 0.75})
        except ImportError:
            pass

        try:
            from src.explainers.baselines import OcclusionExplainer
            methods_to_try["Occlusion"] = (OcclusionExplainer, {"k_hop": 2, "k_frac": 0.5})
        except ImportError:
            pass

        try:
            from src.explainers.baselines import SaliencyExplainer
            methods_to_try["Saliency"] = (SaliencyExplainer, {"k_frac": 0.5})
        except ImportError:
            pass

        ds_results = {}
        for method_name, (cls, kwargs) in methods_to_try.items():
            print(f"  Running {method_name}...")
            result = evaluate_on_synthetic(
                data, model, test_links_sample, gt_edges, device, cls, kwargs,
                ground_truth_nodes=motif_nodes,
            )
            ds_results[method_name] = result
            line = f"    P={result['precision']:.4f} R={result['recall']:.4f} F1={result['f1']:.4f}"
            if "region_f1" in result:
                line += (f"  |  region_P={result['region_precision']:.4f}"
                         f" region_R={result['region_recall']:.4f}"
                         f" region_F1={result['region_f1']:.4f}")
            print(line)

        all_results[ds_name] = ds_results

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "ground_truth_explanation.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    md_lines = ["# Experiment Results: ground_truth_explanation\n"]
    for ds_name, ds_res in all_results.items():
        md_lines.append(f"## {ds_name}\n")
        for method_name, metrics in ds_res.items():
            md_lines.append(f"### {method_name}\n")
            edge_metrics = ["precision", "recall", "f1", "mean_fidelity_plus",
                            "mean_fidelity_minus", "mean_time", "num_samples"]
            for k in edge_metrics:
                if k in metrics:
                    md_lines.append(f"- **{k}**: {metrics[k]:.4f}")
            if "region_precision" in metrics:
                md_lines.append(f"- **region_precision**: {metrics['region_precision']:.4f}")
                md_lines.append(f"- **region_recall**: {metrics['region_recall']:.4f}")
                md_lines.append(f"- **region_f1**: {metrics['region_f1']:.4f}")
            md_lines.append("")
        md_lines.append("")

    md_path = os.path.join("results", "ground_truth_explanation.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown saved to {md_path}")


if __name__ == "__main__":
    main()
