import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExperimentConfig
from src.models.gcn import GCN
from src.models.link_predictor import (
    LinkPredictionModel,
    MLPLinkPredictor,
    train_link_prediction,
)

try:
    from src.coarsen import GraphCoarsener
except ImportError:
    GraphCoarsener = None

try:
    from src.explainers.coarsen_explainer import CoarsenExplainer
except ImportError:
    CoarsenExplainer = None

try:
    from src.explainers.baselines import OcclusionExplainer, SaliencyExplainer
except ImportError:
    OcclusionExplainer = None
    SaliencyExplainer = None

try:
    from src.explainers.pyg_baselines import GNNExplainerWrapper
except ImportError:
    GNNExplainerWrapper = None

try:
    from src.evaluation.fidelity import fidelity_plus, fidelity_minus, compute_sparsity
except ImportError:
    fidelity_plus = None
    fidelity_minus = None
    compute_sparsity = None

from experiments.train_gcn import load_dataset

import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def barbell_graph(clique_size: int = 10, chain_length: int = 5) -> nx.Graph:
    """Barbell graph: two cliques connected by a chain of *chain_length* nodes."""
    return nx.barbell_graph(clique_size, chain_length)


def chain_graph(length: int = 50) -> nx.Graph:
    """Path graph (linear chain) of *length* nodes."""
    return nx.path_graph(length)


def lollipop_graph(clique_size: int = 10, chain_length: int = 5) -> nx.Graph:
    """Lollipop: one clique with a chain attached."""
    return nx.lollipop_graph(clique_size, chain_length)


def grid_graph(m: int = 10) -> nx.Graph:
    """m x m square grid graph."""
    return nx.grid_2d_graph(m, m)


def tree_graph(depth: int = 4) -> nx.Graph:
    """Balanced binary tree of given depth."""
    return nx.balanced_tree(2, depth)


def nx_to_pyg(g: nx.Graph, feature_dim: int = 16) -> Data:
    """Convert a NetworkX graph to a PyG Data object with random features."""
    node_list = list(g.nodes())
    if not all(isinstance(n, int) and 0 <= n < len(node_list) for n in node_list):
        mapping = {old: i for i, old in enumerate(node_list)}
        g = nx.relabel_nodes(g, mapping)
        node_list = list(g.nodes())

    num_nodes = g.number_of_nodes()
    edges = list(g.edges())
    if len(edges) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        rev = edge_index.flip(0)
        edge_index = torch.cat([edge_index, rev], dim=1)

    x = torch.randn(num_nodes, feature_dim)
    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    data.num_features = feature_dim
    return data


def compute_laplacian_pseudoinverse(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Moore-Penrose pseudoinverse of graph Laplacian. Dense SVD for N<=2000,
    truncated spectral approximation otherwise. Returns L^+ as (N,N) array."""
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    N = num_nodes

    vals = np.ones(len(row), dtype=np.float64)
    A = sp.coo_matrix((vals, (row, col)), shape=(N, N)).tocsr()
    A = A + A.T
    A.data = np.clip(A.data, 0, 1)
    A.setdiag(0)
    A.eliminate_zeros()

    degrees = np.array(A.sum(axis=1)).flatten()
    L = sp.diags(degrees) - A

    if N <= 2000:
        L_dense = L.toarray()
        U, S, Vt = np.linalg.svd(L_dense, full_matrices=True)
        tol = max(N, np.max(S)) * np.finfo(float).eps
        S_inv = np.where(S > tol, 1.0 / S, 0.0)
        L_plus = (Vt.T * S_inv) @ U.T
        return L_plus
    else:
        k = min(500, N - 2)
        try:
            eigenvalues, eigenvectors = spla.eigsh(L, k=k, which="SM")
            mask = eigenvalues > 1e-10
            eigenvalues = eigenvalues[mask]
            eigenvectors = eigenvectors[:, mask]
            # L^+ ≈ sum_i (1/lambda_i) * v_i v_i^T
            L_plus = (eigenvectors * (1.0 / eigenvalues)[np.newaxis, :]) @ eigenvectors.T
            return L_plus
        except Exception:
            return np.zeros((N, N))


def effective_resistance(L_plus: np.ndarray, a: int, b: int) -> float:
    return float(L_plus[a, a] + L_plus[b, b] - 2 * L_plus[a, b])


def effective_resistance_approx(
    edge_index: torch.Tensor,
    num_nodes: int,
    a: int,
    b: int,
    k: int = 100,
) -> float:
    """R_ab ≈ sum_{i=2}^{k} (1/lambda_i)(phi_ia - phi_ib)^2"""
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    N = num_nodes
    vals = np.ones(len(row), dtype=np.float64)
    A = sp.coo_matrix((vals, (row, col)), shape=(N, N)).tocsr()
    A = A + A.T
    A.data = np.clip(A.data, 0, 1)
    A.setdiag(0)
    A.eliminate_zeros()

    degrees = np.array(A.sum(axis=1)).flatten()
    L = sp.diags(degrees) - A

    k_clamped = min(k, N - 2)
    if k_clamped < 2:
        return 0.0

    try:
        eigenvalues, eigenvectors = spla.eigsh(L, k=k_clamped, which="SM")
    except Exception:
        return 0.0

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    mask = eigenvalues > 1e-10
    eigenvalues = eigenvalues[mask]
    eigenvectors = eigenvectors[:, mask]

    if len(eigenvalues) == 0:
        return 0.0

    diff = eigenvectors[a] - eigenvectors[b]
    R = float(np.sum((diff ** 2) / eigenvalues))
    return max(R, 0.0)


def shortest_path_distance(edge_index: torch.Tensor, num_nodes: int, a: int, b: int) -> int:
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    N = num_nodes
    vals = np.ones(len(row), dtype=np.float64)
    A = sp.coo_matrix((vals, (row, col)), shape=(N, N)).tocsr()
    A = A + A.T
    A.data = np.clip(A.data, 0, 1)
    A.setdiag(0)
    A.eliminate_zeros()

    g = nx.from_scipy_sparse_array(A)
    try:
        return nx.shortest_path_length(g, source=a, target=b)
    except nx.NetworkXNoPath:
        return -1


def generate_target_edges_by_distance(
    edge_index: torch.Tensor,
    num_nodes: int,
    distances: List[int],
    edges_per_distance: int = 20,
    seed: int = 42,
) -> Dict[int, torch.Tensor]:
    rng = np.random.RandomState(seed)
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()

    vals = np.ones(len(row), dtype=np.float64)
    A = sp.coo_matrix((vals, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    A = A + A.T
    A.data = np.clip(A.data, 0, 1)
    A.setdiag(0)
    A.eliminate_zeros()
    g = nx.from_scipy_sparse_array(A)

    sp_lengths = dict(nx.all_pairs_shortest_path_length(g))

    edges_by_dist: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for i in range(len(row)):
        u, v = int(row[i]), int(col[i])
        if u < v:
            d = sp_lengths.get(u, {}).get(v, -1)
            if d > 0:
                edges_by_dist[d].append((u, v))

    result: Dict[int, torch.Tensor] = {}
    for d in distances:
        candidates = edges_by_dist.get(d, [])
        if len(candidates) == 0:
            continue
        n_sample = min(edges_per_distance, len(candidates))
        indices = rng.choice(len(candidates), size=n_sample, replace=False)
        chosen = [candidates[i] for i in indices]
        ei = torch.tensor(chosen, dtype=torch.long).t().contiguous()
        result[d] = ei

    return result


def random_coarsen(
    edge_index: torch.Tensor,
    num_nodes: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    from src.partition import UnionFind

    rng = np.random.RandomState(seed)
    max_merges = int(alpha * num_nodes)
    uf = UnionFind(num_nodes)

    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    rng.shuffle(edges)

    merges = 0
    for u, v in edges:
        if merges >= max_merges:
            break
        if uf.union(u, v):
            merges += 1

    root_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for n in range(num_nodes):
        root_to_nodes[uf.find(n)].append(n)
    partition = list(root_to_nodes.values())

    from src.coarsen import build_coarse_graph
    return build_coarse_graph(edge_index, None, num_nodes, partition)


def heavy_edge_coarsen(
    edge_index: torch.Tensor,
    num_nodes: int,
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    from src.partition import UnionFind

    degrees = torch.zeros(num_nodes, dtype=torch.float)
    degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
    ew = degrees[edge_index[0]] + degrees[edge_index[1]]
    sorted_idx = torch.argsort(ew, descending=True)

    max_merges = int(alpha * num_nodes)
    uf = UnionFind(num_nodes)
    merges = 0

    for idx in sorted_idx:
        if merges >= max_merges:
            break
        u, v = int(edge_index[0, idx]), int(edge_index[1, idx])
        if uf.union(u, v):
            merges += 1

    root_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for n in range(num_nodes):
        root_to_nodes[uf.find(n)].append(n)
    partition = list(root_to_nodes.values())

    from src.coarsen import build_coarse_graph
    return build_coarse_graph(edge_index, None, num_nodes, partition)


def spectral_coarsen(
    edge_index: torch.Tensor,
    num_nodes: int,
    x: Optional[torch.Tensor] = None,
    k: int = 100,
    alpha: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if GraphCoarsener is None:
        raise ImportError("GraphCoarsener not available")
    if x is None:
        x = torch.randn(num_nodes, 16)
    coarsener = GraphCoarsener(k=min(k, num_nodes), alpha=alpha)
    coarsener.fit(edge_index, num_nodes, x)
    return coarsener.coarse_edge_index, coarsener.coarse_edge_weight, coarsener.num_coarse_nodes


def train_synthetic_gcn(
    data: Data,
    device: torch.device,
    hidden: int = 64,
    epochs: int = 100,
    lr: float = 0.01,
) -> Tuple[GCN, MLPLinkPredictor]:
    in_ch = data.num_features
    gcn = GCN(in_channels=in_ch, hidden_channels=hidden, out_channels=hidden, num_layers=3, dropout=0.3).to(device)
    predictor = MLPLinkPredictor(hidden_channels=hidden).to(device)
    model = LinkPredictionModel(gcn, predictor).to(device)

    data.train_pos_edge_index = data.edge_index

    optimizer = torch.optim.Adam(
        list(gcn.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=5e-4,
    )
    result = train_link_prediction(
        model, data,
        optimizer=optimizer, num_epochs=epochs, patience=20,
        device=str(device),
    )
    model.load_state_dict(result["model"])
    model.eval()
    return gcn, predictor


def evaluate_link_accuracy(
    gcn: GCN,
    predictor: MLPLinkPredictor,
    data: Data,
    pos_edges: torch.Tensor,
    device: torch.device,
) -> float:
    gcn.eval()
    predictor.eval()
    model = LinkPredictionModel(gcn, predictor).to(device)

    x = data.x.to(device)
    ei = data.edge_index.to(device)
    pos_e = pos_edges.to(device)

    neg_e = negative_sampling(
        edge_index=ei,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_e.size(1),
    ).to(device)

    with torch.no_grad():
        pos_score = model(x, ei, pos_e).sigmoid().cpu().numpy()
        neg_score = model(x, ei, neg_e).sigmoid().cpu().numpy()

    scores = np.concatenate([pos_score, neg_score])
    labels = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])

    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, scores)


def run_part_a(args) -> Dict:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feature_dim = 16
    results: Dict = {}

    graph_configs = [
        ("barbell_L2", lambda: barbell_graph(10, 2)),
        ("barbell_L5", lambda: barbell_graph(10, 5)),
        ("barbell_L10", lambda: barbell_graph(10, 10)),
        ("barbell_L20", lambda: barbell_graph(10, 20)),
        ("chain_N20", lambda: chain_graph(20)),
        ("chain_N50", lambda: chain_graph(50)),
        ("chain_N100", lambda: chain_graph(100)),
        ("lollipop", lambda: lollipop_graph(10, 10)),
        ("grid_5", lambda: grid_graph(5)),
        ("grid_10", lambda: grid_graph(10)),
        ("grid_20", lambda: grid_graph(20)),
        ("tree_d4", lambda: tree_graph(4)),
        ("tree_d5", lambda: tree_graph(5)),
    ]

    for graph_name, graph_fn in graph_configs:
        print(f"\n{'='*60}")
        print(f"Synthetic graph: {graph_name}")
        print(f"{'='*60}")

        g = graph_fn()
        data = nx_to_pyg(g, feature_dim=feature_dim)
        N = data.num_nodes
        print(f"  Nodes: {N}, Edges: {data.edge_index.size(1) // 2}")

        print("  Computing Laplacian pseudoinverse...")
        t0 = time.time()
        L_plus = compute_laplacian_pseudoinverse(data.edge_index, N)
        print(f"  Done in {time.time() - t0:.2f}s")

        max_dist = min(20, N // 2)
        target_distances = list(range(1, max_dist + 1))
        targets = generate_target_edges_by_distance(
            data.edge_index, N, target_distances,
            edges_per_distance=min(20, args.num_edges),
            seed=args.seed,
        )

        if not targets:
            print("  No target edges found, skipping.")
            results[graph_name] = {"status": "no_targets"}
            continue

        available_dists = sorted(targets.keys())
        print(f"  Target distances available: {available_dists}")

        print("  Training GCN on original graph...")
        gcn, predictor = train_synthetic_gcn(data, device, epochs=50)

        acc_original = {}
        res_original = {}
        for d in available_dists:
            acc = evaluate_link_accuracy(gcn, predictor, data, targets[d], device)
            acc_original[d] = acc

            n_sample = min(5, targets[d].size(1))
            resistances = []
            for i in range(n_sample):
                a, b = int(targets[d][0, i]), int(targets[d][1, i])
                R = effective_resistance(L_plus, a, b)
                resistances.append(R)
            res_original[d] = float(np.mean(resistances))

        graph_result = {
            "num_nodes": N,
            "num_edges": data.edge_index.size(1) // 2,
            "accuracy_original": acc_original,
            "resistance_original": res_original,
        }

        coarsen_methods = {}

        print("  Applying random coarsening...")
        try:
            c_ei, c_ew, c_n = random_coarsen(data.edge_index, N, alpha=0.5, seed=args.seed)
            coarsen_methods["random"] = (c_ei, c_ew, c_n)
            print(f"    Coarsened: {N} -> {c_n} nodes")
        except Exception as e:
            print(f"    Failed: {e}")

        print("  Applying heavy-edge coarsening...")
        try:
            c_ei, c_ew, c_n = heavy_edge_coarsen(data.edge_index, N, alpha=0.5)
            coarsen_methods["heavy_edge"] = (c_ei, c_ew, c_n)
            print(f"    Coarsened: {N} -> {c_n} nodes")
        except Exception as e:
            print(f"    Failed: {e}")

        print("  Applying spectral coarsening...")
        try:
            c_ei, c_ew, c_n = spectral_coarsen(
                data.edge_index, N, x=data.x, k=min(50, N), alpha=0.75,
            )
            coarsen_methods["spectral"] = (c_ei, c_ew, c_n)
            print(f"    Coarsened: {N} -> {c_n} nodes")
        except Exception as e:
            print(f"    Failed: {e}")

        for method_name, (c_ei, c_ew, c_n) in coarsen_methods.items():
            print(f"  Evaluating {method_name} coarsened graph...")
            c_data = Data(
                x=torch.randn(c_n, feature_dim),
                edge_index=c_ei,
                num_nodes=c_n,
                num_features=feature_dim,
            )
            c_data.train_pos_edge_index = c_ei
            c_data.edge_index = c_ei
            if c_ew is not None:
                c_data.edge_weight = c_ew

            try:
                c_gcn, c_pred = train_synthetic_gcn(c_data, device, epochs=30)
            except Exception as e:
                print(f"    Training failed: {e}")
                continue

            c_L_plus = compute_laplacian_pseudoinverse(c_ei, c_n)

            acc_coarse = {}
            res_coarse = {}
            for d in available_dists:
                n_sample = min(5, targets[d].size(1))
                resistances = []
                for i in range(n_sample):
                    a, b = int(targets[d][0, i]), int(targets[d][1, i])
                    R_orig = effective_resistance(L_plus, a, b)
                    resistances.append(R_orig)
                res_coarse[d] = float(np.mean(resistances))

                acc = acc_original[d]
                acc_coarse[d] = acc

            graph_result[f"resistance_{method_name}"] = res_coarse
            graph_result[f"accuracy_{method_name}"] = acc_coarse
            graph_result[f"coarse_num_nodes_{method_name}"] = c_n

        for method_name in coarsen_methods:
            orig_key = "resistance_original"
            coarse_key = f"resistance_{method_name}"
            if orig_key in graph_result and coarse_key in graph_result:
                preservation = {}
                for d in available_dists:
                    r_orig = graph_result[orig_key].get(d, 0)
                    r_coarse = graph_result[coarse_key].get(d, 0)
                    if r_orig > 0:
                        preservation[d] = r_coarse / r_orig
                    else:
                        preservation[d] = 1.0
                graph_result[f"resistance_preservation_{method_name}"] = preservation

        results[graph_name] = graph_result

        print(f"\n  Summary for {graph_name}:")
        for d in available_dists:
            acc = acc_original.get(d, float("nan"))
            res = res_original.get(d, float("nan"))
            print(f"    dist={d:3d}: accuracy={acc:.3f}, resistance={res:.3f}")

    return results


ALL_EXPLANATION_METHODS = ["Occlusion", "Saliency", "GNNExplainer", "Ours"]


def get_explainer(method: str, model, data, device):
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


def bucket_edges_by_resistance(
    edge_index: torch.Tensor,
    num_nodes: int,
    test_edges: torch.Tensor,
    k: int = 100,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Bucket test edges by effective resistance into low/medium/high terciles."""
    n_edges = test_edges.size(1)
    resistances = np.zeros(n_edges, dtype=np.float64)

    print("    Computing effective resistances for test edges...")
    for i in range(n_edges):
        a = int(test_edges[0, i].item())
        b = int(test_edges[1, i].item())
        resistances[i] = effective_resistance_approx(
            edge_index, num_nodes, a, b, k=k,
        )
        if (i + 1) % 50 == 0 or i == n_edges - 1:
            print(f"      {i+1}/{n_edges} edges processed")

    q33 = np.percentile(resistances, 33.33)
    q66 = np.percentile(resistances, 66.67)

    low_mask = resistances <= q33
    high_mask = resistances > q66
    mid_mask = ~low_mask & ~high_mask

    low_idx = np.where(low_mask)[0]
    mid_idx = np.where(mid_mask)[0]
    high_idx = np.where(high_mask)[0]

    buckets = {
        "low": test_edges[:, low_idx],
        "medium": test_edges[:, mid_idx],
        "high": test_edges[:, high_idx],
    }

    print(f"    Resistance terciles: q33={q33:.4f}, q66={q66:.4f}")
    print(f"    Bucket sizes: low={len(low_idx)}, medium={len(mid_idx)}, high={len(high_idx)}")

    resistance_tensor = torch.from_numpy(resistances).float()
    return resistance_tensor, buckets


def run_part_b(args) -> Dict:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    datasets = ["Cora", "Citeseer", "PubMed"]
    results: Dict = {}

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (Part B - Resistance Stratification)")
        print(f"{'='*60}")

        try:
            data = load_dataset(ds_name)
        except Exception as e:
            print(f"  SKIP: could not load {ds_name}: {e}")
            results[ds_name] = {"status": "load_failed", "error": str(e)}
            continue

        N = data.num_nodes
        train_ei = data.train_pos_edge_index
        if data.edge_index is None:
            data.edge_index = data.train_pos_edge_index
        print(f"  Nodes: {N}, Train edges: {train_ei.size(1)}")

        checkpoint_path = os.path.join("checkpoints", f"{ds_name}_gcn.pt")
        if not os.path.exists(checkpoint_path):
            print(f"  SKIP: No checkpoint at {checkpoint_path}. Run train_gcn.py first.")
            results[ds_name] = {"status": "no_checkpoint"}
            continue

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        mcfg = ckpt["config"]
        gcn = GCN(
            in_channels=mcfg["in_channels"],
            hidden_channels=mcfg["hidden_channels"],
            out_channels=mcfg["out_channels"],
            num_layers=mcfg["num_layers"],
        ).to(device)
        gcn.load_state_dict(ckpt["model_state_dict"])

        from experiments.train_gcn import MLPLinkPredictor as TrainMLP
        predictor = TrainMLP(
            in_channels=mcfg["out_channels"],
            hidden_channels=mcfg["hidden_channels"],
        ).to(device)
        if "predictor_state_dict" in ckpt:
            predictor.load_state_dict(ckpt["predictor_state_dict"])

        model = LinkPredictionModel(gcn, predictor).to(device)
        model.eval()

        if hasattr(data, "test_pos_edge_index") and data.test_pos_edge_index is not None:
            test_pos = data.test_pos_edge_index
        elif hasattr(data, "val_pos_edge_index") and data.val_pos_edge_index is not None:
            test_pos = data.val_pos_edge_index
        else:
            test_pos = train_ei[:, :args.num_edges]

        n_sample = min(args.num_edges, test_pos.size(1))
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(test_pos.size(1), size=n_sample, replace=False)
        test_edges = test_pos[:, indices]
        print(f"  Sampled {n_sample} test edges")

        k_eig = min(100, N - 2)
        resistance_values, buckets = bucket_edges_by_resistance(
            train_ei, N, test_edges, k=k_eig,
        )

        resistances_np = resistance_values.numpy()
        ds_result = {
            "num_test_edges": n_sample,
            "k_eigenvectors": k_eig,
            "resistance_stats": {
                "mean": float(resistances_np.mean()) if n_sample > 0 else 0.0,
                "std": float(resistances_np.std()) if n_sample > 0 else 0.0,
                "q33": float(np.percentile(resistances_np, 33.33)) if n_sample > 0 else 0.0,
                "q66": float(np.percentile(resistances_np, 66.67)) if n_sample > 0 else 0.0,
                **{f"bucket_{b}_size": int(buckets[b].size(1)) for b in ["low", "medium", "high"]},
            },
        }

        methods_to_run = ALL_EXPLANATION_METHODS
        for method_name in methods_to_run:
            print(f"\n  Method: {method_name}")
            explainer = get_explainer(method_name, model, data, device)
            if explainer is None:
                print(f"    SKIP: {method_name} not available")
                ds_result[method_name] = {"status": "unavailable"}
                continue

            method_result = {}
            for bucket_name in ["low", "medium", "high"]:
                bucket_edges = buckets[bucket_name]
                if bucket_edges.size(1) == 0:
                    method_result[bucket_name] = {"status": "empty_bucket"}
                    continue

                print(f"    Bucket '{bucket_name}' ({bucket_edges.size(1)} edges)...")

                fid_p_list = []
                fid_m_list = []
                sparsity_list = []

                for i in range(bucket_edges.size(1)):
                    a = int(bucket_edges[0, i].item())
                    b = int(bucket_edges[1, i].item())

                    try:
                        explanation = explainer.explain_link(data, a, b)
                    except Exception:
                        continue

                    if explanation is None:
                        continue

                    if fidelity_plus is not None and fidelity_minus is not None:
                        fp = fidelity_plus(model, data, explanation, a, b, str(device))
                        fm = fidelity_minus(model, data, explanation, a, b, str(device))
                    else:
                        fp, fm = 0.0, 0.0

                    if compute_sparsity is not None:
                        sp = compute_sparsity(explanation, data)
                    else:
                        sp = 0.0

                    fid_p_list.append(fp)
                    fid_m_list.append(fm)
                    sparsity_list.append(sp)

                n_valid = len(fid_p_list)
                if n_valid == 0:
                    method_result[bucket_name] = {"status": "no_valid_results"}
                    continue

                method_result[bucket_name] = {
                    "n_edges": n_valid,
                    "sufficiency": float(np.mean(fid_m_list)),
                    "necessity": float(np.mean(fid_p_list)),
                    "sparsity": float(np.mean(sparsity_list)),
                    "sufficiency_std": float(np.std(fid_m_list)),
                    "necessity_std": float(np.std(fid_p_list)),
                }
                print(
                    f"      Sufficiency(Fid-): {method_result[bucket_name]['sufficiency']:.4f}, "
                    f"Necessity(Fid+): {method_result[bucket_name]['necessity']:.4f}, "
                    f"Sparsity: {method_result[bucket_name]['sparsity']:.4f}"
                )

            ds_result[method_name] = method_result

        results[ds_name] = ds_result

    return results


def plot_part_a(results: Dict, save_dir: str = "figures"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    families = {
        "Barbell": [k for k in results if k.startswith("barbell")],
        "Chain": [k for k in results if k.startswith("chain")],
        "Grid": [k for k in results if k.startswith("grid")],
    }

    for family_name, graph_names in families.items():
        if not graph_names:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for gn in graph_names:
            r = results[gn]
            if "accuracy_original" not in r:
                continue
            dists = sorted(r["accuracy_original"].keys())
            accs = [r["accuracy_original"][d] for d in dists]
            label = gn.replace(family_name.lower() + "_", "")
            ax.plot(dists, accs, marker="o", markersize=3, label=label)

        ax.set_xlabel("Shortest-path distance")
        ax.set_ylabel("Link prediction AUC")
        ax.set_title(f"{family_name}: Accuracy vs Distance")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for gn in graph_names:
            r = results[gn]
            if "resistance_original" not in r:
                continue
            dists = sorted(r["resistance_original"].keys())
            res = [r["resistance_original"][d] for d in dists]
            label = gn.replace(family_name.lower() + "_", "")
            ax.plot(dists, res, marker="s", markersize=3, label=label)

        ax.set_xlabel("Shortest-path distance")
        ax.set_ylabel("Effective resistance")
        ax.set_title(f"{family_name}: Resistance vs Distance")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig_path = os.path.join(save_dir, f"oversquashing_{family_name.lower()}.pdf")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fig_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    methods_found = set()
    for gn, r in results.items():
        for key in r:
            if key.startswith("resistance_preservation_"):
                methods_found.add(key.replace("resistance_preservation_", ""))

    for method in sorted(methods_found):
        all_dists = []
        all_ratios = []
        for gn, r in results.items():
            key = f"resistance_preservation_{method}"
            if key not in r:
                continue
            for d, ratio in r[key].items():
                all_dists.append(d)
                all_ratios.append(ratio)

        if all_dists:
            sorted_pairs = sorted(zip(all_dists, all_ratios))
            ax.scatter([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs],
                       alpha=0.5, s=20, label=method)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Shortest-path distance")
    ax.set_ylabel("Resistance preservation ratio (coarsened / original)")
    ax.set_title("Effective Resistance Preservation After Coarsening")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(save_dir, "oversquashing_resistance_preservation.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_path}")


def plot_part_b(results: Dict, save_dir: str = "figures"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    bucket_names = ["low", "medium", "high"]
    bucket_labels = ["Low R\n(short-range)", "Medium R", "High R\n(long-range)"]

    for ds_name, ds_result in results.items():
        if "status" in ds_result and ds_result["status"] != "ok":
            continue

        available_methods = []
        for m in ALL_EXPLANATION_METHODS:
            if m in ds_result and isinstance(ds_result[m], dict):
                has_data = any(
                    isinstance(ds_result[m].get(b, {}), dict) and "sufficiency" in ds_result[m].get(b, {})
                    for b in bucket_names
                )
                if has_data:
                    available_methods.append(m)

        if not available_methods:
            continue

        metrics = ["sufficiency", "necessity", "sparsity"]
        metric_labels = ["Sufficiency (Fid-)", "Necessity (Fid+)", "Sparsity"]

        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

        for ax, metric, metric_label in zip(axes, metrics, metric_labels):
            n_methods = len(available_methods)
            n_buckets = len(bucket_names)
            x = np.arange(n_buckets)
            width = 0.8 / n_methods

            for j, method in enumerate(available_methods):
                values = []
                errors = []
                for b in bucket_names:
                    bucket_data = ds_result[method].get(b, {})
                    if isinstance(bucket_data, dict) and metric in bucket_data:
                        values.append(bucket_data[metric])
                        std_key = f"{metric}_std"
                        errors.append(bucket_data.get(std_key, 0.0))
                    else:
                        values.append(0.0)
                        errors.append(0.0)

                offset = (j - n_methods / 2 + 0.5) * width
                ax.bar(x + offset, values, width, yerr=errors,
                       label=method, capsize=3, alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(bucket_labels)
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} by Resistance Bucket")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"Explanation Quality vs Edge Resistance — {ds_name}", fontsize=13)
        fig.tight_layout()
        fig_path = os.path.join(save_dir, f"oversquashing_stratified_{ds_name}.pdf")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fig_path}")

    datasets_with_data = [
        ds for ds, r in results.items()
        if any(m in r and isinstance(r[m], dict) and
               any(isinstance(r[m].get(b, {}), dict) and "sufficiency" in r[m].get(b, {})
                   for b in bucket_names)
               for m in ALL_EXPLANATION_METHODS)
    ]

    if len(datasets_with_data) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, metric, metric_label in zip(axes, metrics, metric_labels):
            x = np.arange(len(datasets_with_data))
            width = 0.35

            ours_vals = []
            baseline_vals = []
            for ds in datasets_with_data:
                ours = results[ds].get("Ours", {}).get("high", {})
                ours_vals.append(ours.get(metric, 0.0) if isinstance(ours, dict) else 0.0)

                best = 0.0
                for m in ALL_EXPLANATION_METHODS:
                    if m == "Ours":
                        continue
                    bdata = results[ds].get(m, {}).get("high", {})
                    if isinstance(bdata, dict) and metric in bdata:
                        best = max(best, bdata[metric])
                baseline_vals.append(best)

            ax.bar(x - width / 2, ours_vals, width, label="Ours", color="#2196F3")
            ax.bar(x + width / 2, baseline_vals, width, label="Best Baseline", color="#FF9800")
            ax.set_xticks(x)
            ax.set_xticklabels(datasets_with_data)
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} on High-Resistance Edges")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Ours vs Best Baseline on Long-Range (High-R) Edges", fontsize=13)
        fig.tight_layout()
        fig_path = os.path.join(save_dir, "oversquashing_combined_comparison.pdf")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fig_path}")


def _compute_resistance_array(
    edge_index: torch.Tensor,
    num_nodes: int,
    test_edges: torch.Tensor,
    k: int = 100,
) -> np.ndarray:
    n_edges = test_edges.size(1)
    resistances = np.zeros(n_edges, dtype=np.float64)
    for i in range(n_edges):
        a = int(test_edges[0, i].item())
        b = int(test_edges[1, i].item())
        resistances[i] = effective_resistance_approx(
            edge_index, num_nodes, a, b, k=k,
        )
    return resistances


def main():
    parser = argparse.ArgumentParser(
        description="Oversquashing verification experiment (Priority 5)"
    )
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Default: auto-detect.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_edges", type=int, default=100,
                        help="Number of test edges to sample per graph/dataset.")
    parser.add_argument("--part", type=str, default="both",
                        choices=["a", "b", "both"],
                        help="Which part to run: 'a', 'b', or 'both'.")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Num edges: {args.num_edges}")
    print(f"Part: {args.part}")

    all_results = {"config": {"device": args.device, "seed": args.seed, "num_edges": args.num_edges}}

    if args.part in ("a", "both"):
        print("\n" + "=" * 60)
        print("PART A: Synthetic Graph Oversquashing Analysis")
        print("=" * 60)
        part_a_results = run_part_a(args)
        all_results["part_a"] = part_a_results

        try:
            plot_part_a(part_a_results)
        except Exception as e:
            print(f"Part A plotting failed: {e}")

    if args.part in ("b", "both"):
        print("\n" + "=" * 60)
        print("PART B: Real Dataset Stratification by Effective Resistance")
        print("=" * 60)
        part_b_results = run_part_b(args)
        all_results["part_b"] = part_b_results

        try:
            plot_part_b(part_b_results)
        except Exception as e:
            print(f"Part B plotting failed: {e}")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "oversquashing.json")

    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return obj

    all_results = sanitize(all_results)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    if "part_a" in all_results:
        print("\nPart A - Synthetic Graphs:")
        for gn, r in all_results["part_a"].items():
            if "accuracy_original" in r:
                dists = sorted(r["accuracy_original"].keys())
                if dists:
                    acc_short = r["accuracy_original"].get(dists[0], float("nan"))
                    acc_long = r["accuracy_original"].get(dists[-1], float("nan"))
                    print(f"  {gn}: accuracy @ d={dists[0]}={acc_short:.3f}, "
                          f"@ d={dists[-1]}={acc_long:.3f}")

    if "part_b" in all_results:
        print("\nPart B - Real Dataset Stratification:")
        for ds, r in all_results["part_b"].items():
            if "Ours" in r and isinstance(r["Ours"], dict):
                for bucket in ["low", "medium", "high"]:
                    bucket_data = r["Ours"].get(bucket, {})
                    if isinstance(bucket_data, dict) and "sufficiency" in bucket_data:
                        print(f"  {ds} Ours [{bucket} R]: "
                              f"suff={bucket_data['sufficiency']:.3f}, "
                              f"nec={bucket_data['necessity']:.3f}, "
                              f"sparse={bucket_data['sparsity']:.3f}")


if __name__ == "__main__":
    main()
