# GraphCoarsening

GraphCoarsening is research code for graph neural network link prediction explanation through prediction-guided pathway-calibrated graph coarsening. The project studies how coarsened structural pathways can identify compact, coherent, and predictive edge explanations by combining spectral graph structure, prediction sensitivity, gradient saliency, and pathway-level redundancy calibration.

## Method

### Prediction-Guided Pathway-Calibrated Gradient Selection

Prediction-Guided Pathway-Calibrated Gradient Selection uses graph coarsening to define structural pathways, then rescales individual edge gradients by pathway-level occlusion effects. The method first builds a prediction-aware partition of the graph, maps local subgraph edges into supernode-pair pathways, measures how much each pathway contributes as a group, and selects edges whose calibrated saliency remains high after accounting for redundancy.

### Algorithm Summary

1. Compute gradient saliency `|∂f/∂w_e|` for all candidate edges.
2. Build a prediction-guided partition with merge cost `C(e) = ρ̂(e) + λ_pred Φ(a,b)`.
3. Reject merges when `Φ(a,b)` exceeds the fidelity threshold.
4. Preserve protected nodes, including 1-hop neighbors of the target, as singleton supernodes.
5. Map subgraph edges to pathways defined by supernode pairs.
6. For each pathway with at least two edges, compute a group occlusion effect and calibration factor.
7. Score each edge as `score(e) = |gradient(e)| × CF(pathway(e))`, then select the top-k calibrated edges.

### Prediction-Guided Partition

The prediction-guided partition extends spectral coarsening by penalizing merges that would collapse prediction-critical endpoints. For an edge `e = (a,b)`, the merge cost is

```text
C(e) = ρ̂(e) + λ_pred Φ(a,b)
Φ(a,b) = ĝ̂(a) · ĝ̂(b)
```

Here, `ρ̂(e)` is the normalized spectral perturbation score, `ĝ̂(a)` and `ĝ̂(b)` are normalized endpoint gradient importances, and `λ_pred` controls the strength of prediction guidance. A hard constraint skips the merge when `Φ(a,b) > fidelity_threshold`, which prevents the partition from merging two highly prediction-critical nodes. The default configuration uses `λ_pred = 1.0` and `fidelity_threshold = 0.8` for Cora and Citeseer, while PubMed uses `λ_pred = 2.0` and `fidelity_threshold = 0.95` because of its larger 2-hop subgraphs.

### Pathway Calibration

Pathway calibration measures whether individual edge gradients overstate the importance of edges that share the same message-passing route. For a pathway `p`, the calibration factor is

```text
CF(p) = group_effect(p) / Σ|gradient(e)|, for e in p
```

The calibrated edge score is then `|gradient(e)| × CF(pathway(e))`. This rescales edge-level saliency by the observed group-level effect of the pathway. The redundancy diagnostic confirms that this correction is necessary: the mean redundancy ratio is `R = 0.61`, meaning the group effect is only 61% of the sum of individual gradients, and 97.5% of pathways are sub-additive. In practice, raw gradient saliency systematically overestimates group importance by about 39%.

## Datasets

### Primary Citation Networks

- **Cora**: citation network used for link prediction explanation and matched-sparsity evaluation.
- **Citeseer**: citation network used for link prediction explanation and matched-sparsity evaluation.
- **PubMed**: larger citation network used to test scaling behavior and stronger prediction guidance.

### Synthetic Ground Truth Datasets

- **BA-Shapes**: motif-augmented graph with ground truth structures for explanation evaluation.
- **Tree-Cycles**: synthetic graph with cycle motifs attached to tree structures.
- **Link-Motif**: synthetic link prediction benchmark with known motif-level explanatory structure.

The configuration also includes larger scalability targets, including Coauthor-CS, Coauthor-Physics, Amazon-Computers, ogbl-ppa, ogbl-collab, and ogbl-ddi.

## Results

The main matched-sparsity comparison evaluates calibrated coarsening explanations against saliency under the same edge budget across Cora, Citeseer, and PubMed.

| Result Type | Wins |
|-------------|------|
| Fidelity wins | 17 |
| Structural coherence wins | 18 |
| Total wins | 35 total wins, 17 fidelity and 18 structural, across 3 datasets |

Key findings:

- **Cora**: 17 total wins, including 11 significant fidelity wins and 6 structural wins. The method wins on sufficiency at all tested budgets and on necessity for most budgets.
- **Citeseer**: 10 total wins, including 4 conservative fidelity wins and 6 structural wins. The strongest gains appear on necessity at moderate-to-large budgets.
- **PubMed**: 8 total wins, including 2 fidelity wins and 6 structural wins. Large-budget necessity and continuous fidelity improve when stronger prediction guidance is used.
- Across all three datasets, pathway-calibrated explanations produce significantly fewer connected components than saliency, with structural coherence wins at every tested budget.
- The method is most effective at moderate-to-large budgets, where removing pathway-calibrated edges causes larger prediction drops across the citation networks.

## Project Structure

```text
src/
├── coarsen.py              # Core coarsening pipeline (Algorithm 1)
├── partition.py            # Union-Find, node partition, prediction-guided partition
├── spectral.py             # Eigenvalue computation, perturbation scores
├── models/
│   ├── gcn.py              # GCN, GraphSAGE encoders
│   └── link_predictor.py   # Link prediction heads and training
├── evaluation/
│   ├── fidelity.py         # Binary fidelity metrics
│   └── comprehensive_metrics.py  # 7-metric evaluation suite
├── explainers/
│   ├── base.py             # BaseExplainer ABC
│   ├── baselines.py        # Occlusion, Saliency explainers
│   ├── coarsen_explainer.py  # Main CoarsenExplainer
│   ├── coarsening_baselines.py  # Coarsening baselines
│   └── pyg_baselines.py    # PyG baseline explainers
config.py                   # Configuration dataclasses
experiments/                # 16 experiment scripts
results/                    # JSON + Markdown result files
```

## Experiment Scripts

The `experiments/` directory contains the main entry points for training, evaluation, ablation, profiling, and result conversion.

| Script | Purpose |
|--------|---------|
| `train_gcn.py` | Train GCN models on citation networks |
| `run_explanations.py` | Explanation fidelity evaluation |
| `run_runtime.py` | Runtime scaling benchmarks |
| `run_ablation.py` | Coarsening ratio ablation |
| `run_baselines_comparison.py` | Baseline explainer comparison |
| `run_comprehensive_metrics.py` | 7-metric evaluation |
| `run_pareto.py` | Pareto curves at matched budgets |
| `run_oversmoothing.py` | Oversmoothing depth sweep |
| `run_oversquashing.py` | Oversquashing analysis |
| `run_hyperparam_ablation.py` | Hyperparameter sensitivity |
| `run_refinement_ablation.py` | Refinement strategy comparison |
| `run_ground_truth.py` | Ground truth evaluation |
| `run_profiling.py` | Memory and time profiling |
| `run_multibackbone.py` | Multi-backbone consistency |
| `run_matched_sparsity_comparison.py` | Main matched-sparsity comparison |
| `run_convert_results_to_md.py` | JSON to Markdown converter |

The full experiment pipeline is available through `run_all_experiments.sh`. It trains citation-network GCN checkpoints when needed, runs explanation fidelity, runtime, ablation, Pareto, oversmoothing, oversquashing, hyperparameter, refinement, ground truth, profiling, and multi-backbone experiments, then converts JSON outputs into Markdown reports.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train a GCN model on one citation network:

```bash
python experiments/train_gcn.py --dataset Cora --epochs 100
```

Run the full experiment pipeline:

```bash
bash run_all_experiments.sh
```

The project depends on PyTorch, PyTorch Geometric, torch-scatter, torch-sparse, SciPy, NumPy, OGB, Matplotlib, tqdm, and scikit-learn. By default, experiments use seed `42` and CUDA when available. If no GPU is detected, the master script falls back to CPU execution.

## Documentation

Additional documentation is available in [`docs/`](docs/). Result summaries and metric definitions are available in [`results/`](results/), including [`results/FINAL_RESULTS.md`](results/FINAL_RESULTS.md) and [`results/METRICS_README.md`](results/METRICS_README.md).

## License

MIT License. See `LICENSE` for details.
