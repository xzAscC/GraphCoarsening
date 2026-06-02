# Method Overview

Status: Approved
Date: 2026-05
Branch: main

## Context

GraphCoarsening studies graph coarsening based explanation for GNN link prediction. The explanatory problem is to identify a sparse edge set that preserves or disrupts a target link prediction in a controlled way, while remaining structurally coherent in the local graph. Standard gradient saliency measures individual edge sensitivity, but it does not account for the fact that multiple edges can support the same message passing pathway. As a result, saliency can overstate the importance of redundant edges and select disconnected explanations.

The project therefore defines the core method as Prediction Guided Pathway Calibrated Gradient Selection. The method combines spectral coarsening, prediction aware partitioning, pathway level group occlusion, and calibrated gradient scoring. Coarsening provides a tractable grouping of structurally related edges into pathways, where a pathway is represented by a pair of supernodes. Group occlusion is then measured at the pathway level rather than through all possible edge interactions.

The method is supported by two formal propositions and three empirical findings recorded in the coarsening implementation.

| Identifier | Statement | Source |
|---|---|---|
| P1 | Protected Partition Correctness. Given a target link (a,b), let N1(a,b) be the one hop neighborhood. A protected partition P prime constrains the greedy partition such that every node in N1(a,b) remains a singleton. This preserves local structure from absorption during coarsening. The proof follows from the skip condition in the merge loop, and the O(E alpha(N)) complexity is preserved. | `src/coarsen.py`, `src/partition.py` |
| P2 | Prediction Guided Merge. Let rho hat(e) be normalized spectral perturbation and g hat(v) be normalized node gradient importance. The merge cost C(e) = rho hat(e) + lambda g hat(a) g hat(b) captures structural and predictive importance. The product Phi(a,b) = g hat(a) g hat(b) penalizes merging two high importance endpoints because Phi is large only when both endpoint importances are high. The hard reject Phi > tau guarantees no merge where both endpoints are in the top (1 tau) importance fraction. | `src/coarsen.py`, `src/partition.py` |
| E1 | Pathway Redundancy. For pathway p, CF(p) = Delta f(p) / sum |g(e)| is approximately 0.61 on average, meaning 39 percent redundancy. Across measured pathways, 97.5 percent are sub additive. Gradient saliency therefore systematically overestimates group importance. | `src/coarsen.py`, `results/FINAL_RESULTS.md` |
| E2 | Structural Sufficiency at Low Sparsity. Pathway calibrated edges form structurally coherent subgraphs with significantly fewer disconnected components than saliency, with p < 0.0001 across all budgets and datasets. On Cora, this coherence gives superior sufficiency at all six budgets, with p <= 0.006. | `src/coarsen.py`, `results/FINAL_RESULTS.md` |
| E3 | Necessity at Moderate to High Sparsity. At budgets k >= 20, removing pathway calibrated edges causes significant prediction drops relative to removing saliency edges. The observed wins include Cora at k = 20 to 100, Citeseer at k = 20 to 200, and PubMed at k = 200. | `src/coarsen.py`, `results/FINAL_RESULTS.md` |

## Decision

The project adopts Prediction Guided Pathway Calibrated Gradient Selection as the explanation method for link prediction. The algorithm is defined as follows.

1. Compute gradient saliency |partial f / partial w_e| for all graph edges.
2. Build a prediction guided partition using the merge cost C(e) = rho hat(e) + lambda_pred Phi(a,b), where Phi(a,b) is the product of normalized endpoint gradient importances. Reject a merge when Phi(a,b) exceeds the fidelity threshold. Keep protected nodes, the one hop neighborhood of the target link, as singleton partitions.
3. Map candidate subgraph edges to pathways by assigning each endpoint to its supernode and using the resulting supernode pair as the pathway identifier.
4. For each pathway with at least two edges, remove all edges in the pathway and measure the group occlusion effect on the target prediction.
5. Compute the pathway calibration factor CF(p) = group_effect(p) / sum |gradient(e)|, with smoothing and clipping in the explainer implementation.
6. Score every candidate edge by score(e) = |gradient(e)| times CF(pathway(e)).
7. Select the top k calibrated edges, where k is implemented through the retained fraction of candidate edges.

This decision makes coarsening part of the explanation objective rather than a post processing step. The partition is shaped by both spectral stability and prediction importance. The pathway calibration step then corrects individual gradient scores using measured group behavior inside each coarsened pathway.

### Configuration Parameters

| Dataclass | Parameter | Default | Role |
|---|---:|---:|---|
| `SpectralConfig` | `k` | `100` | Number of coarse nodes or spectral components used by the coarsening process. |
| `SpectralConfig` | `alpha` | `0.75` | Coarsening ratio that controls the merge budget in partition construction. |
| `ModelConfig` | `hidden_channels` | `128` | Hidden dimension for the GCN link prediction model. |
| `ModelConfig` | `num_layers` | `3` | Number of GCN layers. |
| `ModelConfig` | `dropout` | `0.5` | Dropout probability during model training. |
| `ModelConfig` | `lr` | `0.01` | Learning rate for model optimization. |
| `ModelConfig` | `weight_decay` | `5e-4` | Weight decay for model optimization. |
| `ModelConfig` | `epochs` | `100` | Number of training epochs. |
| `ModelConfig` | `neg_ratio` | `1.0` | Negative sampling ratio for link prediction training. |
| `ExperimentConfig` | `seed` | `42` | Random seed for experiment reproducibility. |
| `ExperimentConfig` | `device` | `cuda` | Default compute device. |
| `ExperimentConfig` | `explanation_datasets` | `Cora`, `Citeseer`, `PubMed`, `ogbl-ppa`, `ogbl-collab`, `ogbl-ddi` | Datasets used for explanation evaluation. |
| `ExperimentConfig` | `scalability_datasets` | `Cora`, `Citeseer`, `PubMed`, `Coauthor-CS`, `Coauthor-Physics`, `Amazon-Computers`, `ogbl-ppa`, `ogbl-collab`, `ogbl-ddi` | Datasets used for scalability evaluation. |
| `ExperimentConfig` | `ablation_ratios` | `0.3`, `0.6`, `0.9`, `0.95`, `0.99` | Ablation ratios used in experiment analysis. |

The explainer also exposes method specific parameters that are not part of the top level dataclasses: `lambda_pred`, with default 1.0, and `fidelity_threshold`, with default 0.8. The matched sparsity results use `lambda_pred=1.0` and `fidelity_threshold=0.8` for Cora and Citeseer, and `lambda_pred=2.0` and `fidelity_threshold=0.95` for PubMed.

### Code Module Mapping

| Source file | Role in the method |
|---|---|
| `src/coarsen.py` | Defines coarse graph construction and the GraphCoarsener pipeline. Its module docstring records P1, P2, and empirical findings E1 to E3. |
| `src/partition.py` | Implements Union Find partitioning, protected node constraints, and `prediction_guided_partition()` with spectral plus predictive merge cost. |
| `src/explainers/coarsen_explainer.py` | Implements Pathway Calibrated Gradient Selection, including protected target neighborhoods, gradient saliency, pathway mapping, group occlusion, calibration factors, and top k edge selection. |
| `config.py` | Defines spectral, model, and experiment dataclasses for default experiment parameters. |
| `results/FINAL_RESULTS.md` | Reports the final matched sparsity evaluation, redundancy diagnostics, statistical wins, losses, and core pipeline modifications. |
| `results/UPDATE_SUMMARY.md` | Records the prediction guided partition update, new explainer parameters, PubMed tuning, and theory update for the merge proposition. |
| `experiments/run_matched_sparsity_comparison.py` | Runs matched sparsity comparisons with `lambda_pred` and `fidelity_threshold` command line parameters, paired t tests, and Wilcoxon signed rank tests. |
| `src/metrics/fidelity.py` | Provides continuous fidelity metrics used to assess prediction changes after explanation edge removal. |
| `src/metrics/comprehensive_metrics.py` | Provides probability based sufficiency and necessity metrics for evaluation. |

## Consequences

The adopted method changes the interpretation of graph coarsening in the project. Coarsening is no longer only a structural compression stage. It is a prediction aware grouping mechanism that defines pathways for explanation and controls which local structures may be merged.

The main benefit is that explanation edges are selected with pathway context. Empirical results show a mean redundancy ratio of R = 0.61, with 97.5 percent sub additive pathways, so the calibration step directly addresses a measured weakness of raw gradient saliency. Structural coherence improves across all tested datasets and budgets, with 40 to 120 times fewer disconnected components reported in the final results.

The fidelity evidence is strongest on Cora, where the method wins on sufficiency at all six budgets and on necessity for five of six budgets. Citeseer shows necessity wins at k = 20, 50, 100, and 200. PubMed requires stronger prediction guidance because its two hop subgraph is larger, but it still shows significant wins at k = 200 on necessity and continuous Fidelity+. The conservative final count is 17 significant fidelity wins and 18 structural coherence wins.

The method also has costs and limits. It performs group occlusion per pathway, so explanation time depends on the number of pathways in the candidate subgraph. It is constrained to the local subgraph selected for explanation, while global saliency can select distant high gradient edges. This explains the observed small budget losses on Fidelity+ when saliency selects from the full graph. PubMed also shows weaker sufficiency, which suggests that identifying necessary edges does not always imply preservation of the original prediction under sparse retention.

Overall, the decision favors explanations that are structurally coherent, prediction aware, and calibrated by group behavior. The formal propositions define correctness for protected local structure and merge safety for prediction critical nodes. The empirical findings support the use of pathway calibration as a correction to individual gradient saliency.
