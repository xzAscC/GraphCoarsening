# Comprehensive Metrics and Baselines Comparison

Status: Approved
Date: 2026-05
Branch: main

## Context

GraphCoarsening evaluates explanations for GNN link prediction. Earlier evaluation relied mainly on binary fidelity, which is useful for asking whether a model prediction changes, but too coarse to rank explanations whose effects are gradual. The project therefore adds probability based metrics, local sparsity, and perturbation curves to measure whether an explanation preserves the original prediction, whether it is necessary for that prediction, and how much structure it uses.

The binary fidelity layer remains part of the evaluation context. In `src/evaluation/fidelity.py`, `fidelity_plus` tests necessity by removing the explanation from the original graph and returning 1.0 when the binary prediction changes. `fidelity_minus` tests sufficiency by running the model on the explanation alone and returning 0.0 when the binary prediction is preserved. These metrics are intentionally binary. They support fair baseline comparison, but they can hide probability changes that do not cross the decision threshold.

The comprehensive metric layer in `src/evaluation/comprehensive_metrics.py` addresses that limitation. It uses sigmoid probabilities for the target link, supports regular subgraph explanations and coarse graph explanations, and returns seven metrics through `compute_all_metrics`: `sufficiency`, `necessity`, `comprehensiveness`, `sparsity`, `sparsity_abs_edges`, `deletion_auc`, and `insertion_auc`. `experiments/run_comprehensive_metrics.py` applies these metrics to Occlusion, Saliency, GNNExplainer, and the GraphCoarsening method. `experiments/run_baselines_comparison.py` provides the broader binary fidelity comparison over trivial, hard, coarsening, GNN, and project methods.

The evaluation also records known constraints. Coarse explanations sometimes lack a clean removal operation, so binary fidelity compares the coarse prediction with the original prediction. For comprehensive necessity, coarse explanations with `original_node_indices` remove original edges among involved nodes. If that mapping is unavailable, the coarse graph prediction is used as `p_remove`. This makes the metric computable while preserving a consistent interpretation: necessity measures the decrease in target probability after removing or replacing the explanatory structure.

## Decision

The project adopts seven comprehensive explanation metrics for link prediction. Let `p_full` denote the target edge probability on the full graph, `p_exp` the target edge probability on the explanation alone, `p_remove` the target edge probability after removing the explanation from the original graph, `E_exp` the explanation edge set, and `E_local` the edge set in the local k hop enclosing subgraph around the target edge.

| Metric | Formula or construction | Direction | Interpretation |
|--------|--------------------------|-----------|----------------|
| Sufficiency | `|p_full - p_exp|` | Lower is better | Measures how closely the explanation alone preserves the full graph prediction. A value near 0 means the explanation is self sufficient. |
| Necessity | `p_full - p_remove` | Higher is better | Measures how much the target probability drops after explanatory structure is removed. Positive values mean the explanation was needed for the prediction. |
| Comprehensiveness | `(p_full - p_remove) / p_full` | Higher is better | Normalizes necessity by the original probability. If `p_full` is near zero, the implementation returns 0.0. |
| Sparsity | `|E_exp| / |E_local|` | Lower is usually better | Measures explanation size relative to the local k hop neighborhood, not the full graph. This avoids rewarding large graph settings where any local explanation appears small against the full edge set. |
| Deletion AUC | Area under the progressive removal curve | Higher is better | Sort explanation edges by `edge_weight` when present, otherwise by a random order. Remove the top fraction of explanation edges step by step and integrate the probability drop curve. |
| Insertion AUC | Area under the progressive addition curve | Higher is better | Start from an empty graph containing the target nodes. Add explanation edges step by step in importance order and integrate the prediction recovery curve. |
| `sparsity_abs_edges` | `|E_exp|` | Context dependent | Reports the absolute explanation edge count so that sparsity ratios can be interpreted with their actual edge budget. |

The comprehensive experiment uses the following metric key order: `sufficiency`, `necessity`, `comprehensiveness`, `sparsity`, `sparsity_abs_edges`, `deletion_auc`, and `insertion_auc`. Each method is evaluated on sampled positive test edges from the selected dataset, and the output stores mean, standard deviation, minimum, and maximum for every metric.

### Baselines Comparison

The approved binary fidelity baseline comparison uses the 12 methods recorded in `results/METRICS_README.md`. It uses Cora, 20 target edges, and binary fidelity for fair comparison.

| Method | Fid+ | Fid- | Sparsity | Samples |
|--------|------|------|----------|---------|
| NoRefine | 0.600 | 0.600 | 0.847 | 20 |
| RandomCoarse | 0.500 | 0.500 | 0.172 | 8 |
| EffResist | 0.500 | 0.500 | 0.448 | 8 |
| FullGraph | 0.400 | 0.000 | 0.000 | 20 |
| KHop | 0.400 | 0.100 | 0.981 | 20 |
| Random | 0.400 | 0.250 | 0.991 | 20 |
| Occlusion | 0.400 | 0.500 | 0.991 | 20 |
| Saliency | 0.400 | 0.500 | 0.500 | 20 |
| Degree | 0.300 | 0.250 | 0.991 | 20 |
| GreedyDel | 0.250 | 0.050 | 0.989 | 20 |
| HeavyEdge | 0.000 | 0.000 | 0.023 | 14 |
| Ours | 0.100 | 0.100 | 0.087 | 20 |

This table should be read together with matched budget Pareto results. The raw Cora baseline table shows that the project method has low binary fidelity at its default settings, partly because it uses about 91 percent of graph edges. The matched budget Pareto comparison gives a fairer view when explanation budgets are aligned.

### Comprehensive Necessity

The approved comprehensive necessity summary compares the project method with neural explanation baselines by dataset. Values are from the coarse prediction comparison recorded in `results/METRICS_README.md`.

| Dataset | Ours | Occlusion | Saliency | GNNExplainer |
|---------|------|-----------|----------|--------------|
| Cora | -0.069 | -0.079 | -0.078 | N/A |
| Citeseer | -0.000 | -0.000 | -0.003 | -0.003 |
| PubMed | -0.211 | +0.208 | +0.191 | +0.191 |

Negative necessity means the probability did not drop after the removal or coarse comparison operation. For Cora, all methods show negative necessity, and the recorded context notes that the underlying model is weak. For PubMed, baselines show positive necessity while the project method is negative under raw comprehensive necessity, so matched budget analysis remains necessary for final interpretation.

### Code Module Mapping

| Module | Function or object | Role in the decision |
|--------|--------------------|----------------------|
| `src/evaluation/comprehensive_metrics.py` | `_get_probability` | Computes the sigmoid probability for the target edge from model logits. |
| `src/evaluation/comprehensive_metrics.py` | `_compute_necessity_components` | Computes shared `p_full` and `p_remove` values for necessity and comprehensiveness, including coarse graph handling. |
| `src/evaluation/comprehensive_metrics.py` | `sufficiency` | Implements `|p_full - p_exp|` for subgraph and coarse explanations. |
| `src/evaluation/comprehensive_metrics.py` | `necessity` | Implements `p_full - p_remove`. |
| `src/evaluation/comprehensive_metrics.py` | `comprehensiveness` | Implements `(p_full - p_remove) / p_full` with a zero guard when `p_full` is near zero. |
| `src/evaluation/comprehensive_metrics.py` | `sparsity` | Computes `|E_exp| / |E_local|` using a k hop enclosing subgraph and returns the absolute edge count. |
| `src/evaluation/comprehensive_metrics.py` | `deletion_auc` | Computes the AUC of progressive explanation edge removal. |
| `src/evaluation/comprehensive_metrics.py` | `insertion_auc` | Computes the AUC of progressive explanation edge insertion. |
| `src/evaluation/comprehensive_metrics.py` | `compute_all_metrics` | Aggregates the seven comprehensive metrics for one target edge. |
| `src/evaluation/comprehensive_metrics.py` | `batch_evaluate` | Aggregates comprehensive metrics over a batch of explanations. |
| `src/evaluation/fidelity.py` | `_predict_binary` | Produces binary predictions and raw scores for fidelity metrics. |
| `src/evaluation/fidelity.py` | `_is_coarse_explanation` | Detects coarse graph explanations through `is_coarse_graph`. |
| `src/evaluation/fidelity.py` | `_to_global_edges` | Maps local explanation edges back to original graph node indices when `original_node_indices` exists. |
| `src/evaluation/fidelity.py` | `fidelity_plus` | Binary necessity metric used in baseline comparison. |
| `src/evaluation/fidelity.py` | `fidelity_minus` | Binary sufficiency metric used in baseline comparison. |
| `src/evaluation/fidelity.py` | `fidelity_plus_continuous` | Continuous score difference variant for removal based fidelity. |
| `src/evaluation/fidelity.py` | `evaluate_fidelity` | Runs binary fidelity evaluation over test edges through an explainer. |
| `src/evaluation/fidelity.py` | `compute_sparsity` | Computes full graph sparsity as `1.0 - |E_exp| / |E_orig|` for the baseline comparison. |

## Consequences

The seven metric suite gives the project a stronger evaluation surface than binary fidelity alone. Sufficiency and necessity separate preservation from causal removal. Comprehensiveness normalizes necessity by the original probability. Sparsity and `sparsity_abs_edges` make explanation size explicit. Deletion AUC and insertion AUC test whether edge ordering carries meaningful importance information, not only whether the final edge set has an effect.

The decision also makes comparison more transparent. The broad baselines comparison remains useful because all 12 methods use the same binary fidelity protocol and test edges. The comprehensive necessity table adds probability level evidence across Cora, Citeseer, and PubMed, but it must be interpreted with model quality and explanation budget in mind. When raw fidelity and matched budget Pareto results disagree, the matched budget view is the more appropriate basis for comparing explanation quality.

There are practical limitations. Coarse explanations require special handling because removing a coarse graph from the original graph is not always well defined. Some methods have fewer valid samples because of coarsening failures. Dataset structure can also distort sparsity, especially when a k hop neighborhood covers most of the graph. These limitations do not invalidate the metric suite, but they require every reported result to include method, dataset, sample count, and edge budget context.
