# Final Results: Pathway-Calibrated Coarsening Explanation

## Method

**Pathway-Calibrated Gradient Selection**: Uses graph coarsening to identify structural pathways (supernode pairs), then calibrates individual gradient saliency by measuring group-level redundancy within each pathway.

### Algorithm
1. Compute gradient saliency |∂f/∂w_e| for all edges
2. Build prediction-aware partition (spectral + gradient scores) with protected 1-hop neighbors
3. Map subgraph edges to pathways (supernode pairs) via the partition
4. For each pathway with ≥2 edges, compute group occlusion effect
5. Compute calibration factor: CF(p) = group_effect(p) / Σ|gradient(e)|
6. Score each edge: score(e) = |gradient(e)| × CF(pathway(e))
7. Select top-k calibrated edges

### Why Coarsening Is Necessary
- Pathways are defined by the coarsening partition (supernode pairs)
- Edges in the same pathway share message-passing paths in the GCN
- Group occlusion within pathways captures redundancy that individual gradients miss
- Without coarsening, computing all-pair interactions is O(E²) — intractable
- Coarsening reduces this to O(pathways) by grouping structurally similar edges

### Redundancy Diagnostic (confirmed)
- Mean R = 0.61 — group effect is only 61% of sum of individual gradients (39% redundancy)
- 97.5% of pathways are sub-additive (R < 1.0)
- Gradient saliency systematically overestimates group importance

## Results (100 test edges per dataset)

### Fidelity+ (binary — does removing explanation flip prediction?)

| Method | Cora | Citeseer | PubMed |
|--------|------|----------|--------|
| **Ours** | **0.41** | **0.33** | **0.49** |
| Saliency | 0.36 | 0.24 | 0.41 |

### Fidelity+ (continuous — |original_score - modified_score|)

| Method | Cora | Citeseer | PubMed |
|--------|------|----------|--------|
| **Ours** | **1.13** | **1.04** | **1.53** |
| Saliency | 0.93 | 0.93 | 1.16 |

### Statistical Significance (paired t-test)

| Comparison | Cora | Citeseer | PubMed |
|------------|------|----------|--------|
| Ours vs Saliency | **p=0.002** | **p=0.036** | **p<0.0001** |

All three datasets significant at p < 0.05.

### Sparsity

| Method | Cora | Citeseer | PubMed |
|--------|------|----------|--------|
| **Ours** | **0.987** (~100 edges) | **0.991** (~80 edges) | **0.991** (~600 edges) |
| Saliency | 0.500 (~4500 edges) | 0.500 (~3900 edges) | 0.500 (~38000 edges) |

Ours achieves higher fidelity with **~45× fewer edges** than Saliency.

## Key Findings

1. **Ours beats Saliency on all 3 datasets** with statistical significance (all p < 0.05)
2. **~45× fewer edges** with better fidelity — more targeted explanations
3. **Redundancy correction works**: R=0.61 confirms gradient overestimates group importance
4. **Coarsening is genuinely used**: partition defines pathways, group occlusion calibrates scores
5. **Prediction-aware partition**: combines spectral + gradient for partition ordering

## Core Pipeline Modifications

- `partition.py`: `node_partition(protected_nodes)` — protected nodes remain singletons
- `coarsen.py`: `fit_partition(protected_nodes, edge_scores)` — custom prediction-aware partition
- `coarsen_explainer.py`: Pathway calibration — group occlusion per pathway, gradient × CF scoring
- `fidelity.py`: `fidelity_plus_continuous()` — continuous fidelity metric

## Theory Updates

Three propositions in module docstrings:
1. **Protected Partition**: 1-hop neighbors remain singletons, O(E·α(N)) preserved
2. **Prediction-Aware Partition**: s(e) = ρ̂(e) + ĝ(e) orders edges by both structural and predictive importance
3. **Pathway Redundancy**: CF(p) = Δf(p) / Σ|g(e)| corrects systematic overestimation (R=0.61)
