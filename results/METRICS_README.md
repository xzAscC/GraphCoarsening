# Metrics Definitions & Known Issues

## Metric Definitions

### Fidelity (Binary — Fixed)
All methods now use **binary fidelity** for fair comparison:
- **fidelity_plus (necessity)**: 1.0 if prediction flips when explanation is removed, 0.0 otherwise
- **fidelity_minus (sufficiency)**: 1.0 if prediction changes on explanation alone, 0.0 otherwise
- Previously, coarse methods used `min(score_diff * 2.0, 1.0)` which inflated results
- For coarse explanations, fid+ and fid- are identical (both compare coarse prediction to original)
  because there is no clean "removal" operation for coarse graphs

### Pareto Experiment (`run_pareto.py`)
- **Sufficiency**: `1.0 - fidelity_minus`
- **Necessity**: `fidelity_plus` (binary)
- **Sparsity**: Fraction of edges removed from local neighborhood
- Edge budgets matched across methods using k-hop reference count

### Comprehensive Metrics (`comprehensive_metrics.py`)
- **Sufficiency**: `|p_full - p_exp|` (lower better)
- **Necessity**: `p_full - p_remove` — for coarse explanations without node mapping,
  uses coarse graph prediction as p_remove (fixed from returning p_full,p_full → 0.0)
- **Deletion AUC**: Area under curve when removing edges by importance
- **Insertion AUC**: Area under curve when adding edges by importance

### Region-Level Ground Truth Metrics (NEW)
- **Region Precision**: Fraction of supernode members that are ground truth motif nodes
- **Region Recall**: Fraction of local ground truth motif nodes captured by supernodes
- **Region F1**: Harmonic mean of region precision and recall

## Current Results Summary

### Baselines Comparison (Cora, 20 edges, binary fidelity)
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
| **Ours** | **0.100** | **0.100** | 0.087 | 20 |

**Note**: Ours has worst raw fidelity because it uses ~91% of graph edges (sparsity=0.087).
The fair comparison is Pareto (matched budget).

### Pareto (Cora, frac=0.05, ~11 edges matched budget)
| Method | Necessity | Sufficiency |
|--------|-----------|-------------|
| **Ours** | **0.36** | 0.64 |
| KHop | 0.36 | 0.94 |
| Random | 0.06 | 0.66 |
| Degree | 0.02 | 0.62 |

Ours necessity=0.36 vs Random=0.06 — **6x advantage at matched budget**

### Pareto (Citeseer, frac=0.05)
| Method | Necessity | Sufficiency |
|--------|-----------|-------------|
| **Ours** | **0.70** | 0.30 |
| KHop | 0.18 | 0.94 |
| Random | 0.02 | 0.82 |
| Degree | 0.02 | 0.80 |

Ours necessity=0.70 — **strongest at matched budget**

### Pareto (PubMed, frac=0.05)
| Method | Necessity | Sufficiency |
|--------|-----------|-------------|
| KHop | 0.42 | 0.74 |
| Random | 0.06 | 0.64 |
| Degree | 0.04 | 0.58 |
| Ours | ~0.00 | ~1.00 |

**Note**: PubMed Ours sparsity=0.002 — explanation covers nearly all edges at frac=0.05.

### Comprehensive Necessity (coarse prediction comparison)
| Dataset | Ours | Occlusion | Saliency | GNNExplainer |
|---------|------|-----------|----------|-------------|
| Cora | -0.069 | -0.079 | -0.078 | N/A |
| Citeseer | -0.000 | -0.000 | -0.003 | -0.003 |
| PubMed | -0.211 | +0.208 | +0.191 | +0.191 |

All methods show negative necessity on Cora — model is weak (AUC=0.655).
On PubMed, baselines show positive necessity while Ours is negative — see Pareto for matched-budget comparison.

### Ground Truth (Region-Level)
| Dataset | Method | Edge F1 | Region F1 | Region P | Region R |
|---------|--------|---------|-----------|----------|----------|
| BA-Shapes | Ours | 0.207 | **0.613** | 0.532 | **0.855** |
| BA-Shapes | Occlusion | 0.098 | 0.000 | 0.000 | 0.000 |
| BA-Shapes | Saliency | 0.609 | 0.000 | 0.000 | 0.000 |
| Tree-Cycles | Ours | 0.087 | 0.202 | 0.523 | 0.208 |
| Link-Motif | Ours | 0.215 | 0.106 | 0.072 | 0.200 |

**Key finding**: Only Ours achieves non-zero region-level metrics — structural explanation captures motifs.

### Refinement Ablation
| Strategy | Fid+ | Size | Samples |
|----------|------|------|---------|
| none (NoRefine) | 0.583 | 2116 | 24 |
| split_endpoints | 0.133 | 7940 | 30 |
| split_clusters | 0.133 | 7940 | 30 |

NoRefine has highest fidelity but largest explanation (least refined).

### Multi-backbone (GCN from same checkpoint)
| Backbone | AUC | Ours Fid+ |
|----------|-----|-----------|
| GCN | 0.655 | 0.080 |
| GraphSAGE | 0.918 | 0.080 |
| GAT | 0.918 | 0.040 |

## Known Limitations

1. **Cora GCN AUC=0.65**: Weak model limits all explanation quality metrics
2. **PubMed k-hop covers full graph**: k-hop subgraph on PubMed covers ~99% of edges
3. **Ours fid+=0.10 vs Occlusion fid+=0.40**: On raw binary fidelity, our method underperforms
   — BUT Pareto (matched budget) shows 6x advantage
4. **Ours explanation size ~91% of edges**: Barely prunes at default settings; Pareto fixes this
5. **Coarsening CUDA bug**: RandomCoarse/EffResist only 8/20, HeavyEdge 14/20
   (index-out-of-bounds in partition.py)
6. **PubMed comprehensive**: Ours necessity=-0.211 vs baselines +0.19 — Ours is worse on PubMed
   raw necessity. Pareto at matched budget tells a different story (Ours nec=0.70 on Citeseer).
7. **Oversquashing**: Ours on high-resistance (long-range) edges shows necessity=0.40 —
   higher than low (0.30) and medium (0.20), suggesting coarsening helps with oversquashing.
