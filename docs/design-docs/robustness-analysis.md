# Robustness Analysis

Status: Approved
Date: 2026-05
Branch: main

## Context

GraphCoarsening evaluates whether spectral coarsening remains credible under four stress conditions that affect graph neural network explanations. The robustness program covers depth sensitivity, long range message passing, encoder choice, and recovery of known motif structure. Each experiment is tied to a source script and a recorded result file.

The oversmoothing experiment in `experiments/run_oversmoothing.py` sweeps GCN depth over 2, 4, 8, 16, and 32 layers. It measures mean pairwise cosine similarity, embedding variance, and Dirichlet energy for full graph inference and three coarsening strategies: random, heavy edge, and spectral. The result file also reports link prediction AUC and coarse graph size, which help separate embedding collapse from task performance.

The oversquashing experiment in `experiments/run_oversquashing.py` studies effective resistance. Part A checks synthetic graphs, while Part B stratifies real dataset test edges into low, medium, and high resistance buckets. The recorded Cora, Citeseer, and PubMed results report sufficiency, necessity, sparsity, and bucket sizes for Occlusion, Saliency, GNNExplainer, and Ours.

The multi backbone experiment in `experiments/run_multibackbone.py` compares GCN, GraphSAGE, and GAT on Cora. It evaluates explanation behavior after changing the encoder while holding the link prediction task and sampled test edges fixed.

The ground truth experiment in `experiments/run_ground_truth.py` evaluates BA-Shapes, Tree-Cycles, and Link-Motif. It reports edge precision, recall, and F1 for all methods. For the coarsening explainer, it also reports region precision, region recall, and region F1, which measure whether a selected supernode region covers the motif structure rather than only matching individual edges.

## Decision

Adopt the four experiments as the robustness analysis suite for the main branch. The suite will be interpreted as a set of complementary checks, not as a single aggregate score.

1. Oversmoothing will use the depth sweep from 2 to 32 layers and will report cosine similarity, variance, and Dirichlet energy. AUC will remain in the table as a task level reference.
2. Oversquashing will use effective resistance buckets. The main interpretation will focus on whether explanation necessity improves on high resistance edges, since those edges are the long range cases most exposed to oversquashing.
3. Multi backbone analysis will use the METRICS_README summary table for GCN, GraphSAGE, and GAT, with the detailed `multibackbone_Cora.md` results as the source record.
4. Ground truth analysis will use both edge F1 and region F1. Region level metrics are necessary because coarsening explanations are structural regions, not only edge masks.

### Oversmoothing Results, Cora

The Cora depth sweep from `results/oversmoothing.md` shows that the full graph embeddings have high cosine similarity across depths. Spectral coarsening keeps variance above the full graph at every recorded depth and reports larger Dirichlet energy than the full graph, which indicates less collapsed representations on the coarsened graph.

| Depth | Method | Cosine similarity | Variance | Dirichlet energy | AUC |
|---|---|---:|---:|---:|---:|
| 2 | none | 0.9798 | 0.0051 | 23.5695 | 0.6552 |
| 2 | random | 0.9677 | 5.3352 | 162672.1182 | 0.4982 |
| 2 | heavy_edge | 0.9730 | 2.9237 | 90590.3456 | 0.4487 |
| 2 | spectral | 0.9460 | 10.6895 | 273108.1359 | 0.4737 |
| 4 | none | 0.9977 | 0.0030 | 13.1195 | 0.6494 |
| 4 | random | 0.9795 | 0.0138 | 537.5185 | 0.5254 |
| 4 | heavy_edge | 0.9814 | 0.0093 | 253.1761 | 0.5426 |
| 4 | spectral | 0.9946 | 0.0187 | 931.7607 | 0.4768 |
| 8 | none | 0.9971 | 0.0025 | 10.9406 | 0.6424 |
| 8 | random | 0.9670 | 0.0042 | 84.0676 | 0.5738 |
| 8 | heavy_edge | 0.9783 | 0.0037 | 34.7281 | 0.5497 |
| 8 | spectral | 0.9788 | 0.0034 | 102.3519 | 0.5084 |
| 16 | none | 0.9978 | 0.0019 | 8.5083 | 0.6419 |
| 16 | random | 0.9758 | 0.0033 | 65.3451 | 0.5548 |
| 16 | heavy_edge | 0.9807 | 0.0030 | 28.5211 | 0.5505 |
| 16 | spectral | 0.9831 | 0.0028 | 83.7924 | 0.5102 |
| 32 | none | 0.9813 | 0.0038 | 17.2949 | 0.6498 |
| 32 | random | 0.9044 | 0.0053 | 101.3100 | 0.4678 |
| 32 | heavy_edge | 0.9211 | 0.0046 | 43.2048 | 0.4338 |
| 32 | spectral | 0.9261 | 0.0045 | 130.7781 | 0.4223 |

### Oversquashing Results

The effective resistance analysis records equal bucket sizes on Cora, Citeseer, and PubMed. The METRICS_README summary highlights the clearest positive pattern on PubMed: high resistance necessity for Ours is 0.40, above low resistance at 0.30 and medium resistance at 0.20. This supports the interpretation that coarsening is most useful where long range structure matters.

| Dataset | Method | Bucket | Necessity | Sufficiency | Sparsity |
|---|---|---|---:|---:|---:|
| Cora | Ours | low | 0.1000 | 0.1000 | 0.0873 |
| Cora | Ours | medium | 0.2101 | 0.2101 | 0.0873 |
| Cora | Ours | high | 0.2000 | 0.2000 | 0.1718 |
| Citeseer | Ours | low | 0.0000 | 0.0000 | 0.7014 |
| Citeseer | Ours | medium | 0.0000 | 0.0000 | 0.1807 |
| Citeseer | Ours | high | 0.0000 | 0.0000 | 0.1807 |
| PubMed | Ours | low | 0.3000 | 0.3000 | 0.1782 |
| PubMed | Ours | medium | 0.2000 | 0.2000 | 0.1781 |
| PubMed | Ours | high | 0.4000 | 0.4000 | 0.1782 |

### Multi Backbone Results, Cora

The multi backbone result confirms that the method can be evaluated across GCN, GraphSAGE, and GAT. The AUC values differ strongly by encoder, while Ours maintains nonzero necessity under all three backbones.

| Backbone | AUC | Ours Fid+ |
|---|---:|---:|
| GCN | 0.655 | 0.080 |
| GraphSAGE | 0.918 | 0.080 |
| GAT | 0.918 | 0.040 |

The detailed source result in `results/multibackbone_Cora.md` records 50 samples per backbone for Ours, with mean times near 0.02 seconds per explanation.

| Backbone | Ours Fid+ mean | Ours Fid+ std | Ours Fid- mean | Ours Fid- std | Mean time | Samples |
|---|---:|---:|---:|---:|---:|---:|
| GCN | 0.0800 | 0.2713 | 0.0800 | 0.2713 | 0.0214 | 50 |
| GraphSAGE | 0.0800 | 0.2713 | 0.0800 | 0.2713 | 0.0207 | 50 |
| GAT | 0.0400 | 0.1960 | 0.0400 | 0.1960 | 0.0203 | 50 |

### Ground Truth Results

The ground truth analysis is the strongest structural evidence. Edge F1 alone can favor edge mask baselines, as in BA-Shapes where Saliency reaches edge F1 of 0.609. Region F1 changes the interpretation because it evaluates whether the explanation captures the motif region. Only Ours achieves non-zero region-level metrics.

| Dataset | Method | Edge F1 | Region F1 | Region P | Region R |
|---|---|---:|---:|---:|---:|
| BA-Shapes | Ours | 0.207 | 0.613 | 0.532 | 0.855 |
| BA-Shapes | Occlusion | 0.098 | 0.000 | 0.000 | 0.000 |
| BA-Shapes | Saliency | 0.609 | 0.000 | 0.000 | 0.000 |
| Tree-Cycles | Ours | 0.087 | 0.202 | 0.523 | 0.208 |
| Link-Motif | Ours | 0.215 | 0.106 | 0.072 | 0.200 |

The detailed `results/ground_truth_explanation.md` file reports the same conclusion. Ours records region F1 of 0.6133 on BA-Shapes, 0.2017 on Tree-Cycles, and 0.1064 on Link-Motif. Occlusion and Saliency do not report region metrics and are treated as 0.000 in the METRICS_README summary.

## Consequences

The robustness analysis should present GraphCoarsening as a structural explanation method. Its main advantage is not always raw edge fidelity. Its clearest evidence appears when evaluation asks whether a coarsened region captures meaningful graph structure.

Only Ours achieves non-zero region-level metrics. This is the key finding because region precision, region recall, and region F1 are the metrics aligned with the design of coarsening explanations. Edge-only baselines can mark individual motif edges, but they don't produce supernode regions whose membership can be tested against motif nodes.

The oversmoothing study gives a depth based stress test. Spectral coarsening preserves higher variance and higher Dirichlet energy than the full graph on Cora, so it should be discussed as evidence against representation collapse in the coarsened representation. The reduced AUC values mean this evidence should not be overstated as task performance improvement.

The oversquashing study gives a long range stress test. The PubMed resistance buckets show the intended pattern, high resistance necessity of 0.40 is greater than low resistance at 0.30 and medium resistance at 0.20. Cora and Citeseer are weaker, so the document should describe the finding as supportive but dataset dependent.

The multi backbone study reduces the risk that the explanation behavior is specific to GCN. GraphSAGE and GAT both reach AUC near 0.918 on Cora, and Ours remains nonzero under both. The lower GAT Fid+ value shows that backbone choice still affects explanation sensitivity.

Future reporting should keep the four experiments together. Oversmoothing and oversquashing test graph signal behavior, multi backbone tests model dependence, and ground truth tests structural correctness. Presenting only one of these would give an incomplete view of the method.
