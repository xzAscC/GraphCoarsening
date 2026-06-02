# Ablation Studies

Status: Approved
Date: 2026-05
Branch: main

## Context

GraphCoarsening evaluates link explanations produced from spectrally informed graph coarsening. The ablation suite separates three design questions that affect the method's empirical claims.

First, the coarsening ratio ablation measures how much spectral structure is lost as the retained graph size changes. The experiment in `experiments/run_ablation.py` uses `ExperimentConfig.ablation_ratios`, set in `config.py` to `alpha=[0.3, 0.6, 0.9, 0.95, 0.99]`. For Cora, Citeseer, and PubMed, it computes the top `k=500` eigenvalues of the normalized adjacency matrix for the original training graph and the coarsened graph. The reported metric is mean relative error, abbreviated MRE, between the two spectra. Lower MRE indicates better spectral reconstruction.

Second, the hyperparameter ablation measures explanation quality across the two main spectral coarsening parameters. The experiment in `experiments/run_hyperparam_ablation.py` evaluates a grid of `k=[20, 50, 100, 200, 500]` and `alpha=[0.3, 0.5, 0.7, 0.75, 0.9, 0.95]` on Cora. Each cell records fidelity plus, fidelity minus, coarsening time, explanation time, spectral error, coarse graph size, and the number of evaluated samples. In the metric definitions, fidelity plus is necessity and fidelity minus is sufficiency. For coarse explanations these two values are identical because there is no clean removal operation for a coarse graph.

Third, the refinement ablation measures the effect of link-wise refinement after global coarsening. The experiment in `experiments/run_refinement_ablation.py` compares `none`, `split_endpoints`, `split_clusters`, `split_1hop`, `split_2hop`, and `full_khop`, while the decision-relevant comparison in `results/METRICS_README.md` focuses on `none`, `split_endpoints`, and `split_clusters`. The measured tradeoff is fidelity against explanation size, with sample count reported for interpretability.

## Decision

The approved ablation design keeps three separate experiments rather than merging them into a single aggregate score. This keeps the spectral reconstruction question, the hyperparameter sensitivity question, and the refinement tradeoff question analytically distinct.

### Coarsening Ratio Ablation

The coarsening ratio ablation is retained as the spectral reconstruction study. It uses the configured ratios `alpha=[0.3, 0.6, 0.9, 0.95, 0.99]` and reports MRE over the top 500 eigenvalues.

| Dataset | alpha=0.3 MRE | alpha=0.3 time_s | alpha=0.6 MRE | alpha=0.6 time_s | alpha=0.9 MRE | alpha=0.9 time_s | alpha=0.95 MRE | alpha=0.95 time_s | alpha=0.99 MRE | alpha=0.99 time_s |
|---------|---------------|------------------|---------------|------------------|---------------|------------------|----------------|-------------------|----------------|-------------------|
| Cora | 0.0966 | 4.1700 | 0.1912 | 1.0100 | 1.0507 | 0.3700 | 1.0000 | 0.3700 | 1.0000 | 0.3000 |
| Citeseer | 1.0000 | 15.1900 | 0.0640 | 0.8100 | 1.0000 | 0.4700 | 1.0000 | 0.4800 | 1.0000 | 0.5300 |
| PubMed | 0.0089 | 65.8300 | 0.0605 | 158.3000 | 0.7019 | 34.4600 | 1.0000 | 29.8600 | 1.0000 | 33.6000 |

These results support reporting the ratio sweep as a stability diagnostic. Moderate ratios can preserve spectral structure on some datasets, but aggressive coarsening at `alpha=0.95` and `alpha=0.99` often reaches MRE near 1.0000.

### Hyperparameter Ablation

The hyperparameter ablation is retained as the sensitivity study over `k x alpha`. The grid reports explanation quality and spectral behavior for Cora.

| k | alpha | Fid+ | Fid- | Coarsening time_s | Explain time_s | Spectral error | Coarse nodes | Samples |
|---|-------|------|------|-------------------|----------------|----------------|--------------|---------|
| 20 | 0.3 | 1.0000 | 1.0000 | 0.3170 | 0.0443 | 0.1008 | 1896 | 10 |
| 20 | 0.5 | 0.4343 | 0.4343 | 0.2332 | 0.0359 | 0.1000 | 1354 | 10 |
| 20 | 0.7 | 0.1511 | 0.1511 | 0.1923 | 0.0305 | 0.1000 | 813 | 10 |
| 20 | 0.75 | 0.1510 | 0.1510 | 0.1775 | 0.0310 | 0.2000 | 677 | 10 |
| 20 | 0.9 | 0.0000 | 0.0000 | 0.2010 | 0.0397 | 0.2000 | 271 | 10 |
| 20 | 0.95 | 0.0000 | 0.0000 | 0.1746 | 0.0732 | 1.0000 | 185 | 10 |
| 50 | 0.3 | 0.8900 | 0.8900 | 0.0847 | 0.0378 | 0.1599 | 1896 | 10 |
| 50 | 0.5 | 0.3815 | 0.3815 | 0.1071 | 0.0237 | 0.0001 | 1354 | 10 |
| 50 | 0.7 | 0.2000 | 0.2000 | 0.1565 | 0.0229 | 0.0400 | 813 | 10 |
| 50 | 0.75 | 0.2000 | 0.2000 | 0.1429 | 0.0218 | 0.1594 | 677 | 10 |
| 50 | 0.9 | 0.0000 | 0.0000 | 0.1628 | 0.0360 | 0.0000 | 271 | 10 |
| 50 | 0.95 | 0.0000 | 0.0000 | 0.1661 | 0.0495 | 1.0000 | 185 | 10 |
| 100 | 0.3 | 0.8234 | 0.8234 | 0.0794 | 0.0251 | 0.0425 | 1896 | 10 |
| 100 | 0.5 | 0.4351 | 0.4351 | 0.1462 | 0.0330 | 0.1948 | 1354 | 10 |
| 100 | 0.7 | 0.3006 | 0.3006 | 0.1605 | 0.0271 | 0.0618 | 813 | 10 |
| 100 | 0.75 | 0.3000 | 0.3000 | 0.1718 | 0.0278 | 0.3857 | 677 | 10 |
| 100 | 0.9 | 0.0000 | 0.0000 | 0.1810 | 0.0362 | 0.0442 | 271 | 10 |
| 100 | 0.95 | 0.0000 | 0.0000 | 0.1750 | 0.0673 | 0.0000 | 185 | 10 |
| 200 | 0.3 | 0.8101 | 0.8101 | 0.1066 | 0.0351 | 978965121351.2986 | 1896 | 10 |
| 200 | 0.5 | 0.5739 | 0.5739 | 0.1039 | 0.0212 | 983876428650.4183 | 1354 | 10 |
| 200 | 0.7 | 0.2211 | 0.2211 | 0.1357 | 0.0203 | 984024766491.7451 | 813 | 10 |
| 200 | 0.75 | 0.3544 | 0.3544 | 0.1544 | 0.0211 | 802326663289.2585 | 677 | 10 |
| 200 | 0.9 | 0.0000 | 0.0000 | 0.3197 | 0.0459 | 775000000000.0000 | 271 | 10 |
| 200 | 0.95 | 0.0000 | 0.0000 | 0.2917 | 0.0541 | 0.0000 | 185 | 10 |
| 500 | 0.3 | 0.7205 | 0.7205 | 0.2940 | 0.1252 | 883949330366.5079 | 1896 | 10 |
| 500 | 0.5 | 0.5515 | 0.5515 | 0.1280 | 0.0322 | 849836429655.5165 | 1354 | 10 |
| 500 | 0.7 | 0.2000 | 0.2000 | 0.2858 | 0.0386 | 750224578026.7069 | 813 | 10 |
| 500 | 0.75 | 0.0000 | 0.0000 | 0.1594 | 0.0255 | 622104526404.7345 | 677 | 10 |
| 500 | 0.9 | 0.0000 | 0.0000 | 0.1880 | 0.0346 | 1.0507 | 271 | 10 |
| 500 | 0.95 | 0.0000 | 0.0000 | 0.1765 | 0.0522 | 1.0000 | 185 | 10 |

The grid shows that explanation fidelity is strongest at lower coarsening ratios, especially `alpha=0.3`, and declines toward zero under aggressive compression. The table also exposes unstable spectral error values for several larger `k` settings, so the hyperparameter study should be interpreted as sensitivity evidence rather than as a single best-setting search.

### Refinement Ablation

The refinement ablation is retained as the fidelity-size tradeoff study. The full Cora result file includes six strategies.

| Strategy | Fid+ | Fid+ std | Fid- | Fid- std | Mean size | Size std | Mean time_s | Samples |
|----------|------|----------|------|----------|-----------|----------|-------------|---------|
| none | 0.5833 | 0.4930 | 0.5833 | 0.4930 | 2116.0000 | 0.0000 | 0.0181 | 24 |
| split_endpoints | 0.1333 | 0.3399 | 0.1333 | 0.3399 | 7939.4667 | 2576.3646 | 0.0160 | 30 |
| split_clusters | 0.1333 | 0.3399 | 0.1333 | 0.3399 | 7939.4667 | 2576.3646 | 0.0161 | 30 |
| split_1hop | 0.0667 | 0.2494 | 0.0667 | 0.2494 | 8445.0333 | 1890.6156 | 0.0181 | 30 |
| split_2hop | 0.0333 | 0.1795 | 0.0333 | 0.1795 | 8697.6667 | 1360.5288 | 0.0201 | 30 |
| full_khop | 0.0333 | 0.1795 | 0.0333 | 0.1795 | 8697.6667 | 1360.5288 | 0.0180 | 30 |

The metrics summary records the approved decision table for the refinement comparison.

| Strategy | Fid+ | Size | Samples |
|----------|------|------|---------|
| none (NoRefine) | 0.583 | 2116 | 24 |
| split_endpoints | 0.133 | 7940 | 30 |
| split_clusters | 0.133 | 7940 | 30 |

This decision preserves the refinement comparison as a diagnostic rather than as a blanket endorsement of additional splitting. On Cora, `none` has the highest fidelity and the smallest reported size among the decision-relevant rows, while `split_endpoints` and `split_clusters` increase explanation size and reduce fidelity.

## Consequences

The ablation suite supports three claims. The ratio experiment quantifies spectral reconstruction loss as compression increases. The hyperparameter grid shows how explanation quality changes across `k` and `alpha`, with lower `alpha` values usually retaining higher fidelity on Cora. The refinement study shows that link-wise splitting is not automatically beneficial under the current binary fidelity metric.

The approved design also carries limits. The hyperparameter grid is reported on Cora only, with 10 samples per cell. The refinement decision table uses Cora and a small sample count, and binary fidelity makes fidelity plus and fidelity minus identical for coarse explanations. Large spectral error values for some high-`k` settings should be reported plainly because they indicate numerical or denominator sensitivity in the MRE calculation.

Future experiment reports should keep these ablations separate. Combining spectral MRE, fidelity, and size into one aggregate would hide the central tradeoffs: structural preservation, explanation quality, and refinement cost answer different questions.
