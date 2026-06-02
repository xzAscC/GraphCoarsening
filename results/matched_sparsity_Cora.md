# Experiment Results: matched_sparsity_Cora

- **dataset**: Cora
- **num_test_edges**: 100
- **seed**: 42
- **budgets**: mean=21.2500, std=17.4553, n=4
- **methods**: ['CoarsenExplainer', 'SaliencyExplainer']
## results

### 5

- **n**: 100
- **budget**: 5
#### sufficiency

- **ours_mean**: 0.2019
- **saliency_mean**: 0.2325
- **ttest_p**: 0.0038
- **wilcoxon_p**: 0.0139
- **cohens_d**: -0.2963
- **ours_wins**: True

#### necessity

- **ours_mean**: 0.0576
- **saliency_mean**: 0.0911
- **ttest_p**: 0.0044
- **wilcoxon_p**: 0.0022
- **cohens_d**: -0.2912
- **ours_wins**: False

#### fidelity_plus_continuous

- **ours_mean**: 0.6924
- **saliency_mean**: 0.9644
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0000
- **cohens_d**: -0.5434
- **ours_wins**: False

#### num_components

- **ours_mean**: 43.4000
- **saliency_mean**: 2243.5400
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0000
- **cohens_d**: -14.6819
- **ours_wins**: True

#### target_connectivity

- **ours_target_connected**: 4
- **saliency_target_connected**: 15
- **ours_pct**: 0.0400
- **saliency_pct**: 0.1500


### 10

- **n**: 100
- **budget**: 10
#### sufficiency

- **ours_mean**: 0.1997
- **saliency_mean**: 0.2298
- **ttest_p**: 0.0075
- **wilcoxon_p**: 0.0230
- **cohens_d**: -0.2728
- **ours_wins**: True

#### necessity

- **ours_mean**: 0.0527
- **saliency_mean**: 0.0598
- **ttest_p**: 0.6267
- **wilcoxon_p**: 0.8554
- **cohens_d**: -0.0488
- **ours_wins**: False

#### fidelity_plus_continuous

- **ours_mean**: 0.8427
- **saliency_mean**: 1.0719
- **ttest_p**: 0.0007
- **wilcoxon_p**: 0.0024
- **cohens_d**: -0.3493
- **ours_wins**: False

#### num_components

- **ours_mean**: 41.5900
- **saliency_mean**: 2239.7800
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0000
- **cohens_d**: -14.6813
- **ours_wins**: True

#### target_connectivity

- **ours_target_connected**: 19
- **saliency_target_connected**: 31
- **ours_pct**: 0.1900
- **saliency_pct**: 0.3100


### 20

- **n**: 100
- **budget**: 20
#### sufficiency

- **ours_mean**: 0.1992
- **saliency_mean**: 0.2158
- **ttest_p**: 0.1105
- **wilcoxon_p**: 0.1581
- **cohens_d**: -0.1610
- **ours_wins**: True

#### necessity

- **ours_mean**: 0.0554
- **saliency_mean**: 0.0068
- **ttest_p**: 0.0172
- **wilcoxon_p**: 0.0085
- **cohens_d**: 0.2424
- **ours_wins**: True

#### fidelity_plus_continuous

- **ours_mean**: 0.9315
- **saliency_mean**: 1.0164
- **ttest_p**: 0.3076
- **wilcoxon_p**: 0.5882
- **cohens_d**: -0.1026
- **ours_wins**: False

#### num_components

- **ours_mean**: 37.6800
- **saliency_mean**: 2233.4200
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0000
- **cohens_d**: -14.6768
- **ours_wins**: True

#### target_connectivity

- **ours_target_connected**: 35
- **saliency_target_connected**: 48
- **ours_pct**: 0.3500
- **saliency_pct**: 0.4800


### 50

- **n**: 100
- **budget**: 50
#### sufficiency

- **ours_mean**: 0.1934
- **saliency_mean**: 0.2047
- **ttest_p**: 0.2526
- **wilcoxon_p**: 0.3007
- **cohens_d**: -0.1151
- **ours_wins**: True

#### necessity

- **ours_mean**: 0.0660
- **saliency_mean**: -0.0085
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0000
- **cohens_d**: 0.4402
- **ours_wins**: True

#### fidelity_plus_continuous

- **ours_mean**: 1.0733
- **saliency_mean**: 1.0022
- **ttest_p**: 0.2925
- **wilcoxon_p**: 0.3498
- **cohens_d**: 0.1058
- **ours_wins**: True

#### num_components

- **ours_mean**: 29.3800
- **saliency_mean**: 2215.9800
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0000
- **cohens_d**: -14.7886
- **ours_wins**: True

#### target_connectivity

- **ours_target_connected**: 54
- **saliency_target_connected**: 66
- **ours_pct**: 0.5400
- **saliency_pct**: 0.6600



