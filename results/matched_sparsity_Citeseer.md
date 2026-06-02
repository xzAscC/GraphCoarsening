# Experiment Results: matched_sparsity_Citeseer

- **dataset**: Citeseer
- **num_test_edges**: 100
- **seed**: 42
- **budgets**: mean=21.2500, std=17.4553, n=4
- **methods**: ['CoarsenExplainer', 'SaliencyExplainer']
## results

### 5

- **n**: 100
- **budget**: 5
#### sufficiency

- **ours_mean**: 0.1424
- **saliency_mean**: 0.1679
- **ttest_p**: 0.0041
- **wilcoxon_p**: 0.0089
- **cohens_d**: -0.2942
- **ours_wins**: True

#### necessity

- **ours_mean**: 0.0508
- **saliency_mean**: 0.0708
- **ttest_p**: 0.0424
- **wilcoxon_p**: 0.0237
- **cohens_d**: -0.2056
- **ours_wins**: False

#### fidelity_plus_continuous

- **ours_mean**: 0.4755
- **saliency_mean**: 0.6557
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0001
- **cohens_d**: -0.4443
- **ours_wins**: False

#### num_components

- **ours_mean**: 22.8900
- **saliency_mean**: 2512.2700
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0000
- **cohens_d**: -17.4749
- **ours_wins**: True

#### target_connectivity

- **ours_target_connected**: 9
- **saliency_target_connected**: 14
- **ours_pct**: 0.0900
- **saliency_pct**: 0.1400


### 10

- **n**: 100
- **budget**: 10
#### sufficiency

- **ours_mean**: 0.1373
- **saliency_mean**: 0.1586
- **ttest_p**: 0.0397
- **wilcoxon_p**: 0.0645
- **cohens_d**: -0.2085
- **ours_wins**: True

#### necessity

- **ours_mean**: 0.0525
- **saliency_mean**: 0.0544
- **ttest_p**: 0.8977
- **wilcoxon_p**: 0.9234
- **cohens_d**: -0.0129
- **ours_wins**: False

#### fidelity_plus_continuous

- **ours_mean**: 0.5570
- **saliency_mean**: 0.7704
- **ttest_p**: 0.0004
- **wilcoxon_p**: 0.0009
- **cohens_d**: -0.3648
- **ours_wins**: False

#### num_components

- **ours_mean**: 21.5700
- **saliency_mean**: 2508.4100
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0000
- **cohens_d**: -17.4539
- **ours_wins**: True

#### target_connectivity

- **ours_target_connected**: 18
- **saliency_target_connected**: 33
- **ours_pct**: 0.1800
- **saliency_pct**: 0.3300


### 20

- **n**: 100
- **budget**: 20
#### sufficiency

- **ours_mean**: 0.1292
- **saliency_mean**: 0.1458
- **ttest_p**: 0.0770
- **wilcoxon_p**: 0.0884
- **cohens_d**: -0.1787
- **ours_wins**: True

#### necessity

- **ours_mean**: 0.0666
- **saliency_mean**: 0.0134
- **ttest_p**: 0.0009
- **wilcoxon_p**: 0.0002
- **cohens_d**: 0.3422
- **ours_wins**: True

#### fidelity_plus_continuous

- **ours_mean**: 0.7057
- **saliency_mean**: 0.7283
- **ttest_p**: 0.7114
- **wilcoxon_p**: 0.9253
- **cohens_d**: -0.0371
- **ours_wins**: False

#### num_components

- **ours_mean**: 19.0400
- **saliency_mean**: 2501.5300
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0000
- **cohens_d**: -17.5667
- **ours_wins**: True

#### target_connectivity

- **ours_target_connected**: 27
- **saliency_target_connected**: 41
- **ours_pct**: 0.2700
- **saliency_pct**: 0.4100


### 50

- **n**: 100
- **budget**: 50
#### sufficiency

- **ours_mean**: 0.1239
- **saliency_mean**: 0.1410
- **ttest_p**: 0.0447
- **wilcoxon_p**: 0.0631
- **cohens_d**: -0.2033
- **ours_wins**: True

#### necessity

- **ours_mean**: 0.0800
- **saliency_mean**: -0.0043
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0000
- **cohens_d**: 0.5799
- **ours_wins**: True

#### fidelity_plus_continuous

- **ours_mean**: 0.7789
- **saliency_mean**: 0.7070
- **ttest_p**: 0.2399
- **wilcoxon_p**: 0.5154
- **cohens_d**: 0.1182
- **ours_wins**: True

#### num_components

- **ours_mean**: 13.9800
- **saliency_mean**: 2481.9900
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0000
- **cohens_d**: -18.3330
- **ours_wins**: True

#### target_connectivity

- **ours_target_connected**: 35
- **saliency_target_connected**: 50
- **ours_pct**: 0.3500
- **saliency_pct**: 0.5000



