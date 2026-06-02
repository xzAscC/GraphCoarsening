# Experiment Results: matched_sparsity_PubMed

- **dataset**: PubMed
- **num_test_edges**: 30
- **seed**: 42
- **budgets**: mean=11.6667, std=6.2361, n=3
- **methods**: ['CoarsenExplainer', 'SaliencyExplainer']
## results

### 5

- **n**: 30
- **budget**: 5
#### sufficiency

- **ours_mean**: 0.2271
- **saliency_mean**: 0.2158
- **ttest_p**: 0.4723
- **wilcoxon_p**: 0.2801
- **cohens_d**: 0.1329
- **ours_wins**: False

#### necessity

- **ours_mean**: 0.1651
- **saliency_mean**: 0.3158
- **ttest_p**: 0.0003
- **wilcoxon_p**: 0.0000
- **cohens_d**: -0.7443
- **ours_wins**: False

#### fidelity_plus_continuous

- **ours_mean**: 1.1170
- **saliency_mean**: 1.9901
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0000
- **cohens_d**: -0.9854
- **ours_wins**: False

#### num_components

- **ours_mean**: 154.3667
- **saliency_mean**: 14771.0667
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0000
- **cohens_d**: -22.6318
- **ours_wins**: True

#### target_connectivity

- **ours_target_connected**: 0
- **saliency_target_connected**: 0
- **ours_pct**: 0.0000
- **saliency_pct**: 0.0000


### 10

- **n**: 30
- **budget**: 10
#### sufficiency

- **ours_mean**: 0.2220
- **saliency_mean**: 0.2115
- **ttest_p**: 0.5892
- **wilcoxon_p**: 0.4771
- **cohens_d**: 0.0997
- **ours_wins**: False

#### necessity

- **ours_mean**: 0.2436
- **saliency_mean**: 0.3379
- **ttest_p**: 0.0008
- **wilcoxon_p**: 0.0000
- **cohens_d**: -0.6831
- **ours_wins**: False

#### fidelity_plus_continuous

- **ours_mean**: 1.7068
- **saliency_mean**: 2.4977
- **ttest_p**: 0.0001
- **wilcoxon_p**: 0.0000
- **cohens_d**: -0.8638
- **ours_wins**: False

#### num_components

- **ours_mean**: 152.3667
- **saliency_mean**: 14766.6333
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0000
- **cohens_d**: -22.6223
- **ours_wins**: True

#### target_connectivity

- **ours_target_connected**: 0
- **saliency_target_connected**: 1
- **ours_pct**: 0.0000
- **saliency_pct**: 0.0333


### 20

- **n**: 30
- **budget**: 20
#### sufficiency

- **ours_mean**: 0.2098
- **saliency_mean**: 0.2057
- **ttest_p**: 0.8373
- **wilcoxon_p**: 0.8078
- **cohens_d**: 0.0378
- **ours_wins**: False

#### necessity

- **ours_mean**: 0.3469
- **saliency_mean**: 0.3464
- **ttest_p**: 0.9927
- **wilcoxon_p**: 0.8712
- **cohens_d**: 0.0017
- **ours_wins**: True

#### fidelity_plus_continuous

- **ours_mean**: 2.5115
- **saliency_mean**: 2.2688
- **ttest_p**: 0.4883
- **wilcoxon_p**: 0.7303
- **cohens_d**: 0.1282
- **ours_wins**: True

#### num_components

- **ours_mean**: 147.4000
- **saliency_mean**: 14759.4667
- **ttest_p**: 0.0000
- **wilcoxon_p**: 0.0000
- **cohens_d**: -22.5846
- **ours_wins**: True

#### target_connectivity

- **ours_target_connected**: 2
- **saliency_target_connected**: 4
- **ours_pct**: 0.0667
- **saliency_pct**: 0.1333



