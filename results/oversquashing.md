# Experiment Results: oversquashing

## config

- **device**: cuda
- **seed**: 42
- **num_edges**: 30

## part_a

### barbell_test

- **num_nodes**: 12
#### accuracy_original

- **1**: 1.0000

#### resistance_original

- **1**: 0.4000

- **coarse_nodes_random**: 6
- **coarse_nodes_spectral**: 3


## part_b

### Cora

- **num_test_edges**: 30
- **k_eigenvectors**: 100
#### resistance_stats

- **mean**: 0.0415
- **std**: 0.0791
- **q33**: 0.0026
- **q66**: 0.0259
- **bucket_low_size**: 10
- **bucket_medium_size**: 10
- **bucket_high_size**: 10

#### Occlusion

##### low

- **n_edges**: 10
- **sufficiency**: 0.4000
- **necessity**: 0.5000
- **sparsity**: 0.9903
- **sufficiency_std**: 0.4899
- **necessity_std**: 0.5000

##### medium

- **n_edges**: 10
- **sufficiency**: 0.7000
- **necessity**: 0.3000
- **sparsity**: 0.9798
- **sufficiency_std**: 0.4583
- **necessity_std**: 0.4583

##### high

- **n_edges**: 10
- **sufficiency**: 0.4000
- **necessity**: 0.3000
- **sparsity**: 0.9918
- **sufficiency_std**: 0.4899
- **necessity_std**: 0.4583


#### Saliency

##### low

- **n_edges**: 10
- **sufficiency**: 0.4000
- **necessity**: 0.5000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.4899
- **necessity_std**: 0.5000

##### medium

- **n_edges**: 10
- **sufficiency**: 0.8000
- **necessity**: 0.3000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.4000
- **necessity_std**: 0.4583

##### high

- **n_edges**: 10
- **sufficiency**: 0.3000
- **necessity**: 0.3000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.4583
- **necessity_std**: 0.4583


#### GNNExplainer

##### low

- **n_edges**: 10
- **sufficiency**: 0.1000
- **necessity**: 0.5000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.3000
- **necessity_std**: 0.5000

##### medium

- **n_edges**: 10
- **sufficiency**: 0.2000
- **necessity**: 0.3000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.4000
- **necessity_std**: 0.4583

##### high

- **n_edges**: 10
- **sufficiency**: 0.0000
- **necessity**: 0.3000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.0000
- **necessity_std**: 0.4583


#### Ours

##### low

- **n_edges**: 10
- **sufficiency**: 0.1000
- **necessity**: 0.1000
- **sparsity**: 0.0873
- **sufficiency_std**: 0.3000
- **necessity_std**: 0.3000

##### medium

- **n_edges**: 10
- **sufficiency**: 0.2101
- **necessity**: 0.2101
- **sparsity**: 0.0873
- **sufficiency_std**: 0.3961
- **necessity_std**: 0.3961

##### high

- **n_edges**: 10
- **sufficiency**: 0.2000
- **necessity**: 0.2000
- **sparsity**: 0.1718
- **sufficiency_std**: 0.4000
- **necessity_std**: 0.4000



### Citeseer

- **num_test_edges**: 30
- **k_eigenvectors**: 100
#### resistance_stats

- **mean**: 0.4954
- **std**: 1.4714
- **q33**: 0.0004
- **q66**: 0.0754
- **bucket_low_size**: 10
- **bucket_medium_size**: 10
- **bucket_high_size**: 10

#### Occlusion

##### low

- **n_edges**: 9
- **sufficiency**: 0.1111
- **necessity**: 0.1111
- **sparsity**: 0.9976
- **sufficiency_std**: 0.3143
- **necessity_std**: 0.3143

##### medium

- **n_edges**: 10
- **sufficiency**: 0.5000
- **necessity**: 0.5000
- **sparsity**: 0.9854
- **sufficiency_std**: 0.5000
- **necessity_std**: 0.5000

##### high

- **n_edges**: 10
- **sufficiency**: 0.0000
- **necessity**: 0.5000
- **sparsity**: 0.9978
- **sufficiency_std**: 0.0000
- **necessity_std**: 0.5000


#### Saliency

##### low

- **n_edges**: 10
- **sufficiency**: 0.1000
- **necessity**: 0.2000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.3000
- **necessity_std**: 0.4000

##### medium

- **n_edges**: 10
- **sufficiency**: 0.5000
- **necessity**: 0.5000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.5000
- **necessity_std**: 0.5000

##### high

- **n_edges**: 10
- **sufficiency**: 0.0000
- **necessity**: 0.5000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.0000
- **necessity_std**: 0.5000


#### GNNExplainer

##### low

- **n_edges**: 10
- **sufficiency**: 0.0000
- **necessity**: 0.2000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.0000
- **necessity_std**: 0.4000

##### medium

- **n_edges**: 10
- **sufficiency**: 0.3000
- **necessity**: 0.5000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.4583
- **necessity_std**: 0.5000

##### high

- **n_edges**: 10
- **sufficiency**: 0.1000
- **necessity**: 0.5000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.3000
- **necessity_std**: 0.5000


#### Ours

##### low

- **n_edges**: 10
- **sufficiency**: 0.0000
- **necessity**: 0.0000
- **sparsity**: 0.7014
- **sufficiency_std**: 0.0000
- **necessity_std**: 0.0000

##### medium

- **n_edges**: 10
- **sufficiency**: 0.0000
- **necessity**: 0.0000
- **sparsity**: 0.1807
- **sufficiency_std**: 0.0000
- **necessity_std**: 0.0000

##### high

- **n_edges**: 10
- **sufficiency**: 0.0000
- **necessity**: 0.0000
- **sparsity**: 0.1807
- **sufficiency_std**: 0.0000
- **necessity_std**: 0.0000



### PubMed

- **num_test_edges**: 30
- **k_eigenvectors**: 100
#### resistance_stats

- **mean**: 0.0030
- **std**: 0.0110
- **q33**: 0.0001
- **q66**: 0.0004
- **bucket_low_size**: 10
- **bucket_medium_size**: 10
- **bucket_high_size**: 10

#### Occlusion

##### low

- **n_edges**: 10
- **sufficiency**: 0.6000
- **necessity**: 0.5000
- **sparsity**: 0.9832
- **sufficiency_std**: 0.4899
- **necessity_std**: 0.5000

##### medium

- **n_edges**: 10
- **sufficiency**: 0.3000
- **necessity**: 0.4000
- **sparsity**: 0.9862
- **sufficiency_std**: 0.4583
- **necessity_std**: 0.4899

##### high

- **n_edges**: 10
- **sufficiency**: 0.3000
- **necessity**: 0.5000
- **sparsity**: 0.9963
- **sufficiency_std**: 0.4583
- **necessity_std**: 0.5000


#### Saliency

##### low

- **n_edges**: 10
- **sufficiency**: 0.6000
- **necessity**: 0.5000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.4899
- **necessity_std**: 0.5000

##### medium

- **n_edges**: 10
- **sufficiency**: 0.2000
- **necessity**: 0.4000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.4000
- **necessity_std**: 0.4899

##### high

- **n_edges**: 10
- **sufficiency**: 0.4000
- **necessity**: 0.5000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.4899
- **necessity_std**: 0.5000


#### GNNExplainer

##### low

- **n_edges**: 10
- **sufficiency**: 0.4000
- **necessity**: 0.5000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.4899
- **necessity_std**: 0.5000

##### medium

- **n_edges**: 10
- **sufficiency**: 0.2000
- **necessity**: 0.4000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.4000
- **necessity_std**: 0.4899

##### high

- **n_edges**: 10
- **sufficiency**: 0.0000
- **necessity**: 0.5000
- **sparsity**: 0.5000
- **sufficiency_std**: 0.0000
- **necessity_std**: 0.5000


#### Ours

##### low

- **n_edges**: 10
- **sufficiency**: 0.3000
- **necessity**: 0.3000
- **sparsity**: 0.1782
- **sufficiency_std**: 0.4583
- **necessity_std**: 0.4583

##### medium

- **n_edges**: 10
- **sufficiency**: 0.2000
- **necessity**: 0.2000
- **sparsity**: 0.1781
- **sufficiency_std**: 0.4000
- **necessity_std**: 0.4000

##### high

- **n_edges**: 10
- **sufficiency**: 0.4000
- **necessity**: 0.4000
- **sparsity**: 0.1782
- **sufficiency_std**: 0.4899
- **necessity_std**: 0.4899




