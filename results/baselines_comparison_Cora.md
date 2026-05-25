# Experiment Results: baselines_comparison_Cora

- **dataset**: Cora
- **num_edges**: 20
## methods

### FullGraph

- **num_samples**: 20
- **mean_fidelity_plus**: 0.4000
- **std_fidelity_plus**: 0.4899
- **mean_fidelity_minus**: 0.0000
- **std_fidelity_minus**: 0.0000
- **mean_sparsity**: 0.0000
- **std_sparsity**: 0.0000
- **mean_time**: 0.0003
- **mean_explanation_size**: 8976.0000

### KHop

- **num_samples**: 20
- **mean_fidelity_plus**: 0.4000
- **std_fidelity_plus**: 0.4899
- **mean_fidelity_minus**: 0.1000
- **std_fidelity_minus**: 0.3000
- **mean_sparsity**: 0.9813
- **std_sparsity**: 0.0242
- **mean_time**: 0.0030
- **mean_explanation_size**: 340.0000

### Random

- **num_samples**: 20
- **mean_fidelity_plus**: 0.4000
- **std_fidelity_plus**: 0.4899
- **mean_fidelity_minus**: 0.2500
- **std_fidelity_minus**: 0.4330
- **mean_sparsity**: 0.9906
- **std_sparsity**: 0.0121
- **mean_time**: 0.0031
- **mean_explanation_size**: 170.0000

### Degree

- **num_samples**: 20
- **mean_fidelity_plus**: 0.3000
- **std_fidelity_plus**: 0.4583
- **mean_fidelity_minus**: 0.2500
- **std_fidelity_minus**: 0.4330
- **mean_sparsity**: 0.9906
- **std_sparsity**: 0.0121
- **mean_time**: 0.0035
- **mean_explanation_size**: 170.0000

### GreedyDel

- **num_samples**: 20
- **mean_fidelity_plus**: 0.2500
- **std_fidelity_plus**: 0.4330
- **mean_fidelity_minus**: 0.0500
- **std_fidelity_minus**: 0.2179
- **mean_sparsity**: 0.9893
- **std_sparsity**: 0.0225
- **mean_time**: 8.8845
- **mean_explanation_size**: 240.0000

### RandomCoarse

- **num_samples**: 8
- **mean_fidelity_plus**: 0.5000
- **std_fidelity_plus**: 0.5000
- **mean_fidelity_minus**: 0.5000
- **std_fidelity_minus**: 0.5000
- **mean_sparsity**: 0.1720
- **std_sparsity**: 0.2617
- **mean_time**: 0.0345
- **mean_explanation_size**: 8320.0000

### HeavyEdge

- **num_samples**: 14
- **mean_fidelity_plus**: 0.0000
- **std_fidelity_plus**: 0.0000
- **mean_fidelity_minus**: 0.0000
- **std_fidelity_minus**: 0.0000
- **mean_sparsity**: 0.0234
- **std_sparsity**: 0.0000
- **mean_time**: 0.0288
- **mean_explanation_size**: 8766.0000

### EffResist

- **num_samples**: 8
- **mean_fidelity_plus**: 0.5000
- **std_fidelity_plus**: 0.5000
- **mean_fidelity_minus**: 0.5000
- **std_fidelity_minus**: 0.5000
- **mean_sparsity**: 0.4477
- **std_sparsity**: 0.2908
- **mean_time**: 0.6679
- **mean_explanation_size**: 6979.0000

### NoRefine

- **num_samples**: 20
- **mean_fidelity_plus**: 0.6000
- **std_fidelity_plus**: 0.4899
- **mean_fidelity_minus**: 0.6000
- **std_fidelity_minus**: 0.4899
- **mean_sparsity**: 0.8473
- **std_sparsity**: 0.0000
- **mean_time**: 0.0171
- **mean_explanation_size**: 1371.0000

### Occlusion

- **num_samples**: 20
- **mean_fidelity_plus**: 0.4000
- **std_fidelity_plus**: 0.4899
- **mean_fidelity_minus**: 0.5000
- **std_fidelity_minus**: 0.5000
- **mean_sparsity**: 0.9906
- **std_sparsity**: 0.0121
- **mean_time**: 0.1467
- **mean_explanation_size**: 170.0000

### Saliency

- **num_samples**: 20
- **mean_fidelity_plus**: 0.4000
- **std_fidelity_plus**: 0.4899
- **mean_fidelity_minus**: 0.5000
- **std_fidelity_minus**: 0.5000
- **mean_sparsity**: 0.5000
- **std_sparsity**: 0.0000
- **mean_time**: 0.0118
- **mean_explanation_size**: 4488.0000

### Ours

- **num_samples**: 20
- **mean_fidelity_plus**: 0.1000
- **std_fidelity_plus**: 0.3000
- **mean_fidelity_minus**: 0.1000
- **std_fidelity_minus**: 0.3000
- **mean_sparsity**: 0.0873
- **std_sparsity**: 0.2533
- **mean_time**: 0.0266
- **mean_explanation_size**: 8950.0000


## baseline_groups

- **trivial**: ['FullGraph', 'KHop', 'Random', 'Degree']
- **hard**: ['GreedyDel']
- **coarsening**: ['RandomCoarse', 'HeavyEdge', 'EffResist', 'NoRefine']
- **gnn**: ['Occlusion', 'Saliency']
- **ours**: ['Ours']

