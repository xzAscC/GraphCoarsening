# Experiment Results: baselines_comparison_Citeseer

- **dataset**: Citeseer
- **num_edges**: 50
## methods

### FullGraph

- **num_samples**: 50
- **mean_fidelity_plus**: 0.1400
- **std_fidelity_plus**: 0.3470
- **mean_fidelity_minus**: 0.0000
- **std_fidelity_minus**: 0.0000
- **mean_sparsity**: 0.0000
- **std_sparsity**: 0.0000
- **mean_time**: 0.0048
- **mean_explanation_size**: 7740.0000

### KHop

- **num_samples**: 50
- **mean_fidelity_plus**: 0.1400
- **std_fidelity_plus**: 0.3470
- **mean_fidelity_minus**: 0.0800
- **std_fidelity_minus**: 0.2713
- **mean_sparsity**: 0.9813
- **std_sparsity**: 0.0309
- **mean_time**: 0.0007
- **mean_explanation_size**: 28.0000

### Random

- **num_samples**: 50
- **mean_fidelity_plus**: 0.1600
- **std_fidelity_plus**: 0.3666
- **mean_fidelity_minus**: 0.1600
- **std_fidelity_minus**: 0.3666
- **mean_sparsity**: 0.9906
- **std_sparsity**: 0.0155
- **mean_time**: 0.0008
- **mean_explanation_size**: 14.0000

### Degree

- **num_samples**: 50
- **mean_fidelity_plus**: 0.1400
- **std_fidelity_plus**: 0.3470
- **mean_fidelity_minus**: 0.1800
- **std_fidelity_minus**: 0.3842
- **mean_sparsity**: 0.9906
- **std_sparsity**: 0.0155
- **mean_time**: 0.0007
- **mean_explanation_size**: 14.0000


## baseline_groups

- **trivial**: ['FullGraph', 'KHop', 'Random', 'Degree']
- **hard**: ['GreedyDel']
- **coarsening**: ['RandomCoarse', 'HeavyEdge', 'EffResist', 'NoRefine']
- **gnn**: ['Occlusion', 'Saliency']
- **ours**: ['Ours']

