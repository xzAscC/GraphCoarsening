# Experiment Results: baselines_comparison_Cora

- **dataset**: Cora
- **num_edges**: 50
## methods

### FullGraph

- **num_samples**: 50
- **mean_fidelity_plus**: 0.2600
- **std_fidelity_plus**: 0.4386
- **mean_fidelity_minus**: 0.0000
- **std_fidelity_minus**: 0.0000
- **mean_sparsity**: 0.0000
- **std_sparsity**: 0.0000
- **mean_time**: 0.0001
- **mean_explanation_size**: 8976.0000

### KHop

- **num_samples**: 50
- **mean_fidelity_plus**: 0.2600
- **std_fidelity_plus**: 0.4386
- **mean_fidelity_minus**: 0.0000
- **std_fidelity_minus**: 0.0000
- **mean_sparsity**: 0.9737
- **std_sparsity**: 0.0340
- **mean_time**: 0.0005
- **mean_explanation_size**: 136.0000

### Random

- **num_samples**: 50
- **mean_fidelity_plus**: 0.1800
- **std_fidelity_plus**: 0.3842
- **mean_fidelity_minus**: 0.1200
- **std_fidelity_minus**: 0.3250
- **mean_sparsity**: 0.9869
- **std_sparsity**: 0.0170
- **mean_time**: 0.0005
- **mean_explanation_size**: 68.0000

### Degree

- **num_samples**: 50
- **mean_fidelity_plus**: 0.2600
- **std_fidelity_plus**: 0.4386
- **mean_fidelity_minus**: 0.0800
- **std_fidelity_minus**: 0.2713
- **mean_sparsity**: 0.9869
- **std_sparsity**: 0.0170
- **mean_time**: 0.0008
- **mean_explanation_size**: 68.0000


## baseline_groups

- **trivial**: ['FullGraph', 'KHop', 'Random', 'Degree']
- **hard**: ['GreedyDel']
- **coarsening**: ['RandomCoarse', 'HeavyEdge', 'EffResist', 'NoRefine']
- **gnn**: ['Occlusion', 'Saliency']
- **ours**: ['Ours']

