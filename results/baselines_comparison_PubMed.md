# Experiment Results: baselines_comparison_PubMed

- **dataset**: PubMed
- **num_edges**: 10
## methods

### FullGraph

- **num_samples**: 10
- **mean_fidelity_plus**: 0.6000
- **std_fidelity_plus**: 0.4899
- **mean_fidelity_minus**: 0.0000
- **std_fidelity_minus**: 0.0000
- **mean_sparsity**: 0.0000
- **std_sparsity**: 0.0000
- **mean_time**: 0.0038
- **mean_explanation_size**: 75352.0000

### KHop

- **num_samples**: 10
- **mean_fidelity_plus**: 0.6000
- **std_fidelity_plus**: 0.4899
- **mean_fidelity_minus**: 0.1000
- **std_fidelity_minus**: 0.3000
- **mean_sparsity**: 0.9701
- **std_sparsity**: 0.0306
- **mean_time**: 0.0015
- **mean_explanation_size**: 256.0000

### Ours

- **num_samples**: 10
- **mean_fidelity_plus**: 0.7000
- **std_fidelity_plus**: 0.4583
- **mean_fidelity_minus**: 0.2000
- **std_fidelity_minus**: 0.4000
- **mean_sparsity**: 0.9851
- **std_sparsity**: 0.0153
- **mean_time**: 11.9699
- **mean_explanation_size**: 128.0000


## baseline_groups

- **trivial**: ['FullGraph', 'KHop', 'Random', 'Degree']
- **hard**: ['GreedyDel']
- **coarsening**: ['RandomCoarse', 'HeavyEdge', 'EffResist', 'NoRefine']
- **gnn**: ['Occlusion', 'Saliency']
- **ours**: ['Ours']

