# Experiment Results: explanation_PubMed

- **dataset**: PubMed
- **num_edges**: 10
## methods

### Ours

- **fidelity_plus**: mean=0.8866, std=0.2664, n=10
- **fidelity_minus**: mean=0.8866, std=0.2664, n=10
- **mean_fidelity_plus**: 0.8866
- **mean_fidelity_minus**: 0.8866
- **std_fidelity_plus**: 0.2664
- **std_fidelity_minus**: 0.2664
- **mean_time**: 34.0331

### Occlusion

- **fidelity_plus**: mean=0.5000, std=0.5000, n=20
- **fidelity_minus**: mean=0.4500, std=0.4975, n=20
- **mean_fidelity_plus**: 0.5000
- **mean_fidelity_minus**: 0.4500
- **std_fidelity_plus**: 0.5000
- **std_fidelity_minus**: 0.4975
- **mean_time**: 10.0510
- **note**: num_edges=20

### Saliency

- **mean_fidelity_plus**: 0.5000
- **mean_fidelity_minus**: 0.4000
- **std_fidelity_plus**: 0.5000
- **std_fidelity_minus**: 0.4899
- **mean_time**: 0.0296
- **note**: num_edges=20

### GNNExplainer

- **mean_fidelity_plus**: 0.5000
- **mean_fidelity_minus**: 0.1000
- **std_fidelity_plus**: 0.5000
- **std_fidelity_minus**: 0.3000
- **mean_time**: 1.2300
- **note**: num_edges=20


- **num_edges_note**: Baselines run with num_edges=20, Ours with num_edges=10
