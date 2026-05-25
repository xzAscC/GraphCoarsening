# Baselines Comparison: Citeseer

## Experimental Setup
- Dataset: Citeseer
- Test edges: 50
- Evaluation: Binary fidelity (prediction flip = 1, no flip = 0)
- Our method: Spectral-Predictive (SP) scoring = gradient × (1 + spectral)

## Results

| Method | Fidelity+ | Fidelity- | Sparsity | Avg Time (s) | Samples |
|--------|-----------|-----------|----------|---------------|---------|
| Ours | 0.2600 | 0.1600 | 0.9906 | 0.0269 | 50 |
| Occlusion | 0.1702 | 0.1702 | 0.9900 | 0.1297 | 47 |
| Random | 0.1400 | 0.1800 | 0.9906 | 0.0005 | 50 |
| Saliency | 0.1800 | 0.1600 | 0.5000 | 0.0031 | 50 |
| FullGraph | 0.1800 | 0.0000 | 0.0000 | 0.0001 | 50 |
| KHop | 0.1800 | 0.0600 | 0.9813 | 0.0005 | 50 |
| Degree | 0.2200 | 0.2000 | 0.9906 | 0.0005 | 50 |

## Key Findings
- **Ours achieves highest Fidelity+ across all baselines**
- SP scoring combines spectral perturbation (structural) with gradient saliency (predictive)
- High sparsity (~0.99) indicates compact explanations (~1% of original edges)
- Faster than Occlusion (no per-edge forward passes needed)
