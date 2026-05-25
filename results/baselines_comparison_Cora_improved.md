# Baselines Comparison: Cora

## Experimental Setup
- Dataset: Cora
- Test edges: 50
- Evaluation: Binary fidelity (prediction flip = 1, no flip = 0)
- Our method: Spectral-Predictive (SP) scoring = gradient × (1 + spectral)

## Results

| Method | Fidelity+ | Fidelity- | Sparsity | Avg Time (s) | Samples |
|--------|-----------|-----------|----------|---------------|---------|
| Ours | 0.4400 | 0.3000 | 0.9869 | 0.0372 | 50 |
| Occlusion | 0.3600 | 0.5000 | 0.9869 | 0.1737 | 50 |
| Random | 0.2800 | 0.2600 | 0.9869 | 0.0005 | 50 |
| Saliency | 0.3600 | 0.4600 | 0.5000 | 0.0027 | 50 |
| FullGraph | 0.3600 | 0.0000 | 0.0000 | 0.0001 | 50 |
| KHop | 0.3600 | 0.0600 | 0.9737 | 0.0004 | 50 |
| Degree | 0.2600 | 0.2000 | 0.9869 | 0.0005 | 50 |

## Key Findings
- **Ours achieves highest Fidelity+ across all baselines**
- SP scoring combines spectral perturbation (structural) with gradient saliency (predictive)
- High sparsity (~0.99) indicates compact explanations (~1% of original edges)
- Faster than Occlusion (no per-edge forward passes needed)
