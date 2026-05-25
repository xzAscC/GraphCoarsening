# Baselines Comparison: PubMed

## Experimental Setup
- Dataset: PubMed
- Test edges: 50
- Evaluation: Binary fidelity (prediction flip = 1, no flip = 0)
- Our method: Spectral-Predictive (SP) scoring = gradient × (1 + spectral)

## Results

| Method | Fidelity+ | Fidelity- | Sparsity | Avg Time (s) | Samples |
|--------|-----------|-----------|----------|---------------|---------|
| Ours | 0.4600 | 0.3200 | 0.9900 | 1.1657 | 50 |
| Occlusion | 0.4200 | 0.3000 | 0.9900 | 1.9919 | 50 |
| Random | 0.3600 | 0.2600 | 0.9900 | 0.0006 | 50 |
| Saliency | 0.4200 | 0.3400 | 0.5000 | 0.0044 | 50 |
| FullGraph | 0.4200 | 0.0000 | 0.0000 | 0.0001 | 50 |
| KHop | 0.4200 | 0.2600 | 0.9800 | 0.0006 | 50 |
| Degree | 0.2800 | 0.3400 | 0.9900 | 0.0005 | 50 |

## Key Findings
- **Ours achieves highest Fidelity+ across all baselines**
- SP scoring combines spectral perturbation (structural) with gradient saliency (predictive)
- High sparsity (~0.99) indicates compact explanations (~1% of original edges)
- Faster than Occlusion (no per-edge forward passes needed)
