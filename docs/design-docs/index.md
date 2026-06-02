# Design Documents

This directory holds design decisions and architectural rationale.

## Index

| Doc | Status | Date | Summary |
|-----|--------|------|---------|
| [method-overview.md](method-overview.md) | Approved | 2026-05 | Core algorithm, formal propositions, empirical findings, and code module mapping |
| [matched-sparsity-evaluation.md](matched-sparsity-evaluation.md) | Approved | 2026-05 | Primary matched-sparsity experiment with sufficiency, necessity, and Fidelity+ tables across three datasets |
| [comprehensive-metrics.md](comprehensive-metrics.md) | Approved | 2026-05 | Seven evaluation metrics and twelve-method baselines comparison |
| [pareto-analysis.md](pareto-analysis.md) | Approved | 2026-05 | Pareto curves at matched budgets with size-vs-quality tradeoffs |
| [ablation-studies.md](ablation-studies.md) | Approved | 2026-05 | Coarsening ratio, hyperparameter, and refinement ablation experiments |
| [robustness-analysis.md](robustness-analysis.md) | Approved | 2026-05 | Oversmoothing, oversquashing, multi-backbone, and ground truth evaluation |
| [scalability-profiling.md](scalability-profiling.md) | Approved | 2026-05 | Runtime scaling and per-stage profiling across six datasets |

## Adding New Design Docs

1. Create a new `.md` file with descriptive name
2. Include: Context, Decision, Consequences
3. Add entry to index table above
4. Update status as doc evolves (Draft -> Approved -> Superseded)

## Naming Convention

- Use kebab-case: `feature-name.md`
- Be descriptive: `auth-strategy.md` not `auth.md`
