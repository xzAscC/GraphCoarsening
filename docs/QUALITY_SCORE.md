# Quality Tracking Framework

## Layer 1: Automated Gates (Pass/Fail)

| Check | Command | Status |
|-------|---------|--------|
| Lint | N/A | N/A (research codebase, no linter configured) |
| Type Check | N/A | N/A (research codebase, no type checker configured) |
| Tests | N/A | N/A (no formal test suite; validated via experiments) |
| Dependencies | `pip install -r requirements.txt` | OK (torch, torch-geometric, scipy, etc.) |
| Experiment Runner | `bash run_all_experiments.sh` | OK |

## Layer 2: Trend Metrics

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Experiment Coverage | 16/16 | 16/16 | All planned experiments implemented |
| Dataset Coverage | 6/6 | 6/6 | 3 primary (Cora, Citeseer, PubMed) + 3 synthetic (BA-Shapes, Tree-Cycles, Link-Motif) |
| Baseline Coverage | 12 methods | 12 methods | All comparison methods from the paper |
| Statistical Rigor | paired t-test + Wilcoxon | paired t-test + Wilcoxon | Applied to all experiment comparisons |

## Layer 3: Human Rubric (1-5 Scale)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Code Readability | 4 | Well-documented with formal propositions alongside implementation |
| Architecture Fitness | 4 | Clean module separation (src/, experiments/, config.py) |
| Documentation Freshness | 5 | Comprehensive docs added (design docs, execution plans) |
| Result Reproducibility | 4 | Fixed seed=42, deterministic pipelines where possible |

## Scoring History

| Date | Gates | Exp Coverage | Dataset Coverage | Readability | Architecture | Docs | Reproducibility |
|------|-------|-------------|-----------------|-------------|--------------|------|-----------------|
| 2026-06 | 2/2 | 16/16 | 6/6 | 4 | 4 | 5 | 4 |

## Update Schedule

- **Automated Gates**: Every experiment run
- **Trend Metrics**: Per experiment milestone
- **Human Rubric**: Per paper revision or major refactor
