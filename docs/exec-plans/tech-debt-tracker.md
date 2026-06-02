# Technical Debt Tracker

Track technical debt items for periodic cleanup.

## Active Debt

| ID | Description | Severity | Category | Resolution Plan | Status |
|----|-------------|----------|----------|-----------------|--------|
| TD-001 | Cora GCN AUC=0.655 — weak model limits explanation quality | High | model | Consider deeper GCN or GraphSAGE for Cora | Open |
| TD-002 | PubMed k-hop covers full graph (sparsity=0.002) | High | evaluation | Use adaptive k-hop or local subgraph sampling | Open |
| TD-003 | Ours raw fidelity+=0.10 vs Occlusion=0.40 (unfair at unmatched budgets) | Medium | metrics | Always report matched-budget Pareto curves | Open |
| TD-004 | Ours explanation size ~91% of edges at default settings | Medium | algorithm | Tune alpha and k_frac for each dataset | Open |
| TD-005 | Coarsening CUDA bug: RandomCoarse/EffResist only 8/20 samples | Medium | code | Fix index-out-of-bounds in partition.py | Open |
| TD-006 | PubMed comprehensive necessity negative (-0.211) | Medium | evaluation | Requires lambda_pred=2.0; document parameter sensitivity | Open |
| TD-007 | No formal test suite | Low | test | Add pytest infrastructure and unit tests for core modules | Open |

## Severity Levels

- **High**: Blocks development or causes frequent issues
- **Medium**: Causes friction but has workarounds
- **Low**: Minor inconvenience, fix when convenient

## Categories

- **model**: Model architecture or training issues
- **evaluation**: Evaluation methodology or metric issues
- **metrics**: Metric definitions or reporting issues
- **algorithm**: Algorithm design or parameter issues
- **code**: Code quality issues (duplication, complexity, bugs)
- **test**: Missing or inadequate tests
- **docs**: Missing or outdated documentation
- **infra**: Build, deploy, or infrastructure issues

## Adding New Debt

1. Assign unique ID (TD-001, TD-002, etc.)
2. Describe the issue and its impact
3. Categorize and set severity
4. Note when/where it was introduced
5. Plan resolution approach

## Resolving Debt

1. Create exec plan in `docs/exec-plans/active/`
2. Implement fix
3. Move entry to **Resolved** section below
4. Archive exec plan to `docs/exec-plans/completed/`

## Resolved Debt

| ID | Description | Resolved | PR |
|----|-------------|----------|-----|
| - | - | - | - |

## Periodic Maintenance

This tracker supports the periodic agent workflow:
- Review weekly during sprint planning
- Prioritize High severity items
- Schedule dedicated debt-reduction sprints
