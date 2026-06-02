# Project Plans & Roadmap

Graph coarsening research: "Prediction-Guided Graph Coarsening via Pathway-Calibrated Gradient Selection"

## Backlog

- [ ] [High] Run experiments on larger OGB datasets (ogbl-ppa, ogbl-collab, ogbl-ddi)
- [ ] [High] Add GNNExplainer and SubgraphX baselines
- [ ] [Medium] Hyperparameter tuning for PubMed (current lambda_pred=2.0 is manually tuned)
- [ ] [Medium] Improve Cora GCN model quality (current AUC=0.655)
- [ ] [Low] Add visualization tools for coarse graphs
- [ ] [Low] Benchmark on directed graphs

## Done

- [x] [High] Prediction-Guided Pathway-Calibrated Gradient Selection
- [x] [High] Matched sparsity comparison across Cora, Citeseer, PubMed
- [x] [High] Comprehensive experiment suite (16 experiments)
- [x] [High] Formal propositions (P1, P2) and empirical findings (E1-E3)
- [x] [Medium] Ground truth evaluation (BA-Shapes, Tree-Cycles, Link-Motif)

## How to Use

1. Add new items to **Backlog** with priority (High/Medium/Low)
2. Link to execution plan in `docs/exec-plans/`
3. Move to **In Progress** when work begins
4. Move to **Done** when complete and verified

## Maintaining This Plan

- Review weekly
- Archive completed items quarterly
- Keep linked exec-plans updated
