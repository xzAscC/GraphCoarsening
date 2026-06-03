# GraphCoarsening — Agent Notes

## What This Is

Research code for GNN link-prediction explanation via prediction-guided pathway-calibrated graph coarsening. No package manager — flat Python project installed via `pip install -r requirements.txt`.

## Setup

```bash
pip install -r requirements.txt
```

Requires PyTorch, PyG (`torch-geometric`), `torch-scatter`, `torch-sparse`, scipy, numpy, ogb, matplotlib, tqdm, scikit-learn.

No virtual environment or lockfile is committed. No linter, formatter, type checker, or CI is configured.

## Execution Flow

**Mandatory first step**: train a GCN checkpoint before any explanation/experiment.

```bash
# Step 1 — train (creates checkpoints/{Dataset}_gcn.pt)
python experiments/train_gcn.py --dataset Cora --epochs 100

# Step 2 — run experiments (require checkpoint to exist)
python experiments/run_explanations.py --dataset Cora --num_edges 50

# Full pipeline (trains missing checkpoints, runs all experiments)
bash run_all_experiments.sh
```

Experiment scripts **must** be run from the repo root. They add the parent directory to `sys.path` and import directly from `config` and `src.*`.

## Common CLI Arguments

All experiment scripts share these argparse flags:
- `--dataset {Cora,Citeseer,PubMed,...}` — required for most scripts
- `--device {cuda,cpu}` — auto-detected; override with `--device cpu`
- `--seed INT` — default 42
- `--num_edges INT` — number of test edges to explain (default varies)

## Architecture

```
config.py              ← dataclass configuration (ExperimentConfig, SpectralConfig, ModelConfig)
src/                   ← library code (imported by experiments)
  coarsen.py           ← GraphCoarsener: fit() → explain_link(), project_back_edges()
  partition.py         ← UnionFind, node_partition(), prediction_guided_partition()
  spectral.py          ← eigenvalue decomposition, perturbation scores (scipy)
  models/
    gcn.py             ← GCN encoder
    link_predictor.py  ← LinkPredictionModel wrapper
  explainers/
    base.py            ← BaseExplainer ABC
    coarsen_explainer.py ← CoarsenExplainer (main method)
    baselines.py       ← OcclusionExplainer, SaliencyExplainer
    pyg_baselines.py   ← GNNExplainer, PGExplainer, SubgraphX wrappers
  evaluation/
    fidelity.py         ← binary fidelity_plus / fidelity_minus
    comprehensive_metrics.py ← 7-metric evaluation suite
experiments/           ← 17 entry-point scripts (all use argparse)
results/               ← JSON + Markdown result pairs (tracked in git)
```

**Key dependency chain**: `experiments/` → `config.py` + `src.*`. The `train_gcn.py` script exports `load_dataset()` and `MLPLinkPredictor`, which many other experiment scripts import directly.

## Data

- Datasets auto-download to `data/` on first use via PyG (`Planetoid`, `Coauthor`, `Amazon`) and OGB (`ogbl-*`).
- `data/`, `checkpoints/`, `figures/` are gitignored.
- Results (`results/`) are committed — JSON for machine consumption, Markdown for human readability.

## No Test Suite

There are no tests. Verification is running experiment scripts and comparing results.

## Conventions

- Default seed: 42. All experiments respect `--seed`.
- GPU preferred; all scripts auto-fallback to CPU.
- Experiment results follow `{type}_{Dataset}.json` / `.md` naming in `results/`.
- `run_all_experiments.sh` uses `|| echo "FAILED: ..."` per step — individual experiment failures do not halt the pipeline.
- The `convert_results_to_md.py` script regenerates all `.md` files from `.json` outputs.
- PubMed uses different hyperparameters (`lambda_pred=2.0`, `fidelity_threshold=0.95`) due to larger 2-hop subgraphs; Cora and Citeseer use `lambda_pred=1.0`, `fidelity_threshold=0.8`.

## Metrics Caveat

Fidelity metrics in `src/evaluation/fidelity.py` use **binary** fidelity (1.0 if prediction flips, 0.0 otherwise), not continuous score differences. This was changed from an earlier continuous version that inflated results. See `results/METRICS_README.md` for details.
