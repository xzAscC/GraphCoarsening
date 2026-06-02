# Scalability Profiling

Status: Approved
Date: 2026-05
Branch: main

## Context

GraphCoarsening needs runtime evidence for two related claims. First, the one time coarsening pass should remain practical as graph size grows. Second, per link explanation should be fast enough to support repeated evaluation over sampled target edges. The project measures these claims with `experiments/run_runtime.py` and `experiments/run_profiling.py`.

The runtime experiment loads each dataset in `ExperimentConfig.scalability_datasets`, times `GraphCoarsener.fit()`, and then measures average `GraphCoarsener.explain_link()` latency over sampled positive edges. The coarsening timer covers normalized adjacency construction, spectral decomposition, perturbation scoring, node partitioning, and coarse graph construction as one setup cost. The per link timer measures the cached explanation path after that setup cost has been paid.

The profiling experiment compares Occlusion, Saliency, GNNExplainer, and Ours on a selected dataset. It records explainer construction time, construction memory, mean explanation time, forward pass time, peak memory, batch query latency, and sample count. On Cora, the recorded profiling artifact uses 10 valid samples per method and reports peak memory through CUDA allocation counters when CUDA is available, otherwise through Python `tracemalloc`.

## Decision

The project treats scalability as a two level measurement problem. Dataset level runtime scaling is reported from `results/runtime.md`, while method level profiling is reported from `results/profiling_Cora.md`. The approved scalability table uses the six datasets present in the runtime artifact. The broader experiment code can attempt nine datasets, including `ogbl-ppa`, `ogbl-collab`, and `ogbl-ddi`, but those entries are not present in the approved result file and are therefore not reported as measured results here.

### Runtime Scaling

| Dataset | Nodes | Edges | Coarsening time (s) | Avg explanation time (s) |
|---|---:|---:|---:|---:|
| Cora | 2,708 | 8,976 | 0.4912 | 0.0120 |
| Citeseer | 3,327 | 7,740 | 0.4930 | 0.0280 |
| PubMed | 19,717 | 75,352 | 2.5134 | 0.0528 |
| Coauthor-CS | 18,333 | 139,222 | 2.2279 | 0.1434 |
| Coauthor-Physics | 34,493 | 421,536 | 3.1264 | 0.2999 |
| Amazon-Computers | 13,752 | 417,964 | 1.6594 | 0.0548 |

The measured setup time stays below 3.2 seconds for all six reported datasets. Per link explanation time grows with graph and neighborhood size, reaching 0.2999 seconds on Coauthor-Physics, the largest reported graph by nodes and edges.

### Profiling Breakdown

The profiling output records method level timing and memory. For the project method, `CoarsenExplainer` fits the coarsener lazily during `explain_link()`, so spectral decomposition, partitioning, and coarse graph construction are included in the measured explanation path rather than in the constructor time. The table below maps the pipeline stages to the available measurements on Cora.

| Pipeline stage | Measurement source | Time on Cora (s) | Memory on Cora (MB) | Interpretation |
|---|---|---:|---:|---|
| Spectral decomposition | Included in `Ours.mean_explain_time_s` and runtime `coarsening_time_s` | 0.0510 mean explanation, 0.4912 full fit | 212.28 peak | The stage is part of the cached coarsener fit. The profiler does not isolate eigenpair computation from the rest of the fit. |
| Partition | Included in `GraphCoarsener.fit()` and prediction guided partition inside `CoarsenExplainer.explain_link()` | 0.0510 mean explanation, 0.4912 full fit | 212.28 peak | The runtime script measures default partitioning in the setup timer. The profiling path also includes target protected prediction guided partitioning. |
| Coarse graph construction | Included in `GraphCoarsener.fit()` | 0.4912 full fit | 212.28 peak | Coarse edge construction is part of the setup cost saved before repeated link explanations. |
| Explanation | `Ours.mean_explain_time_s`, `Ours.mean_forward_time_s`, and `Ours.batch_times` | 0.0510 mean explanation, 0.0016 forward | 212.28 peak | Per query explanation includes pathway mapping, group occlusion, calibration, and top edge selection. |

### Method Profiling on Cora

| Method | Preprocess time (s) | Preprocess memory (MB) | Mean explanation time (s) | Mean forward time (s) | Peak memory (MB) | Samples |
|---|---:|---:|---:|---:|---:|---:|
| Occlusion | 0.0004 | 1.78 | 0.3758 | 0.0015 | 50.76 | 10 |
| Saliency | 0.0004 | 33.03 | 0.0162 | 0.0022 | 92.85 | 10 |
| GNNExplainer | 0.0005 | 42.05 | 0.4023 | 0.0020 | 93.40 | 10 |
| Ours | 0.0003 | 42.05 | 0.0510 | 0.0016 | 212.28 | 10 |

The Cora profiling run places the project method between the two cheaper gradient baselines and the slower perturbation or optimization baselines. It is slower than Saliency because it computes pathway level group effects, but it is faster than Occlusion and GNNExplainer in the recorded mean explanation time.

### Batch Query Timing on Cora

| Method | 1 query (s) | 10 queries (s) | 100 query setting (s) | 1000 query setting (s) |
|---|---:|---:|---:|---:|
| Occlusion | 0.1331 | 2.3865 | 2.2984 | 2.1782 |
| Saliency | 0.0082 | 0.0459 | 0.0495 | 0.0470 |
| GNNExplainer | 0.2821 | 2.8338 | 3.1455 | 3.0608 |
| Ours | 0.0072 | 0.1803 | 0.1810 | 0.1815 |

The larger batch columns reuse the available 10 sampled edges because `run_profiling.py` sets `actual_size = min(batch_size, n)`. They should therefore be read as repeated timing of the capped sample set, not as successful profiling of 100 or 1000 distinct Cora queries.

### Experiment Parameters

| Parameter | Runtime experiment | Profiling experiment | Source |
|---|---|---|---|
| Dataset list | `ExperimentConfig.scalability_datasets` | Command line `--dataset`, default `Cora` | `config.py`, `experiments/run_runtime.py`, `experiments/run_profiling.py` |
| Reported datasets | Cora, Citeseer, PubMed, Coauthor-CS, Coauthor-Physics, Amazon-Computers | Cora | `results/runtime.md`, `results/profiling_Cora.md` |
| Device | `ExperimentConfig.device`, default `cuda`, falls back to `cpu` when CUDA is unavailable | Same | `config.py` and experiment scripts |
| Seed | `42` by default | `42` by default | `ExperimentConfig.seed` and CLI defaults |
| Sampled edges | `--num_edges`, default `50` | `--num_edges`, default `50`; recorded Cora output has 10 valid samples | `run_runtime.py`, `run_profiling.py` |
| Coarsening `k` | `100` | `100` for Ours | `SpectralConfig.k`, `profile_method()` |
| Coarsening `alpha` | `0.75` | `0.75` for Ours | `SpectralConfig.alpha`, `profile_method()` |
| Baseline methods | Not used | Occlusion, Saliency, GNNExplainer, Ours | `ALL_METHODS` in `run_profiling.py` |
| Baseline sparsity | Not used | `k_frac=0.5` for Occlusion and Saliency; `k_frac=0.5` for GNNExplainer | `profile_method()` |
| GNNExplainer epochs | Not used | `100` | `profile_method()` |
| Batch sizes | Not used | 1, 10, 100, 1000, capped by available sampled edges | `BATCH_SIZES` in `run_profiling.py` |
| Output files | `results/runtime.json`, `figures/runtime_scaling.pdf` | `results/profiling_<dataset>.json`, `figures/profiling_<dataset>.pdf` | Experiment scripts |

### Code References

| Source file | Role |
|---|---|
| `experiments/run_runtime.py` | Runs dataset level scalability evaluation, measures `GraphCoarsener.fit()`, samples target edges, measures average `explain_link()` time, writes `results/runtime.json`, and plots `figures/runtime_scaling.pdf`. |
| `experiments/run_profiling.py` | Runs method level profiling, records preprocessing time, memory, mean explanation time, forward pass time, batch timing, peak memory, and writes `results/profiling_<dataset>.json`. |
| `config.py` | Defines `SpectralConfig.k`, `SpectralConfig.alpha`, the default seed and device, and the scalability dataset list. |
| `src/coarsen.py` | Implements `GraphCoarsener.fit()`, including normalized adjacency construction, spectral decomposition, perturbation scoring, node partitioning, and coarse graph construction. |
| `src/explainers/coarsen_explainer.py` | Implements the project explanation path, including lazy coarsener fitting, prediction guided partitioning, pathway group occlusion, calibration, and top edge selection. |

## Consequences

The approved evidence supports reporting coarsening as a small one time setup cost for the six measured datasets. Repeated explanations then run from the cached coarsener state, with the highest recorded average explanation time remaining below one third of a second per link. This supports the use of GraphCoarsening in sampled link prediction explanation studies where many target links are evaluated after a shared graph setup pass.

The profiling evidence also clarifies the cost of the method. The project method is not as cheap as direct Saliency because pathway calibration requires group occlusion over coarsened pathways. It is still materially faster than Occlusion and GNNExplainer in the recorded Cora mean explanation time. The higher peak memory for Ours reflects the cached coarsening state and pathway explanation work, and should be reported with the runtime gains rather than hidden behind a single latency number.

There are two reporting limits. The approved runtime artifact contains six measured datasets even though the experiment configuration lists nine scalability datasets. Any claim about `ogbl-ppa`, `ogbl-collab`, or `ogbl-ddi` requires a new completed runtime result. Also, the current profiling output does not isolate spectral decomposition, partitioning, and coarse graph construction as separate timers. It supports stage attribution through code mapping and aggregate timings, but a future profiling run should add explicit internal timers if separate stage level percentages are needed.
