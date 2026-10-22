# CTBoost Agent Notes

## Project Summary

CTBoost is a gradient boosting library centered on Conditional Inference Trees. The repository combines a C++17 core, `pybind11` bindings, a Python API layer, optional CUDA sources, and optional scikit-learn compatible estimators.

## Non-Negotiable Constraint

- Preserve the current conditional-inference-tree split logic and the conditional evaluation of candidate features during tree growth.
- Prefer work in data plumbing, preprocessing, metrics, orchestration, serialization, profiling, packaging, and deployment.

## Current Baseline

- Low-level `ctboost.train(...)` plus `Booster` persistence, staged prediction, warm start via `init_model`, leaf-index output, and path-based contributions
- Optional scikit-learn compatible `CTBoostClassifier`, `CTBoostRegressor`, and `CTBoostRanker`
- NumPy, pandas, and SciPy sparse ingestion without dense conversion
- External-memory pool staging through `ctboost.prepare_pool(..., external_memory=True)` and reusable preparation through `ctboost.prepare_training_data(...)`
- Native feature pipeline support for categorical, text, and embedding inputs through `FeaturePipeline`
- Callable eval metrics, multi-metric reporting, early stopping, per-iteration callbacks, and learning-rate schedules or callback-driven learning-rate changes
- Built-in `log_evaluation(...)`, `checkpoint_callback(...)`, `snapshot_path`, and `resume_from_snapshot` with strict config and schema validation
- Richer schema metadata via `feature_names`, `column_roles`, `feature_metadata`, `categorical_schema`, and `Booster.data_schema`
- Standalone Python export plus lightweight JSON predictor export through `export_format="json_predictor"`
- Regression, classification, ranking, and survival objectives already present in the native package
- Ranking metadata in `Pool`: `group_id`, `group_weight`, `subgroup_id`, `pairs`, `pairs_weight`, and `baseline`
- TCP-based distributed training, optional CUDA builds, and Python CV helpers

## Active Generic Backlog

- Tighten the exact-equivalence guarantee surface for snapshot / resume beyond the current warm-start-based convenience flow
- Broaden export and deployment surfaces with clearer manifests and downstream contracts
- Add higher-level platform integration helpers for workflows such as Dask, Spark handoff, or Kaggle setup without changing learner semantics

See [BACKLOG.md](BACKLOG.md) for the prioritized roadmap.

## Already Covered, Not Open Backlog Items

- Native sparse ingestion and sparse-preserving pool plumbing
- Callback-based training control, including learning-rate schedules and callback-driven learning-rate changes
- Callable evaluation metrics with safe early-stopping flags
- Schema metadata round-tripping for boosters and sklearn estimators
- Lightweight JSON predictor export
- Text and embedding preprocessing layered around the existing learner

## Deferred Work

- Any feature that would replace or materially rewrite the conditional-inference-tree split-selection principle
- Native custom objectives inside the optimization loop
- Full distributed algorithm redesign beyond the current single-model orchestration paths
- Multi-output objectives and losses
- New survival or ranking objective families that require deeper native optimization changes
- Constraint redesign beyond the current `monotone_constraints` and `interaction_constraints` support

## Repository Layout

- `ctboost/`: Python API surface
- `include/`: public C++ headers
- `src/core/`: core training, data, objectives, trees, statistics
- `src/bindings/`: `pybind11` extension module
- `cuda/`: optional CUDA backend sources
- `tests/`: Python test suite

## Maintenance Notes

- Keep the Python package surface, native build metadata, and release workflows aligned when Python-version support changes
- Treat wheel coverage as part of the public install contract
- Keep standard `pip install ctboost` CPU-friendly; do not make default installation depend on CUDA or a local compiler toolchain
- Preserve JSON booster compatibility when changing native persistence internals, or version the schema explicitly
- Be explicit when documentation is describing warm-start-based resume semantics, strict config and schema validation, versus stronger exact-equivalence guarantees
- When adding generic library features, prefer changes in data handling, preprocessing, metrics, orchestration, or export rather than in the core learner
