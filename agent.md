# CTBoost Agent Notes

## Project Summary

CTBoost is a gradient boosting library centered on Conditional Inference Trees. The repository combines a C++17 core, `pybind11` Python bindings, a Python API layer, and optional CUDA sources for accelerated source builds.

## Core Principle

- CTBoost's defining feature is its conditional-inference-tree learner and the conditional evaluation of candidate features and splits during tree growth.
- That conditional tree logic is the product identity and must not be replaced with XGBoost-style or CatBoost-style split selection just to chase generic feature parity.
- New capabilities should be layered around that core through objectives, weighting, data plumbing, prediction paths, profiling, categorical preprocessing, and systems work while preserving the existing conditional evaluation principle.

## Current State

- The public Python package exposes low-level training primitives plus optional scikit-learn compatible estimators.
- The low-level `Booster` API now supports stable JSON persistence, staged prediction, validation metric history, iteration-limited prediction, configurable `eval_metric`, row weights, explicit missing-value handling through `nan_mode`, warm-start continuation via `init_model`, leaf-index prediction, and path-based contribution output.
- The high-level estimator API supports persistence, staged prediction, `evals_result_`, `best_score_`, `sample_weight`, `class_weight`, `scale_pos_weight`, `eval_metric`, `nan_mode`, `warm_start`, and grouped ranking through `CTBoostRanker` when `scikit-learn` is installed.
- A Python-side `ctboost.cv(...)` helper is available for generic cross-validation workflows, including grouped ranking splits, when `scikit-learn` is installed.
- Native training now supports generic regression objectives beyond RMSE: `MAE`, `Huber`, and `Quantile`.
- Native training now also supports `Poisson` and `Tweedie` regression objectives with configurable `tweedie_variance_power`.
- Native training now also supports survival objectives `Cox` and `SurvivalExponential`.
- Native validation metrics now include regression metrics plus generic classification metrics such as `Accuracy`, `Precision`, `Recall`, `F1`, and `AUC`, ranking-oriented `NDCG`, `MAP`, and `MRR`, and survival-oriented `CIndex`.
- `Pool` now accepts pandas inputs, SciPy sparse matrices, row weights, categorical feature indices, and optional `group_id` metadata.
- The Python boundary preserves column-major input layout for dense pandas matrices, SciPy sparse input now stays sparse through the native pool boundary instead of being densified first, and disk-backed memmap pools can now be staged through `ctboost.prepare_pool(..., external_memory=True)`.
- A native C++ `FeaturePipeline`, exposed through thin Python wrappers plus both low-level and sklearn-estimator integration, now provides ordered CTRs, categorical feature combinations, text hashing, and embedding-stat expansion around the existing learner without replacing the conditional split criterion.
- The low-level `ctboost.train(...)` path now accepts raw inputs with feature-pipeline parameters directly, persists the fitted preprocessing on low-level boosters, and exposes `ctboost.prepare_pool(...)` for explicit raw-data preparation.
- Histogram storage now compacts to 1-byte bins when the fitted feature bin counts permit it, reducing resident fit memory while preserving the existing conditional-inference split logic.
- Histogram building now writes directly into final-width compact storage when the fitted schema permits `<=256` bins, avoiding the old transient `uint16` plus compact-copy host-memory spike.
- Booster state now stores quantization metadata once per fitted model, with trees sharing that schema instead of duplicating it per tree.
- CUDA source builds now keep the binned feature matrix and histogram buffers resident on device for the full fit instead of rebuilding node-local bin matrices on every split.
- GPU histogram construction now uses shared-memory accumulation, with a chunked shared-memory path for larger per-feature bin counts.
- GPU tree building now batches histogram work across feature chunks, computes node aggregate statistics on device, reuses a single row-index buffer with in-place partitioning, avoids per-class gradient and hessian re-uploads in the multiclass GPU path, and performs node-level feature search on device so full node histograms no longer have to round-trip to host.
- GPU fit now releases the host training histogram bin matrix immediately after the device histogram workspace and any warm-start predictions are initialized.
- Tree growth now uses histogram subtraction on both CPU and GPU so one child histogram can be derived from `parent - sibling` instead of rescanning both children.
- GPU split search now keeps best-feature selection on device and copies back only the winning feature summary instead of the full per-feature search buffer per node.
- CUDA source builds now also support process-local feature-parallel multi-GPU training by sharding histogram and split-search work across the requested `devices` list without changing the conditional tree logic itself.
- The earlier `v0.1.15` fold-memory bottleneck list is therefore closed: shared-schema trees, post-upload host-bin release, GPU histogram subtraction, and device-side best-feature reduction are all in the live code path.
- GPU raw-score prediction is implemented for regression, binary classification, and multiclass models built with `task_type="GPU"`.
- Histogram building now parallelizes across features and uses a hybrid exact quantile strategy that chooses between full sort and order-statistic selection per feature; large-row approximation remains available through environment overrides.
- Feature selection now has a fast score-only linear-statistic path for large tree builds, avoiding the full covariance solve on the hot path while preserving the conditional-inference split criterion.
- Native training now supports row subsampling through `subsample`, `bootstrap_type="No"|"Bernoulli"|"Poisson"`, `boosting_type="RandomForest"` for bagged trees against a fixed gradient reference, and `boosting_type="DART"` for dropout-style tree normalization around the same conditional tree learner.
- Native CPU training now also supports `monotone_constraints` and `interaction_constraints`, implemented as split-feasibility and path-level feature-filtering layers around the existing conditional split search rather than a replacement for it.
- Native profiling hooks are available through `verbose=True` and `CTBOOST_PROFILE=1`, including histogram build, per-node histogram, per-tree, and overall fit timing.
- The repository now includes `run_kaggle_kernel_session.py` for source-build validation on Kaggle GPU environments.
- The repository now also includes `run_kaggle_yx_benchmark_session.py` for replaying the heavy ordered-target-encoding Playground benchmark on Kaggle GPUs against the current source tree.
- The native extension is built with `scikit-build-core` and CMake.
- Release wheels are CPU-only by default.
- CUDA remains an optional source-build capability rather than the default PyPI wheel path.
- The release workflow publishes dedicated Linux `x86_64` CUDA wheels as GitHub release assets for CPython `3.10` through `3.14`.
- The dedicated CUDA wheels are built in `manylinux2014`, require a detected CUDA toolkit via `CTBOOST_REQUIRE_CUDA=ON`, and run a dedicated GPU smoke test before upload.
- As of April 13, 2026, the repository version is bumped to `0.1.19`, following the `0.1.18` PyPI release and the merged CI and release follow-up work.
- The base wheel no longer hard-depends on `scikit-learn`; estimator and CV entry points are lazy-loaded and raise a clear import error if `scikit-learn` is absent.

## Release and Wheel Policy

- Wheel builds are driven by `cibuildwheel`.
- CPU wheel coverage targets Windows `amd64`, Linux `x86_64` and `aarch64`, and macOS `universal2`.
- Wheel coverage targets current supported CPython releases declared in project metadata.
- Linux CPU wheels target the current `cibuildwheel` manylinux baseline (`manylinux_2_28`).
- Linux GPU wheels target `manylinux2014` `x86_64` because the CUDA toolchain is provisioned inside that container.
- GPU release assets use the `1gpu` wheel build tag so CPU and GPU Linux wheels can coexist for the same Python and platform tags.
- PyPI publishing is handled through GitHub Actions with trusted publishing.
- PyPI publishing is gated on the CPU wheel jobs, source distribution job, and GPU wheel matrix so broken GPU release assets do not accompany a published version.

## Validation

- CI validates CMake-based CPU builds across Windows, Linux, and macOS.
- Wheel builds run a dedicated smoke test against built artifacts.
- The broader Python test suite runs in the regular CI matrix.
- Source builds can optionally enable CUDA when a suitable toolkit is available.
- The merged GPU histogram/prediction pass was validated locally with:
  `python -m pytest tests/test_build.py tests/test_data_and_loss.py tests/test_sparse_input.py -q`
  `python -m pytest tests/test_booster.py tests/test_multiclass.py tests/test_sklearn.py -q`
- The post-`0.1.9` tree-build optimization pass was revalidated locally on April 12, 2026 with:
  `python -m pytest tests/test_build.py tests/test_booster.py tests/test_multiclass.py tests/test_sklearn.py -q`
- The `0.1.15` compact-bin and GPU node-search follow-up was validated locally on April 12, 2026 with:
  `python -X faulthandler -m pytest tests/test_build.py tests/test_booster.py tests/test_multiclass.py -q`
- The `0.1.16` shared-schema and GPU memory-reduction follow-up was validated locally on April 13, 2026 with:
  `python -m pytest -q`
- The sparse-ingestion, survival, DART, and Python feature-pipeline follow-up was revalidated locally on April 13, 2026 with:
  `python -m pytest -q`
- Kaggle source-build validation on `playground-series-s6e4` succeeded on April 12, 2026 using notebook `maiernator/ctboost-gpu-source-validate-s6e4` version `8`.
- That Kaggle run reported `cuda_enabled=True`, `cuda_runtime="12.8"`, `630000` training rows, and `270000` test rows.
- The latest merged Kaggle timings are:
  multiclass fit `21.50s`, predict `0.353s`
  binary fit `5.77s`, predict `0.348s`
  regression fit `7.82s`, predict `0.326s`
- The heavier ordered-target-encoding Kaggle replay also succeeded on April 12, 2026 using notebook `maiernator/ps-s6e4-yx-ctb-src-bench-v0-1-11` version `1`.
- That one-fold replay reported source build `55.41s`, fold preprocess `57.17s`, fold fit `2107.10s`, fold predict `5.89s`, fold total `2170.17s`, and validation score `0.9732134730`.
- In this Windows environment, native local install/build verification still depends on manually setting the MSVC include/lib environment before rebuilding the extension.

## Local Plan

- Preserve the current conditional-inference-tree learner and conditional feature-evaluation path as the non-negotiable core while closing generic usability gaps around it.
- Treat the shared-schema and GPU fit-memory reduction pass as complete baseline behavior, not open roadmap work.
- Treat the native feature pipeline for ordered CTRs, categorical crosses, text hashing, and embedding expansion as the current generic-data path around the existing learner; Python should stay limited to pandas extraction, Pool orchestration, and estimator-facing API glue rather than owning the transform logic itself.
- Treat native sparse training plus disk-backed external-memory pool staging as present baseline behavior; fully streaming quantized fit beyond that staging layer remains open.
- Keep extending tree-growth and constraint controls around the existing learner. Row subsampling, bootstrap controls, `max_leaves`, minimum leaf-size controls, minimum split-loss controls, a random-forest mode, monotonic constraints, and interaction constraints are now present; GPU support for the new constraints remains open.
- Broaden booster and objective coverage only where it can reuse the existing conditional tree backbone. Count objectives, extra ranking metrics, DART-style dropout, survival objectives, low-level raw-data preprocessing, and local multi-GPU execution are now present; distributed multi-host execution remains open.
- Add production tooling expected from mature libraries, including callbacks, snapshot or checkpoint resume, richer SHAP tooling, first-class export targets, and deeper distributed execution.

## Repository Layout

- `ctboost/`: Python API surface
- `include/`: public C++ headers
- `src/core/`: core training, data, objectives, trees, statistics
- `src/bindings/`: `pybind11` extension module
- `cuda/`: optional CUDA backend sources
- `tests/`: Python test suite

## Maintenance Notes

- Keep Python package metadata, native build metadata, and release workflows aligned when changing supported Python versions.
- Treat wheel coverage as part of the public install contract.
- Keep CPU wheels easy to install with plain `pip install ctboost`; avoid making standard PyPI installation depend on local CUDA or platform-specific compiler setup.
- Preserve backward compatibility of the JSON booster schema when changing native tree or booster internals, or version the persistence format explicitly if that becomes necessary.
- Keep the CTBoost conditional-inference split-selection principle intact when adding generic training features; additions should sit in weighting, metrics, objectives, data handling, categorical statistics, systems work, and API plumbing rather than replacing the core conditional tree logic.
