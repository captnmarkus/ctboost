# CTBoost Agent Notes

## Project Summary

CTBoost is a gradient boosting library centered on Conditional Inference Trees. The repository combines a C++17 core, `pybind11` Python bindings, a Python API layer, and optional CUDA sources for accelerated source builds.

## Current State

- The public Python package exposes low-level training primitives plus optional scikit-learn compatible estimators.
- The low-level `Booster` API now supports stable JSON persistence, staged prediction, validation metric history, iteration-limited prediction, configurable `eval_metric`, row weights, explicit missing-value handling through `nan_mode`, warm-start continuation via `init_model`, leaf-index prediction, and path-based contribution output.
- The high-level estimator API supports persistence, staged prediction, `evals_result_`, `best_score_`, `sample_weight`, `class_weight`, `scale_pos_weight`, `eval_metric`, `nan_mode`, `warm_start`, and grouped ranking through `CTBoostRanker` when `scikit-learn` is installed.
- A Python-side `ctboost.cv(...)` helper is available for generic cross-validation workflows, including grouped ranking splits, when `scikit-learn` is installed.
- Native training now supports generic regression objectives beyond RMSE: `MAE`, `Huber`, and `Quantile`.
- Native validation metrics now include regression metrics plus generic classification metrics such as `Accuracy`, `Precision`, `Recall`, `F1`, and `AUC`, plus ranking-oriented `NDCG`.
- `Pool` now accepts pandas inputs, SciPy sparse matrices, row weights, categorical feature indices, and optional `group_id` metadata.
- The Python boundary preserves column-major input layout for dense pandas and SciPy-ingested matrices so large pools can be handed to the native layer without an extra full-table transpose.
- CUDA source builds now keep the binned feature matrix and histogram buffers resident on device for the full fit instead of rebuilding node-local bin matrices on every split.
- GPU histogram construction now uses shared-memory accumulation, with a chunked shared-memory path for larger per-feature bin counts.
- GPU tree building now batches histogram work across feature chunks, computes node aggregate statistics on device, reuses a single row-index buffer with in-place partitioning, and avoids per-class gradient and hessian re-uploads in the multiclass GPU path.
- Tree growth now uses histogram subtraction so one child histogram can be derived from `parent - sibling` instead of rescanning both children.
- GPU raw-score prediction is implemented for regression, binary classification, and multiclass models built with `task_type="GPU"`.
- Histogram building now parallelizes across features and uses a hybrid exact quantile strategy that chooses between full sort and order-statistic selection per feature; large-row approximation remains available through environment overrides.
- Feature selection now has a fast score-only linear-statistic path for large tree builds, avoiding the full covariance solve on the hot path while preserving the conditional-inference split criterion.
- Native profiling hooks are available through `verbose=True` and `CTBOOST_PROFILE=1`, including histogram build, per-node histogram, per-tree, and overall fit timing.
- The repository now includes `run_kaggle_kernel_session.py` for source-build validation on Kaggle GPU environments.
- The repository now also includes `run_kaggle_yx_benchmark_session.py` for replaying the heavy ordered-target-encoding Playground benchmark on Kaggle GPUs against the current source tree.
- The native extension is built with `scikit-build-core` and CMake.
- Release wheels are CPU-only by default.
- CUDA remains an optional source-build capability rather than the default PyPI wheel path.
- The release workflow publishes dedicated Linux `x86_64` CUDA wheels as GitHub release assets for CPython `3.10` through `3.14`.
- The dedicated CUDA wheels are built in `manylinux2014`, require a detected CUDA toolkit via `CTBOOST_REQUIRE_CUDA=ON`, and run a dedicated GPU smoke test before upload.
- As of April 12, 2026, PR `#1` (`Accelerate GPU histograms and add Kaggle validation`) is merged on `master`, and the repository version is bumped to `0.1.10` after the live `0.1.9` PyPI release.
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
- Kaggle source-build validation on `playground-series-s6e4` succeeded on April 12, 2026 using notebook `maiernator/ctboost-gpu-source-validate-s6e4` version `8`.
- That Kaggle run reported `cuda_enabled=True`, `cuda_runtime="12.8"`, `630000` training rows, and `270000` test rows.
- The latest merged Kaggle timings are:
  multiclass fit `21.50s`, predict `0.353s`
  binary fit `5.77s`, predict `0.348s`
  regression fit `7.82s`, predict `0.326s`
- In this Windows environment, native local install/build verification still depends on manually setting the MSVC include/lib environment before rebuilding the extension.

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
- Keep the CTBoost split-selection principle intact when adding generic training features; generic additions should sit in weighting, metrics, objectives, data handling, and API plumbing rather than replacing the conditional-inference tree logic.
