# CTBoost Agent Notes

## Project Summary

CTBoost is a gradient boosting library centered on Conditional Inference Trees. The repository combines a C++17 core, `pybind11` Python bindings, a Python API layer, and optional CUDA sources for accelerated source builds.

## Current State

- The public Python package exposes low-level training primitives plus scikit-learn compatible estimators.
- The low-level `Booster` API now supports stable JSON persistence, staged prediction, validation metric history, iteration-limited prediction, configurable `eval_metric`, row weights, explicit missing-value handling through `nan_mode`, warm-start continuation via `init_model`, leaf-index prediction, and path-based contribution output.
- The high-level estimator API now supports persistence, staged prediction, `evals_result_`, `best_score_`, `sample_weight`, `class_weight`, `scale_pos_weight`, `eval_metric`, `nan_mode`, `warm_start`, and grouped ranking through `CTBoostRanker`.
- A Python-side `ctboost.cv(...)` helper is available for generic cross-validation workflows, including grouped ranking splits.
- Native training now supports generic regression objectives beyond RMSE: `MAE`, `Huber`, and `Quantile`.
- Native validation metrics now include regression metrics plus generic classification metrics such as `Accuracy`, `Precision`, `Recall`, `F1`, and `AUC`, plus ranking-oriented `NDCG`.
- `Pool` now accepts pandas inputs, SciPy sparse matrices, row weights, categorical feature indices, and optional `group_id` metadata.
- The native extension is built with `scikit-build-core` and CMake.
- Release wheels are CPU-only by default.
- CUDA remains an optional source-build capability rather than the default PyPI wheel path.
- The release workflow also publishes a dedicated Linux CPython 3.12 CUDA wheel as a GitHub release asset for Kaggle-style installs.

## Release and Wheel Policy

- Wheel builds are driven by `cibuildwheel`.
- CPU wheel coverage targets Windows `amd64`, Linux `x86_64` and `aarch64`, and macOS `universal2`.
- Wheel coverage targets current supported CPython releases declared in project metadata.
- Linux CPU wheels target the current `cibuildwheel` manylinux baseline (`manylinux_2_28`).
- PyPI publishing is handled through GitHub Actions with trusted publishing.

## Validation

- CI validates CMake-based CPU builds across Windows, Linux, and macOS.
- Wheel builds run a dedicated smoke test against built artifacts.
- The broader Python test suite runs in the regular CI matrix.
- Source builds can optionally enable CUDA when a suitable toolkit is available.
- Local validation for the current pass was completed with a manual Windows CPU CMake build and the Python test suite (`48` tests passing).
- In this environment, plain `pip install -e .` was blocked by MSVC discovery; the local validation path used explicit compiler and Windows SDK environment variables instead.

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
