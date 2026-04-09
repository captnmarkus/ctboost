# CTBoost

Phase 1 scaffold for a conditional boosting library with a C++17/CUDA backend and a Python
API modeled after XGBoost and CatBoost.

Current scope:
- `scikit-build-core` packaging
- conditional CUDA compilation via CMake
- `pybind11` native extension skeleton
- Python API shells for `Pool`, `train`, `CTBoostClassifier`, and `CTBoostRegressor`
- pytest smoke tests for import and build metadata
