# CTBoost

CTBoost is a gradient boosting library built around Conditional Inference Trees, with a native C++17 core, Python bindings via `pybind11`, optional CUDA support for source builds, and a scikit-learn style API.

The current codebase supports end-to-end training and prediction for regression, binary classification, and multiclass classification, plus native pandas `DataFrame` ingestion with automatic categorical detection, row weights and class imbalance controls, explicit missing-value handling, configurable validation metrics, model persistence, staged prediction, and a built-in cross-validation helper.

## Current Status

- Language mix: Python + C++17, with optional CUDA
- Python support: `3.8` through `3.14`
- Packaging: `scikit-build-core`
- CI/CD: GitHub Actions for CMake validation and `cibuildwheel` release builds
- Status: actively evolving native + Python package

## What Works Today

- Native gradient boosting backend exposed as `ctboost._core`
- `Pool` abstraction for dense tabular data and categorical feature indices
- Native pandas `DataFrame` and `Series` support
- Automatic categorical detection for pandas `category` and `object` columns
- Regression training with `ctboost.train(...)`
- scikit-learn compatible `CTBoostClassifier` and `CTBoostRegressor`
- Binary and multiclass classification
- Row weights through `Pool(..., weight=...)` and `sample_weight` on sklearn estimators
- Class imbalance controls through `class_weight`, `class_weights`, `auto_class_weights="balanced"`, and `scale_pos_weight`
- Explicit missing-value handling through `nan_mode`
- Early stopping with `eval_set` and `early_stopping_rounds`
- Separate `eval_metric` support for validation history and early stopping
- Validation loss/metric history and `evals_result_`
- Per-iteration prediction through staged prediction and `num_iteration`
- Model persistence for low-level boosters and scikit-learn style estimators
- Cross-validation with `ctboost.cv(...)`
- Regression objectives: `RMSE`, `MAE`, `Huber`, `Quantile`
- Generic eval metrics including `RMSE`, `MAE`, `Accuracy`, `Precision`, `Recall`, `F1`, and `AUC`
- Feature importance reporting
- Build metadata reporting through `ctboost.build_info()`
- CPU builds on standard CI runners
- Optional CUDA compilation when building from source with a suitable toolkit

## Current Limitations

- The Python package currently targets dense NumPy-compatible input
- Dedicated GPU wheel automation currently targets Linux CPython 3.10
- CUDA wheel builds in CI depend on container-side toolkit provisioning

## Installation

For local development or source builds:

```bash
pip install .
```

Install development dependencies:

```bash
pip install -e .[dev]
```

### Wheels vs Source Builds

`pip install ctboost` works without a compiler only when PyPI has a prebuilt wheel for your exact Python/OS tag. If no matching wheel exists, `pip` falls back to the source distribution and has to compile the native extension locally.

The release workflow is configured to publish CPU wheels for current CPython releases on Windows, Linux, and macOS so standard `pip install ctboost` usage does not depend on a local compiler.

### CPU-Only Source Build

To force a CPU-only native build:

```bash
CMAKE_ARGS="-DCTBOOST_ENABLE_CUDA=OFF" pip install .
```

On PowerShell:

```powershell
$env:CMAKE_ARGS="-DCTBOOST_ENABLE_CUDA=OFF"
pip install .
```

Windows source builds require a working C++ toolchain. In practice that means Visual Studio Build Tools 2022 or a compatible MSVC environment, plus CMake. `ninja` is recommended, but it does not replace the compiler itself.

### CUDA Source Build

CTBoost can compile a CUDA backend when the CUDA toolkit and compiler are available. CUDA is enabled by default in CMake, but the build automatically falls back to CPU-only when no toolkit is detected.

```bash
pip install .
```

You can inspect the compiled package after installation:

```python
import ctboost
print(ctboost.build_info())
```

## Quick Start

### scikit-learn Style Classification

```python
import pandas as pd
from sklearn.datasets import make_classification

from ctboost import CTBoostClassifier

X, y = make_classification(
    n_samples=256,
    n_features=8,
    n_informative=5,
    n_redundant=0,
    random_state=13,
).astype("float32")
X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
X["segment"] = pd.Categorical(["a" if i % 2 == 0 else "b" for i in range(len(X))])
y = y.astype("float32")

model = CTBoostClassifier(
    iterations=256,
    learning_rate=0.1,
    max_depth=3,
    alpha=1.0,
    lambda_l2=1.0,
    task_type="CPU",
)

model.fit(
    X.iloc[:200],
    y[:200],
    eval_set=[(X.iloc[200:], y[200:])],
    early_stopping_rounds=20,
)
proba = model.predict_proba(X)
pred = model.predict(X)
importance = model.feature_importances_
best_iteration = model.best_iteration_
```

### Low-Level Training API

```python
import numpy as np

import ctboost

X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
y = np.array([0.0, 1.0, 0.5], dtype=np.float32)

pool = ctboost.Pool(X, y)
booster = ctboost.train(
    pool,
    {
        "objective": "Huber",
        "learning_rate": 0.2,
        "max_depth": 2,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "max_bins": 64,
        "huber_delta": 1.5,
        "eval_metric": "MAE",
        "nan_mode": "Min",
        "task_type": "CPU",
    },
    num_boost_round=10,
)

predictions = booster.predict(pool)
loss_history = booster.loss_history
eval_loss_history = booster.eval_loss_history
```

### Working With Categorical Features

Categorical columns can still be marked manually through the `Pool` API:

```python
import numpy as np
import ctboost

X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

pool = ctboost.Pool(X, y, cat_features=[0])
```

For pandas inputs, categorical/object columns are detected automatically:

```python
import pandas as pd
import ctboost

frame = pd.DataFrame(
    {
        "value": [1.0, 2.0, 3.0, 4.0],
        "city": pd.Categorical(["berlin", "paris", "berlin", "rome"]),
        "segment": ["retail", "enterprise", "retail", "enterprise"],
    }
)
label = pd.Series([0.0, 1.0, 0.0, 1.0], dtype="float32")

pool = ctboost.Pool(frame, label)
assert pool.cat_features == [1, 2]
```

### Model Persistence And Cross-Validation

```python
import ctboost

booster.save_model("regression-model.pkl")
restored = ctboost.load_model("regression-model.pkl")
restored_predictions = restored.predict(pool)

cv_result = ctboost.cv(
    pool,
    {
        "objective": "RMSE",
        "learning_rate": 0.2,
        "max_depth": 2,
        "alpha": 1.0,
        "lambda_l2": 1.0,
    },
    num_boost_round=25,
    nfold=3,
)
```

The scikit-learn compatible estimators also expose:

- `save_model(...)`
- `load_model(...)`
- `staged_predict(...)`
- `staged_predict_proba(...)` for classifiers
- `evals_result_`
- `best_score_`
- `sample_weight` on `fit(...)`
- `class_weight`, `scale_pos_weight`, `eval_metric`, and `nan_mode`

## Public Python API

The main entry points are:

- `ctboost.Pool`
- `ctboost.train`
- `ctboost.cv`
- `ctboost.Booster`
- `ctboost.CTBoostClassifier`
- `ctboost.CTBoostRegressor`
- `ctboost.CBoostClassifier`
- `ctboost.CBoostRegressor`
- `ctboost.build_info`
- `ctboost.load_model`

## Build and Test

Run the test suite:

```bash
pytest tests
```

Build an sdist:

```bash
python -m build --sdist
```

Configure and build the native extension directly with CMake:

```bash
python -m pip install pybind11 numpy pandas scikit-learn pytest
cmake -S . -B build -DCTBOOST_ENABLE_CUDA=OFF -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build --config Release --parallel
```

Wheel builds are configured through `cibuildwheel` for:

- Windows `amd64`
- Linux `x86_64` and `aarch64` using the current manylinux baseline
- macOS `universal2`
- CPython `3.8`, `3.9`, `3.10`, `3.11`, `3.12`, `3.13`, and `3.14`

GitHub Actions workflows:

- `.github/workflows/cmake.yml`: configures, builds, installs, and tests CPU builds on Ubuntu, Windows, and macOS for pushes and pull requests
- `.github/workflows/publish.yml`: builds release wheels and the sdist, runs wheel smoke tests on built artifacts, publishes tagged releases, and uploads a dedicated Linux GPU wheel artifact

The standard release workflow builds CPU-only wheels by setting:

```text
CMAKE_ARGS='-DCTBOOST_ENABLE_CUDA=OFF'
```

## Project Layout

```text
ctboost/      Python API layer
include/      public C++ headers
src/core/     core boosting, objectives, trees, statistics
src/bindings/ pybind11 extension bindings
cuda/         optional CUDA backend
tests/        Python test suite
```

## License

Apache 2.0. See `LICENSE`.
