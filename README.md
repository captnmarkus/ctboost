# CTBoost

CTBoost is an alpha-stage gradient boosting library built around Conditional Inference Trees, with a native C++17 core, Python bindings via `pybind11`, and optional CUDA support for source builds.

The current codebase already supports end-to-end training and prediction for regression, binary classification, and multiclass classification through both a low-level training API and scikit-learn style estimators.

## Current Status

- Version: `0.1.0`
- Language mix: Python + C++17, with optional CUDA
- Python support: `3.8` through `3.12`
- Packaging: `scikit-build-core`
- Release automation: GitHub Actions + `cibuildwheel`
- Stability: alpha

## What Works Today

- Native gradient boosting backend exposed as `ctboost._core`
- `Pool` abstraction for dense tabular data and categorical feature indices
- Regression training with `ctboost.train(...)`
- scikit-learn compatible `CTBoostClassifier` and `CTBoostRegressor`
- Binary and multiclass classification
- Feature importance reporting
- Build metadata reporting through `ctboost.build_info()`
- CPU builds on standard CI runners
- Optional CUDA compilation when building from source with a suitable toolkit

## Current Limitations

- Public wheel builds are CPU-only for `0.1.0`
- `eval_set` is not implemented yet
- `early_stopping_rounds` is not implemented yet
- The Python package currently targets dense NumPy-compatible input

## Installation

For local development or source builds:

```bash
pip install .
```

Install development dependencies:

```bash
pip install -e .[dev]
```

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
import numpy as np
from sklearn.datasets import make_classification

from ctboost import CTBoostClassifier

X, y = make_classification(
    n_samples=256,
    n_features=8,
    n_informative=5,
    n_redundant=0,
    random_state=13,
)

X = X.astype(np.float32)
y = y.astype(np.float32)

model = CTBoostClassifier(
    iterations=32,
    learning_rate=0.1,
    max_depth=3,
    alpha=1.0,
    lambda_l2=1.0,
    task_type="CPU",
)

model.fit(X, y)
proba = model.predict_proba(X)
pred = model.predict(X)
importance = model.feature_importances_
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
        "objective": "RMSE",
        "learning_rate": 0.2,
        "max_depth": 2,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "max_bins": 64,
        "task_type": "CPU",
    },
    num_boost_round=10,
)

predictions = booster.predict(pool)
loss_history = booster.loss_history
```

### Working With Categorical Features

Categorical columns can be marked through the `Pool` API:

```python
import numpy as np
import ctboost

X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

pool = ctboost.Pool(X, y, cat_features=[0])
```

## Public Python API

The main entry points are:

- `ctboost.Pool`
- `ctboost.train`
- `ctboost.Booster`
- `ctboost.CTBoostClassifier`
- `ctboost.CTBoostRegressor`
- `ctboost.CBoostClassifier`
- `ctboost.CBoostRegressor`
- `ctboost.build_info`

## Build and Test

Run the test suite:

```bash
pytest tests
```

Build an sdist:

```bash
python -m build --sdist
```

Wheel builds are configured through `cibuildwheel` for:

- Windows
- Linux
- macOS
- CPython `3.8`, `3.9`, `3.10`, `3.11`, and `3.12`

The release workflow builds CPU-only wheels by setting:

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
