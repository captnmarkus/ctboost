# CTBoost

CTBoost is a gradient boosting library built around Conditional Inference Trees, with a native C++17 core, `pybind11` bindings, optional CUDA support, and optional scikit-learn compatible estimators.

The project constraint is deliberate: keep the conditional-inference-tree split logic intact. New features should land in data ingestion, preprocessing, metrics, orchestration, serialization, and deployment layers rather than by replacing the learner with CatBoost-style or XGBoost-style tree growth.

## What CTBoost Supports

- Regression, classification, grouped ranking, and survival training
- Low-level `ctboost.train(...)` plus `CTBoostClassifier`, `CTBoostRegressor`, and `CTBoostRanker`
- NumPy, pandas, and SciPy sparse input without dense conversion
- Native categorical, text, and embedding preprocessing through `FeaturePipeline`
- Row weights, class imbalance controls, missing-value handling, quantization controls, and generic regularization or growth settings
- Validation watchlists, multiple eval metrics, callable eval metrics, early stopping, per-iteration callbacks, and learning-rate schedules or callback-driven learning-rate changes
- Stable JSON and pickle persistence, warm start via `init_model`, snapshot-path resume with config and schema validation, staged prediction, and standalone Python or JSON predictor export
- Richer `Pool` schema metadata via `feature_names`, `column_roles`, `feature_metadata`, and `categorical_schema`
- Ranking metadata in `Pool`: `group_id`, `group_weight`, `subgroup_id`, `pairs`, `pairs_weight`, and `baseline`
- External-memory pool staging plus optional TCP-based distributed training
- Feature importance, leaf indices, and path-based prediction contributions

See [BACKLOG.md](BACKLOG.md) for the remaining generic feature roadmap and deferred items.

## Installation

Install from source:

```bash
pip install .
```

Install development dependencies:

```bash
pip install -e .[dev]
```

Install the optional scikit-learn wrappers and `ctboost.cv(...)` support:

```bash
pip install -e .[sklearn]
```

`pip install ctboost` uses CPU wheels when a matching wheel exists on PyPI. Tagged GitHub releases also publish CUDA wheel assets for supported Linux and Windows targets.

To force a CPU-only native source build:

```bash
CMAKE_ARGS="-DCTBOOST_ENABLE_CUDA=OFF" pip install .
```

On PowerShell:

```powershell
$env:CMAKE_ARGS="-DCTBOOST_ENABLE_CUDA=OFF"
pip install .
```

## Quick Start

### scikit-learn API

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
)

frame = pd.DataFrame(X.astype("float32"), columns=[f"f{i}" for i in range(X.shape[1])])
frame["segment"] = pd.Categorical(["a" if i % 2 == 0 else "b" for i in range(len(frame))])

model = CTBoostClassifier(
    iterations=256,
    learning_rate=0.1,
    max_depth=3,
    alpha=1.0,
    lambda_l2=1.0,
    eval_metric="AUC",
)

model.fit(
    frame.iloc[:200],
    y[:200].astype("float32"),
    eval_set=[(frame.iloc[200:], y[200:].astype("float32"))],
    early_stopping_rounds=20,
)

proba = model.predict_proba(frame)
pred = model.predict(frame)
importance = model.feature_importances_
```

### Low-Level API

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
        "learning_rate": 0.1,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "eval_metric": "MAE",
    },
    num_boost_round=32,
)

predictions = booster.predict(pool)
```

### Learning-Rate Schedules And Callbacks

```python
schedule = [0.2, 0.2, 0.1, 0.1, 0.05, 0.05]

booster = ctboost.train(
    pool,
    {
        "objective": "RMSE",
        "learning_rate": schedule[0],
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 1.0,
    },
    num_boost_round=len(schedule),
    learning_rate_schedule=schedule,
    callbacks=[ctboost.log_evaluation(2)],
)

print(booster.learning_rate_history)
```

Callbacks receive `env.learning_rate` and may call `env.model.set_learning_rate(...)` to change the step size used for later rounds. The scikit-learn estimators accept the same `learning_rate_schedule=` keyword on `fit(...)`.

### Categorical, Text, And Embedding Inputs

```python
import numpy as np
import pandas as pd

from ctboost import CTBoostRegressor

frame = pd.DataFrame(
    {
        "city": ["berlin", "paris", "berlin", "rome"],
        "headline": ["red fox", "blue fox", "red hare", "green fox"],
        "embedding": [
            np.array([0.1, 0.4, 0.2], dtype=np.float32),
            np.array([0.7, 0.1, 0.3], dtype=np.float32),
            np.array([0.2, 0.5, 0.6], dtype=np.float32),
            np.array([0.9, 0.2, 0.4], dtype=np.float32),
        ],
        "value": [1.0, 2.0, 1.5, 3.0],
    }
)
y = np.array([0.5, 1.2, 0.7, 1.6], dtype=np.float32)

model = CTBoostRegressor(
    iterations=64,
    learning_rate=0.1,
    max_depth=3,
    ordered_ctr=True,
    cat_features=["city"],
    text_features=["headline"],
    embedding_features=["embedding"],
)
model.fit(frame, y)
```

## Persistence, Resume, And Export

```python
import ctboost

metric = ctboost.make_eval_metric(
    lambda predictions, label, **_: float(((predictions >= 0.0) == label).mean()),
    name="SignedAccuracy",
    higher_is_better=True,
    allow_early_stopping=True,
)

booster = ctboost.train(
    pool,
    {
        "objective": "Logloss",
        "learning_rate": 0.1,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "eval_metric": [metric, "AUC"],
    },
    num_boost_round=64,
    eval_set=[(X_valid, y_valid)],
    snapshot_path="run_snapshot.ctb",
)

resumed = ctboost.train(
    pool,
    {
        "objective": "Logloss",
        "learning_rate": 0.1,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 1.0,
    },
    num_boost_round=128,
    snapshot_path="run_snapshot.ctb",
    resume_from_snapshot=True,
)

booster.export_model("predictor.json", export_format="json_predictor")
predictor = ctboost.load_exported_predictor("predictor.json")
exported_predictions = predictor.predict(X_numeric)
```

`resume_from_snapshot=True` validates the saved training configuration and data schema before loading the checkpoint. It remains a warm-start-based convenience flow rather than a blanket exact-equivalence guarantee for every training path. For per-iteration checkpoint emission and logging hooks, use `callbacks=[ctboost.log_evaluation(...), ctboost.checkpoint_callback(...)]`.

## Metadata

```python
pool = ctboost.Pool(
    X,
    y,
    feature_names=["score", "ratio", "city_code"],
    column_roles=["numeric", "numeric", "categorical"],
    feature_metadata={"score": {"description": "normalized score"}},
    categorical_schema={"city_code": {"categories": ["berlin", "paris", "rome"]}},
)

booster = ctboost.train(pool, {"objective": "RMSE"}, num_boost_round=16)
print(booster.data_schema)
```

The scikit-learn estimators expose the same persisted schema through `data_schema_`.

## Build And Test

Run the Python tests:

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

## Project Layout

```text
ctboost/      Python API surface
include/      public C++ headers
src/core/     core training, data, objectives, trees, statistics
src/bindings/ pybind11 extension bindings
cuda/         optional CUDA backend
tests/        Python test suite
```

## License

Apache 2.0. See `LICENSE`.
