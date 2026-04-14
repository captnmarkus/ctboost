# CTBoost

CTBoost is a gradient boosting library built around Conditional Inference Trees, with a native C++17 core, Python bindings via `pybind11`, optional CUDA support for source builds, and an optional scikit-learn style API.

The current codebase supports end-to-end training and prediction for regression, classification, grouped ranking, and survival, plus pandas and SciPy sparse ingestion without dense expansion, row weights and class imbalance controls, explicit missing-value handling, configurable validation metrics, stable JSON model persistence, standalone Python export for prepared numeric models, staged prediction, warm-start continuation, a native C++ feature pipeline for categorical/text/embedding transforms with thin Python wrappers, reusable prepared training-data bundles, and a built-in cross-validation helper.

## Current Status

- Language mix: Python + C++17, with optional CUDA
- Python support: `3.8` through `3.14`
- Packaging: `scikit-build-core`
- CI/CD: GitHub Actions for CMake validation and `cibuildwheel` release builds
- Repository version: `0.1.28`
- Status: actively evolving native + Python package

## What Works Today

- Native gradient boosting backend exposed as `ctboost._core`
- `Pool` abstraction for dense tabular data, SciPy sparse input, categorical feature indices, and optional `group_id`
- Native pandas `DataFrame` and `Series` support
- Automatic categorical detection for pandas `category` and `object` columns
- Regression training with `ctboost.train(...)`, including raw array/DataFrame inputs plus optional preprocessing and external-memory staging
- scikit-learn compatible `CTBoostClassifier`, `CTBoostRegressor`, and `CTBoostRanker` when `scikit-learn` is installed
- Binary and multiclass classification
- Grouped ranking with `PairLogit` and `NDCG`
- Row weights through `Pool(..., weight=...)` and `sample_weight` on sklearn estimators
- Class imbalance controls through `class_weight`, `class_weights`, `auto_class_weights="balanced"`, and `scale_pos_weight`
- Explicit missing-value handling through `nan_mode`
- Quantization controls through `max_bins`, `max_bin_by_feature`, `border_selection_method`, `feature_borders`, and `nan_mode_by_feature`
- Row subsampling through `subsample` plus `bootstrap_type="No"|"Bernoulli"|"Poisson"`
- Bayesian bagging through `bootstrap_type="Bayesian"` plus `bagging_temperature`
- `boosting_type="RandomForest"` on top of the existing conditional-inference tree learner
- `boosting_type="DART"` with dropout-style tree normalization on top of the existing conditional-inference tree learner
- Monotonic constraints through `monotone_constraints`
- Path-level interaction constraints through `interaction_constraints`
- Additional generic regularization and tree-growth controls through `feature_weights`, `first_feature_use_penalties`, `random_strength`, `grow_policy`, `min_samples_split`, and `max_leaf_weight`
- GPU tree growth now also supports monotonic constraints, interaction constraints, `feature_weights`, `first_feature_use_penalties`, `random_strength`, and `grow_policy="LeafWise"` without replacing the conditional-inference split gate
- Survival objectives: `Cox`, `SurvivalExponential`
- Survival evaluation through `CIndex`
- Early stopping with `eval_set`, `eval_names`, `early_stopping_rounds`, `early_stopping_metric`, and `early_stopping_name`
- Single- and multi-watchlist evaluation through one or many `eval_set` entries
- Single- and multi-metric evaluation through string or sequence `eval_metric` values
- Per-iteration callback hooks through `callbacks`, plus built-in `ctboost.log_evaluation(...)` and `ctboost.checkpoint_callback(...)`
- Validation loss/metric history and `evals_result_`
- Per-iteration prediction through staged prediction and `num_iteration`
- Stable JSON and pickle model persistence for low-level boosters and scikit-learn style estimators
- Cross-validation with `ctboost.cv(...)` when `scikit-learn` is installed
- Regression objectives: `RMSE`, `MAE`, `Huber`, `Quantile`, `Poisson`, `Tweedie`
- Generic eval metrics including `RMSE`, `MAE`, `Poisson`, `Tweedie`, `Accuracy`, `BalancedAccuracy`, `Precision`, `Recall`, `F1`, `AUC`, `NDCG`, `MAP`, `MRR`, and `CIndex`
- Native `ctboost.FeaturePipeline` logic in `_core.NativeFeaturePipeline`, with low-level and sklearn integration for ordered CTRs, frequency-style CTRs, categorical crosses, low-cardinality one-hot expansion, rare-category bucketing, text hashing, and embedding-stat expansion
- Generic categorical controls around the existing conditional tree learner: `one_hot_max_size` / `max_cat_to_onehot`, `max_cat_threshold`, `simple_ctr`, `combinations_ctr`, and `per_feature_ctr`
- `ctboost.prepare_pool(...)` for low-level raw-data preparation, optional feature-pipeline fitting, and disk-backed external-memory pool staging
- `ctboost.prepare_training_data(...)` plus `PreparedTrainingData` for one-time raw train/eval preparation that can be reused across repeated fits
- Native CPU out-of-core fit through `ctboost.train(..., external_memory=True)`, which now spills quantized feature-bin columns to disk instead of keeping the full histogram matrix resident in RAM
- Multi-host distributed training through `distributed_world_size`, `distributed_rank`, `distributed_root`, and `distributed_run_id`, with a native per-node histogram reduction path and a TCP collective backend available through `distributed_root="tcp://host:port"`
- Distributed `eval_set`, multi-watchlist or multi-metric evaluation, callbacks, `early_stopping_rounds`, `init_model`, grouped ranking shards, and sklearn-estimator wrappers on the TCP collective backend
- Filesystem-backed distributed runs now also fall back to a rank-0 coordinator path for advanced eval, callback, ranking, and GPU compatibility flows when TCP is not configured
- Distributed GPU training when CUDA is available and `distributed_root` uses the TCP collective backend
- Distributed raw-data feature-pipeline fitting across ranks for native categorical, text, and embedding preprocessing
- Feature importance reporting
- Leaf-index introspection and path-based prediction contributions
- Continued training through `init_model` and estimator `warm_start`
- Standalone pure-Python deployment export through `Booster.export_model(..., export_format="python")` and matching sklearn-estimator wrappers for numeric or already-prepared features
- Build metadata reporting through `ctboost.build_info()`
- CPU builds on standard CI runners
- Optional CUDA compilation when building from source with a suitable toolkit
- GPU source builds now keep fit-scoped histogram data resident on device, support shared-memory histogram accumulation, and expose GPU raw-score prediction for regression, binary classification, and multiclass models
- Histogram building now writes directly into final-width compact storage when the fitted schema permits `<=256` bins, avoiding the old transient `uint16 -> uint8` duplication spike
- Fitted models now store quantization metadata once per booster instead of duplicating the same schema in every tree
- Low-level boosters can export reusable fitted borders through `Booster.get_borders()` and expose the full shared quantization schema through `Booster.get_quantization_schema()`
- GPU fit now drops the host training histogram bin matrix immediately after the device histogram workspace has been created and warm-start predictions have been seeded
- GPU tree building now uses histogram subtraction in the device path as well, so only one child histogram is built explicitly after each split
- GPU node search now keeps best-feature selection on device and returns a compact winner instead of copying the full per-feature search buffer back to host each node
- Training can emit native histogram/tree timing via `verbose=True` or `CTBOOST_PROFILE=1`

## Current Limitations

- Ordered CTRs, frequency-style CTRs, categorical crosses, low-cardinality one-hot expansion, rare-category bucketing, text hashing, and embedding expansion now run through a native C++ pipeline, while pandas extraction, raw-data routing, and Pool orchestration remain thin Python glue; `ctboost.prepare_training_data(...)` reduces that repeated Python work when you need to fit multiple times on the same raw train/eval split
- There is now a native sparse training path plus disk-backed quantized-bin staging through `ctboost.train(..., external_memory=True)` on both CPU and GPU, and distributed training can also use a standalone TCP collective coordinator through `distributed_root="tcp://host:port"`
- The legacy filesystem-based distributed path still exists for the native shard-reduction path; advanced eval, callback, ranking, and GPU compatibility workflows now fall back to a rank-0 coordinator path, while the TCP backend remains the true multi-rank path for those features
- Distributed grouped/ranking training requires each `group_id` to live entirely on one worker shard; cross-rank query groups are rejected
- Dedicated GPU wheel automation now targets Linux `x86_64` and Windows `amd64` CPython `3.10` through `3.14` release assets
- CUDA wheel builds in CI depend on container-side toolkit provisioning

## Resolved Fold-Memory Hotspots

The older `v0.1.15` GPU fit-memory bottleneck list is now closed in the current tree:

- Quantization metadata is stored once per fitted booster and shared by all trees instead of being duplicated per tree
- GPU fit releases the host training histogram bin matrix immediately after device workspace creation and warm-start seeding
- GPU tree growth uses histogram subtraction, so only one child histogram is built explicitly after a split
- GPU split search keeps best-feature selection on device and copies back only the winning feature summary

That means the old per-node GPU bin-materialization issue is no longer the main resident-memory problem in the current codebase. The remaining generic backlog is now in broader distributed runtime ergonomics and additional export or deployment tooling.

## Benchmark Snapshot

The heavy ordered-target-encoding `playground-series-s6e4` replay was last measured on April 12, 2026 with the `v0.1.11` source tree. The one-fold Kaggle source-build replay completed successfully with:

- build `55.41s`
- fold preprocess `57.17s`
- fold fit `2107.10s`
- fold predict `5.89s`
- fold total `2170.17s`
- validation score `0.973213`

Since that replay, the source tree has removed additional fit-memory overhead by sharing quantization schema per model, building compact train bins without a second host copy, releasing host train-bin storage after GPU upload, and adding GPU histogram subtraction plus device-side best-feature reduction.

## Installation

For local development or source builds:

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

### Wheels vs Source Builds

`pip install ctboost` works without a compiler only when PyPI has a prebuilt wheel for your exact Python/OS tag. If no matching wheel exists, `pip` falls back to the source distribution and has to compile the native extension locally.

The release workflow is configured to publish CPU wheels for current CPython releases on Windows and Linux, plus macOS `x86_64` CPU wheels for CPython `3.10` through `3.14`, so standard `pip install ctboost` usage does not depend on a local compiler.

Each tagged GitHub release also attaches the CPU wheels, the source distribution, and dedicated Linux `x86_64` plus Windows `amd64` CUDA wheels for CPython `3.10` through `3.14`. The GPU wheel filenames carry a `1gpu` build tag so the release can publish CPU and GPU artifacts for the same Python and platform tags without filename collisions.

The GPU release jobs install the CUDA toolkit in CI, export the toolkit paths into the build environment, and set `CTBOOST_REQUIRE_CUDA=ON` so the wheel build fails instead of silently degrading to a CPU-only artifact. The release smoke test also checks that `ctboost.build_info()["cuda_enabled"]` is `True` before the GPU wheel is uploaded.

### Kaggle GPU Install

`pip install ctboost` still resolves to the CPU wheel on PyPI. On Kaggle, install the matching GPU release wheel from GitHub instead:

```python
import json
import subprocess
import sys
import urllib.request

tag = "v0.1.28"
py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
api_url = f"https://api.github.com/repos/captnmarkus/ctboost/releases/tags/{tag}"

with urllib.request.urlopen(api_url) as response:
    release = json.load(response)

asset = next(
    item
    for item in release["assets"]
    if item["name"].endswith(".whl") and f"-1gpu-{py_tag}-{py_tag}-" in item["name"]
)

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-U", asset["browser_download_url"]]
)
```

After installation, confirm the wheel really contains CUDA support:

```python
import ctboost

info = ctboost.build_info()
if not info["cuda_enabled"]:
    raise RuntimeError(f"Expected a CUDA-enabled CTBoost wheel, got: {info}")
print(info)
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
exported_borders = booster.get_borders()
```

Per-feature quantization controls are available on the same low-level API:

```python
booster = ctboost.train(
    pool,
    {
        "objective": "RMSE",
        "learning_rate": 0.1,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "max_bins": 128,
        "max_bin_by_feature": {0: 16, 1: 8},
        "border_selection_method": "Uniform",
        "feature_borders": {1: [-0.5, 0.0, 0.5]},
        "nan_mode_by_feature": {0: "Max"},
    },
    num_boost_round=32,
)
```

`feature_borders` lets selected numeric features reuse explicit cut values, `max_bin_by_feature` overrides the global `max_bins` budget per column, `border_selection_method` currently supports `Quantile` and `Uniform`, and `Booster.get_borders()` returns an importable border bundle keyed by fitted feature index.

The same low-level API also exposes generic regularization and growth controls around the existing conditional tree learner:

```python
booster = ctboost.train(
    pool,
    {
        "objective": "RMSE",
        "learning_rate": 0.1,
        "max_depth": 4,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,
        "feature_weights": {0: 2.0, 3: 0.5},
        "first_feature_use_penalties": {2: 1.5},
        "random_strength": 0.2,
        "grow_policy": "LeafWise",
        "max_leaves": 16,
        "min_samples_split": 8,
        "max_leaf_weight": 2.0,
    },
    num_boost_round=64,
)
```

`feature_weights` rescales feature preference without replacing the conditional test, `first_feature_use_penalties` discourages the first use of selected features at the model level, `random_strength` adds seeded noise to break near-ties in split gain after the conditional gate has already accepted a candidate, and `grow_policy="LeafWise"` currently means a best-child-first heuristic under the existing `max_leaves` budget rather than a separate split criterion.

The same low-level API can now prepare raw categorical/text/embedding inputs directly:

```python
import numpy as np
import ctboost

X = np.empty((4, 4), dtype=object)
X[:, 0] = ["berlin", "paris", "berlin", "rome"]
X[:, 1] = [1.0, 2.0, 1.5, 3.0]
X[:, 2] = ["red fox", "blue fox", "red hare", "green fox"]
X[:, 3] = [
    np.array([0.1, 0.4, 0.2], dtype=np.float32),
    np.array([0.7, 0.1, 0.3], dtype=np.float32),
    np.array([0.2, 0.5, 0.6], dtype=np.float32),
    np.array([0.9, 0.2, 0.4], dtype=np.float32),
]
y = np.array([0.5, 1.2, 0.7, 1.6], dtype=np.float32)

booster = ctboost.train(
    X,
    {
        "objective": "RMSE",
        "learning_rate": 0.1,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "ordered_ctr": True,
        "one_hot_max_size": 4,
        "max_cat_threshold": 16,
        "cat_features": [0],
        "simple_ctr": ["Mean", "Frequency"],
        "per_feature_ctr": {0: ["Mean"]},
        "text_features": [2],
        "embedding_features": [3],
    },
    label=y,
    num_boost_round=32,
)

raw_predictions = booster.predict(X)
```

If you want to reuse the raw-data preparation work across repeated fits on the same split, prepare it once and then train against the prepared bundle:

```python
prepared = ctboost.prepare_training_data(
    X_train,
    {
        "objective": "RMSE",
        "ordered_ctr": True,
        "cat_features": [0],
        "text_features": [2],
    },
    label=y_train,
    eval_set=[(X_valid, y_valid)],
    eval_names=["holdout"],
)

booster = ctboost.train(
    prepared,
    {
        "objective": "RMSE",
        "learning_rate": 0.1,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "ordered_ctr": True,
        "cat_features": [0],
        "text_features": [2],
    },
    num_boost_round=64,
    early_stopping_rounds=10,
)
```

For disk-backed pool staging on large folds:

```python
pool = ctboost.prepare_pool(
    X_numeric,
    y,
    external_memory=True,
    external_memory_dir="ctboost-cache",
)

booster = ctboost.train(
    X_numeric,
    {
        "objective": "RMSE",
        "learning_rate": 0.1,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "external_memory": True,
        "external_memory_dir": "ctboost-cache",
    },
    label=y,
    num_boost_round=64,
)
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

For estimator-side ordered CTRs, categorical crosses, one-hot expansion, rare-category bucketing, text hashing, and embedding expansion, use the Python feature pipeline parameters:

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
label = np.array([0.5, 1.2, 0.7, 1.6], dtype=np.float32)

model = CTBoostRegressor(
    iterations=32,
    learning_rate=0.1,
    max_depth=3,
    ordered_ctr=True,
    one_hot_max_size=8,
    max_cat_threshold=32,
    cat_features=["city"],
    categorical_combinations=[["city", "headline"]],
    simple_ctr=["Mean", "Frequency"],
    per_feature_ctr={"city": ["Mean"]},
    text_features=["headline"],
    embedding_features=["embedding"],
)
model.fit(frame, label)
```

`one_hot_max_size` keeps low-cardinality categoricals as explicit indicator columns, `max_cat_threshold` buckets higher-cardinality levels down to a capped native categorical domain before the conditional tree learner sees them, and `per_feature_ctr` lets specific base features or categorical combinations opt into CTR generation without changing the underlying conditional split logic.

### Model Persistence, Warm Start, And Cross-Validation

```python
import ctboost

booster.save_model("regression-model.json")
restored = ctboost.load_model("regression-model.json")
restored_predictions = restored.predict(pool)
booster.export_model("standalone_predictor.py", export_format="python")

continued = ctboost.train(
    pool,
    {"objective": "RMSE", "learning_rate": 0.2, "max_depth": 2, "alpha": 1.0, "lambda_l2": 1.0},
    num_boost_round=10,
    init_model=restored,
)

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
- `export_model(..., export_format="python")` for standalone numeric or already-prepared deployment scoring
- `load_model(...)`
- `staged_predict(...)`
- `staged_predict_proba(...)` for classifiers
- `predict_leaf_index(...)`
- `predict_contrib(...)`
- `evals_result_`
- `best_score_`
- `sample_weight` on `fit(...)`
- `class_weight`, `scale_pos_weight`, `eval_metric`, `nan_mode`, `nan_mode_by_feature`, and `warm_start`
- `max_bins`, `max_bin_by_feature`, `border_selection_method`, and `feature_borders`
- `bagging_temperature`, `feature_weights`, `first_feature_use_penalties`, `random_strength`, `grow_policy`, `min_samples_split`, and `max_leaf_weight`

## Public Python API

The main entry points are:

- `ctboost.Pool`
- `ctboost.FeaturePipeline`
- `ctboost.PreparedTrainingData`
- `ctboost.prepare_pool`
- `ctboost.prepare_training_data`
- `ctboost.train`
- `ctboost.cv`
- `ctboost.Booster`
- `ctboost.CTBoostClassifier`
- `ctboost.CTBoostRanker`
- `ctboost.CTBoostRegressor`
- `ctboost.CBoostClassifier`
- `ctboost.CBoostRanker`
- `ctboost.CBoostRegressor`
- `ctboost.build_info`
- `ctboost.load_model`

## Build and Test

Run the test suite:

```bash
pytest tests
```

The latest local release-candidate validation on April 13, 2026 was:

```bash
python -m pytest -q
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
- macOS `x86_64`
- CPython `3.8`, `3.9`, `3.10`, `3.11`, `3.12`, `3.13`, and `3.14`

GitHub Actions workflows:

- `.github/workflows/cmake.yml`: configures, builds, installs, and tests CPU builds on Ubuntu, Windows, and macOS for pushes and pull requests
- `.github/workflows/publish.yml`: builds release wheels and the sdist, runs wheel smoke tests on built artifacts, publishes CPU wheels to PyPI, and attaches both CPU and Linux/Windows GPU wheels to tagged GitHub releases

The standard PyPI release wheel workflow builds CPU-only wheels by setting:

```text
cmake.define.CTBOOST_ENABLE_CUDA=OFF
```

The GPU release-wheel matrices enable CUDA separately with:

```text
cmake.define.CTBOOST_ENABLE_CUDA=ON
cmake.define.CTBOOST_REQUIRE_CUDA=ON
cmake.define.CMAKE_CUDA_COMPILER=/usr/local/cuda-12.0/bin/nvcc
cmake.define.CUDAToolkit_ROOT=/usr/local/cuda-12.0
cmake.define.CMAKE_CUDA_ARCHITECTURES=60;70;75;80;86;89
wheel.build-tag=1gpu
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
