# CTBoost

CTBoost is a gradient boosting library built around Conditional Inference Trees, with a native C++17 core, `pybind11` bindings, optional CUDA support, and optional scikit-learn compatible estimators.

CTBoost is focused on making conditional-inference-tree boosting practical for real datasets and production workflows. Development is centered on data ingestion, preprocessing, metrics, orchestration, serialization, and deployment around the existing learner.

## What CTBoost Supports

- Regression, classification, grouped ranking, and survival training
- Low-level `ctboost.train(...)` plus `CTBoostClassifier`, `CTBoostRegressor`, and `CTBoostRanker`
- NumPy, pandas, and SciPy sparse input without dense conversion
- Native categorical, text, and embedding preprocessing through `FeaturePipeline`
- Row weights, class imbalance controls, missing-value handling, quantization controls, and generic regularization or growth settings
- Validation watchlists, callable objectives, multiple or callable eval metrics, early stopping, per-iteration callbacks, and learning-rate schedules or callback-driven learning-rate changes
- Stable JSON and pickle persistence, warm start via `init_model`, snapshot-path resume with config and schema validation, staged prediction, and standalone Python or JSON predictor export
- Richer `Pool` schema metadata via `feature_names`, `column_roles`, `feature_metadata`, and `categorical_schema`
- Ranking metadata in `Pool`: `group_id`, `group_weight`, `subgroup_id`, `pairs`, `pairs_weight`, and `baseline`
- External-memory pool staging plus optional TCP-based distributed training
- Feature importance/statistics, leaf indices, approximate shared-leaf object influence, prediction/tree plots, fast path contributions, and exact background-based TreeSHAP values or interactions

See [BACKLOG.md](BACKLOG.md) for the remaining generic feature roadmap and deferred items.

## Demos

The repository ships local Kaggle-oriented demos in [`demo/`](demo/) instead of remote kernel automation:

- `demo/kaggle_titanic.py` for binary classification on Titanic
- `demo/kaggle_house_prices.py` for regression on House Prices

See [`demo/README.md`](demo/README.md) for expected data layouts and run commands.

## Installation

Install the current release from PyPI:

```bash
python -m pip install --upgrade ctboost
```

Starting with CTBoost 0.1.54, that ordinary pip command installs a
CUDA-enabled wheel on manylinux-compatible x86-64 systems and Windows AMD64
when using CPython 3.10 through 3.14. The same wheel continues to work for CPU
training on a machine without an NVIDIA GPU. It bundles the CUDA 12.8 runtime
library, so GPU use requires an NVIDIA driver compatible with CUDA 12.x
(525.60.13 or newer on Linux; 528.33 or newer on Windows), but does not require
a locally installed CUDA toolkit. The bundled NVIDIA runtime remains subject
to the NVIDIA CUDA Toolkit license included in each CUDA-enabled wheel.

Released CUDA wheels target NVIDIA compute capability 6.0 or newer, with
native code for Pascal through Blackwell and forward-compatible PTX for future
architectures. macOS, Linux aarch64, and the CPython 3.8/3.9 wheels remain
CPU-only. Inspect the installed build before selecting `task_type="GPU"`:

```bash
python -c "import ctboost; print(ctboost.build_info())"
```

GPU-capable builds report `cuda_enabled: True`; a driver or device error is
reported only when GPU work is requested. The legacy `ctboost-install-gpu`
command is retained for CTBoost 0.1.52 and earlier GitHub Release assets. It is
deprecated and is not needed for 0.1.54 or later.

Install from a source checkout:

```bash
python -m pip install .
```

Install development dependencies from a checkout:

```bash
python -m pip install -e ".[dev]"
```

Install the optional scikit-learn wrappers and `ctboost.cv(...)` support:

```bash
python -m pip install -e ".[sklearn]"
```

To force a CPU-only native source build:

```bash
CMAKE_ARGS="-DCTBOOST_ENABLE_CUDA=OFF" python -m pip install .
```

On PowerShell:

```powershell
$env:CMAKE_ARGS="-DCTBOOST_ENABLE_CUDA=OFF"
python -m pip install .
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

The estimators also accept familiar XGBoost/CatBoost parameter names, so existing
model-selection code can usually switch libraries without a parameter rewrite:

```python
model = CTBoostClassifier(
    n_estimators=256,
    depth=3,
    reg_lambda=1.0,       # l2_leaf_reg is accepted too
    random_state=13,
)
model.fit(frame.iloc[:200], y[:200])

leaf_indices = model.apply(frame)
history = model.get_evals_result()
best_iteration = model.get_best_iteration()
native_booster = model.get_booster()
```

`is_fitted()`, `get_best_score()`, `evals_result()`, and
`calc_leaf_indexes()` are available as convenience aliases. The low-level
`train(...)` API likewise accepts `n_estimators`/`num_trees`, `eta`, `depth`,
`reg_lambda`/`l2_leaf_reg`, `random_state`/`seed`, and `max_bin`. Conflicting
aliases and unknown parameter names fail early with a useful error instead of
being silently ignored.

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

For inference-only data, labels are optional: `prediction_pool = ctboost.Pool(X_new)`.

### Callable Objectives And Metrics

Custom objectives receive raw predictions followed by labels and return the
mathematical gradient and non-negative Hessian for every prediction:

```python
def squared_error(predictions, label, *, weight, **_):
    # Sample weights are passed for context and are applied by CTBoost's tree
    # builder, so they must not be multiplied into these derivatives again.
    return predictions - label, np.ones_like(predictions)

objective = ctboost.make_objective(
    squared_error,
    name="MySquaredError",
    native_objective="RMSE",
)

booster = ctboost.train(
    pool,
    {"objective": objective, "max_depth": 3, "alpha": 1.0},
    num_boost_round=32,
)
```

A bare callable is also accepted in `params["objective"]`, and the
XGBoost-style `ctboost.train(..., obj=squared_error)` form uses the native
objective named in `params` for output-shape and inference semantics.
The scikit-learn estimators accept the same callable or `ObjectiveSpec` as
`loss_function=` and choose their regression, classification, or ranking
semantics for a bare callable.
Gradients and Hessians must be finite arrays with exactly the prediction
shape; Hessians must be non-negative. Multiclass objectives receive a
`(rows, classes)` prediction matrix. Callables may optionally accept
`weight`, ranking metadata, `num_classes`, and `params` keyword arguments.
Because a derivative-only callable does not define a scalar loss,
`loss_history` uses the selected native objective's metric; configure a
callable eval metric when you need an objective-specific reported score.

Use `make_eval_metric(...)` for a named callable metric and declare
`higher_is_better`; set `allow_early_stopping=True` when it should control
early stopping. Model artifacts store the learned trees and custom objective
name, but never embed Python code. They are therefore self-contained for
inference; pass the same callable again when continuing custom-objective
training from an `init_model` or snapshot.

Gamma, Poisson, and Tweedie use a log link. The existing `predict` contract
continues to return the additive raw score; `predict_raw` makes that explicit,
while `predict_mean` (or its `predict_response` alias) applies `exp(raw)` to
return the positive response-scale mean.

LambdaMART uses standard, unweighted NDCG inside each query. Row weights must
therefore be uniform within a `group_id`; that common value is treated as a
query weight (and may be combined with `group_weight`). Nonuniform per-document
weights are rejected instead of producing an NDCG value outside `[0, 1]`.

### Compact multiclass vector leaves

Use `CTBoostClassifier(multi_strategy="multi_output_tree")`, or the same
parameter in `train(..., {"objective": "MultiClass", ...})`, to store one shared
tree with a score vector in each leaf per boosting round. This CPU multiclass
mode preserves CTBoost's existing conditional inference tests and predictions
while avoiding duplicated class-tree topology. The default remains
`multi_strategy="one_output_per_tree"`.

See [vector leaves](docs/guides/vector-leaves.md) for examples, persistence/export
compatibility, limits, and a reproducible comparison benchmark.

### Multi-output, multilabel, and AFT estimators

The sklearn API includes independent-tree wrappers for targets that do not fit
in a native one-dimensional `Pool.label`:

```python
multi_reg = ctboost.CTBoostMultiOutputRegressor(
    ctboost.CTBoostRegressor(iterations=300),
    n_jobs=-1,
).fit(X_train, y_train_2d)

multi_label = ctboost.CTBoostMultiLabelClassifier(
    ctboost.CTBoostClassifier(iterations=300),
    n_jobs=-1,
).fit(X_train, binary_labels_2d)
```

These estimators fit one native CTBoost booster per output. Trees, conditional
split tests, early-stopping state, and optional target-aware preprocessing are
independent rather than shared across outputs. A one-dimensional sample weight
is shared; a `(rows, outputs)` matrix applies weights per output. CPU child fits
can use joblib process parallelism, which requires serializable custom losses,
metrics, and schedules. Sequential `n_jobs=1` fitting can use non-picklable
Python objectives; persisted models retain the learned inference semantics but
not that Python code. Callbacks and GPU child fits also require `n_jobs=1`, and
these wrappers do not orchestrate distributed training.

Log-normal accelerated-failure-time survival training accepts exact,
left-censored, right-censored, and interval-censored observations:

```python
# A flat sequence contains exact event times. Alternatively, bounds[i] is
# [event_time, event_time] for an observed event,
# [0, upper] for left censoring, [lower, np.inf] for right censoring,
# or [lower, upper] for interval censoring.
aft = ctboost.CTBoostAFTRegressor(
    ctboost.CTBoostRegressor(iterations=300),
    scale=0.8,
    prediction_type="time",
).fit(X_train, bounds, eval_set=(X_valid, valid_bounds))

median_time = aft.predict_time(X_test)
mean_time = aft.predict_time(X_test, kind="mean")
log_time_location = aft.predict_log_time(X_test)
```

Bounds may also be supplied as a `(lower_vector, upper_vector)` tuple. A flat
two-value tuple such as `(1.0, 2.0)` is two exact observations, not one interval.

`scale` is the fixed standard deviation of `log(T)`. The reported `AFTNLL`
metric and `negative_log_likelihood` use the censoring-aware log-normal
likelihood. Internally, `RMSE` supplies only the scalar model/output and export
contract; the custom AFT gradient and Hessian drive every tree. AFT convenience
training supports CPU or a single GPU but not distributed fitting, and currently
requires numeric or already-prepared features rather than target-aware
categorical/text/embedding preprocessing.

All three wrappers use pickle for resumable Python persistence. Their
`export_model(directory)` method writes a manifest plus one standalone JSON
predictor per output; `load_exported_model(directory)` loads that inference-only
bundle and records the independent-tree semantics explicitly.

### Model selection conveniences

The sklearn estimators expose `grid_search`, `randomized_search`,
`select_features`, and `plot_metrics`. Searches use sklearn's cross-validation
contracts and can refit the same estimator object. Feature selection reports
permutation importance in raw input space, including categorical/text columns.
Use `compare_estimators(...)`, or `model.compare(...)`, to evaluate CTBoost and
other sklearn-compatible estimators on identical folds with per-fold scores,
timing means, and standard deviations.

### Exact TreeSHAP explanations

`predict_shap` computes exact interventional TreeSHAP values against an
explicit empirical background distribution. This is distinct from
`predict_contrib`, which remains available as a faster path-based additive
decomposition. The final SHAP column is the expected raw model output over the
background, and every explanation row sums to the corresponding raw
prediction:

```python
background = X[:32]
shap_values = booster.predict_shap(X[32:40], background)
shap_interactions = booster.predict_shap_interactions(X[32:40], background)

np.testing.assert_allclose(
    shap_values.sum(axis=-1),
    booster.predict(X[32:40]),
    rtol=1e-6,
    atol=1e-6,
)
```

For a single-output model, SHAP values have shape
`(rows, features + 1)` and interaction values have shape
`(rows, features + 1, features + 1)`. Multiclass models add an output
dimension after the row dimension. The interaction matrix follows the
XGBoost-style bias convention: the expected value is at `[..., -1, -1]`, each
feature row sums to its SHAP value, and the whole matrix sums to the raw model
prediction. `Pool.weight` supplies optional background weights. Estimator
aliases `predict_shap_values` and `predict_shap_interaction_values` are also
available. When a categorical/text/embedding `FeaturePipeline` expands raw
columns, explanations are returned in that transformed feature space and the
names are available from `model.get_booster().feature_names`.

### Object influence and diagnostic plots

`calc_leaf_influence` provides a transparent object-attribution approximation.
For each tree, it distributes the explained row's signed leaf contribution
among reference rows that reach the same leaf. A weighted reference `Pool`
uses its row weights for that distribution:

```python
influence, coverage = booster.calc_leaf_influence(
    X_test[:8],
    train_pool,
    return_coverage=True,
)
indices, scores = booster.get_object_importance(
    X_test[:8],
    train_pool,
    top_size=10,
    importance_type="PerObject",
)
```

This is deliberately not advertised as exact training influence. It performs
no deletion/upweighting refits and does not differentiate the training loss.
Positive or negative scores mean co-membership in leaves that raise or lower
the raw model output. With the original training rows as the reference, scores
sum to the covered tree component of the raw prediction; input baselines are
not attributed. `coverage` reports the fraction of trees whose explained leaf
was represented by a positive-weight reference row. The returned dense matrix
uses `rows × reference_rows` memory (plus an output dimension for multiclass),
so batch large explanation jobs.

Matplotlib conveniences return their axes and are also available on fitted
scikit-learn estimators:

```python
prediction_ax = booster.plot_predictions(X_test, y_test)
residual_ax = booster.plot_predictions(X_test, y_test, kind="residual")
feature_ax = booster.plot_feature_statistics(X_test, y_test, feature=0)
tree_ax = booster.plot_tree(0)
```

Prediction diagnostics show raw model output. Multiclass plots and ranked
object importance therefore require `prediction_dimension`. Install plotting
support with `pip install "ctboost[plot]"`.

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
    text_tokenizer="word",           # word, whitespace, or character
    text_ngram_range=(1, 2),
    text_min_token_count=2,
    text_max_dictionary_size=20_000,  # 0 keeps fixed-size feature hashing
    text_feature_calcer="tfidf",     # count, binary, or tfidf
    embedding_features=["embedding"],
    embedding_target_features=True,
    embedding_target_regularization=1.0,
    embedding_target_mode="auto",     # auto, regression, or classification
)
model.fit(frame, y)
```

Text columns use the original deterministic, fixed-size count hashing by default.
Setting `text_max_dictionary_size` learns a frequency-ranked dictionary on the
training split; `text_min_token_count` filters rare tokens, and the fitted
vocabulary and TF-IDF weights are stored with the model. `text_lowercase=False`
preserves case. Character tokenization applies `text_ngram_range` to non-space
characters, while word and whitespace tokenization apply it to token sequences.

`embedding_target_features=True` adds a regularized supervised projection for
each embedding column (one projection per class for multiclass targets). The
projection is fitted only from the training data and labels; validation and
prediction inputs never require labels. Embedding vectors must have a consistent
dimension when the target-aware transform is enabled. Estimators resolve `auto`
from their task; direct `FeaturePipeline` use treats contiguous integer targets
with three or more values as multiclass and can be made explicit with
`embedding_target_mode`. These transformations only
add input features: CTBoost's conditional-inference split selection is unchanged.

### Streaming and columnar input

`Pool` accepts eager PyArrow, Polars, cuDF, NumPy/DLPack, pandas, and SciPy
inputs. For a source that is only available as an iterator, assemble it into a
disk-backed numeric pool without retaining all source batches in RAM:

```python
from ctboost import PoolBatch, pool_from_batches, train

def batches():
    for features, target in source:
        yield PoolBatch(features, target)

pool = pool_from_batches(batches(), directory="./ctboost-spill")
model = train(pool, {"iterations": 200, "depth": 7})
```

Batch items can also be `(data, label)` tuples, mappings, existing pools, or
feature matrices. Metadata must be consistently present across batches. The
streaming bridge is numeric: apply text/embedding preprocessing before yielding
batches. `Pool.from_batches(...)` is an equivalent convenience constructor.

## Dask, Ray, And Spark

Install only the integration you use:

```bash
pip install "ctboost[dask]"  # or ctboost[ray], ctboost[spark]
```

Dask DataFrames and row-chunked Dask Arrays can train through CTBoost's native
TCP collective. One rank is pinned to each selected Dask worker, while row
partitions are combined locally on that worker. Workers must be separate
processes (the normal `dask worker`/Nanny deployment), not threads in one
Python process. Predictions stay lazy and partitioned:

```python
from dask.distributed import Client
import ctboost.dask as ctd

client = Client("tcp://scheduler:8786")
booster = ctd.train(
    client,
    dask_frame,
    label="target",
    params={"objective": "RMSE", "cat_features": ["city"]},
    num_boost_round=200,
    num_workers=4,
    mode="distributed",
)
dask_predictions = ctd.predict(booster, dask_frame.drop(columns=["target"]))
```

For small data, `mode="materialize"` is an explicit driver-memory fallback.
String labels and metadata arguments such as `weight="sample_weight"` are
treated as columns and removed from the feature frame. Feature names, pandas
dtypes, and CTBoost categorical/text pipeline configuration are retained.

Ray uses the same CTBoost TCP collective over disjoint `Ray Dataset` shards;
the label and optional metadata are named columns. Prediction uses Ray Data's
lazy `map_batches` execution:

```python
import ctboost.ray as ctr

booster = ctr.train(
    ray_dataset,
    {"objective": "Logloss", "cat_features": ["segment"]},
    label="target",
    num_boost_round=200,
    num_workers=4,
)
prediction_dataset = ctr.predict(
    booster,
    ray_dataset,
    feature_columns=["age", "income", "segment"],
)
```

Automatic Dask and Ray endpoints bind the selected worker's concrete network
address and attach a cryptographically random, per-run bearer token. For a
manually coordinated run, create one root and pass that same value to every
rank:

```python
from ctboost.distributed import authenticated_tcp_root

root = authenticated_tcp_root("10.0.0.12", 19091)
```

The token is kept only in live runtime configuration; model exports, estimator
pickles, and snapshots store the redacted `tcp://host:port` endpoint. The TCP
collective does not provide TLS encryption, so use it only on a trusted private
network or through a protected network overlay/firewall. Manually supplied bare
or wildcard TCP roots are rejected.

`ctboost.spark.train(...)` intentionally requires `mode="collect"` and calls
`DataFrame.toPandas()` for fitting, making the driver-memory boundary explicit.
It returns `SparkCTBoostModel`; its `transform(...)` method performs partitioned
Arrow pandas-UDF inference. Native multi-worker training is exposed for Dask and
Ray, whose task APIs can safely pin and coordinate the required TCP ranks.

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

# Versioned deployment contract: feature schema, objective/output semantics,
# build identity, and a deterministic model fingerprint.
manifest = booster.get_inference_manifest()
booster.export_inference_manifest("inference-manifest.json")
assert ctboost.load_inference_manifest("inference-manifest.json") == manifest

# A dependency-free Python scorer can also be generated.
booster.export_model("standalone_predictor.py", export_format="python")
```

Standalone predictors expose `predict_raw` (and its `predict` alias). Classification
exports additionally expose `predict_proba` and `predict_class`; estimator exports
preserve the fitted class-label order. If a model uses CTBoost categorical, text, or
embedding preprocessing, pass `prepared_features=True` when exporting a standalone
scorer and feed it the fitted pipeline's transformed numeric features. The manifest
records that preprocessing requirement explicitly.

`resume_from_snapshot=True` validates the saved training configuration and data schema before loading the checkpoint. It remains a warm-start-based convenience flow rather than a blanket exact-equivalence guarantee for every training path. For per-iteration checkpoint emission and logging hooks, use `callbacks=[ctboost.log_evaluation(...), ctboost.checkpoint_callback(...)]`.

## Command-Line Deployment

Install the table/estimator dependencies with `pip install "ctboost[cli]"` and
use either the installed `ctboost` command or `python -m ctboost`. The CLI calls
the same public estimators, model persistence, prediction, and export APIs as
Python code:

```bash
ctboost train \
  --task classification \
  --input train.csv \
  --target outcome \
  --categorical country,device \
  --params @training-params.json \
  --model churn.ctb

ctboost predict \
  --model churn.ctb \
  --input scoring.parquet \
  --prediction-type probability \
  --output probabilities.parquet

ctboost inspect --model churn.ctb --output model-info.json
ctboost export --model churn.ctb --format manifest --output inference-manifest.json
ctboost info
```

`--params` accepts an inline JSON object, a JSON file path, or `@path`; explicit
flags such as `--iterations`, `--learning-rate`, and `--random-seed` override
the JSON values. Regression, classification, and ranking are supported. Ranking
also requires `--group` or `--group-file`. Target, group, categorical, and
prediction drop columns can be selected by table name or zero-based feature
index. For NPZ archives, use `--array-key` for the feature matrix and select a
target/group array by key; standalone `.npy` targets use `--target-file`.

Inputs may be `.npy`, `.npz`, `.csv`, `.tsv`, `.parquet`/`.pq`, or `.feather`.
Parquet and Feather require `pyarrow`. Prediction output additionally supports
JSON. `raw` works for every model; `probability` and `class` reject
non-classification objectives. NumPy inputs are loaded with pickling disabled,
and object-valued predictions must use CSV, TSV, JSON, Parquet, or Feather.
Commands refuse to replace an existing artifact unless `--force` is supplied,
write stable JSON command summaries, and return exit status 2 with an actionable
error for expected input/model failures. JSON `.ctb` models are the safe default;
loading or creating pickle models requires `--allow-unsafe-pickle` and should be
limited to artifacts from a trusted source.

Standalone Python, C++, JSON-predictor, ONNX, and inference-manifest exports are
available through `ctboost export --format ...`. Models with categorical, text,
or embedding preprocessing need `--prepared-features` for standalone scorers;
the resulting artifact then expects numeric features already transformed by the
fitted pipeline. ONNX additionally requires `ctboost[onnx]`.

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
demo/         local example workflows, including Kaggle demos
include/      public C++ headers
src/core/     core training, data, objectives, trees, statistics
src/bindings/ pybind11 extension bindings
cuda/         optional CUDA backend
tests/        Python test suite
```

## Acknowledgments

CTBoost draws methodological inspiration from the original conditional inference tree work by
Torsten Hothorn, Kurt Hornik, and Achim Zeileis, along with the subsequent `partykit` work on
CRAN by Torsten Hothorn, Achim Zeileis, and Heidi Seibold. If you are using CTBoost in research
or want the statistical background behind the learner, start with these references:

- Hothorn, T., Hornik, K., and Zeileis, A. (2006). *Unbiased Recursive Partitioning: A Conditional Inference Framework.*
- Hothorn, T. and Zeileis, A. (2015). *partykit: A Modular Toolkit for Recursive Partytioning in R.*

## License

Apache 2.0. See `LICENSE`.
