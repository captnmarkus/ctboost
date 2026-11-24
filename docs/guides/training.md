# Training workflows

CTBoost supports both a low-level `ctboost.train(...)` interface and
scikit-learn-compatible estimators. Start with the
[getting-started examples](../getting-started.md); use the workflows below when
you need custom optimization, training control, model selection, or structured
targets.

For safeguarded Newton updates and coupled multiclass learning, see
[leaf backtracking and joint feature tests](learning-options.md).

The estimator examples require the scikit-learn extra:

```bash
python -m pip install "ctboost[sklearn]"
```

## Validation and early stopping

For binary classification, put `eval_metric` in the estimator constructor and
`early_stopping_rounds` in `.fit()`. A list containing `(X_valid, y_valid)` is
supported, as is a single tuple:

```python
from ctboost import CTBoostClassifier, log_evaluation

model = CTBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    max_depth=5,
    eval_metric="AUC",
    random_seed=42,
    task_type="CPU",
)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=100,
    callbacks=[log_evaluation(period=100)],
)
valid_probabilities = model.predict_proba(X_valid)[:, 1]
```

Early stopping retains the best iteration, so ordinary `predict_proba(...)`
uses the selected model. `best_iteration_` is zero-based. For regression,
use `CTBoostRegressor` and an appropriate metric such as `eval_metric="RMSE"`.
With several validation sets or metrics, the first controls stopping by default;
`early_stopping_name` and `early_stopping_metric` select a different one.

### Switching from LightGBM

Keep your feature preparation and fold indices. Replace the estimator and its
training controls: LightGBM's `early_stopping` callback expects a different
callback environment and cannot be passed to CTBoost. Use
`early_stopping_rounds` as above and CTBoost's `log_evaluation` for logging.

Constructor parameters also differ. Use `eval_metric` instead of `metric`,
`max_bins` instead of `max_bin`, and `min_data_in_leaf` instead of
`min_child_samples`. Recheck their meaning and defaults when tuning.
LightGBM's `num_leaves`, `reg_alpha`, `feature_pre_filter` and `n_jobs` are not accepted by
`CTBoostClassifier` or `CTBoostRegressor`; CTBoost's `alpha` controls the
statistical feature test, not L1 regularization. The
[API reference](../reference/api.md) lists the supported parameters.

## Expectile regression

Use expectile regression when a prediction that is too low has a different
cost from an equally large prediction that is too high. With `tau=0.8`,
underprediction costs four times as much as overprediction of the same
size. The fitted value is a conditional expectile. This loss family is
also available in R's mboost. [mboost ExpectReg](https://cran.r-project.org/web/packages/mboost/mboost.pdf)

```python
import numpy as np
import ctboost

objective, metric = ctboost.make_expectile_loss(0.8)
model = ctboost.CTBoostRegressor(
    loss_function=objective,
    eval_metric=metric,
    base_score=float(np.mean(y_train)),
    iterations=200,
)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=25)
```

The metric averages squared error with weight `tau` for underprediction
and `1 - tau` for overprediction, also respecting sample weights. At
`tau=0.5` the derivatives match squared-error training with the same
initial score, and the metric reports half the mean squared error. The
example initializes the callable objective at the training mean. Without
an explicit `base_score`, callable objectives start at zero. For low-level
training, pass the pair as `params["objective"]` and `params["eval_metric"]`.
The native model records the level. For portable predictor exports, store
`tau` alongside the artifact so downstream users can interpret its predictions.
To continue a saved fit, recreate the pair at the same level.

## Callable objectives and metrics

A callable objective receives raw predictions followed by labels and returns
the mathematical gradient and a non-negative Hessian for every prediction.
Sample weights are provided for context but are applied by CTBoost's tree
builder, so do not multiply them into the derivatives again.

```python
import numpy as np
import ctboost

train_pool = ctboost.Pool(X_train, label=y_train)
valid_pool = ctboost.Pool(X_valid, label=y_valid)

def squared_error(predictions, label, *, weight, **_):
    return predictions - label, np.ones_like(predictions)

objective = ctboost.make_objective(
    squared_error,
    name="MySquaredError",
    native_objective="RMSE",
)

booster = ctboost.train(
    train_pool,
    {
        "objective": objective,
        "eval_metric": "RMSE",
        "max_depth": 3,
    },
    num_boost_round=100,
    eval_set=valid_pool,
    early_stopping_rounds=10,
)
```

A bare callable is also accepted in `params["objective"]`. The
XGBoost-style `ctboost.train(..., obj=squared_error)` form instead uses the
native objective named in `params` to define output shape and inference
semantics. Estimators accept a callable or `ObjectiveSpec` through
`loss_function=`.

Callable objectives must return finite arrays with exactly the prediction
shape, and every Hessian must be non-negative. Multiclass objectives receive a
`(rows, classes)` prediction matrix. A callable may opt into keyword arguments
such as `weight`, ranking metadata, `num_classes`, and `params`.

Callable metrics return one finite scalar. Wrap one with
`make_eval_metric(...)` to give it a stable name and direction:

```python
def weighted_mae(predictions, label, *, weight, **_):
    return np.average(np.abs(predictions - label), weights=weight)

metric = ctboost.make_eval_metric(
    weighted_mae,
    name="WeightedMAE",
    higher_is_better=False,
    allow_early_stopping=True,
)
```

Set `allow_early_stopping=True` only when the metric should be eligible to
control early stopping. A derivative-only objective has no scalar loss of its
own, so `loss_history` uses the associated native objective's metric unless a
callable evaluation metric is configured.

Model artifacts store the learned trees, native inference semantics, and the
custom objective name, but not Python code. Inference is self-contained; pass
the same callable again when continuing custom-objective training from an
`init_model` or snapshot.

## Positive-link objectives and ranking weights

Gamma, Poisson, and Tweedie use a log link. `predict(...)` retains CTBoost's
additive raw-score contract; `predict_raw(...)` is the explicit alias, while
`predict_mean(...)` (also available as `predict_response(...)`) applies
`exp(raw)` and returns the positive response-scale mean.

LambdaMART computes standard unweighted NDCG within each query. Row weights
must therefore be uniform inside a `group_id`; that common value is treated as
a query weight and can be combined with `group_weight`. Nonuniform
per-document weights are rejected rather than producing invalid NDCG values.

## Learning-rate schedules and callbacks

`learning_rate_schedule` accepts either one positive value per boosting round
or a callable receiving the zero-based iteration and optional
`total_iterations` keyword:

```python
schedule = [0.2, 0.2, 0.1, 0.1, 0.05, 0.05]

booster = ctboost.train(
    train_pool,
    {
        "objective": "RMSE",
        "learning_rate": schedule[0],
        "max_depth": 3,
    },
    num_boost_round=len(schedule),
    learning_rate_schedule=schedule,
    callbacks=[ctboost.log_evaluation(2)],
)

print(booster.learning_rate_history)
```

Callbacks receive a `TrainingCallbackEnv` after each round. It exposes the
model, iteration bounds, current evaluations and history, best iteration and
score, and current learning rate. Returning a truthy value requests an early
stop. A callback may call `env.model.set_learning_rate(...)` to change the rate
used by later rounds.

CTBoost provides `log_evaluation(period)` and
`checkpoint_callback(path, interval=...)`. The estimator `fit(...)` methods
accept the same `callbacks=` and `learning_rate_schedule=` arguments. Parallel
multi-output child fits require serializable schedules and do not allow
callbacks; use `n_jobs=1` for callback-driven multi-output training.

## Model selection

To choose one boosting-round count for a refit, enable
`select_num_boost_round` in `ctboost.cv`. Each fold fits on its training rows
and scores its held-out rows after each round. CTBoost selects the earliest
round with the best mean validation metric, weighting folds equally. The
returned `best_num_boost_round` starts at one and can be passed directly to
`train`. This follows the common validation-risk approach in
[mboost's cvrisk](https://github.com/boost-R/mboost/blob/52a515bce45542a85f9de0e03417ca6db22c77e0/man/cvrisk.Rd).

```python
params = {"objective": "Logloss", "eval_metric": "AUC"}
summary = ctboost.cv(
    train_pool, params, num_boost_round=200, nfold=5, select_num_boost_round=True,
)
model = ctboost.train(train_pool, params, num_boost_round=summary["best_num_boost_round"])
print(summary["best_valid_score"])
```

With several evaluation metrics, the first controls round selection. CTBoost
uses its declared optimization direction, so it maximizes AUC and minimizes
RMSE. Only rounds with a finite validation score in every fold are eligible.
With fold-level early stopping, the shortest retained history limits the
search. Omit `early_stopping_rounds` to compare the full path. Assess the
refitted model on separate test data. Fit any label-dependent preprocessing
inside each training fold before scoring its validation rows.

The primary scikit-learn estimators expose `grid_search(...)`,
`randomized_search(...)`, `select_features(...)`, and `plot_metrics(...)`.
Searches follow scikit-learn's cross-validation and scoring contracts. With
`refit=True`, the estimator adopts the best fitted model:

```python
model = ctboost.CTBoostClassifier(iterations=300, random_seed=7)

search = model.grid_search(
    {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.1],
    },
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc",
    refit=True,
)

print(search["params"], search["best_score"])
probabilities = model.predict_proba(X_test)
```

`select_features(...)` ranks raw input columns by permutation importance, so
categorical, text, and embedding columns remain visible as users supplied
them. Use `ctboost.compare_estimators(...)`, or `model.compare(...)`, to run
CTBoost and other compatible estimators on identical folds and report per-fold
scores plus fit and scoring times. Plotting helpers require the `plot` extra.

## Multi-output regression and multilabel classification

Targets with multiple independent outputs use one native booster per output:

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

Trees, conditional split tests, preprocessing, and early-stopping state are
independent for each output. A one-dimensional `sample_weight` is shared;
`(rows, outputs)` weights apply per output. `predict_proba(...)` on the
multilabel classifier follows the scikit-learn multi-output convention and
returns one `(rows, 2)` probability matrix per label.

CPU child fits can use joblib process parallelism. GPU child fits, callbacks,
and non-picklable Python objectives require `n_jobs=1`; these wrappers do not
orchestrate distributed training. Parallel schedules, objectives, and metrics
must be serializable.

## Accelerated-failure-time survival regression

`CTBoostAFTSurvivalRegressor` fits a log-normal accelerated-failure-time model
for exact, left-censored, right-censored, and interval-censored observations:

```python
bounds = np.array(
    [
        [5.0, 5.0],       # exact event
        [0.0, 4.0],       # left-censored
        [7.0, np.inf],    # right-censored
        [3.0, 8.0],       # interval-censored
    ]
)

aft = ctboost.CTBoostAFTSurvivalRegressor(
    ctboost.CTBoostRegressor(iterations=300),
    scale=0.8,
    prediction_type="time",
).fit(X_train, bounds, eval_set=(X_valid, valid_bounds))

median_time = aft.predict_time(X_test)
mean_time = aft.predict_time(X_test, kind="mean")
log_time_location = aft.predict_log_time(X_test)
```

A flat target sequence represents exact event times. Bounds can be an
`(rows, 2)` array or a `(lower_vector, upper_vector)` tuple; a flat two-value
tuple represents two exact observations, not one interval. `scale` is the
fixed standard deviation of `log(T)`, and the reported `AFTNLL` metric uses the
censoring-aware log-normal likelihood.

AFT convenience training supports CPU or one GPU, but not distributed fitting.
It currently requires numeric or already-prepared features rather than
target-aware categorical, text, or embedding preprocessing.

The multi-output, multilabel, and AFT wrappers use pickle for resumable Python
persistence. Their `export_model(directory)` method writes an inference bundle
containing a manifest and standalone JSON predictor for every output; load it
with the corresponding wrapper class's `load_exported_model(directory)`
classmethod. See the [deployment guide](deployment.md)
for single-model persistence and export workflows, and the
[Python API reference](../reference/api.md) for the canonical public names.
