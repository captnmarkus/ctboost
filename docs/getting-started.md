# Getting started

## Install

```bash
python -m pip install -U ctboost
```

The base runtime depends only on NumPy. Install optional integrations explicitly:

```bash
python -m pip install "ctboost[sklearn,plot,dataframes]"
```

See [GPU installation](gpu.md) for the CUDA support matrix.

## Classification

```python
from ctboost import CTBoostClassifier

model = CTBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    max_depth=6,
    random_seed=7,
)
model.fit(X_train, y_train)

labels = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## Regression

```python
from ctboost import CTBoostRegressor

model = CTBoostRegressor(
    loss_function="RMSE",
    iterations=500,
    learning_rate=0.05,
)
model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=40,
)
predictions = model.predict(X_test)
```

## Fixed-structure leaf refinement

For single-output objectives, `leaf_estimation_iterations` can run 1–5 Newton
or objective-defined gradient/Hessian passes after each conditional-inference
tree structure has been selected:

```python
model = CTBoostClassifier(leaf_estimation_iterations=3)
```

The default is `1`, which is the legacy training path. Additional passes keep
the split topology fixed and update only its leaves. They evaluate the current
tree as an unshrunk raw-margin delta; the outer learning rate and any DART scale
are applied once afterward. Sample/bootstrap weights, ranking metadata, leaf
caps, monotone constraints, and snapshots follow the same contract. GPU tree
construction uses these same host-side leaf passes, and distributed
single-output training reduces per-leaf statistics across workers. Extra passes
add training work and should be selected on validation data. The value is
persisted in model state and snapshots; exact snapshot resume requires the same
value and rejects configuration drift (use `init_model` when intentionally
changing it).

Multiclass objectives currently reject values greater than `1`: independent
per-class diagonal Newton steps can overshoot because softmax classes are
coupled. This fails closed until a coupled, safeguarded multiclass solver is
available.

## Optional grouped feature test

High-resolution numeric histograms can opt into an approximately
equal-node-weight grouped independence test while retaining the original bins
for the final cut search:

```python
model = CTBoostClassifier(
    feature_test="grouped",
    feature_test_bins=8,
    feature_test_adjustment="bonferroni",  # optional; default is "none"
)
```

The legacy `feature_test="quadratic"` path remains the default. Categorical
features always retain that nominal quadratic test. See
[Conditional split statistics](guides/split-statistics.md) for semantics,
persistence, and current GPU limits.

## pandas categoricals

Keep categorical values as strings or pandas `category` columns and identify them
by name or position:

```python
model = CTBoostClassifier(
    cat_features=["country", "plan"],
    ordered_ctr=True,
)
model.fit(frame, target)
```

The fitted feature pipeline is stored with the estimator, including category
dictionaries, CTR statistics, text dictionaries, and embedding transforms.

## Check the build

```python
import ctboost

print(ctboost.__version__)
print(ctboost.build_info())
```

`build_info()` reports the compiled version, compiler, C++ standard, and whether
the installed wheel contains CUDA support.
