# Compact multiclass vector leaves

Set `multi_strategy="multi_output_tree"` to store one conditional inference tree
per boosting round, with one score per class in each leaf:

```python
from ctboost import CTBoostClassifier

model = CTBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    max_depth=4,
    multi_strategy="multi_output_tree",
)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=20)
probabilities = model.predict_proba(X_test)  # (rows, classes)
leaves = model.apply(X_test)                # (rows, retained rounds)
```

The low-level API accepts the same parameter with `objective="MultiClass"`
and `num_classes=K`. `Booster.predict` continues to return raw class scores.
`Booster.multi_strategy` reports the fitted layout.

## Preserving conditional inference

CTBoost already builds a shared multiclass partition. At each round it selects
the class gradient with the greatest weighted variance as the structure target,
runs the existing conditional inference feature tests and split selection,
then estimates a Newton update for each class in each leaf.

The default `one_output_per_tree` stores that partition in K scalar trees.
`multi_output_tree` stores it once and traverses it once per raw-score prediction row.
Both layouts use the same structure target, conditional feature tests,
significance threshold, feature sampling, categorical/missing routing and leaf
estimates. With the same configuration and seed, their predictions and split
topologies match. The default remains unchanged.

This feature is inspired by the [XGBoost vector-leaf model](https://xgboost.ai/2026/08/25/introducing-the-xgboost-vector-leaf-model).
CTBoost retains its existing conditional inference algorithm. It does not add
a joint multivariate split statistic or claim the accuracy gains from XGBoost's
different training algorithm. The benefits here are reduced topology storage
and shared prediction traversal; timing gains depend on the workload.

## Supported workflows

- CPU multiclass objectives with more than two classes, including compatible
  callable derivatives; class/row weights and baselines work as before.
- Native categorical and sparse inputs, missing values, external-memory pools,
  and the existing categorical/text/embedding feature pipeline.
- Sampling, DART, feature and interaction controls, learning-rate schedules,
  callbacks, early stopping, warm starts and snapshots.
- Raw/probability predictions, staged prediction, path contributions, exact
  background-based TreeSHAP, leaf influence and diagnostic plots.
- JSON and pickle persistence, compact standalone Python/JSON predictors,
  and C++/ONNX exports that expand to scalar trees at the export boundary.

`apply`/`predict_leaf_index` returns one column per physical tree: R columns
for R vector rounds, versus R × K columns for the default multiclass layout.
SHAP and contribution output shapes retain the class dimension.

Vector model documents and compact predictor artifacts use schema version 2;
scalar artifacts remain version 1, and this release still loads old scalar
models. Older releases cannot read vector artifacts. A warm start or snapshot
must retain the saved strategy; omitting the low-level parameter inherits it.

GPU and distributed vector training are rejected explicitly. Regression with
multiple targets and multilabel classification continue to use the independent
estimator wrappers. `leaf_estimation_iterations > 1` remains unsupported for
multiclass, and existing multiclass constraint restrictions still apply.

## Reproducible comparison

From a source checkout, run:

```bash
python -m benchmarks.vector_leaf --rows 3000 --classes 8 --rounds 24
```

The benchmark asserts exact prediction equality, then reports held-out log loss,
physical tree/split counts, serialized state size and local training/prediction
timings. Compare on your own workload before choosing a layout.
