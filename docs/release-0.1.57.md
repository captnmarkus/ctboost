# CTBoost 0.1.57

CTBoost 0.1.57 adds optional compact multiclass vector leaves and fixes training,
resume, and export correctness issues. It retains the conditional-inference
feature tests and split selection, including the default quadratic feature test
and opt-in grouped testing. The default multiclass storage strategy remains
`one_output_per_tree`.

## Compact multiclass vector leaves

```python
from ctboost import CTBoostClassifier

model = CTBoostClassifier(multi_strategy="multi_output_tree")
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
```

For CPU classification with more than two classes, this strategy stores a
shared tree once per boosting round and keeps one score per class in each leaf.
It uses the same multiclass structure target, statistical feature selection,
split topology, and leaf estimates as the scalar strategy. It reduces repeated
topology storage and shares prediction traversal without introducing a new
split statistic or claiming an accuracy improvement.

Supported workflows include compatible callable objectives, categorical and
sparse inputs, fitted preprocessing, DART, callbacks, early stopping, snapshots,
warm starts, staged predictions, and explanations. The
[vector-leaf guide](guides/vector-leaves.md) documents the full contract and a
reproducible storage and prediction comparison.

## Correctness fixes

- Final multiclass leaf updates now respect `max_leaf_weight` in both layouts.
- Distributed multiclass training uses globally reduced structure moments and
  class leaf statistics, including unequal shard weights.
- CPU DART computes gradients from the retained ensemble, base score and row
  baseline, avoiding cancellation errors when previous trees are dropped.
- Python-controlled DART early stopping restores the weights from the best
  round along with its retained trees.
- Completed snapshots still validate the requested training configuration,
  feature schema, and learning-rate schedule before returning a saved model.
- Learning-rate callbacks are no longer invoked for an unused round after
  callbacks or early stopping have stopped training.
- Empty leaf-index predictions retain the physical tree dimension.
- Generated C++ scorers emit valid floating-point literals for integral values.
- Generated Python scorers enforce `nan_mode="Forbidden"` for missing input.

## Compatibility and boundaries

Existing scalar model documents and JSON predictors retain their format
versions and remain readable. Vector model documents and JSON predictors use
format version 3; vector inference manifests use version 2. Older readers
cannot consume vector artifacts. The Python JSON runtime supports fitted raw
preprocessing for vector models, while standalone Python, C++, and ONNX scorers
require prepared numeric features when a fitted pipeline is present.

The R and JVM source inference packages continue to support prepared scalar
predictor versions 1 and 2 and reject version-3 vector predictors. C++ and ONNX
exports expand vector trees into scalar trees at the export boundary.

`apply` and `predict_leaf_index` return one column per physical tree: one per
vector round, versus one per class per scalar round. Prediction, probability,
and explanation outputs preserve their class dimension. Warm starts and
snapshots must retain the saved strategy.

GPU and distributed vector training are explicitly unsupported. Multi-target
regression and multilabel wrappers continue to fit independent boosters.
Existing multiclass constraints and the restriction on
`leaf_estimation_iterations > 1` still apply.

The native tree and booster C++ layouts changed. Applications linking the
repository's native headers or static core must rebuild; ordinary wheel
installations receive a matching extension. This release retains the CUDA,
distributed, preprocessing-validation, and portable-inference foundations from
0.1.56.
