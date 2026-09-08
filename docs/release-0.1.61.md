# CTBoost 0.1.61 (unreleased)

This release candidate reduces CPU prediction overhead and corrects two
regression-loss edge cases. Conditional-inference feature selection still
precedes cut-point optimization. Learning defaults remain unchanged.

## CPU prediction

- Fitted preprocessing no longer serializes its metadata on every prediction.
  Numeric input uses typed conversion, and multiclass categorical target
  statistics reuse category lookups across output columns.
- Native prediction uses a compact traversal cache. Scalar multiclass trees
  with identical topology share traversal; root-only trees add their updates
  directly. Physical trees and leaf-index columns retain their original order.
- The cache is built lazily and invalidated when model state changes. It adds
  memory and first-call work. Fast-math builds and GCC targets with native FMA
  use the original score path to preserve their rounding behavior.
- GPU leaf-index prediction retains its CPU traversal without building the
  otherwise unused CPU score cache.

The fixed 14-model development comparison measured a geometric **1.76x**
speedup over public 0.1.60, including input preprocessing, with exact saved-model
predictions on that panel. Development CTBoost was faster than the separately
fitted CatBoost models on 11 of 14 tasks. Model sizes and accuracy differ, and
concurrent workloads affected individual timings; this is not an
accuracy-matched speed comparison or an official TabArena result. The
[archived protocol, per-task timings and controls](https://github.com/captnmarkus/ctboost/tree/8cc75aa/benchmarks/results/inference_accuracy_20260908)
give the limits of these measurements.

## Correctness and compatibility

- Huber initialization ignores zero-weight target extrema and converges across
  the finite target range instead of stopping after 100 bisections.
- Quantile loss uses a zero subgradient at an exact fit, so training does not
  move constant targets away from an already optimal prediction.
- Numeric preprocessing retains the old signaling-NaN behavior under NumPy
  error policies, including errors raised from pandas numeric conversion.
- Direct C++ tree prediction preserves bounds checks when a histogram does
  not contain every feature referenced by the tree.

These loss fixes affect newly trained models using those objectives. Saved
model and preprocessing formats are unchanged; existing scalar/vector models,
prefix prediction, exports and physical leaf-index outputs remain supported.
Applications linking native headers or the static core must rebuild.

## Accuracy evidence

The development studies do not establish an Elo increase or a general win
over default XGBoost. Existing grouped feature tests and joint multiclass tests
showed gains on some tasks and regressions on others, so they remain opt-in.
The experimental joint-cut change failed its quality gate and is excluded.
The loss fixes above do not explain or close the measured RMSE gap.

The frozen Full HPO25 evaluation continues to measure public **0.1.60**. It
does not measure this candidate, and no historical result has been relabeled.
