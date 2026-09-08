# CTBoost 0.1.61

This release reduces CPU prediction overhead and corrects two
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

The finalized local Windows CPU wheel measured a geometric **1.79x** warm-prediction
speedup over public 0.1.60 across 14 fixed saved models, including input
preprocessing for 1,000-row batches. All 14 improved with bit-for-bit identical
predictions. The two ordering-specific comparisons measured 1.83x and 1.76x.
The summary uses geometric means across the two passes and then across tasks.

CTBoost was faster than the separately fitted CatBoost models on 11 of 14 tasks.
Model sizes and accuracy differ, and concurrent workloads affected individual
timings; this is not an accuracy-matched comparison or an official TabArena
result. The [final protocol, per-task timings and controls](https://github.com/captnmarkus/ctboost/tree/833a775/benchmarks/results/release_readiness_20260908/inference_timing_phase3)
retain all four public/candidate/candidate/public passes. Earlier development
measurements remain in the separate historical archive.

Compatibility hardening retains object conversion for float16 inputs and mixed
numeric DataFrames that contain float32 columns. Uniform float32 frames and
other supported numeric NumPy arrays retain typed conversion. This tradeoff
preserves the original NumPy warning, callback and exception behavior. Conversion
cost therefore depends on the input dtypes as well as the model.

## Correctness and compatibility

- Huber initialization ignores zero-weight target extrema and converges across
  the finite target range instead of stopping after 100 bisections.
- Quantile loss uses a zero subgradient at an exact fit, so training does not
  move constant targets away from an already optimal prediction.
- Numeric preprocessing retains the old signaling-NaN behavior under NumPy
  error policies, including errors raised from pandas numeric conversion.
- CPU training on GNU FMA targets uses the same score-update arithmetic as
  prediction. Previously, different rounding in the training shortcut could
  change resumed training and later split decisions. The fix traverses each
  newly built tree once on these targets, adding some training work.
- Direct C++ tree prediction preserves bounds checks when a histogram does
  not contain every feature referenced by the tree.

These loss fixes affect training, including continued Quantile training. Saved
model and preprocessing formats are unchanged; existing scalar/vector models,
prefix prediction, exports and physical leaf-index outputs remain supported.
Applications linking native headers or the static core must rebuild.

Warm start with a nonzero `Pool.baseline` is still not guaranteed to reproduce
uninterrupted training exactly: floating-point summation order can change later
split decisions. This existing limitation is separate from the GNU FMA fix.

## Accuracy evidence

The development studies do not establish an Elo increase or a general win
over default XGBoost. Existing grouped feature tests and joint multiclass tests
showed gains on some tasks and regressions on others, so they remain opt-in.
The experimental joint-cut change failed its quality gate and is excluded.
The loss fixes above do not explain or close the measured RMSE gap.

The frozen Full HPO25 evaluation continues to measure public **0.1.60**. It
does not measure 0.1.61, and no historical result has been relabeled.

## Release validation

The final installed Windows/Python 3.12 CPU wheel passed **1,916 tests**, with
38 skips for optional dependencies, GPU capabilities or checkout-only checks.
The native API fixture passed 230 checks. All 14 public-0.1.60 saved models
retain bit-for-bit raw and prepared predictions across 6,610 development rows,
with unchanged serialized states.

Focused preprocessing suites passed 323 cases on both NumPy 1.26/pandas 2.1
and NumPy 2.5/pandas 3.0. All 13 CPU platform/compiler jobs passed, including
Linux ARM, GNU FMA with and without contraction, and the scoped MSVC fast-math
compatibility check. The pre-release dry run built and smoke-tested all 26 wheels,
validated the complete matrix and strict package metadata, passed R/JVM checks,
and rebuilt and smoke-tested the source distribution. Both publication jobs
were skipped. CUDA-enabled wheel checks did not execute on GPU hardware. The
[pre-release verification archive](https://github.com/captnmarkus/ctboost/tree/833a775/benchmarks/results/release_readiness_20260908)
retains original failures, corrective checks and artifact identities.
