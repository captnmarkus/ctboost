# CTBoost 0.1.59

CTBoost 0.1.59 adds optional safeguards and multiclass learning improvements
while retaining statistical feature selection before cut-point optimization.
Existing learning defaults and prediction formats remain unchanged.

- **Safeguarded scalar leaves:** `leaf_estimation_backtracking=True` checks
  the initial Newton proposal and subsequent refinements against the weighted
  RMSE/SquaredError or LogLoss objective plus L2. Overshooting proposals are
  reduced or rejected after applying constraints.
- **Coupled multiclass leaves:** `multiclass_leaf_solver="full"` uses the full
  softmax Hessian, a sum-zero representation, damping, and loss backtracking.
  It supports 1–5 refinement steps and both scalar and vector tree storage.
- **Joint multiclass feature tests:** `multiclass_feature_test="joint"` uses
  every class-gradient coordinate with a rank-aware covariance calculation.
  Feature selection remains separate from the existing scalar cut objective.
- **Faster grouped scoring:** the production grouped statistic avoids temporary
  vectors and unused Hessian aggregation. Bin grouping and score arithmetic
  retain exact agreement with the previous implementation.
- **Correct profiling:** feature-statistic and cut-search timings now measure
  their actual phases. Diagnostics distinguish the minimum-p feature from a
  different gain-selected feature in constrained or penalized ranked searches.
- **Documentation security:** require MkDocs Material 9.7.7 or newer to fix the
  search-suggestion DOM XSS vulnerability
  [GHSA-xvg9-69gf-fjrf](https://github.com/squidfunk/mkdocs-material/security/advisories/GHSA-xvg9-69gf-fjrf).
- Fixed an internal evaluation-pool helper that raised `NameError` for tuple
  inputs, and missing typing imports in distributed/evaluation helpers.

The new learning controls require built-in objectives and non-distributed CPU
training. Full multiclass leaves and joint tests support 3–32 classes. The
joint training test requires integer frequency weights; fractional sample,
class, and Bayesian bootstrap weights are rejected. Full leaf fitting and
scalar backtracking continue to support fractional weights.

Loss acceptance checks unshrunk leaf increments, before learning-rate or DART
scaling. It does not guarantee improved validation loss or every final ensemble
update. See [learning options](https://captnmarkus.github.io/ctboost/guides/learning-options/) for usage, persistence,
constraints, and the statistical interpretation.

## Validation and interpretation

The isolated Windows CPU regression suite passed **1,088 tests**, with 22
skips for unavailable GPU/framework/source-checkout capabilities. This includes
compiled C++ export prediction checks and ONNX Runtime comparisons. Strict
documentation builds passed with Material 9.7.7.

The compatibility comparison with the published 0.1.58 wheel covers 32 fixed
regression, binary, multiclass, vector, grouped, weighted/missing/categorical,
leaf-wise, and DART cases. Predictions, leaf indices, loss histories, and all
previous tree fields match exactly.

The grouped audit compares 20,000 scores with the previous implementation.
Its null cases cover small and sparse nodes, missing values, and literal
frequency weights. A separate 1,000-trial joint-test audit observed 4.7%
rejection at a nominal 5% threshold for unweighted and equivalent frequency
data. Fractional-weight continuation observed 15.7%, motivating the explicit
joint-training restriction. These are fixed-node experiments, not a proof of
error control over an entire adaptively fitted ensemble; chi-square tails
remain asymptotic.

A fixed 42-fit development comparison used bundled scikit-learn datasets,
three stratified 75/25 splits, 40 rounds, depth 3, and alpha 0.05. Mean held-out
log loss was:

| Dataset | Existing defaults | Optional configuration |
|---|---:|---:|
| Breast cancer | 0.146208 | 0.144528 |
| Iris | 0.207686 | 0.108978 |
| Wine | 0.191139 | 0.107759 |
| Digits | 0.563504 | 0.469233 |

The binary configuration uses three backtracked leaf steps; the multiclass
configuration combines three full-solver steps, joint testing, and grouped-8
bins. That combination changes several controls and is not an isolated joint-test
ablation. All predefined variants and splits are retained in the
[raw comparison](https://github.com/captnmarkus/ctboost/blob/master/benchmarks/results/learning_options_0159.json).
Full leaf fitting alone slightly worsened mean wine loss (0.191741), and binary
backtracking worsened one of three splits. Combined multiclass fitting took
roughly 3–5 times the baseline training time on this small local panel. These
results support task-specific validation, not changing the defaults.

The isolated grouped-score microbenchmark retained exact results in 12 cases.
At 256 raw bins and eight groups, score calls took 55–61% of the previous time,
with and without a separate missing bin. This does not measure whole-model
training speed. Reproducible evidence includes the
[timing record](https://github.com/captnmarkus/ctboost/blob/master/benchmarks/results/grouped_score_microbenchmark_0159.json),
[compatibility record](https://github.com/captnmarkus/ctboost/blob/master/benchmarks/results/release_compatibility_0158_0159.json),
[grouped calibration](https://github.com/captnmarkus/ctboost/blob/master/benchmarks/split_research/GROUPED_CALIBRATION_V1.md),
and [joint calibration](https://github.com/captnmarkus/ctboost/blob/master/benchmarks/results/multivariate_calibration_0159.json).

The existing [0.1.58 TabArena-Lite results](https://captnmarkus.github.io/ctboost/benchmarks/) remain unchanged.
They do not measure 0.1.59, and no new official leaderboard claim is made.
The quadratic feature test remains the default; the earlier grouped-test
promotion gate has not been reopened by this release.
