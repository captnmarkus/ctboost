# Grouped score: unchanged-math diagnostics for 0.1.59

This check evaluates the native grouped statistic at a fixed node. It does not
fit trees, search parameters, or reopen the external panel's promotion gates.
The quadratic default, group assignment, missing-value group, feature ranking,
stopping correction, and subsequent raw-bin cut search remain unchanged.

The optimization keeps grouped gradients and weights in fixed-size storage.
It removes grouped Hessian accumulation, which the feature score does not use,
and avoids heap allocation in the score path, including its rare dense fallback.
The existing grouping API and its diagnostic arrays are preserved.

## Frozen protocol and reproduction

`grouped_calibration.py` fixes seed 20260907, eight non-missing groups,
alpha 0.05, and 2,000 repetitions for each of ten cases before evaluation.
Its protocol SHA-256 is
`a6e716c4a0bd44fe866467a688dab0c5b7c193c102720a7c2e11c650c62850d6`.
The machine-readable result records that protocol, the harness and native
extension hashes, and the build and environment versions.

From the repository root, using the interpreter of an isolated installed build:

```bash
python -I benchmarks/split_research/grouped_calibration.py --protocol-only
python -I benchmarks/split_research/grouped_calibration.py \
  --output benchmarks/split_research/results/grouped_calibration_v1_seed_20260907.json
```

Scores and feature bins are generated independently. Sparse cases put 90% of
observations into the first bin; missing cases independently mark 30% missing.
The integer-weight case starts with 320 independent binary-score observations
and losslessly collapses equal `(feature bin, score)` pairs into frequency
counts. It does not duplicate one random score to invent independent samples.

The three fractional-weight cases reuse exactly the same observations and
independent lognormal weight draws. Weights are normalized to mean one and
then multiplied by 0.1, 1, or 10. These are sensitivity checks for importance
weights, not exchangeable frequency samples.

## Observed fixed-node results

Every one of the 20,000 comparisons produced exactly the same p-value,
chi-square statistic, and degrees of freedom through the diagnostic and
optimized paths. All p-values were finite and in `[0, 1]`.

| Scenario | Rejections / 2,000 | Rejection rate | Marginal 95% Wilson interval |
|---|---:|---:|---:|
| 24 rows, 255 possible bins | 24 | 1.20% | 0.81%–1.78% |
| 96 rows, 255 possible bins | 68 | 3.40% | 2.69%–4.29% |
| 320 rows, 255 possible bins | 90 | 4.50% | 3.68%–5.50% |
| 96 rows, sparse occupancy | 106 | 5.30% | 4.40%–6.37% |
| 96 rows, missing bin first | 76 | 3.80% | 3.05%–4.73% |
| 96 rows, missing bin last | 85 | 4.25% | 3.45%–5.23% |
| Literal integer frequencies, 320 observations | 88 | 4.40% | 3.59%–5.39% |
| Fractional weights, mean 1 | 1,099 | 54.95% | 52.76%–57.12% |
| Same fractional weights, mean 0.1 | 0 | 0.00% | 0.00%–0.19% |
| Same fractional weights, mean 10 | 1,998 | 99.90% | 99.64%–99.97% |

The small-sample checks show conservatism; the larger, sparse, missing, and
literal-frequency cases have the displayed marginal rates. These finite
experiments do not establish exact null calibration for all distributions,
feature-family error control, or adaptive-tree family-wise error control.

The fractional-weight results expose a pre-existing limitation: this statistic
treats the sum of weights as a frequency count. Arbitrary importance weights
and their rescaling therefore do not inherit frequency-null calibration.
The optimization reproduces that behavior exactly. It neither normalizes user
weights silently nor introduces a new claim about their p-values.

See the [JSON ledger](results/grouped_calibration_v1_seed_20260907.json) for
degrees of freedom, median p-values, and complete provenance. The reported
effective weight sample size is descriptive and is not used to adjust a test.

## Compatibility, profiling, and timing scope

`tests/test_grouped_calibration.py` compares the two score paths over 2, 8, 16,
and 64 groups, both missing-bin positions, sparse and fractional weights,
degenerate nodes, and an extreme-weight dense-fallback case. Its profiler test
forces the minimum-p feature to have no feasible cut under a monotonicity
constraint, verifying that the minimum-p feature and selected feature remain
distinguishable and that Bonferroni stopping uses the former.

`benchmarks/release_compatibility.py` compares an isolated published 0.1.58 wheel
against 0.1.59 on 32 deterministic small training cases: regression, binary,
scalar multiclass, and vector multiclass; quadratic and grouped tests; and
ordinary, weighted/missing/categorical, leafwise, and DART variants. Predictions,
leaf indices, loss histories, and all existing serialized tree fields matched
exactly. Additional serialization keys are excluded from that comparison.
The [portable compatibility record](../results/release_compatibility_0158_0159.json)
contains each case's parameters and common output hashes, together with the
two native extension hashes and source-report provenance.

The profiler now measures feature-statistic evaluation and cut search in their
actual phases, including the ranked candidate path. Its log distinguishes
`minimum_p_feature`/`minimum_p_value` from the selected `feature`/`p_value`,
and includes the selected test's `degrees_of_freedom`.

`benchmarks/grouped_score_microbenchmark.cpp` measures the same native score
before and after the optimization, with 32/128/256 raw bins, 8/64 groups, and
with/without a dedicated missing bin. It checks exact output equality before
timing and alternates execution order over seven rounds. This isolates node
scoring overhead; it does not measure histogram construction or the extra
nodes and cuts that a more powerful statistic may cause. It cannot establish
that the previous external panel's fit-time gate now passes.

On the local Ryzen 7 5800X3D/MSVC `/O2` run, optimized score calls took
54–57% of the previous time for 256 raw bins and eight groups, without/with
a separate missing bin. All 12 case comparisons were exact. The
[timing record](../results/grouped_score_microbenchmark_0159.json) includes
each measured case and source/executable hashes.

The 0.1.55 panel remains the latest registered grouped-vs-default predictive
evaluation. Grouped-8 remains opt-in; no new TabArena score or default promotion
is claimed by these checks.
