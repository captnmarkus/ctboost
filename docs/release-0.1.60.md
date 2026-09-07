# CTBoost 0.1.60

CTBoost 0.1.60 corrects categorical target statistics and early-stopping model
selection. Conditional-inference feature selection, cut-point optimization,
and existing learning defaults remain unchanged.

## Corrections

- **Fractional CTR priors:** positive `ctr_prior_strength` values below one no
  longer scale the target prior down at first occurrences or unseen categories.
  Ordered frequency CTRs also use the intended prior at their first row.
  The default strength of one and zero-prior cold-start behavior are unchanged.
- **Best-model selection at the iteration limit:** native training with early
  stopping now retains the best validation checkpoint when the tree budget
  expires before patience does. DART restores the recorded ensemble, including
  its historical tree weights.
- **DART warm starts:** when a supplied untrimmed ensemble cannot recover its
  historical best round, early stopping evaluates that full ensemble on the
  current validation data and selects between it and subsequent improvements.
  It no longer treats a tree prefix as the historical DART checkpoint. Native
  and callback training paths keep selected-model state consistent.
- **Benchmark helper loading:** the sealed adapter loader includes its existing
  learning-option dependency. Frozen protocols and prior benchmark results are
  preserved.

## Saved-model compatibility

Newly fitted or explicitly refitted preprocessing pipelines use **format 4**
to identify corrected CTR smoothing. Pipelines loaded from formats 1–3 retain
their historical smoothing for prediction and warm starts, and reserialize as
format 3. Categorical key encoding remains unchanged.

The Python raw-feature export reader accepts pipeline formats 3 and 4 with
categorical key encoding 2. Older runtimes reject format 4 rather than silently
using different preprocessing. Existing prepared-feature export boundaries
and scalar/vector predictor formats remain unchanged. Applications linking
the native headers or static core must rebuild; wheels include a matching
extension. See [deployment](https://captnmarkus.github.io/ctboost/guides/deployment/)
for DART resume semantics and preprocessing compatibility.

## Validation and evidence

Pre-release Windows/Python 3.12 implementation validation passed **1,345
distinct tests**, with 30 skips for unavailable optional capabilities. This
includes 33 CTR regression cases, 22 early-stopping cases, and compiled C++
export checks. Genuine regression, binary and multiclass models saved by the
public 0.1.59 wheel retain bitwise predictions after loading and resaving;
their raw JSON predictors retain identical predictions as well.

The local development studies used training-only inner validation splits.
CTR smoothing at strength 0.2 had 6 wins and 4 losses across 10 datasets, with
a +0.0818% median relative error improvement and a −12.1872% regression on
splice. All 20 strength-one control pairs had identical predictions. A separate
64-versus-256-bin comparison had 5 wins and 9 losses across 14 datasets, with
−0.5381% median improvement. These mixed results do not establish a broad score
gain; `max_bins=256` and other learning defaults remain unchanged.

The [audit report and original artifacts](https://github.com/captnmarkus/ctboost/tree/master/benchmarks/results/score_audit_20260907)
document the study binaries, local freezing before fits, reused baselines,
and every regression. The studies were not publicly preregistered or
independent holdouts and do not constitute HPO admission or leaderboard results.
Published TabArena scores still measure **0.1.58**, not this release; see
[benchmark status](https://captnmarkus.github.io/ctboost/benchmarks/).
