# Multiclass score-test bounded-screen protocol

## Status and invariant

This protocol was fixed before generating the curated multiclass result ledger.
It is a bounded synthetic screen, not the full pre-registered 10,000-draw
calibration study and not evidence from TabArena or another external benchmark.

An independent pre-run review replaced the original K=3 mathematical tie with
a K=4 dyadic exact tie before any curated ledger completed. Uniform K=4 scores
use exactly representable `0.25` and `-0.75` values, so native row-order
accumulation retains an actual variance tie and the normal strict-`>` rule—not
a forced class index—selects coordinate zero. The same review required native
integer-bin ordering in the scalar reference and atomic progress-ledger writes.
No scenario outcome or completed frozen table informed these corrections.

Nothing in this experiment is connected to CTBoost's native learner.  Any
future candidate must retain the conditional-inference sequence:

1. test every eligible feature on a comparable p-value scale;
2. select the feature with the smallest p-value, subject to the existing
   global stopping rule; and
3. only then search that selected feature for a gain-maximizing cut.

The released/default multiclass structure target remains the single class
coordinate with the largest global weighted gradient variance.  This screen
compares that rule with a full-score reference; it does not authorize a default
change.

## Frozen configuration

- seed: `20260820`
- observations per draw: `480`
- main repetitions per scenario and bin profile: `400`
- permutation-oracle repetitions per null/profile cell: `80`
- score-vector permutations per oracle test: `499`
- class-permutation repetitions: `200`
- numeric-cut repetitions per power scenario: `200`
- midpoint recovery tolerance for single-boundary scenarios: `0.05` on the
  normalized raw-bin scale
- alpha: `0.05`
- raw selection-bin profiles: `2`, `8`, `32`, and `255`
- grouped profile: 8 contiguous equal-weight groups formed from the 255 raw
  numeric bins; grouping affects the feature test only
- missing values: a positive-weight missing level is a dedicated selection bin
- weights: the bounded screen is unweighted; exact integer frequency-weight
  replication and zero-weight behavior are correctness tests. Non-integer
  weights remain an explicitly working-asymptotic diagnostic, not an exact
  conditional permutation interpretation
- score covariance eigenvalue tolerance:
  `sqrt(machine_epsilon) * largest_eigenvalue`

The deterministic command is:

```bash
python -m benchmarks.split_research.multiclass_simulation \
  --repetitions 400 \
  --oracle-repetitions 80 \
  --permutations 499 \
  --class-permutation-repetitions 200 \
  --cut-repetitions 200 \
  --n-observations 480 \
  --seed 20260820 \
  --output benchmarks/split_research/results/multiclass_reference_seed_20260820.json
```

The output path is atomically replaced after each completed top-level section
and once more on completion. An interrupted file with
`completion.complete=false` is an audit/progress snapshot, not a resumable
simulation state; rerunning always starts from the frozen seed.

## Methods

For scores `s_i` and non-negative frequency weights `a_i`, let
`W = sum(a_i)`, `s_bar = sum(a_i s_i) / W`, and

`V = sum(a_i (s_i - s_bar)(s_i - s_bar)') / W`.

For active selection bin `b`, define `n_b = sum(i in b, a_i)`,
`S_b = sum(i in b, a_i s_i)`, and `D_b = S_b - n_b s_bar`.
The full statistic is

`Q = ((W - 1) / W) sum_b D_b' V+ D_b / n_b`,

where `V+` is the Moore-Penrose inverse after the frozen relative eigenvalue
cutoff.  If `r = rank(V)` and there are `B` active bins, the asymptotic p-value
uses `chi-square((B - 1) r)`.  The full K softmax coordinates are retained;
their sum-to-zero dependence is removed by the pseudoinverse rather than by
choosing a label-dependent reference class.

The legacy reference first chooses the lowest-index coordinate among those
with the largest global weighted variance, then applies CTBoost's scalar
nominal quadratic reference to that coordinate.

The separate numeric cut reference evaluates, after a feature has already
been selected,

`sum_k [G_Lk^2/(H_Lk + lambda) + G_Rk^2/(H_Rk + lambda)
        - G_Tk^2/(H_Tk + lambda)]`

over ordered cutpoints.  Missing rows are evaluated all-left and all-right.
This is a cut-stage diagnostic only; it is never used to select among
features.  No multivariate categorical ordering is proposed here.

## Correctness requirements

All must pass before interpreting the screen:

- binary/rank-one equivalence with the scalar quadratic form, allowing only
  the scalar production reference's documented diagonal epsilon;
- class-label permutation and orthonormal-contrast invariance of the full
  statistic;
- integer frequency weights equal explicit row replication;
- zero-weight rows are irrelevant;
- constant scores produce rank zero and p-value one;
- deliberately rank-deficient scores produce the expected rank and degrees
  of freedom;
- bin relabeling is irrelevant, while a positive-weight missing level adds one
  active bin;
- exact legacy ties choose coordinate zero;
- full numeric gain equals the sum of per-class scalar gains and is invariant
  to class-column permutation.

## Scenario matrix

The screen uses root/intercept softmax scores `p - one_hot(y)`:

- balanced K=3 null with fixed class margins independent of the feature;
- long-tail K=5 null with fixed class margins independent of the feature;
- balanced K=4 dyadic exact tie: classes 0 and 3 are independent, while classes
  1 and 2 swap across the midpoint; fixed equal margins and exactly
  representable scores make the native legacy selector choose class 0;
- imbalanced hidden K=3 signal: class 0 has probability 0.5 and is independent,
  while classes 1 and 2 swap;
- diffuse K=5 signal: the highest-variance class is independent and four
  lower-frequency classes carry rotating, individually modest quartile shifts;
- aligned K=3 control: the highest-variance class carries the midpoint shift;
- rare K=4 signal: the highest-variance classes are independent while a 5%
  class swaps mass with another class across the midpoint.

Every scenario is evaluated at raw 2/8/32/255 selection bins and grouped 8
from raw 255.  The permutation oracle covers both nulls at raw 8 and grouped 8.
All 24 K=4 label permutations are evaluated on paired tie-trap draws.  The
numeric gain screen uses raw 255 bins and reports midpoint recovery where a
single midpoint is the data-generating boundary.

## Bounded-screen decision rule

Proceed to the full pre-registered run only if all correctness requirements
pass and all of the following hold:

- for both nulls at raw 2/8/32 and grouped 8, the full asymptotic rejection
  rate lies in `[0.02, 0.08]`; raw 255 is reported but is not a gate because
  the large degrees of freedom are an explicit stress condition;
- at grouped 8, the full test exceeds legacy power by at least 0.10 on at
  least two of the tie-trap, imbalanced-hidden, and diffuse alternatives;
- at grouped 8, full-test aligned-control power is no more than 0.10 below
  legacy;
- full-test rejection is identical across the 24 paired class permutations,
  and the largest paired statistic difference is at most `1e-10`;
- no full-test permutation-oracle null cell lies outside `[0.01, 0.09]` and
  no absolute asymptotic-versus-permutation rejection difference above 0.04;
  these deliberately broad bounds reflect only 80 oracle draws;
- the full numeric gain is class-permutation invariant to `1e-10` and its
  midpoint-recovery rate is no worse than legacy on the aligned control.

Passing this gate warrants the larger synthetic calibration/power matrix and
an independently reviewed experimental CPU design.  It does not warrant a
native implementation, a GPU/distributed claim, or a default change by itself.

## Primary sources

- [Hothorn, Hornik, and Zeileis (2006), *Unbiased Recursive
  Partitioning*](https://www.zeileis.org/papers/Hothorn+Hornik+Zeileis-2006.pdf)
  defines vector-valued linear statistics, conditional covariance, the
  Moore-Penrose quadratic form, rank-based chi-square degrees of freedom, and
  the separation of variable and cut selection.
- [Strasser and Weber (1999), *On the Asymptotic Theory of Permutation
  Statistics*](https://research.wu.ac.at/ws/portalfiles/portal/19841038/document.pdf)
  supplies the conditional multivariate-normal expectation/covariance result.
- [Zeileis, Hothorn, and Hornik (2008), *Model-Based Recursive
  Partitioning*](https://www.zeileis.org/papers/Zeileis+Hothorn+Hornik-2008.pdf)
  motivates testing the complete vector of observation-wise objective scores.
- [libcoin's official vignette](https://stat.ethz.ch/CRAN/web/packages/libcoin/vignettes/libcoin.pdf)
  documents the Moore-Penrose quadratic statistic, rank degrees of freedom,
  and the relative eigenvalue tolerance used by the reference.
