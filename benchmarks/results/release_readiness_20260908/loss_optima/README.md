# Huber and Quantile objective correctness - 8 September 2026

Two supported regression objectives violated simple optimum invariants in
CTBoost 0.1.60. Release candidate 0.1.61 contains narrow fixes. Neither change
modifies conditional feature selection, cut selection, RMSE, or training
defaults. These unit cases establish correctness; they do not establish better
TabArena scores or Elo.

**CPU and release dry-run validation passed.** The historical phase-2
Linux ARM64 cp311 wheel smoke run failed 12 float16 warning/callback policy
cases (312 passed, five skipped). Its
[failed job](https://github.com/captnmarkus/ctboost/actions/runs/34268643220/job/102204604662)
and [retained log](cross-platform-arm64-smoke-failure.log) remain available.
A separate CPU training/prediction reconstruction discrepancy also required
a fix. These failures are preserved, rather than erased by later passes.

Phase 3 uses source `5780b41` with those CPU portability fixes. Its
[installed-wheel loss recheck](phase3-loss-tests.log) passed **18/18** in 0.12s,
and its [complete local suite](phase3-full-suite.log) passed **1916 tests with
38 skips and zero failures**. The loss source and test bytes remain unchanged.
The [phase-3 receipt](phase3-loss-verification.json) and
[wheel identity](phase3-wheel-identity.json) identify the rebuilt artifact.
All **13 CPU CI jobs** have now passed, including GNU FMA, FMA-contract-off,
ARM64, MSVC-fast, and all standard Windows/macOS variants; the
[archived CPU report](../ci_final/cpu-34270193368.md) retains job and log evidence.
The [complete release dry run](../ci_final/README.md) also passed: 26 wheels
and one source distribution validated, nine JVM tests passed, R reported
`Status: OK`, and the rebuilt source archive passed four smoke tests. Both
publication jobs were skipped; this dry run did not publish a package or
release asset. The [final gate receipt](../ci_final/final-gates.json) binds
these results and the validated artifact to source `5780b41`.

| Objective | Reproduction in 0.1.60 | Restored invariant |
| --- | --- | --- |
| Huber | Fifteen targets of 7 and a zero-weight target of `1e38` initialized at `39443051`, instead of 7. Giving the outlier weight 0.1 produced the same wrong initializer, although the analytic optimum is approximately `7.006666667`. | Only positive-weight observations bracket the optimum. Bisection continues until its double-precision midpoint cannot advance, resolving even an extreme finite label range. |
| Quantile | For constant targets of 7 and `alpha=0.5`, one round moved predictions to approximately `6.95294094`, increasing mean pinball loss from zero to `0.023529529571533203`. | An exact prediction has zero subgradient and remains at the zero-loss constant optimum. |

Huber still validates every label and preserves its existing upper-edge choice
when the minimum is a flat interval. Quantile retains the same derivatives
above and below the target and the same Hessian surrogate. Choosing zero at
individual ties does not claim to solve every aggregate quantile tie case.

The [18 focused tests](../../../../tests/test_loss_optima.py) compare native
behavior with analytic Huber roots and an independently calculated pinball
loss. They include both extreme signs, zero and positive outlier weights,
ordinary weighted controls, three quantiles, and weighted/unweighted constant
targets. Against the installed public 0.1.60 wheel, **14 failed and four ordinary
Huber controls passed**; the original output is retained in the
[failure log](public-0.1.60-failures.log). All 18 focused cases then passed in the first candidate installed-wheel run.
That full run had **1833 passed, three failed, and 38 skipped**: its three
failures concern optional-input pipeline cases, so it does not establish
overall release readiness. The [full log](candidate-first-suite.log) is
preserved separately. After the pipeline fix, the phase-2 installed 0.1.61
wheel passed the [complete suite](candidate-final-suite.log): **1908 passed,
38 skipped, zero failures**, including all 18 loss cases again. The loss
source and tests were unchanged between these candidate runs.
[verification.json](verification.json) records the wheel identities and
retains the failed phase rather than replacing it.

The [model-free transcription](huber_initializer_mechanism.py.txt) and its
[numeric results](huber_initializer_mechanism.json) were saved before the source
fix. Their historical `native_fix_implemented=false` and
`native_candidate_verified=false` fields are intentionally preserved. They
explain why 100 bisections cannot resolve an interval spanning approximately
`1e38`; the native tests are the implementation check.

An independent read-only review found no numerical or mathematical blocker.
Finite float labels keep the double interval difference finite, midpoint
stagnation terminates the loop, and zero is a valid Quantile subgradient at an
exact prediction. Source and wheel identities are recorded in
[verification.json](verification.json); [manifest.json](manifest.json) hashes
the archive. The red run imports the installed package before invoking pytest,
so it does not accidentally exercise the checkout's Python package.
