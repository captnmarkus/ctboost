# Multiclass score-test bounded-screen protocol v2

## Freeze status

This v2 protocol was frozen after an independent audit of the completed v1
bounded screen and before generating any v2 result. It inherits every v1 data
generating mechanism, random seed, sample size, repetition count, statistic,
and decision threshold except for the two harness corrections documented
below. It remains reference-only and changes no CTBoost native learner, tree,
GPU, serialization, or public API code.

The immutable v1 ledger remains at
`results/multiclass_reference_seed_20260820.json`, with SHA-256
`fcdc3e3761fb08ddbdabd88d329c1bceac436e9c6d255fb302136a5c51c7f2ed`.
It is retained as negative evidence, not reinterpreted or overwritten. V1
robustly failed two preregistered aligned-control guards:

- grouped-8 aligned power delta: `-0.135`, below the unchanged `-0.10` floor;
- aligned numeric-cut recovery: full `0.53` versus legacy `0.56`, below the
  unchanged requirement that full be no worse than legacy.

Those failures remain decisive regardless of the audit corrections.

## Post-run audit and erratum

The audit found two research-harness defects and one floating-point tie issue:

1. The v1 protocol gates class-permuted **numeric gain** equality to `1e-10`.
   The implementation additionally gated selected-threshold equality, an
   outcome-independent but unfrozen requirement. V2 gates gain only.
   Threshold differences remain in the ledger as a diagnostic.
2. The v1 protocol requires identical full-test rejection decisions for every
   paired draw across all 24 K=4 label orders. The implementation compared
   aggregate rejection rates, which can conceal offsetting per-draw
   mismatches. V2 records mismatch counts against label order `0,1,2,3` for
   every order and requires the maximum paired mismatch count to be zero.
3. Mathematically tied full-class numeric gains could differ by class-column
   summation roundoff and select adjacent thresholds. V2 sums coordinate terms
   with `math.fsum`; gains within
   `64 * machine_epsilon * max(1, abs(candidate), abs(best))` are ties; and a
   tie selects the lowest threshold, then missing-right. The scalar legacy
   comparator retains its original `1e-15` tie rule.

These corrections were fixed from the protocol text and class-order invariant,
not from a preferred outcome. In particular, neither robust aligned guard is
weakened.

## Frozen v2 configuration

- seed: `20260820`
- observations per draw: `480`
- main repetitions per scenario/profile: `400`
- permutation-oracle repetitions per null/profile cell: `80`
- score-vector permutations per oracle test: `499`
- class-permutation repetitions: `200`
- numeric-cut repetitions per power scenario: `200`
- alpha: `0.05`
- midpoint recovery tolerance: `0.05`
- raw profiles: `2`, `8`, `32`, and `255`
- grouped profile: 8 contiguous equal-weight groups from raw 255 bins
- scenario matrix, missing policy, frequency-weight interpretation, covariance
  tolerance, and conditional-inference ordering: exactly as frozen in
  [v1](MULTICLASS_PROTOCOL.md)

The v2 output is deliberately distinct from v1:

```bash
python -m benchmarks.split_research.multiclass_simulation \
  --repetitions 400 \
  --oracle-repetitions 80 \
  --permutations 499 \
  --class-permutation-repetitions 200 \
  --cut-repetitions 200 \
  --n-observations 480 \
  --seed 20260820 \
  --output benchmarks/split_research/results/multiclass_reference_v2_seed_20260820.json
```

## Frozen decision rule

V2 passes only if every requirement below passes:

- for both nulls at raw 2/8/32 and grouped 8, full asymptotic rejection is in
  `[0.02, 0.08]`; raw 255 remains a reported stress profile;
- grouped-8 full power exceeds legacy by at least `0.10` on at least two of
  exact-tie, imbalanced-hidden, and diffuse alternatives;
- grouped-8 aligned-control full power is no more than `0.10` below legacy;
- the maximum paired full-test rejection mismatch count across all 24 label
  orders is zero and the maximum paired statistic difference is at most
  `1e-10`;
- every full-test permutation-oracle null cell has both rejection rates in
  `[0.01, 0.09]` and absolute asymptotic/permutation difference at most `0.04`;
- maximum class-permuted full numeric-gain difference is at most `1e-10`;
- aligned-control full midpoint recovery is no worse than legacy.

Selected-threshold equality is reported but is not a decision gate. Passing
would warrant only a larger independently reviewed synthetic study. Failing
any gate rejects this candidate for native integration or a default change.
