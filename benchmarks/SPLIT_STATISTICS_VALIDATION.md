# Grouped split-statistics validation ledger

This ledger records the staged native gate for the opt-in grouped feature test.
It does not authorize adding grouped settings to the frozen TabArena portfolio;
that remains an external validation decision.

The behavioral list below is the historical gate recorded when grouped testing
was CPU-only. Current development now routes GPU quadratic, grouped, and
Bonferroni modes through the same host candidate selector while retaining GPU
histogram construction and partitioning; see `docs/guides/split-statistics.md`.
The historical fail-closed item is retained as evidence of the earlier boundary,
not as a description of the current development tree.

## Legacy-default compatibility

Baseline: `origin/master` at
`2dca18d905395eb7fc3b32aa73b3b908133f1886`. Candidate: the native grouped
feature-test worktree based on that commit. Both probes used the same CPython
3.10 / NumPy environment and `benchmarks/default_compatibility_probe.py`.

| Case | Raw prediction SHA-256 | Canonical tree JSON SHA-256 | Result |
|---|---|---|---|
| Numeric | `fe005097d85d035c52bccf396a505928790b156a6e7adf032d2dd655dd667b77` | `a13edaaa18eca6119c3e9f1f4a98b7606afbec5541190c916c0e62a35ec46f00` | Exact match |
| Categorical + missing (`nan_mode=Max`) | `454d1bb75c531ac816f2ba0bbff1a07f0eb7fc78244f25c750ecc820b0ca209b` | `4de431ccabf909c875d4adb3613d9488c7fbd075f3f468eb403b910092f36f36` | Exact match |
| Constraints + LeafWise + penalties/random strength | `5391b66bfae20a924893bee3249436eb2361f96bc7ea6767235c1bf98f204b6c` | `4264f875a0094cdcdce27a550d80aefaf9ebdda37eb16cff1b642b1e1c3858bf` | Exact match |

Whole-model hashes are intentionally not compared because the candidate adds
backward-compatible configuration keys. The gate is raw predictions and tree
structure/value JSON.

## Native/reference parity

The focused tests compare native aggregation and the resulting degrees of
freedom, quadratic statistic, and p-value against an independent Python
implementation of the audited atomic-level midpoint rule:

```text
floor(G * (cumulative_weight_before + 0.5 * level_weight) / total_weight)
```

Coverage includes highly unequal frequency weights, gap compaction caused by a
heavy atomic level, no missing level, missing-Min, and missing-Max. All cases
match within numerical solver tolerance; grouped weight sums match exactly.

## Behavioral gates

- smooth, step, and U-shaped signal detection with 255 raw bins;
- high- versus low-cardinality null behavior at the same seven degrees of
  freedom;
- raw-bin cut selection beyond the eight selection-only groups;
- weighted nodes and categorical-feature legacy parity;
- Bonferroni root stopping, both LeafWise child candidates, constrained ranked
  search, and random-strength ranked search;
- raw and adjusted stopping p-values in profiler output;
- persistence, legacy-state defaults, sklearn cloning, CLI, init-model warm
  start, and exact snapshot resume including mixed-case aliases;
- CPU distributed parity, including a numeric feature whose NaNs occur on only
  one shard for both missing-Min and missing-Max;
- explicit fail-closed behavior for grouped/adjusted GPU training at that stage.

## Dispatch microbenchmark

Command:

```bash
python benchmarks/split_statistics_microbenchmark.py \
  --rows 20000 --features 24 --iterations 3 --repeats 7
```

Median native fit times in the clean isolated CPython 3.10 Release validation
build:

| Branch | Median seconds | Ratio to implicit default |
|---|---:|---:|
| Implicit legacy default | 0.022317 | 1.0000 |
| Explicit `quadratic/8/none` | 0.022262 | 0.9975 |
| Opt-in `grouped/8/none` | 0.023365 | 1.0470 |

The explicit default produced identical predictions and trees. The benchmark
has a configurable default-path regression threshold and randomized case order;
it is a local microbenchmark, not a public accuracy or throughput claim.
