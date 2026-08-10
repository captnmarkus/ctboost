# Split-statistic reference experiments

This package is an executable research specification. It does **not** alter
CTBoost's native conditional-inference tree, defaults, TabArena adapter, or
model format.

Run the deterministic experiment from the repository root:

```bash
python -m benchmarks.split_research.simulation \
  --repetitions 250 \
  --global-repetitions 200 \
  --n-observations 320 \
  --permutations 199 \
  --effect-size 0.6 \
  --seed 20260810 \
  --output benchmarks/split_research/results/reference_seed_20260810.json
```

The committed JSON is a synthetic reference ledger, not benchmark evidence.
It contains the full configuration, null calibration, power results, global
Bonferroni results, category-ordering diagnostics, and reference runtimes.
See [the decision record](../../docs/split-statistics-research.md) for the
interpretation and literature links.

Reference implementations include:

- the current nominal `k - 1` quadratic test, with explicit
  `production_bin` and `ctree_omit` missing policies;
- a one-degree-of-freedom ordered weighted-midrank statistic;
- an edge-trimmed permutation maxstat whose shared response permutations are
  evaluated over every eligible cutpoint and whose p-value uses the inclusive
  plus-one correction;
- global Bonferroni stopping;
- selection-only compression of raw ordered bins into 8 or 16 contiguous,
  approximately equal-weight test groups;
- a within-feature Bonferroni hybrid of the ordered and grouped tests; and
- binary Newton, matched smoothed-WoE, and cross-fitted category scores.

Integer weights mean literal frequency counts. The permutation reference
rejects non-integer case weights because arbitrary real weights do not define
an exchangeable replicated sample.
