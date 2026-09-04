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

The [external ablation panel](EXTERNAL_PANEL.md) freezes task IDs, folds,
profiles, metrics, and promotion thresholds before native results are run.
If grouped-8 passes that panel, the
[grouped-statistic TabArena scout](TABARENA_GROUPED_SCOUT.md) fixes the later
three-task paired integration check before either treatment is evaluated.
Its runner is split into protocol, OpenML/provenance, isolated-worker, and
aggregation modules. Inspect its exact machine-readable schedule without
network access or a fit:

```bash
python -m benchmarks.split_research.external_panel metadata
python -m benchmarks.split_research.external_panel preflight
```

The result-producing `run` command is documented in the frozen panel ledger and
is intentionally not part of the synthetic reference experiment above.

The final-source CTBoost 0.1.55 panel completed all 294 jobs, with all 42
implicit-control checks exact. Its
[sanitized release evidence](results/grouped_external_panel_v2_fb65b685.json)
records nine wins, no ties, three losses, and a 5.63% median primary-loss
improvement, but the 1.1708 median paired fit-time ratio exceeded the frozen
1.15 ceiling. Grouped-8 therefore does not advance; the TabArena scout was not
triggered under the frozen protocol and therefore was not run. This result
supersedes the earlier 0.1.54 positive artifact only for 0.1.55 release
qualification; the earlier sealed artifact remains historical evidence.

The original [multiclass bounded-screen protocol](MULTICLASS_PROTOCOL.md)
separately froze a reference-only comparison of the current highest-variance
class coordinate against a full K-1 quadratic score test. Its completed v1
ledger is retained byte-for-byte at
`results/multiclass_reference_seed_20260820.json` (SHA-256
`fcdc3e3761fb08ddbdabd88d329c1bceac436e9c6d255fb302136a5c51c7f2ed`).
It is negative evidence: the candidate failed the aligned grouped-power and
aligned cut-recovery guards.

The post-run audit found an extra, unfrozen threshold-equality gate and an
aggregate-only implementation of a paired rejection-invariance requirement.
The corrected [v2 protocol and erratum](MULTICLASS_PROTOCOL_V2.md) freezes both
repairs, a scale-aware deterministic numeric-gain tie rule, the unchanged
aligned guards, and a distinct output before rerunning the same screen:

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

That runner also evaluates a class-permutation-invariant diagonal-Newton
numeric cut gain, but only after the feature is fixed. It does not use gain to
select among features and proposes no multiclass categorical ordering.

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
- a within-feature Bonferroni hybrid of the ordered and grouped tests;
- binary Newton, matched smoothed-WoE, and cross-fitted category scores;
- the current multiclass highest-variance-coordinate response rule;
- a full-score Moore-Penrose quadratic statistic with rank-based chi-square
  degrees of freedom; and
- a post-selection numeric gain summed over class-specific diagonal-Newton
  improvements.

Integer weights mean literal frequency counts. The permutation reference
rejects non-integer case weights because arbitrary real weights do not define
an exchangeable replicated sample.
