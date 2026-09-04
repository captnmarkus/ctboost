# Benchmarks

## Current public-evidence status

CTBoost has two separate evidence tracks. They must not be combined into one
score:

1. a default-configuration TabArena-v0.1 Lite evaluation for the ordinary adapter; and
2. a pre-registered external panel for the opt-in grouped split statistic.

### CTBoost 0.1.56: TabArena-v0.1 Lite

CTBoost has a committed adapter for the official TabArena protocol. It delegates
folds, task metrics, bagging, tuning, ensembling, timing, and memory measurement to
TabArena rather than recreating them locally.

The latest [published result bundle](https://huggingface.co/datasets/Maiernator/ctboost-tabarena-lite-0.1.56)
covers all 51 TabArena-v0.1 Lite datasets with **CTBoost 0.1.56**, using one
default configuration and outer split `r0f0` only. It contains no measured HPO
configurations or TabArena-Full results. CTBoost 0.1.58 has no new measured
TabArena score; the vector-leaf and correctness changes do not inherit this
version's score.

| Scope | Result |
|---|---:|
| Lite Elo | 1166.7 (+52.1 / -67.5) |
| Win rate | 0.3957 |
| Position among default-configuration rows | 23 / 38 |
| Successful CTBoost outer splits | 51 / 51 |
| Imputed CTBoost tasks | 0 / 51 |
| Bagging | 8 children per task; 408 child fits |
| Kaggle CPU allocation | 4 |
| Kaggle memory limit | 28 GB |
| Per-fit limit | 3,600 seconds |

!!! note "Scope of the score"
    This is a local default-only Lite evaluation, not an official leaderboard
    entry. Elo depends on the dataset and method roster: this evaluation uses
    42 reference-method artifacts and 85 evaluated rows, including defaults,
    tuned configurations, ensembles, and systems. Kaggle's 4 CPUs and 28 GB
    differ from the canonical 8 CPUs and 32 GB, so its recorded timings are not
    directly comparable to official TabArena runtimes.

Provenance:

- Released `ctboost==0.1.56`, measured as `CTBoost_c1_default_BAG_L1`;
- 30 binary, 8 multiclass, and 13 regression datasets, with no missing or
  duplicate tasks in the published canonical tables;
- Six Kaggle CPU shards, Python 3.12.13, AutoGluon Tabular 1.6.2b20260821;
- TabArena integration/evaluation commit
  [`31026f7d758390994353eba79fbfa6747616f365`](https://github.com/captnmarkus/tabarena/commit/31026f7d758390994353eba79fbfa6747616f365),
  associated with [upstream PR #479](https://github.com/autogluon/tabarena/pull/479).

The [aggregate manifest](https://huggingface.co/datasets/Maiernator/ctboost-tabarena-lite-0.1.56/blob/main/validation/aggregate_manifest.json)
records task coverage, resource limits, artifact hashes, and version provenance.
The [evaluation table](https://huggingface.co/datasets/Maiernator/ctboost-tabarena-lite-0.1.56/blob/main/reports/leaderboard_lite.csv)
records the score and comparison roster. Despite its canonical filename,
`hpo_results.parquet` contains only the measured default configuration.

The earlier [0.1.55 three-dataset smoke record](https://github.com/captnmarkus/ctboost/blob/master/benchmarks/tabarena/smoke_0155_public_wheel.json)
(1054.0 provisional Elo) and [0.1.53 smoke record](https://github.com/captnmarkus/ctboost/blob/master/benchmarks/tabarena/smoke_fd187da.json)
remain historical integration checks. Their different task scope prevents
interpreting the newer Lite score as a version-to-version Elo improvement.

### Grouped split-statistic qualification

The final-source 0.1.55 grouped-statistic panel completed 294/294 isolated
fits and 42/42 exact implicit-control checks. It recorded nine wins, no ties,
and three losses across the twelve decision datasets, with an observed 5.63%
median primary-loss improvement. Its task-bootstrap 95% interval was -1.23%
to +13.64%, so the point estimate is not a claim of a uniformly positive
population effect.

The frozen promotion rule was conjunctive. Its median paired fit-time ratio
was 1.1708, above the pre-registered 1.15 ceiling, so grouped-8 did not advance.
The conditional grouped TabArena scout was therefore not applicable and was
not run. There is no grouped-8 TabArena Elo to report, and the quadratic test
remains the production default.

See the [research ledger](split-statistics-research.md) and the
[sanitized machine-readable panel evidence](https://github.com/captnmarkus/ctboost/blob/master/benchmarks/split_research/results/grouped_external_panel_v2_fb65b685.json).

## Reproduce the lite run

Follow the pinned environment instructions in
[`benchmarks/tabarena/README.md`](https://github.com/captnmarkus/ctboost/blob/master/benchmarks/tabarena/README.md),
then run:

```bash
ctboost-tabarena \
  --subset lite \
  --device cpu \
  --num-cpus 8 \
  --memory-limit-gb 24 \
  --time-limit 3600 \
  --results-dir benchmark-results/tabarena/raw \
  --output-dir benchmark-results/tabarena/report
```

The runner writes a manifest, split metrics, resource measurements, a leaderboard,
and plots. Local result directories are ignored by Git because raw TabArena artifacts
can be large.

## What a full result requires

TabArena-v0.1 currently covers 51 datasets and 816 outer splits. Each configuration
uses eight-fold bagging:

| Result | Configurations | Outer jobs | Child fits |
|---|---:|---:|---:|
| default | 1 | 816 | 6,528 |
| tuned + ensemble | default + frozen 200-config portfolio | 164,016 | 1,312,128 |

The full tuned run belongs on a resumable Linux cluster. Official publication also
requires upstream model registration, raw-artifact review, and TabArena maintainer
verification. Public CatBoost/XGBoost baselines should be reused; rerunning them would
multiply cost without improving CTBoost's evaluation.

## Benchmark policy

1. Search spaces are defined before reading TabArena test results.
2. Validation folds select configurations and ensemble weights; test folds do not.
3. Failed tasks remain failures and are reported.
4. Every published number includes code, package, hardware, resource, and task scope.
5. Smoke, default, tuned, and tuned-plus-ensemble results are labeled separately.
