# Benchmarks

## Current public-evidence status

CTBoost has a committed adapter for the official TabArena protocol. It delegates
folds, task metrics, bagging, tuning, ensembling, timing, and memory measurement to
TabArena rather than recreating them locally.

The current completed result is a **protocol smoke test**, not a full leaderboard entry:

| Scope | Result |
|---|---:|
| TabArena-v0.1 lite, three datasets, default configuration | 1058.7 provisional Elo |
| Successful CTBoost outer splits | 3 / 3 |
| Imputed CTBoost splits | 0 |
| CPU allocation | 8 |
| Memory limit | 24 GiB |
| Per-fit limit | 3,600 seconds |

!!! warning "Do not compare this number with the full leaderboard"
    Elo is relative to its entrant and dataset pool. Three datasets and one split per
    dataset produce extremely wide uncertainty. CTBoost does not yet have an official
    full TabArena Elo.

### Smoke-task metrics

| Dataset | Task | Metric | Test error | Train time |
|---|---|---|---:|---:|
| anneal | multiclass | log loss | 0.041871 | 150.18 s |
| blood-transfusion-service-center | binary | ROC-AUC error | 0.314912 | 2.35 s |
| QSAR_fish_toxicity | regression | RMSE | 0.961281 | 3.94 s |

Provenance:

- CTBoost `fd187da60ec1844ef8d83c95b2d2ac6ccd839cd3`, clean tree;
- TabArena `50f8ab1bbc6e7f7e5dd9b19d8b643ac284ae9b3c`, clean tree;
- CTBoost 0.1.53, Python 3.12.11, CPU build, Windows AMD64.

The sanitized [machine-readable smoke record](https://github.com/captnmarkus/ctboost/blob/master/benchmarks/tabarena/smoke_fd187da.json)
contains the exact per-split metrics, timing, memory, package versions, and commit
identities without machine-local artifact paths.

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
| tuned + ensemble | default + 200 sampled | 164,016 | 1,312,128 |

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
