# CTBoost on TabArena

This adapter evaluates CTBoost on the public
[TabArena](https://github.com/autogluon/tabarena) folds and compares its output
with the benchmark's cached CatBoost, XGBoost, and other baselines. It does not
replace TabArena's splitting, metrics, timing, bagging, tuning, or leaderboard
logic.

The integration is audited against TabArena commit
`50f8ab1bbc6e7f7e5dd9b19d8b643ac284ae9b3c`. TabArena currently requires Python
3.11–3.13 and recommends `uv`.

The default CTBoost run and its public TabArena/OpenML data and cached baselines
do not require a Hugging Face token. A token is only relevant if you separately
enable a gated foundation-model configuration or upload benchmark artifacts.

## Environment

From a fresh checkout of CTBoost, follow TabArena's upstream benchmark install
path: clone the complete repository, pin the audited revision, and install the
editable `packages/tabarena[benchmark]` package in an isolated Python 3.12
environment. The explicit `--python` arguments avoid accidentally installing
into another active environment:

```bash
git clone https://github.com/autogluon/tabarena.git ../tabarena
git -C ../tabarena checkout --detach 50f8ab1bbc6e7f7e5dd9b19d8b643ac284ae9b3c
uv venv --seed --python 3.12 .venv-tabarena
uv pip install --python .venv-tabarena/bin/python --prerelease=allow \
  -e "../tabarena/packages/tabarena[benchmark]"
uv pip install --python .venv-tabarena/bin/python .
```

On Windows, replace `.venv-tabarena/bin/python` with
`.venv-tabarena/Scripts/python.exe`.

To deliberately evaluate against the latest TabArena `main`, omit the pinned
checkout (or update the clone before installing). That is useful for integration
testing but is not a reproducible benchmark target and may require adapter
changes. The run manifest records the actual TabArena checkout commit, dirty
status, and dirty-tree fingerprint when installed from this editable clone.

Run the three-dataset smoke benchmark with the default CTBoost configuration:

```bash
.venv-tabarena/bin/ctboost-tabarena --subset lite
```

TabArena tracks wall time plus peak CPU and GPU memory around every fit and
inference call. CTBoost consumes AutoGluon's remaining finite positive
`--time-limit` from adapter entry, including data normalization and model setup.
It declines to start boosting when setup leaves 40% or less of that budget, and
a training callback stops between trees when the deadline cannot safely fit two
average iterations. A single unusually slow tree can still cross this soft
deadline. Make the hardware/resource contract explicit when producing a report:

```bash
.venv-tabarena/bin/ctboost-tabarena --subset lite --device cpu \
  --num-cpus 8 --memory-limit-gb 32 --time-limit 3600
```

If `--time-limit` is omitted, the runner derives it from the installed
`TabArenaV0pt1ExperimentBundle.DEFAULT_TIME_LIMIT` (3,600 seconds at the
audited revision). The manifest records both the requested CLI value and the
effective per-fit value.

On a CUDA worker with the matching GPU wheel, `--device gpu` publishes a
separate `CTBoostGPU` method. `--device both` runs CPU and GPU methods in the
same experiment bundle. Add `--rerun-competitors` to execute CatBoost and
XGBoost on those exact folds and resource limits too; without it, `compare`
uses TabArena's public cached competitor baselines. GPU modes fail fast when
the installed CTBoost build is CPU-only.

Sample 20 additional author-defined configurations:

```bash
.venv-tabarena/bin/ctboost-tabarena --subset lite --n-configs 20
```

Run the full benchmark only on appropriately provisioned infrastructure:

```bash
.venv-tabarena/bin/ctboost-tabarena --subset all --n-configs 200 --ray
```

On Windows, the executable is `.venv-tabarena/Scripts/ctboost-tabarena.exe`.

TabArena reports that raw artifacts can require roughly 100 GB per method. Keep
`benchmark-results/` and `TABARENA_CACHE` outside source control and place them
on storage sized for the chosen run.

Every completed run writes `resources_per_split.csv` and JSON beside the
leaderboard. They contain the official metric error, train/inference time,
incremental and absolute peak CPU/GPU memory, requested resources, disk size,
and raw artifact path for every method/task/split. Do not average raw metric
errors across unlike task metrics; use TabArena's generated leaderboard for
cross-dataset rank and normalized-error comparisons.

## Fairness contract

- TabArena supplies the official train/validation/test folds and metrics.
- Validation data is used only for early stopping.
- AutoGluon injects each fold/configuration seed through CTBoost's
  `random_seed`; an explicitly configured `random_seed` remains authoritative.
- When no `eval_metric` is configured, supported AutoGluon stopping metrics are
  translated to CTBoost (`roc_auc` to `AUC`, classification `log_loss` to
  `Logloss`/`MultiClass`, and regression `rmse` to `RMSE`).
- Categorical columns remain categorical and are processed by CTBoost's fitted
  feature pipeline; they are not ordinal-encoded by this adapter.
- Default and tuned results are reported separately.
- Failed or timed-out folds remain failures; the runner does not impute them.
- The exact CTBoost and TabArena commits (including dirty status), hardware
  information, training time, inference time, and peak memory must accompany
  published results.

Official leaderboard submission requires processing and hosting the generated
artifacts according to TabArena's current maintainer instructions. Do not submit
the three-dataset smoke result as a full benchmark result.
