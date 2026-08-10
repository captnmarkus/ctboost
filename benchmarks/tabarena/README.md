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

On a CUDA worker, CTBoost 0.1.54's ordinary unified PyPI wheel publishes a
separate `CTBoostGPU` method with `--device gpu`; install it with
`python -m pip install --upgrade --only-binary=:all: "ctboost>=0.1.54"`.
`--device both` runs CPU
and GPU methods in the same experiment bundle. Add `--rerun-competitors` to
execute CatBoost and XGBoost on those exact folds and resource limits too;
without it, `compare` uses TabArena's public cached competitor baselines. GPU
modes fail fast when the installed CTBoost build is CPU-only. The deprecated
`ctboost-install-gpu` helper is not needed for 0.1.54 or later.

Sample 20 additional author-defined configurations:

```bash
.venv-tabarena/bin/ctboost-tabarena --subset lite --n-configs 20
```

The HPO entry is a frozen, progressively ordered 200-configuration portfolio.
TabArena's official `CustomAGConfigGenerator` contract accepts a deterministic
configuration callable, so the adapter uses a fixed Latin-hypercube design
instead of depending on ConfigSpace's unconstrained random sampler. Smaller
`--n-configs` runs are prefixes of the same portfolio. Numeric ranges are
stratified, conditional knobs are absent when inactive, and the full portfolio
balances 100 DepthWise/100 LeafWise and 150 ordered-CTR/50 unordered-CTR
configurations.

Learning-rate configurations receive a bounded 400-1,600 tree cap and 30-80
round validation patience; early stopping chooses the actual retained tree
count. A subset of ordered-CTR configurations requests two or four categorical
pairs. At fit time the adapter selects no more than that number from the 16
lowest-cardinality non-constant categorical columns, rejects pairs whose
cardinality product exceeds 4,096, and uses only the training fold to make the
choice. It never enables CTBoost's unbounded all-pairs switch.

These ranges and rules were generated from the fixed seed `1234` and frozen from
implementation/resource constraints plus non-TabArena development evidence.
They do not inspect TabArena dataset metadata, validation outcomes, or test
metrics. Do not revise them after inspecting full test metrics; start a newly
versioned portfolio instead. The adaptive tree caps do not change TabArena's
3,600-second per-fit limit or its deadline callback.

The clean default-only smoke behind the provisional `1058.7` three-dataset Elo is
recorded in [`smoke_fd187da.json`](smoke_fd187da.json). The file contains exact
per-split metrics, timing, memory, versions, and commits without machine-local
artifact paths. It is explicitly not a TabArena-Full or official leaderboard result;
an Elo at or above 1,300 remains an unproven target until the frozen tuned portfolio
completes the full protocol.

Run the full benchmark only on appropriately provisioned infrastructure:

```bash
.venv-tabarena/bin/ctboost-tabarena --subset all --n-configs 200 --ray
```

For a distributed or interruptible run, execute disjoint outer-job shards into
the same results directory, then evaluate once all shards have completed. Existing
`results.pkl` files are reused, so rerunning a shard resumes rather than refits it:

```bash
# Submit indices 0..63 as separate workers.
.venv-tabarena/bin/ctboost-tabarena --stage run --subset all --n-configs 200 \
  --shard-count 64 --shard-index 0 --job-batch-size 32 --ray

# Run once after every worker succeeds.
.venv-tabarena/bin/ctboost-tabarena --stage evaluate --subset all \
  --n-configs 200 --shard-count 64 --ray
```

The runner builds one dataset's lightweight job objects at a time and retains at
most `--job-batch-size` raw result objects in the driver. Evaluation uses
TabArena's task-by-task `EndToEnd.from_path_raw` path, filters unrelated configs,
and requires exactly one artifact for every expected `(config, task, split)` key.
`--allow-incomplete` is available only for diagnostics; such a report is not
leaderboard-valid. TabArena's cache key does not include the package commit or
hyperparameter payload, so resume a directory only with the same frozen source and
search space. Start with a fresh `--results-dir` after either changes.

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
- AutoGluon injects disjoint fold/configuration seeds through CTBoost's
  `random_seed`; the search portfolio never hard-codes a model seed, while an
  explicitly configured `random_seed` remains authoritative.
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

The exact upstream file list, validation gates, scale calculation, and PR/run-request
templates are in [`UPSTREAM_SUBMISSION.md`](UPSTREAM_SUBMISSION.md).
