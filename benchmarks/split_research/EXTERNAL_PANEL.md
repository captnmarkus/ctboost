# Frozen external ablation panel

This protocol is frozen before the native grouped-statistic results are run.
It is deliberately disjoint from the TabArena-v0.1 dataset names and is not a
substitute for TabArena. Its purpose is to decide whether the synthetic
grouped-8 result is strong enough to justify a public, opt-in learner feature
and a later frozen TabArena scout.

## Protocol correction

The first execution attempt was stopped after 90 of 294 scheduled jobs. Its
27 multiclass jobs failed because the runner incorrectly supplied the binary
training evaluator name `Logloss` to the `MultiClass` objective. No promotion
summary was produced and none of those partial results are reused. Protocol
v2 freezes the corrected CTBoost training pair as
`objective="MultiClass", eval_metric="MultiClass"` and restarts in a fresh
source/result identity. This correction is applied identically to control and
candidate and does not change the pre-registered datasets, schedules,
hyperparameters, held-out metrics, or decision thresholds.

## Dataset tasks

The classification tasks come from OpenML-CC18 (suite 99); the regression
tasks come from OpenML-CTR23 (suite 353). Use repeat 0 and folds 0, 1, and 2
from each OpenML task's published split. Dataset identity is the task ID, not a
mutable name lookup.

| Task ID | Dataset | Problem | Intended stress |
|---:|---|---|---|
| 15 | breast-w | binary | small numeric |
| 29 | credit-approval | binary | mixed types and missing values |
| 9952 | phoneme | binary | low-dimensional numeric |
| 53 | vehicle | multiclass | small numeric multiclass |
| 2074 | satimage | multiclass | medium numeric multiclass |
| 2079 | eucalyptus | multiclass | small mixed/missing multiclass |
| 361234 | abalone | regression | mixed-type regression |
| 361236 | auction_verification | regression | small numeric regression |
| 361243 | geographical_origin_of_music | regression | high-dimensional small data |
| 361249 | white_wine | regression | medium numeric regression |
| 361258 | kin8nm | regression | smooth numeric regression |
| 361617 | energy_efficiency | regression | small numeric regression |

Tasks 7592 (adult) and 361255 (california_housing) are runtime/memory stress
checks only and do not affect the promotion decision.

## Compared treatments

Use the same released-source build, fold, preprocessing, initialization, and
hyperparameters for both treatments:

- control: `feature_test="quadratic"` (also verify implicit and explicit
  control are exactly identical on the first fold of every dataset);
- candidate: `feature_test="grouped", feature_test_bins=8`;
- `feature_test_adjustment="none"` for both. Bonferroni is a separate future
  ablation and must not be mixed into this decision.

The later gain/cut search keeps all 256 raw bins in both treatments.

Run these three fixed profiles with `max_bins=256`, `alpha=0.05`,
`boost_from_average=True`, no row/column subsampling, and early stopping after
50 rounds:

| Profile | Iterations | Learning rate | Growth |
|---|---:|---:|---|
| depthwise-default | 600 | 0.05 | `max_depth=6`, `lambda_l2=1` |
| depthwise-regularized | 800 | 0.03 | `max_depth=4`, `lambda_l2=3` |
| leafwise | 600 | 0.05 | `max_depth=8`, `grow_policy="LeafWise"`, `max_leaves=31`, `lambda_l2=1` |

Use deterministic seed `20260815 + task_id + 100 * fold`. Do not add a
profile, remove a dataset, or alter a threshold after inspecting results.

## Frozen execution details

These operational details are part of the ledger and are fixed before any
external-panel result is produced:

- Resolve data only by OpenML task ID. Use the task's published
  `repeat=0`, `fold in {0, 1, 2}`, `sample=0` indices; never reproduce a fold
  from a dataset name or a new splitter.
- Split each published outer-training fold once into 80% inner training and
  20% inner validation with that fit's deterministic seed. Classification is
  stratified and shuffled; regression is shuffled without stratification.
  The published outer-test fold is never supplied as an evaluation set and is
  used only for the final held-out prediction and metrics.
- Retain the OpenML pandas DataFrame, convert OpenML-declared categorical
  columns to pandas categorical dtype when needed, and preserve missing values.
  Do not impute, one-hot encode, or coerce the whole frame to a numeric array in
  the runner.
- Set `CTBOOST_HIST_THREADS=8`. Launch every fit in a fresh subprocess and
  record that process's peak RSS. Run the subprocesses sequentially so another
  panel fit cannot contaminate peak process memory.
- Alternate treatment order by the frozen dataset/fold/profile ordinal:
  control then candidate for even ordinals, candidate then control for odd
  ordinals. Resuming skips completed identities but does not reorder the
  remaining ledger.
- OpenML download/cache access, staged-data validation, fold slicing, and Pool
  construction happen before the fit timer. Fit time covers only the
  `ctboost.train(...)` call. Model serialization and outer-test prediction are
  also outside fit time.
- On fold 0 of every profile and every listed dataset, run an additional
  implicit-control fit that omits `feature_test`, `feature_test_bins`, and
  `feature_test_adjustment`. It must match the explicit quadratic control's raw
  outer-test prediction bytes, canonical tree JSON, and best iteration exactly.
  Whole serialized-model hashes are recorded but are not the equality gate,
  because explicit configuration metadata may legitimately differ.
- The task bootstrap uses 10,000 task-level resamples, seed `20260815`, and a
  two-sided 95% percentile interval for the median relative primary-loss
  improvement.
- Freeze CTBoost's training objective/evaluator pairs in the manifest:
  binary `Logloss`/`AUC`, multiclass `MultiClass`/`MultiClass`, and regression
  `RMSE`/`RMSE`. The multiclass held-out primary metric remains scikit-learn
  multiclass log loss; `MultiClass` is CTBoost's name for the training and
  early-stopping evaluator.

The runner records the actual OpenML data and published-index fingerprints,
the full fit configuration, the source working-tree fingerprint, the installed
CTBoost package fingerprint, and the native-extension SHA-256. A completed fit
is reusable only when the hash of all three identities (source, data, and full
configuration) is unchanged. Each fit result is added through an atomic JSON
replacement. Public manifests and summaries contain no absolute paths.

## Metrics and aggregation

- Binary: ROC-AUC is primary; log loss is a diagnostic.
- Multiclass: log loss is primary; accuracy is a diagnostic.
- Regression: RMSE divided by the training-fold target standard deviation is
  primary.
- Record fit time, best iteration, peak process RSS, serialized model bytes,
  failures, and non-finite predictions.

For each dataset and treatment, take the median primary loss across the three
folds for each profile, then the median relative change across the three
profiles. Convert ROC-AUC to loss as `1 - AUC` before calculating relative
changes. Report task-macro win/tie/loss, the median relative change, and a
fixed-seed task bootstrap interval. A tie is an absolute relative change below
0.1%.

Define relative improvement as `(control_loss - candidate_loss) / control_loss`,
so positive values favor grouped-8. Regression normalization uses the
published outer-training target standard deviation with `ddof=0`. The reported
fit-time ratio is the median of candidate/control ratios across the 108 matched
fold/profile pairs in the 12 decision datasets. The two stress tasks and their
control checks are reported separately and do not enter any promotion gate.

## Pre-registered decision

Grouped-8 advances to a frozen TabArena scout only if all of these hold:

1. every fit completes with finite, correctly shaped predictions;
2. it wins at least 7 of the 12 datasets after task-level aggregation;
3. the task-macro median relative primary loss improves by at least 0.25%;
4. no more than three datasets worsen by more than 1%; and
5. median fit-time ratio is at most 1.15.

If these gates fail, grouped-8 remains an experimental user option or is
rejected. The panel must not be revised using TabArena outcomes. A later full
TabArena result is reported regardless of whether it supports the hypothesis.

## Runner

The implementation is intentionally separate from the learner and can be
inspected without OpenML or a native build:

```bash
python -m benchmarks.split_research.external_panel metadata
python -m benchmarks.split_research.external_panel preflight
```

`metadata` is dependency-free apart from CTBoost's normal NumPy requirement.
`preflight` imports installed dependencies and probes the grouped native API,
but does not request an OpenML task or fit a model. A result-producing run is a
separate explicit command and should be invoked only after reviewing this
ledger and its generated metadata:

```bash
python -m pip install pandas scikit-learn openml
python -m benchmarks.split_research.external_panel run \
  --results-dir benchmark-results/split_research/external-panel \
  --cache-dir benchmark-results/split_research/openml-cache
```

The default directories are ignored by Git. Re-running the command resumes the
atomic full-identity ledger; `--rerun-failures` explicitly retries recorded
failures. `summarize` regenerates the sanitized public summary from existing
artifacts without contacting OpenML.
