# Benchmarks

## Official TabArena leaderboard

The 17 November 2026 snapshot includes 110 rows in the models-only view with
all datasets, task types and splits and with imputed entries included. All three
CTBoost rows are marked verified and have 0% imputation.

| Configuration | CTBoost Elo (95% interval) | CatBoost Elo | LightGBM Elo | XGBoost Elo |
|---|---:|---:|---:|---:|
| Default | 1177 (1111 to 1241) | 1363 | 1183 | 1208 |
| Tuned | 1298 (1246 to 1337) | 1391 | 1368 | 1336 |
| Tuned + ensembled | 1327 (1279 to 1367) | 1402 | 1411 | 1360 |

CTBoost ranks fourth among these four libraries in each configuration by Elo
point estimate. The six-point gap to default LightGBM is small relative to the
reported uncertainty. The Space table omits package versions. Its snapshot
predates 0.1.63, and the upstream CTBoost integration pins 0.1.61.
The older author-run Lite study below used a different task and competitor
roster. [Versioned leaderboard table](https://huggingface.co/spaces/TabArena/leaderboard/blob/8ce4c3037c2b8f9385f6b35375776f610af2e9dd/data/entrants_models/imputation_yes/splits_all/tasks_all/datasets_all/website_leaderboard.csv) and
[upstream version pin](https://github.com/autogluon/tabarena/blob/cc507b84d1f32d2ca5b5704326bf4f086ee6519a/packages/tabarena/src/tabarena/models/ctboost/info.py)

## CTBoost 0.1.64 learning options on TabArena development splits

On the 13 regression tasks, CTBoost's expectile objective reduced asymmetric
squared error on 12 tasks when underprediction cost four times as much as
overprediction. Across two training seeds, the median reduction was **18.75%**.
Ordinary RMSE rose on all 13 tasks, with a median increase of **12.18%**.
On the broader 51-task comparison, shallower trees increased median task error
by **5.14%**, and fitting each tree on all rows increased it by **2.08%**.
CTBoost retains the existing TabArena settings. [Study and per-task results](https://github.com/captnmarkus/ctboost/blob/612328f74cd792ace7f7bdd96ef50f583e903f52/benchmarks/tabarena/BLACKBOOST_OPTIONS.md)

## CTBoost 0.1.63 development comparisons

In a separate four-library comparison on the 51 cached development splits,
the public 0.1.63 wheel ranked **third of four** by mean task rank. CatBoost
averaged 1.86, LightGBM 2.41, CTBoost 2.73 and XGBoost 3.00 (lower is better).
CTBoost won nine tasks outright. It placed second on binary classification,
third on multiclass classification and fourth on regression. All 204 fits
completed using one seed and a maximum of 200 boosting rounds. Replaying
CTBoost after the pandas schema fixes reproduced the published wheel's
predictions on all 51 tasks. [Four-model protocol and per-task results](https://github.com/captnmarkus/ctboost/blob/702058fe652ae31312aba9189c8530ba47ab4027/benchmarks/tabarena/FOUR_MODEL_DEVELOPMENT.md)

In the earlier comparison against CTBoost 0.1.62 on these tasks, 0.1.63 improved
three, produced identical predictions on 46 and worsened two. All 102 fits
completed. Median relative error reduction across tasks was **0.00%** and the
unweighted mean was **+0.17%**. Each fit used a 200-round cap, 25 rounds of
validation patience and a separate development split for scoring.

The quantile-border correction restores a numeric split when rare values fell
in the same bin as a column's minimum. A screen of the training features from
51 cached TabArena tasks found 110 affected raw numeric columns on six tasks.
All six were evaluated at two seeds. Three improved, one tied and two worsened.
The median relative error reduction was 0.25% after averaging the two errors for
each model within each task. The largest improvement was on coil insurance
(6.81%), and the largest regression was on Bioresponse (1.13%). The study used
held-out development rows and capped single fits. [Paired results and reproduction](https://github.com/captnmarkus/ctboost/blob/v0.1.63/benchmarks/tabarena/QUANTILE_TIES_SCOUT.md)

An independent 36-fit scout compared log loss with ROC-AUC for binary early
stopping. Across nine small tasks, selecting by ROC-AUC increased mean
development error (1 minus ROC-AUC) from 0.2063 to 0.2160. The adapter retains
its existing stopping policy. [Results by task](https://github.com/captnmarkus/ctboost/blob/v0.1.63/benchmarks/tabarena/BINARY_STOPPING_SCOUT.md)

## CTBoost 0.1.62: local evidence and scope

[Release 0.1.62](release-0.1.62.md) reduces prediction preparation overhead,
improves categorical target-statistic precision, and writes compact JSON.
The conditional-inference feature-test-before-cut design and tree-growth and
hyperparameter defaults remain unchanged. The [October follow-up archive](https://github.com/captnmarkus/ctboost/tree/master/benchmarks/results/improvements_20261016_followup) records the
production changes, tests, fixed-model latency protocols and accuracy studies.

The accuracy study contains twelve tasks: four binary, four regression and
four multiclass. It compares CTBoost, XGBoost and LightGBM with explicit fitting,
selection and replication budgets. Development data were reused across
successive hypotheses; their scores are development evidence, not unbiased
estimates or an official TabArena ranking. Equal model-fit counts do not imply
equal compute or model capacity. The immutable accuracy runtime is recorded
separately from the final inference runtime; the CTR precision fix does not
retroactively change the training of those frozen models.

The final current-data round-count experiment completed 540 fold fits and 36
selected full-data fits. CTBoost achieved a median relative error reduction of
**+0.2075% versus XGBoost** (7/12 wins) and **-0.8217% versus LightGBM** (5/12
wins), with no ties. The combined development target was not met. The original
rules therefore kept the second-seed and official-test stages closed; the
separate confirmation role remained sealed. The archive preserves the earlier
resource failure and externally interrupted attempt, plus the unchanged-plan
recovery that completed this experiment.

Serving-cost comparisons use one complete, prospectively selected model policy
and balanced library orders. They distinguish raw-input preparation from
normalized-input prediction, and require exact prediction checks. Their task
scope and differing model capacities must accompany any reported speedup;
latency alone does not establish accuracy superiority.

The completed three-pass diagnostic measured batches of 1, 64 and 1,000 rows
on one CPU/thread. Across the 36 task/batch cases, CTBoost's geometric mean
raw-input speed ratio was **2.247x versus XGBoost** and **1.915x versus LightGBM**;
CTBoost was faster in 36/36 and 35/36 cases respectively. Full-development and
every batch prediction matched the frozen references exactly. These selected
models have the accuracy gap described above. See the
[complete final timing report](https://github.com/captnmarkus/ctboost/tree/master/benchmarks/results/improvements_20261016_followup/inference/final_execution_20261022)
for every case, normalized-input costs and source/runtime identities.

## Historical public evaluations

The [0.1.61 accuracy archive](https://github.com/captnmarkus/ctboost/tree/8cc75aa/benchmarks/results/inference_accuracy_20260908/accuracy)
retains earlier local studies and failed qualification gates. Those studies
did not establish an Elo increase or justify changing learning defaults.

The status recorded on **9 September 2026** was that a fresh Full HPO25
evaluation of public **0.1.61** was prepared but had **not started**. Its design
covered the default plus 25 frozen HPO configurations across 816 outer splits,
with eight-fold bagging. The 0.1.60 run was retired. This is a historical status
record, not a completed 0.1.62 Full evaluation or an official leaderboard entry.

On 8 September 2026, Lennart Purucker separately
[reported TabArena-Lite results with 25 configurations and offered a Full run with 200](https://github.com/autogluon/tabarena/pull/479#issuecomment-5588951890).
That report predates 0.1.61 and is not evidence for 0.1.62.

The [0.1.60 CTR and numeric-bin audit](https://github.com/captnmarkus/ctboost/tree/master/benchmarks/results/score_audit_20260907)
and earlier grouped-statistic panels remain historical development evidence,
with their own protocols and limitations.

### CTBoost 0.1.58: TabArena-v0.1 Lite with 25 HPO configurations

CTBoost has a committed adapter for the official TabArena protocol. It delegates
folds, task metrics, bagging, tuning, ensembling, timing, and memory measurement to
TabArena rather than recreating them locally.

The historical [downloadable author-run result bundle](https://huggingface.co/datasets/Maiernator/ctboost-tabarena-lite-hpo25-0.1.58)
covers all 51 TabArena-v0.1 Lite datasets with **CTBoost 0.1.58**, using the
manual default plus the first 25 configurations of the frozen 200-configuration
portfolio. Each configuration uses outer split `r0f0` and eight-fold bagging.
All 156 shards completed: 1,326 parent results and 10,608 child fits, with zero
imputed CTBoost tasks. TabArena selects configurations and ensemble weights
using validation predictions; test outcomes do not change the frozen portfolio.

| Evaluation | Lite Elo | Win rate |
|---|---:|---:|
| Default | 1161.9 (+52.2 / -66.8) | 0.3900 |
| Tuned | 1262.8 (+44.3 / -53.8) | 0.5006 |
| Tuned + ensemble | 1296.9 (+47.8 / -58.2) | 0.5383 |

| Run scope | Value |
|---|---:|
| Datasets | 51 |
| Measured configurations | 26 (default + 25 HPO) |
| Successful parent results | 1,326 / 1,326 |
| Bagging | 8 children per parent; 10,608 child fits |
| Worker CPU allocation | 4 |
| Worker memory limit | 28 GB |
| Per-fit limit | 3,600 seconds |

!!! note "Scope of the score"
    This is an author-run Lite evaluation with 25 HPO configurations, not a full
    200-configuration run, TabArena-Full result, or official leaderboard entry.
    Elo depends on the dataset and method roster: this evaluation uses 87 rows,
    including defaults, tuned configurations, ensembles, and systems. The workers' 4 CPUs and 28 GB
    differ from the canonical 8 CPUs and 32 GB, so its recorded timings are not
    directly comparable to official TabArena runtimes.

Provenance:

- Released `ctboost==0.1.58`, measured as `CTBoost_c1_default_BAG_L1` and
  `CTBoost_r1_default_BAG_L1` through `CTBoost_r25_default_BAG_L1`;
- 30 binary, 8 multiclass, and 13 regression datasets, with no missing or
  duplicate tasks in the published canonical tables;
- 156 CPU shards, AutoGluon Tabular 1.6.2b20260821;
- TabArena integration/evaluation commit
  [`31026f7d758390994353eba79fbfa6747616f365`](https://github.com/captnmarkus/tabarena/commit/31026f7d758390994353eba79fbfa6747616f365),
  associated with [upstream PR #479](https://github.com/autogluon/tabarena/pull/479).

The [aggregate manifest](https://huggingface.co/datasets/Maiernator/ctboost-tabarena-lite-hpo25-0.1.58/blob/main/validation/aggregate_manifest.json)
records task coverage, resource limits, artifact hashes, and version provenance.
The [evaluation table](https://huggingface.co/datasets/Maiernator/ctboost-tabarena-lite-hpo25-0.1.58/blob/main/reports/leaderboard_lite.csv)
records the scores and comparison roster. Canonical result tables contain all
26 measured configurations and the default, tuned, and tuned-plus-ensemble
evaluation results.

### Historical TabArena results

The [0.1.56 default-only Lite result](https://huggingface.co/datasets/Maiernator/ctboost-tabarena-lite-0.1.56)
covered the same 51 datasets and reported 1166.7 Elo (+52.1 / -67.5), a 0.3957
win rate, and position 23/38 among default-configuration rows, with no imputation.
Its comparison roster contained 85 rows, versus 87 in the 0.1.58 evaluation.
The two Elo values therefore do not quantify a version-to-version change in
accuracy. The historical bundle contains only the default configuration and
408 child fits; its `hpo_results.parquet` filename does not imply an HPO run.

The earlier [0.1.55 three-dataset smoke record](https://github.com/captnmarkus/ctboost/blob/master/benchmarks/tabarena/smoke_0155_public_wheel.json)
(1054.0 provisional Elo) and [0.1.53 smoke record](https://github.com/captnmarkus/ctboost/blob/master/benchmarks/tabarena/smoke_fd187da.json)
remain historical integration checks. Their different task scope prevents
interpreting the newer Lite score as a version-to-version Elo improvement.

### Historical grouped split-statistic qualification

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

## Reproduce the evaluations

The published [aggregate manifest](https://huggingface.co/datasets/Maiernator/ctboost-tabarena-lite-hpo25-0.1.58/blob/main/validation/aggregate_manifest.json)
preserves the completed evaluation's resource and artifact provenance. For a local
default-only Lite run, follow the pinned environment instructions in the
[adapter guide](https://github.com/captnmarkus/ctboost/blob/master/benchmarks/tabarena/README.md),
then run:

```bash
ctboost-tabarena \
  --subset lite \
  --device cpu \
  --num-cpus 8 \
  --memory-limit-gb 32 \
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
| Full HPO25 | default + 25 frozen HPO configurations | 21,216 | 169,728 |
| Full HPO200 | default + 200 frozen HPO configurations | 164,016 | 1,312,128 |

Full refers to outer-split coverage; the HPO count is a separate part of the
protocol. These counts require resumable execution and complete coverage checks.
Official publication also requires upstream model registration, raw-artifact
review, and TabArena maintainer verification. Resource allocations and timing
comparability must be reported alongside results.

## Benchmark policy

1. Search spaces are defined before reading TabArena test results.
2. Validation folds select configurations and ensemble weights; test folds do not.
3. Failed tasks remain failures and are reported.
4. Every published number includes code, package, hardware, resource, and task scope.
5. Smoke, default, tuned, and tuned-plus-ensemble results are labeled separately.
