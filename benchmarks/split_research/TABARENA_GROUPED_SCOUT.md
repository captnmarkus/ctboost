# Frozen grouped-statistic TabArena scout

This scout is pre-registered before the external-panel result is known. It is
an integration and directional check, not TabArena-Full and not a source of
hyperparameter tuning decisions. Run it only if grouped-8 passes every frozen
promotion gate in [EXTERNAL_PANEL.md](EXTERNAL_PANEL.md).

## Immutable scope

Use TabArena commit
`50f8ab1bbc6e7f7e5dd9b19d8b643ac284ae9b3c`, CPU execution, and only repeat 0,
fold 0, sample 0 of these official lite tasks:

| Task | Dataset | Problem | Primary error |
|---:|---|---|---|
| 363614 | `anneal` | multiclass | log loss |
| 363621 | `blood-transfusion-service-center` | binary | `1 - ROC-AUC` |
| 363698 | `QSAR_fish_toxicity` | regression | RMSE |

Let `P50 = [{}, *generate_configs_ctboost(50)]`. Before scheduling, require
its canonical 51-entry SHA-256 to equal:

`edf9ec119040cf687220737a8410c3406979ced4b6c47b969ca6f4f5a8e8b6fe`

Abort if it differs or if a base configuration already contains
`feature_test`, `feature_test_bins`, or `feature_test_adjustment`. Deep-copy
each entry and create exactly two paired methods:

- `CTBQS1` / `CTBoostQuadraticScoutV1`: add
  `feature_test="quadratic"`, `feature_test_bins=8`, and
  `feature_test_adjustment="none"`;
- `CTBG8S1` / `CTBoostGrouped8ScoutV1`: add
  `feature_test="grouped"`, `feature_test_bins=8`, and
  `feature_test_adjustment="none"`.

The paired effective configurations may differ only in `feature_test`. Do not
test another bin count, multiplicity adjustment, selected subset, or ordering.
Do not modify the existing frozen 200-configuration CTBoost portfolio.

Expected coverage is `2 * 3 * 51 = 306` outer artifacts and 2,448 eight-fold
bagged child fits. Expected model keys are the two method names followed by
`_c1_default_BAG_L1` or `_r1_default_BAG_L1` through
`_r50_default_BAG_L1`.

## Cache and resources

Use a clean merged CTBoost commit and a clean TabArena checkout. Derive a new
empty result namespace as:

`benchmark-results/tabarena/g8s1-p50-cpu8-ct<SHA12>-ta50f8ab1/{raw,report}`

Record the full CTBoost commit, installed-package fingerprint, native-extension
SHA-256, full TabArena commit, and P50 hash. Any source, package, configuration,
or dependency identity change requires a new namespace. Public dataset caches
may be reused; no previous CTBoost fit artifact may be copied or linked. The
old n50 control results are ineligible because their source, method, and
effective configuration identities differ.

The fixed resource contract is:

- CPU only, `num_cpus=8`, `num_gpus=0`, and `CTBOOST_HIST_THREADS=8`;
- 32 GiB memory limit and 3,600 seconds per TabArena fit;
- no Ray, one shard, job batch size 8, and no competitor reruns;
- both treatments on the same host;
- all 306 jobs finish before evaluated outcomes are inspected.

TabArena retains its eight bag folds and fold/config-wise seed injection. Do
not add a model seed.

## Evaluation and decision

Use TabArena `metric_error` (lower is better). For endpoint `e` and task `t`,
report:

`improvement(t, e) = (quadratic_error - grouped_error) / quadratic_error`

The primary endpoint is `tuned + ensemble`; `tuned` and `default` are
secondary. An absolute relative change below 0.1% is a tie. Also publish all
51 paired configuration errors per task, validation-selected configuration
IDs, median and total training-time ratios, inference time, incremental and
absolute peak RSS, model bytes, timeouts, and failures. Configurations are not
independent statistical observations.

Integration passes only if coverage is exactly 306/306 with no duplicate or
stale artifact; all predictions and metrics are finite and correctly shaped;
all 51 configurations exist for every task/treatment; every resource and seed
contract matches; paired effective configs differ only in `feature_test`; and
all six default/tuned/ensemble summaries exist.

A full grouped ablation is supported only if integration passes and:

1. grouped tuned+ensemble wins at least two of three tasks;
2. its task-macro median improvement is at least 0.25%;
3. no task worsens by more than 1%;
4. tuned (without ensembling) has non-negative median improvement and no task
   worsens by more than 2%; and
5. the median paired training-time ratio is at most 1.15.

Otherwise report `integration failure` or `performance not supportive`; do
not revise and rerun this protocol. A negative three-task scout does not by
itself overturn a passing external panel or remove a user-facing opt-in.

Any Elo is labeled **local three-task provisional scout Elo**. It is not
TabArena-Full, official Elo, or evidence of reaching 1300/1400. Elo is not a
decision criterion. Publish favorable or unfavorable sanitized JSON/CSV with
full provenance and coverage; keep raw artifacts ignored and exclude absolute
paths and credentials.

## Later official run

Keep the official `CTB` quadratic default and its frozen 200 configurations
unchanged. If maintainers approve a full grouped ablation, register `CTBG8` as
a separate method with the same 201 base configurations plus the fixed grouped
override. Report `CTB` and `CTBG8` independently. Each costs 164,016 outer
jobs. Combining them as one tuner or ensemble is a 402-configuration method
and must not be presented as the existing CTBoost portfolio.

If only one full run is funded, run unchanged `CTB`. Any future mixed
quadratic/grouped search needs a new portfolio version, method identity, and
empty cache.
