# Frozen external ablation panel

This protocol is frozen before the native grouped-statistic results are run.
It is deliberately disjoint from the TabArena-v0.1 dataset names and is not a
substitute for TabArena. Its purpose is to decide whether the synthetic
grouped-8 result is strong enough to justify a public, opt-in learner feature
and a later frozen TabArena scout.

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
