# Quantile cuts on sparse numerical features

When most values in a numerical feature equal its minimum, every sampled quantile can equal that minimum. The old binning code then put both the minimum and the rare larger values in the same bin. The correction moves the cut to the next distinct observed training value so the learner can evaluate the feature.

On 17 November 2026, a training-only screen found this condition in **110 raw numerical columns across six of the 51 cached TabArena tasks** at the default 256 bins. All six tasks were evaluated. Averaging the errors of two learner seeds within each task, the correction reduced median relative development error by **0.25%**. Three tasks improved, one tied and two worsened.

| Dataset | Affected columns | Development rows | Seed 0 baseline | Seed 0 fixed | Seed 1 baseline | Seed 1 fixed | Relative reduction across seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| coil2000_insurance_policies | 8 | 818 | 0.34839840 | 0.31986943 | 0.33920278 | 0.32087790 | +6.81% |
| APSFailure | 3 | 2,500 | 0.01527231 | 0.01509514 | 0.01497998 | 0.01500656 | +0.50% |
| Bioresponse | 89 | 624 | 0.19513179 | 0.20037655 | 0.19948690 | 0.19870071 | −1.13% |
| QSAR-TID-11 | 8 | 956 | 0.86756871 | 0.86802097 | 0.87533843 | 0.87785073 | −0.17% |
| kddcup09_appetency | 1 | 2,500 | 0.19486375 | 0.19486375 | 0.21649215 | 0.21649215 | 0.00% |
| taiwanese_bankruptcy_prediction | 1 | 1,136 | 0.07158842 | 0.07026043 | 0.07456410 | 0.06969481 | +4.24% |

Errors are 1 − ROC-AUC for binary tasks and RMSE for the QSAR-TID-11 regression task. Relative reduction compares each task’s mean errors as `1 − fixed / baseline`. Seed 0 was the planned single-seed comparison and had a median reduction of **0.58%**. A follow-up check repeated all six tasks with seed 1 on the same splits and had a median reduction of **0.20%**.

## Evaluation scope

The screen inspected numerical columns of the cached training frames before target-based fitting or scoring. It did not screen categorical expansions. Each run used the TabArena manual default with a 200-round cap and 25 rounds of patience. It selected the iteration on the stopping split using log loss for binary tasks and RMSE for regression. Each runtime finished all six fits before predicting its separate development split. Task selection used training features alone. The results cover individual models on reused development data. Measuring an overall Elo change requires evaluation against the same TabArena roster and splits.

## Follow-up across all 51 tasks

Both builds were also fitted on **all 51 tasks**, including those with no affected raw columns. All 102 fits completed at seed 0 using the same cap, patience and stopping roles. Multiclass models used multiclass log loss for stopping and scoring. The correction improved three tasks, tied on 46 and worsened two. All 46 ties produced byte-identical development predictions. The median relative error reduction was **0.00%** across all tasks, and the unweighted mean reduction was **0.166%**. The [paired result file](../results/quantile_ties_all_tasks_20261117.json) gives the errors, selected iterations and prediction hashes for both builds on every task.

| Task type | Datasets | Improved | Identical predictions | Worsened |
|---|---:|---:|---:|---:|
| Binary | 30 | 3 | 26 | 1 |
| Multiclass | 8 | 0 | 8 | 0 |
| Regression | 13 | 0 | 12 | 1 |
| **Total** | **51** | **3** | **46** | **2** |

All 45 tasks without affected raw columns reproduced the baseline predictions exactly. One of the six tasks with affected columns, kddcup09_appetency, also tied. This single-seed development panel used one model per task and build. Its taskwise losses do not give a TabArena Elo rating.

## Reproduction

The [runner](quantile_ties_scout.py) reads the two September cache plans, verifies their SHA-256 hashes and checks each role file against its plan. It prints every task’s affected raw columns and each evaluated task’s error, selected iteration and prediction hash. Run the commands for each installed build from the repository root, pointing `--cache` at the directory containing `accuracy-grouped-scout-v1` and `accuracy-remaining-scout-v1`. An optional `--progress` path saves one JSON line per task as it finishes.

```shell
python -I benchmarks/tabarena/quantile_ties_scout.py --cache .tmp/inference-accuracy-20260908 --seed 0
python -I benchmarks/tabarena/quantile_ties_scout.py --cache .tmp/inference-accuracy-20260908 --seed 1
python -I benchmarks/tabarena/quantile_ties_scout.py --cache .tmp/inference-accuracy-20260908 --seed 0 --all-tasks
```

Both CPU builds used MSVC 1950 and C++17. The environment used Python 3.12.11, NumPy 2.0.2, pandas 2.3.3 and scikit-learn 1.6.1. The installed 0.1.62 baseline extension had SHA-256 `4e2924a2259ad08f072be6b92d2bd0a9173e8c3c55d8ca0bcafab7f47c50d41f`. The final 0.1.63 candidate extension had SHA-256 `21b735aa0437881e9b315edfcb29203539f64ff8fcfdca134a4ec389f62b14a4` and reproduced the first corrected build’s probability and prediction hashes for every task and seed.
