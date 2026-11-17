# Four-model comparison on TabArena development splits

This CPU study compares the public CTBoost 0.1.63 wheel with CatBoost 1.2.10,
LightGBM 4.7.0 and XGBoost 3.4.1 on the same 51 cached datasets. The
[per-task results](../results/four_model_development_20261117.json) record each
model's development error, selected boosting round, prediction hash and rank.

CTBoost placed **third of four overall** with a mean task rank of **2.73**.
It had the best error on nine tasks. All 204 model–task pairs were scored.

| Model | All 51 | Binary (30) | Multiclass (8) | Regression (13) | First places |
|---|---:|---:|---:|---:|---:|
| CatBoost | 1.86 | 1.73 | 2.38 | 1.85 | 22 |
| LightGBM | 2.41 | 2.77 | 1.88 | 1.92 | 14 |
| CTBoost | 2.73 | 2.47 | 2.50 | 3.46 | 9 |
| XGBoost | 3.00 | 3.03 | 3.25 | 2.77 | 6 |

CTBoost beat CatBoost on 14 tasks and lost on 37. Against LightGBM it won
24 and lost 27. Against XGBoost it won 27 and lost 24. There were no
head-to-head ties. CTBoost's strongest relative result was on binary
classification, where it ranked second. It ranked fourth on regression.

## Evaluation

The two September cache plans fix the dataset roster and the training, stopping
and development rows. Each role file is checked against its plan's SHA-256.
The scoring rows come from previously used development splits inside the outer
training data. The official outer test and the separate confirmation split
were excluded. Training sets range from 312 to 6,250 rows.

Each model used one CPU thread, seed 0, a cap of 200 boosting rounds and a
patience of 25 rounds on the same stopping rows. Binary and multiclass models
stopped on log loss, and regressors stopped on RMSE. On the development split,
the primary errors are 1 minus ROC-AUC for binary classification, log loss for
multiclass classification and RMSE for regression. Lower errors rank higher.
Each task gets equal weight in the mean rank. Equal scores share the average
of their rank positions. This comparison uses individual fits and omits the
bagging, tuning, ensembling and opponent roster used to calculate official Elo.
Class probabilities follow sorted training-label order. Multiclass log loss
was evaluated in float64. The released CTBoost wheel
reproduced the prediction hashes from the earlier 51-task 0.1.63 candidate
study on every task.

CTBoost used its TabArena manual default with the shared round and patience
limits. The other libraries used their public CPU training defaults for tree
shape, regularization and learning rate. CatBoost can choose its default
learning rate from the fitting data and iteration cap. Equal round limits
allow different runtimes and model sizes.

CTBoost received the raw frames with native ordered categorical processing.
CatBoost used integer identifiers for known training categories, with distinct
codes for missing and unseen values. LightGBM and XGBoost received pandas
categorical columns whose vocabulary was fitted on training rows. Unseen
values in those columns take the missing route. All models use the same
column order and examples. Each library's native categorical learner still
determines how those features enter its trees.

## Reproduction

The run used Windows, a Ryzen 7 5800X3D and Python 3.12.11. Package versions
are recorded in the results. Run the following from the source checkout with
the September caches under the named directory. Use a new, empty directory
for the per-library receipts. The four library commands can run in separate
environments with the versions recorded in the results.

```powershell
New-Item -ItemType Directory results-four-model
foreach ($library in @("ctboost", "catboost", "lightgbm", "xgboost")) {
    python -I benchmarks/tabarena/four_model_development.py --library $library --cache .tmp/inference-accuracy-20260908 --output "results-four-model/$library.jsonl"
}
python -I benchmarks/tabarena/four_model_development.py --summarize results-four-model --output results-four-model/summary.json
```
