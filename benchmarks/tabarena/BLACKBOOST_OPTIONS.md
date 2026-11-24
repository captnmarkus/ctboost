# BlackBoost-inspired options in CTBoost

R's `mboost::blackboost` combines gradient boosting with conditional inference
trees. This study transfers two ideas independently. CTBoost can fit a
conditional expectile and choose a refit round from the mean validation loss
across folds. Both use CTBoost's feature test and Hessian-weighted search for
a cut on the selected feature. [BlackBoost manual](https://github.com/boost-R/mboost/blob/52a515bce45542a85f9de0e03417ca6db22c77e0/man/blackboost.Rd) / [Loss families](https://github.com/boost-R/mboost/blob/52a515bce45542a85f9de0e03417ca6db22c77e0/man/Family.Rd) / [Cross-validated risk](https://github.com/boost-R/mboost/blob/52a515bce45542a85f9de0e03417ca6db22c77e0/man/cvrisk.Rd)

## Asymmetric prediction costs

For both objectives we selected the boosting round on the same stopping rows
using the mean asymmetric squared error at `tau=0.8`. An error of a given size
costs four times as much below the observed value as above it. Both models
started at the training-label mean and were scored on the same held-out
development rows in the original target units. We measured ordinary RMSE
alongside the asymmetric loss. The study uses all 13 regression tasks in the
frozen roster, with 200 rounds at most and stopping patience of 25.

| Target metric | Lower error | Ties | Higher error | Median change in error |
|---|---:|---:|---:|---:|
| Asymmetric squared error at 0.8 | 12 | 0 | 1 | -18.75% |
| Ordinary RMSE | 0 | 0 | 13 | +12.18% |

Each row compares the expectile objective with squared-error training. We
averaged each objective's errors over seeds 0 and 1 within each dataset
before computing its percentage change. Averaging those changes across
datasets gives a 15.37% decrease in asymmetric loss and a 14.41% increase
in RMSE. At each seed, asymmetric loss fell on the same 12 tasks and rose
on QSAR fish toxicity. The two seeds reuse the same data splits. [Per-task
scores, seeds and prediction hashes](../results/expectile_regression_20261124.json)

## Tree depth and row sampling

We tested two common BlackBoost choices independently on all 51 cached TabArena
tasks. The first uses depth 2, at least 10 rows to consider a split and at least
4 rows per leaf. The second fits each tree to all training rows. The CTBoost
reference uses depth 6, a minimum split size of 2, the default minimum leaf
setting, and Bernoulli sampling at 0.8. Every arm uses the same significance
threshold of 0.05, learning rate of 0.05, one seed, one thread, a cap of
200 boosting rounds, and stopping patience of 25.

| Change from CTBoost reference | Lower error | Ties | Higher error | Median change in error |
|---|---:|---:|---:|---:|
| Shallow trees | 8 | 0 | 43 | +5.14% |
| Use all rows per tree | 7 | 0 | 44 | +2.08% |

Percent changes give each dataset equal weight. The task errors are one minus
ROC-AUC for binary classification, log loss for multiclass classification,
and RMSE for regression. The reference run reproduced the earlier saved
predictions and selected rounds for all 51 tasks. Against the same three
frozen competitors, CTBoost averaged rank 2.73 with its reference settings,
3.20 with shallow trees and 3.06 with full-row sampling. CTBoost retains its existing
TabArena settings. These development results cover single-model fits and the
fixed round budget. Official Elo uses a separate evaluation.

## Reproduction

Both runners verify the train, stopping and development files against the two
September cache plans. The official outer test and the separate confirmation
rows were excluded. The Python and native binary versions are recorded in
the receipts. The expectile runner requires a CTBoost installation containing
`make_expectile_loss`. Install this checkout in the benchmark environment
and run these commands with the September cache directory and new output paths.

```sh
python -I benchmarks/tabarena/blackboost_controls_scout.py --cache CACHE --output controls.json
python benchmarks/tabarena/expectile_regression.py --cache CACHE --output expectile.json
```

The [tree and sampling results](../results/blackboost_controls_20261124.json)
include the three arms on each task. The [earlier four-model comparison](FOUR_MODEL_DEVELOPMENT.md)
documents the original splits, competitors and scoring protocol.
