# Binary checkpoint selection on small TabArena tasks

On 17 November 2026, selecting a checkpoint by ROC-AUC increased mean development error from **0.206329 to 0.216042** across nine datasets. Keep the existing binary stopping policy. AutoGluon uses log loss by default for ROC-AUC tasks, and the CTBoost adapter passes the selected stopping metric to the estimator. The upstream implementation is available in [TabArena](https://github.com/autogluon/tabarena/blob/cc507b84d1f32d2ca5b5704326bf4f086ee6519a/packages/tabarena/src/tabarena/models/ctboost/model.py).

The roster included every binary task in the two September development caches with at most 2,000 original outer-training rows and 128 features. CTBoost 0.1.62 ran once per stopping policy and learner seed (0 and 1), for 36 fits. Each used the TabArena manual default, a 1,000-round cap and 50 rounds of patience. Only the stopping metric differed. All models finished before scoring the separate development rows. These are single-model development results on reused tasks. An official leaderboard change requires a new benchmark evaluation.

| Dataset | Development rows | Log-loss stopping | AUC stopping |
|---|---:|---:|---:|
| blood-transfusion-service-center | 62 | 0.426241 | 0.426241 |
| Fitness_Club | 250 | 0.215199 | 0.216178 |
| Is-this-a-good-customer | 286 | 0.281804 | 0.297859 |
| Marketing_Campaign | 372 | 0.210160 | 0.210104 |
| credit-g | 166 | 0.196293 | 0.209698 |
| diabetes | 128 | 0.174175 | 0.198864 |
| hazelnut-spread-contaminant-detection | 400 | 0.062300 | 0.076688 |
| qsar-biodeg | 174 | 0.089031 | 0.086690 |
| seismic-bumps | 430 | 0.201759 | 0.222059 |

Each cell is mean development **1 − ROC-AUC** across the two seeds. Lower is better. AUC stopping won on two tasks, tied on one within 1e-12, and lost on six. Its median relative error reduction was **−5.70%**.

## Reproduction

The [runner](binary_stopping_scout.py) accepts the original local cache containing `accuracy-grouped-scout-v1` and `accuracy-remaining-scout-v1`. It verifies both source plans and each training, stopping and development file against its SHA-256 digest and prints the error and selected iteration for each task, seed and policy.

```shell
python -I benchmarks/tabarena/binary_stopping_scout.py --cache .tmp/inference-accuracy-20260908
```

The recorded Windows CPU environment used Python 3.12.11, CTBoost 0.1.62, NumPy 2.0.2, pandas 2.3.3 and scikit-learn 1.6.1. The native extension SHA-256 was `4e2924a2259ad08f072be6b92d2bd0a9173e8c3c55d8ca0bcafab7f47c50d41f`.
