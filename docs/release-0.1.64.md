# CTBoost 0.1.64

## Expectile regression and cross-validated round selection

Use `make_expectile_loss(tau)` when underprediction and overprediction have
different costs. It returns a training objective and a matching weighted
validation metric. At `tau=0.8`, a prediction below the target costs four
times as much as an equally distant prediction above it. CTBoost fits the
resulting gradients and Hessians with its conditional-inference trees.

The new `select_num_boost_round=True` option in `ctboost.cv` selects one
round count for refitting. It compares the mean primary validation metric
at rounds with finite scores in every fold. Ties go to the earliest round.
The returned count starts at one and can be passed directly to `ctboost.train`.

Install the scikit-learn extra to run this example. It chooses a round count
on the training data, refits the expectile model, and predicts for a new row.

```sh
python -m pip install "ctboost[sklearn]==0.1.64"
```

```python
import numpy as np
import ctboost

rng = np.random.default_rng(42)
X = rng.normal(size=(150, 2))
y = 2 * X[:, 0] + rng.exponential(size=len(X))
pool = ctboost.Pool(X, label=y)
objective, metric = ctboost.make_expectile_loss(0.8)
params = {"objective": objective, "eval_metric": metric}

result = ctboost.cv(pool, params, num_boost_round=40, nfold=3,
                    select_num_boost_round=True)
model = ctboost.train(pool, params, num_boost_round=result["best_num_boost_round"])
print(model.predict([[1.0, 0.0]]))
```

For portable prediction, record the expectile level alongside the exported
predictor so its outputs can be interpreted. [Training guide](https://captnmarkus.github.io/ctboost/guides/training/)

## Pandas columns with colliding names

Previously, two inferred categorical columns with the same display name could
use the same set of category labels and produce incorrect predictions. This
can happen with duplicate column names or with names such as the integer
`1` and the string `"1"`. New fits store each affected vocabulary by column
position. Preserve the training column order when validating, predicting, or
continuing a fit.

A 0.1.63 model saved only the last inferred vocabulary under a repeated name.
Supply the original numeric training codes for the earlier affected columns,
or refit on the original labels. Inference manifests now also assign metadata
for a numeric-looking column name to the column that owns it.
[Categorical input guide](https://captnmarkus.github.io/ctboost/guides/categorical-text/)

## Benchmark evidence

Across 13 TabArena regression development datasets and two seeds, expectile
training at `tau=0.8` reduced asymmetric squared error on 12 datasets compared
with squared-error training. The median reduction was **18.75%**. Ordinary
RMSE increased on all 13, with a median increase of **12.18%**. These figures
measure the cost tradeoff on the chosen development splits. Official TabArena Elo
uses a separate evaluation. [Study and reproduction](https://github.com/captnmarkus/ctboost/blob/612328f74cd792ace7f7bdd96ef50f583e903f52/benchmarks/tabarena/BLACKBOOST_OPTIONS.md)

The R predictor package metadata now names Markus Maier as its maintainer.
