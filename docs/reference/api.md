# Python API

The canonical signatures live in the installed package and remain available through
Python's standard help and inspection tools:

```python
import inspect
from ctboost import CTBoostClassifier

print(inspect.signature(CTBoostClassifier))
help(CTBoostClassifier.fit)
```

## Estimators

- `CTBoostRegressor`
- `CTBoostClassifier`
- `CTBoostRanker`
- `CTBoostMultiOutputRegressor`
- `CTBoostMultiOutputClassifier`
- `CTBoostAFTSurvivalRegressor`

The sklearn-style estimators provide `fit`, `predict`, staged prediction, persistence,
feature importance, SHAP, plotting, grid/random search, feature selection, and estimator
comparison. Classifiers also provide `predict_proba`; positive-link regressors provide
raw and response-scale prediction helpers.

## Low-level API

- `Pool` and `PoolBatch`
- `train`
- `cv`
- `Booster`
- `FeaturePipeline`
- inference-manifest and export helpers

Use the low-level API when you need explicit Pools, ranking groups, survival metadata,
distributed roots, custom objectives, or direct booster-state control.

## Optional modules

- `ctboost.dask`
- `ctboost.ray`
- `ctboost.spark`
- `ctboost.streaming`

These modules import their framework dependencies lazily and provide actionable errors
when an optional extra is absent.
