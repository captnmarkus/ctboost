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
- `CTBoostMultiLabelClassifier`
- `CTBoostAFTSurvivalRegressor`

`CTBoostAFTRegressor` is an alias of `CTBoostAFTSurvivalRegressor`. The
`CBoostClassifier`, `CBoostRegressor`, and `CBoostRanker` names are compatibility
aliases for their `CTBoost...` counterparts.

The primary estimators provide `fit`, `predict`, staged prediction, persistence,
feature importance, SHAP, plotting, grid/random search, feature selection, and
estimator comparison. Classifiers also provide `predict_proba`; positive-link
regressors provide raw and response-scale prediction helpers. The independent-output
and survival wrappers expose the subset appropriate to their target semantics. See
[Training workflows](../guides/training.md) for custom objectives, callbacks, model
selection, multi-output targets, and AFT censoring.

## Low-level API

- Data: `Pool`, `PoolBatch`, `pool_from_batches`, `FeaturePipeline`,
  `PreparedTrainingData`, `prepare_training_data`, and `prepare_pool`
- Training: `train`, `cv`, `Booster`, and `load_model`
- Customization: `ObjectiveSpec`, `make_objective`, `EvalMetricSpec`,
  `make_eval_metric`, `TrainingCallbackEnv`, `log_evaluation`, and
  `checkpoint_callback`
- Inference contracts: `load_exported_predictor`, `load_inference_manifest`, and
  `validate_inference_manifest`
- Runtime identity: `build_info` and `__version__`

Use the low-level API when you need explicit Pools, ranking groups, survival metadata,
distributed roots, custom objectives, streaming input, or direct booster-state control.
Most of these names are available directly from `ctboost`; inspect the corresponding
object for its installed-version signature.

## Framework integrations

- `ctboost.dask`
- `ctboost.ray`
- `ctboost.spark`

These modules import their framework dependencies lazily and provide actionable errors
when an optional extra is absent. Streaming helpers are part of the base package under
`ctboost.streaming` and are also exported as `PoolBatch` and `pool_from_batches`.
