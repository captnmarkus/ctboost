# Explainability

## Exact empirical interventional SHAP

```python
values = model.predict_shap_values(X_explain, X_background)
interactions = model.predict_shap_interaction_values(X_explain, X_background)
```

These values are exact for CTBoost's empirical interventional game over the supplied
background rows. They support repeated-feature paths, missing and categorical routes,
sample weights, multiclass models, and iteration limits. Runtime grows with foreground
rows, background rows, and trees; this implementation is currently CPU-oriented.

When a fitted categorical, text, or embedding `FeaturePipeline` expands raw
columns, SHAP values are reported in the transformed feature space. The
corresponding names are available from `model.get_booster().feature_names`.

`predict_contrib` remains available as a faster path-based additive
decomposition. It is additive, but it is not an interventional SHAP value.
For a single-output model, SHAP output has shape `(rows, features + 1)`; the
final column is the expected raw output over the background. Interaction
output has shape `(rows, features + 1, features + 1)` and uses the final
row/column for the bias term.

```python
shap_values = model.get_booster().predict_shap(X_explain, X_background)
np.testing.assert_allclose(
    shap_values.sum(axis=-1),
    model.get_booster().predict(X_explain),
    rtol=1e-6,
    atol=1e-6,
)
```

## Diagnostics

```python
model.plot_tree(tree_index=0)
model.plot_predictions(X_valid, y_valid, kind="residual")
model.plot_feature_statistics(X_train, y_train, feature="age")
model.plot_metrics()
```

`get_object_importance` is a signed shared-leaf approximation. It does not refit the
model after deleting a training object and should not be described as exact influence.

`calc_leaf_influence` distributes each explained row's signed leaf
contribution among reference rows reaching the same leaf. Positive and
negative values describe shared-leaf contribution direction; they are not
deletion or upweighting counterfactuals. The dense result scales with
`explained_rows × reference_rows` (and an output dimension for multiclass), so
batch large jobs.
