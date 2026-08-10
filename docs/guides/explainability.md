# Explainability

## Exact empirical interventional SHAP

```python
values = model.get_feature_importance(
    data=X_explain,
    reference_data=X_background,
    type="ShapValues",
)

interactions = model.get_feature_importance(
    data=X_explain,
    reference_data=X_background,
    type="ShapInteractionValues",
)
```

These values are exact for CTBoost's empirical interventional game over the supplied
background rows. They support repeated-feature paths, missing and categorical routes,
sample weights, multiclass models, and iteration limits. Runtime grows with foreground
rows, background rows, and trees; this implementation is currently CPU-oriented.

## Diagnostics

```python
model.plot_tree(tree_index=0)
model.plot_predictions(X_valid, y_valid, kind="residual")
model.plot_feature_statistics(X_train, y_train, feature="age")
model.plot_metrics()
```

`get_object_importance` is a signed shared-leaf approximation. It does not refit the
model after deleting a training object and should not be described as exact influence.
