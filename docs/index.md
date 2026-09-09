# Conditional inference, boosted

<div class="ctb-hero" markdown>

CTBoost is a gradient-boosting library built around **conditional-inference trees**.
It combines alpha-stopped feature-test selection, optional per-node Bonferroni
adjustment, a familiar Python and scikit-learn interface, native C++ training,
and optional CUDA acceleration.

[Install CTBoost](getting-started.md){ .md-button .md-button--primary }
[See benchmark status](benchmarks.md){ .md-button }

</div>

The current release, [0.1.61](release-0.1.61.md), is available on
[PyPI](https://pypi.org/project/ctboost/0.1.61/) and
[GitHub](https://github.com/captnmarkus/ctboost/releases/tag/v0.1.61).
It reduces CPU inference overhead and fixes
Huber/Quantile edge cases, NumPy conversion compatibility, and GNU FMA
training-rounding differences. Saved-model formats, learning defaults, and
conditional-inference feature selection are preserved.

!!! info "Project status"
    CTBoost is an alpha project. Its API and model format are tested extensively,
    but it does not yet have the independent production history of CatBoost or
    XGBoost. Benchmark claims on this site state their dataset and protocol scope.

As of 9 September 2026, the fresh 0.1.61 Full HPO25 evaluation is prepared but
has not started; the 0.1.60 run is retired. The latest downloadable author-run
TabArena bundle measures 0.1.58. See [benchmark status](benchmarks.md) for the
separate maintainer report, evidence, and limits.

<div class="grid cards" markdown>

-   :material-chart-tree:{ .lg .middle } **A different tree learner**

    ---

    Conditional tests select features before cut points are optimized. CTBoost
    preserves that ordering across its supported learning options.

-   :material-speedometer:{ .lg .middle } **Native CPU and CUDA paths**

    ---

    Histogram training, stochastic boosting, DART, early stopping, callbacks,
    ranking, survival objectives, and task-aware feature preprocessing.

-   :material-shield-check:{ .lg .middle } **Auditable artifacts**

    ---

    Versioned inference manifests, deterministic fingerprints, JSON/Python/C++/ONNX
    exports, exact empirical interventional TreeSHAP, and reproducible benchmark metadata.

-   :material-language-python:{ .lg .middle } **Python ecosystem fit**

    ---

    NumPy, pandas, SciPy sparse, Arrow, Polars, cuDF/CuPy/DLPack adapters,
    scikit-learn helpers, and Dask/Ray/Spark integration surfaces.

</div>

## A small example

```python
from ctboost import CTBoostClassifier

model = CTBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    max_depth=6,
    ordered_ctr=True,
    random_seed=42,
)
model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=50,
)

probability = model.predict_proba(X_test)
explanation = model.predict_shap_values(X_test, X_train)
```

## Where to go next

- [Install and train a first model](getting-started.md).
- [Understand what is different](why-ctboost.md).
- [Read the benchmark protocol and current evidence](benchmarks.md).
- [Choose a deployment format](guides/deployment.md).
- [Check explicit compatibility limits](reference/compatibility.md).
