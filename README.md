# CTBoost

CTBoost is an alpha gradient-boosting library built around
**conditional-inference trees**. At each node it selects a feature with a
conditional statistical test before optimizing that feature's split point. A
native C++17 core provides CPU training, optional CUDA acceleration, and
Python/scikit-learn interfaces.

[Documentation](https://captnmarkus.github.io/ctboost/) ·
[Getting started](https://captnmarkus.github.io/ctboost/getting-started/) ·
[Benchmarks](https://captnmarkus.github.io/ctboost/benchmarks/) ·
[Compatibility](https://captnmarkus.github.io/ctboost/reference/compatibility/)

The [0.1.62 release](https://captnmarkus.github.io/ctboost/release-0.1.62/)
reduces CPU prediction preparation costs for eligible categorical pandas inputs
and wide fitted schemas, writes compact JSON, and improves the precision of
categorical target statistics. Conditional feature testing still selects a
feature before its cut point is optimized; tree-growth and hyperparameter
defaults remain unchanged.
Newly fitted pipelines use format 5 for precise CTR accumulation. Formats 1–4
remain loadable with their original arithmetic; older CTBoost versions cannot
load format 5. The JVM binding also updates Jackson to 2.21.7 to address five
reported security alerts. See the release notes for compatibility, validation,
and the scope of the measured results.

## Highlights

- Regression, classification, query-group ranking, survival, multi-output,
  multilabel, and callable-objective training.
- Low-level `Pool`/`train` APIs plus familiar scikit-learn estimators.
- NumPy, pandas, SciPy sparse, Arrow, and Polars input, with fitted
  categorical, text, and embedding preprocessing.
- Native CPU and CUDA training, plus Dask, Ray, and Spark barrier adapters;
  Spark retains an explicit collect-to-driver fallback for small jobs.
- Snapshots, warm starts, staged prediction, callbacks, model selection, and
  deterministic inference manifests.
- Exact empirical interventional TreeSHAP and JSON, Python, C++, C ABI, ONNX,
  pickle, and prepared-feature R/JVM inference choices.

## Install

```bash
python -m pip install -U ctboost
python -m pip install -U "ctboost[sklearn,dataframes]"  # optional integrations
```

See the [GPU installation guide](https://captnmarkus.github.io/ctboost/gpu/)
for the released wheel, driver, architecture, and platform matrix. Inspect an
installation with:

```bash
python -c "import ctboost; print(ctboost.build_info())"
```

## Quick start

```python
import numpy as np
from ctboost import CTBoostClassifier

X = np.array(
    [
        [0.2, 1.0],
        [0.8, 0.1],
        [0.1, 0.9],
        [0.9, 0.2],
        [0.3, 0.7],
        [0.7, 0.3],
    ],
    dtype=np.float32,
)
y = np.array([0, 1, 0, 1, 0, 1])

model = CTBoostClassifier(
    iterations=100,
    learning_rate=0.05,
    max_depth=4,
    random_seed=42,
)
model.fit(X, y)

probability = model.predict_proba(X)
labels = model.predict(X)
```

The [getting-started guide](https://captnmarkus.github.io/ctboost/getting-started/)
covers validation sets, early stopping, categorical data, grouped feature
tests, and the low-level API.

## Compact multiclass vector leaves

Use `CTBoostClassifier(multi_strategy="multi_output_tree")` to store one shared
CPU tree per multiclass boosting round, with one score per class in each leaf.
It preserves CTBoost's conditional-inference feature tests and split selection;
the default scalar-tree layout remains unchanged.

See the [vector-leaf guide](https://captnmarkus.github.io/ctboost/guides/vector-leaves/)
for supported workflows, artifact compatibility, and a reproducible comparison.

## Evidence and project status

The [October follow-up archive](https://github.com/captnmarkus/ctboost/tree/master/benchmarks/results/improvements_20261016_followup) separates production correctness,
fixed-model inference measurements, and a twelve-task development accuracy
study. It retains failed hypotheses, resource limits, source identities, and
the distinction between model training and inference runtimes. These local
studies do not establish a new TabArena Elo or a universal accuracy advantage.

The final current-data experiment completed 540 fold fits and 36 selected
full-data fits. CTBoost's median relative error reduction was +0.21% against
XGBoost (7/12 wins) and -0.82% against LightGBM (5/12 wins). The combined
development target was not met, so the gated second-seed and official-test
stages were not run. This release publishes the validated implementation
improvements with that remaining accuracy gap documented.

The [0.1.61 verification archive](https://github.com/captnmarkus/ctboost/tree/833a775/benchmarks/results/release_readiness_20260908)
and [earlier accuracy studies](https://github.com/captnmarkus/ctboost/tree/8cc75aa/benchmarks/results/inference_accuracy_20260908/accuracy)
remain historical evidence. The status recorded on **9 September 2026** was
that a fresh 0.1.61 Full HPO25 evaluation had been prepared but had not started,
and the 0.1.60 run was retired. That record is not a completed 0.1.62 evaluation.
The published author-run 0.1.58 TabArena-Lite bundle is also separate from this
release and is not an official leaderboard entry. The
[benchmark page](https://captnmarkus.github.io/ctboost/benchmarks/) retains its
scores, resource limits, and historical comparisons.

For opt-in CPU learning controls, see
[safeguarded leaves and joint multiclass tests](https://captnmarkus.github.io/ctboost/guides/learning-options/).
The full solver and joint test support 3–32 classes. The
[split-statistics research ledger](https://captnmarkus.github.io/ctboost/split-statistics-research/)
records qualification gates and the literature-to-implementation boundary.

CTBoost is an alpha project. Its API and model formats are tested extensively,
but it does not yet have the independent production history of CatBoost or
XGBoost.

## Documentation

| Topic | Guide |
|---|---|
| Installation and first model | [Getting started](https://captnmarkus.github.io/ctboost/getting-started/) |
| Training, objectives, callbacks, and wrappers | [Training workflows](https://captnmarkus.github.io/ctboost/guides/training/) |
| GPU wheels and runtime requirements | [GPU installation](https://captnmarkus.github.io/ctboost/gpu/) |
| Data, streaming, and schema metadata | [Data input](https://captnmarkus.github.io/ctboost/guides/data-input/) |
| Categorical, text, and embeddings | [Feature preprocessing](https://captnmarkus.github.io/ctboost/guides/categorical-text/) |
| SHAP, influence, and diagnostics | [Explainability](https://captnmarkus.github.io/ctboost/guides/explainability/) |
| Dask, Ray, and Spark | [Distributed training](https://captnmarkus.github.io/ctboost/guides/distributed/) |
| Persistence, exports, and CLI | [Deployment](https://captnmarkus.github.io/ctboost/guides/deployment/) |
| Prepared-feature R/JVM inference | [Portable inference](https://captnmarkus.github.io/ctboost/guides/portable-inference/) |
| Python symbols and signatures | [API reference](https://captnmarkus.github.io/ctboost/reference/api/) |
| Source builds and tests | [Development](https://captnmarkus.github.io/ctboost/development/) |

## Development

```bash
python -m pip install -e ".[dev]"
pytest tests
```

See the [development guide](https://captnmarkus.github.io/ctboost/development/)
for native CMake builds, repository layout, and documentation checks.

## Methodology and license

CTBoost draws on the conditional-inference framework of Hothorn, Hornik, and
Zeileis (2006) and the modular `partykit` work of Hothorn and Zeileis (2015).
The [research ledger](https://captnmarkus.github.io/ctboost/split-statistics-research/)
records the precise literature-to-implementation boundary.

Apache-2.0 licensed. See the
[license text](https://github.com/captnmarkus/ctboost/blob/master/LICENSE).
