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

CTBoost 0.1.64 adds expectile regression for asymmetric prediction costs and
can select a boosting-round count for refitting from cross-validation. It also
fixes the handling of pandas categorical columns with identical display names.
[Release notes](https://captnmarkus.github.io/ctboost/release-0.1.64/) and
[training examples](https://captnmarkus.github.io/ctboost/guides/training/)

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

On 17 November 2026, the TabArena models-only leaderboard reported **1177 Elo**
for CTBoost's default, **1298** for tuned CTBoost and **1327** for the tuned
ensemble across all tasks, datasets and splits. All three entries are verified
with 0% imputation. The Space does not identify a package version in these
rows. [TabArena results at the checked revision](https://huggingface.co/spaces/TabArena/leaderboard/blob/8ce4c3037c2b8f9385f6b35375776f610af2e9dd/data/entrants_models/imputation_yes/splits_all/tasks_all/datasets_all/website_leaderboard.csv)

CTBoost ranked fourth among CatBoost, LightGBM, XGBoost and CTBoost by Elo
in each configuration in that snapshot. The public 0.1.63 wheel ranked third on 51 local
TabArena development splits, ahead of XGBoost and behind CatBoost and LightGBM.
All four models used one seed and a 200-round cap. [Dataset results and methodology](https://github.com/captnmarkus/ctboost/blob/702058fe652ae31312aba9189c8530ba47ab4027/benchmarks/tabarena/FOUR_MODEL_DEVELOPMENT.md)

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
