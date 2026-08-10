# Getting started

## Install

```bash
python -m pip install -U ctboost
```

The base runtime depends only on NumPy. Install optional integrations explicitly:

```bash
python -m pip install "ctboost[sklearn,plot,dataframes]"
```

See [GPU installation](gpu.md) for the CUDA support matrix.

## Classification

```python
from ctboost import CTBoostClassifier

model = CTBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    max_depth=6,
    random_seed=7,
)
model.fit(X_train, y_train)

labels = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## Regression

```python
from ctboost import CTBoostRegressor

model = CTBoostRegressor(
    loss_function="RMSE",
    iterations=500,
    learning_rate=0.05,
)
model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=40,
)
predictions = model.predict(X_test)
```

## pandas categoricals

Keep categorical values as strings or pandas `category` columns and identify them
by name or position:

```python
model = CTBoostClassifier(
    cat_features=["country", "plan"],
    ordered_ctr=True,
)
model.fit(frame, target)
```

The fitted feature pipeline is stored with the estimator, including category
dictionaries, CTR statistics, text dictionaries, and embedding transforms.

## Check the build

```python
import ctboost

print(ctboost.__version__)
print(ctboost.build_info())
```

`build_info()` reports the compiled version, compiler, C++ standard, and whether
the installed wheel contains CUDA support.
