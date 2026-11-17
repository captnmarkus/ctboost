"""CatBoost control for the cached TabArena development comparison."""

import numpy as np
import pandas as pd


def _prepare(frame, vocabulary):
    columns = {}
    for name, series in frame.items():
        if name in vocabulary:
            # Known labels have nonnegative training codes. CatBoost accepts
            # integers as categorical values but rejects missing float labels.
            codes = pd.Categorical(series, categories=vocabulary[name]).codes.copy()
            codes[series.isna().to_numpy()] = -2  # Unseen, nonmissing labels keep -1.
            columns[name] = codes
        else:
            columns[name] = series.to_numpy(dtype=np.float32, na_value=np.nan)
    return pd.DataFrame(columns, index=frame.index)


def fit_predict(case, train, stop, dev, seed):
    from catboost import CatBoostClassifier, CatBoostRegressor

    problem = case["problem_type"]
    loss = {"binary": "Logloss", "multiclass": "MultiClass", "regression": "RMSE"}[
        problem
    ]
    frame = train["X"]
    vocabulary = {
        name: pd.unique(series.dropna().astype(object))
        for name, series in frame.items()
        if isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
    }
    model_type = CatBoostRegressor if problem == "regression" else CatBoostClassifier
    model = model_type(
        iterations=200,
        early_stopping_rounds=25,
        loss_function=loss,
        eval_metric=loss,
        random_seed=seed,
        thread_count=1,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        _prepare(frame, vocabulary),
        train["y"],
        cat_features=[frame.columns.get_loc(name) for name in vocabulary],
        eval_set=(_prepare(stop["X"], vocabulary), stop["y"]),
        use_best_model=True,
    )
    evaluation = _prepare(dev["X"], vocabulary)
    if problem == "regression":
        return model.predict(evaluation, thread_count=1), model.tree_count_
    probabilities = model.predict_proba(evaluation, thread_count=1)
    positions = {label: i for i, label in enumerate(model.classes_)}
    order = [positions[label] for label in np.unique(train["y"])]
    probabilities = probabilities[:, order]
    return (
        probabilities[:, 1] if problem == "binary" else probabilities
    ), model.tree_count_
