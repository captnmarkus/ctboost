"""LightGBM control for the cached TabArena development comparison."""

import numpy as np
import pandas as pd


def prepare_native_frames(train_frame, stop_frame, dev_frame):
    """Encode observed training labels in a shared categorical dtype for all roles."""
    roles = (train_frame, stop_frame, dev_frame)
    columns = [{} for _ in roles]
    for i, (name, series) in enumerate(train_frame.items()):
        key = f"f{i}"
        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ):
            levels = pd.unique(series.dropna().astype(object))
            for role, converted in zip(roles, columns):
                codes = pd.Categorical(role[name], categories=levels).codes
                converted[key] = pd.Categorical.from_codes(
                    codes, categories=range(len(levels))
                )
        else:
            for role, converted in zip(roles, columns):
                converted[key] = role[name].to_numpy(dtype=np.float32, na_value=np.nan)
    return tuple(
        pd.DataFrame(converted, index=role.index)
        for role, converted in zip(roles, columns)
    )


def fit_predict(case, train, stop, dev, seed):
    import lightgbm as lgb

    problem = case["problem_type"]
    metric = {
        "binary": "binary_logloss",
        "multiclass": "multi_logloss",
        "regression": "rmse",
    }[problem]
    X, X_stop, X_dev = prepare_native_frames(train["X"], stop["X"], dev["X"])
    model_type = lgb.LGBMRegressor if problem == "regression" else lgb.LGBMClassifier
    model = model_type(
        n_estimators=200,
        metric=metric,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(
        X,
        train["y"],
        eval_set=[(X_stop, stop["y"])],
        callbacks=[lgb.early_stopping(25, first_metric_only=True, verbose=False)],
    )
    retained = min(model.best_iteration_, model.booster_.current_iteration())
    if problem == "regression":
        return model.predict(X_dev, num_iteration=retained), retained
    probabilities = model.predict_proba(X_dev, num_iteration=retained)
    positions = {label: i for i, label in enumerate(model.classes_)}
    probabilities = probabilities[
        :, [positions[label] for label in np.unique(train["y"])]
    ]
    return (probabilities[:, 1] if problem == "binary" else probabilities), retained
