import numpy as np
import pytest
from sklearn.datasets import make_regression

import ctboost


def test_regressor_handles_missing_values_and_predict_time_nans():
    X, y = make_regression(
        n_samples=180,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=53,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X[::9, 0] = np.nan
    X[5::13, 2] = np.nan

    reg = ctboost.CTBoostRegressor(
        iterations=14,
        learning_rate=0.15,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    reg.fit(X, y)

    predictions = reg.predict(X)
    assert np.all(np.isfinite(predictions))

    predict_rows = X[:8].copy()
    predict_rows[:, 4] = np.nan
    predict_predictions = reg.predict(predict_rows)
    assert np.all(np.isfinite(predict_predictions))


def test_nan_mode_forbidden_rejects_missing_values():
    X, y = make_regression(
        n_samples=96,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=59,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X[0, 1] = np.nan

    with pytest.raises(ValueError, match="nan_mode='Forbidden'"):
        ctboost.train(
            ctboost.Pool(X, y),
            {
                "objective": "RMSE",
                "learning_rate": 0.2,
                "max_depth": 2,
                "alpha": 1.0,
                "lambda_l2": 1.0,
                "nan_mode": "Forbidden",
            },
            num_boost_round=8,
        )
