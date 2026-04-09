import numpy as np
from sklearn.datasets import make_regression

import ctboost


def test_booster_reduces_training_loss_on_regression():
    X, y = make_regression(
        n_samples=128,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=7,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    pool = ctboost.Pool(X, y)
    booster = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda": 1.0,
            "max_bins": 64,
        },
        num_boost_round=10,
    )

    predictions = booster.predict(pool)
    baseline_loss = np.mean(y**2)
    trained_loss = np.mean((predictions - y) ** 2)

    assert booster.loss_history
    assert booster.loss_history[-1] < booster.loss_history[0]
    assert trained_loss < baseline_loss
