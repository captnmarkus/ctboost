import os

import numpy as np

import ctboost


def test_build_info_smoke():
    info = ctboost.build_info()

    assert info["version"] == ctboost.__version__
    assert isinstance(info["cuda_enabled"], bool)
    assert isinstance(info["compiler"], str)
    expected_cuda = os.environ.get("CTBOOST_EXPECT_CUDA", "0") == "1"
    assert info["cuda_enabled"] is expected_cuda


def test_low_level_training_smoke():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.9, 0.8],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 1.0, 1.0, 2.0, 0.3, 1.1, 1.2, 1.9], dtype=np.float32)

    pool = ctboost.Pool(X, y)
    booster = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "iterations": 12,
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "task_type": "CPU",
        },
        num_boost_round=12,
    )

    predictions = booster.predict(pool)

    assert predictions.shape == (X.shape[0],)
    assert np.isfinite(predictions).all()
    assert booster.loss_history
