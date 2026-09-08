"""Training updates must match inference on the same compiler and input rows."""

import numpy as np
import pytest

import ctboost


@pytest.mark.parametrize(
    "objective,multi_strategy",
    [
        ("RMSE", "one_output_per_tree"),
        ("LogLoss", "one_output_per_tree"),
        ("MultiClass", "one_output_per_tree"),
        ("MultiClass", "multi_output_tree"),
    ],
)
@pytest.mark.parametrize("root_only", [False, True])
def test_training_eval_and_resumed_predictions_use_the_same_arithmetic(
    objective, multi_strategy, root_only
):
    rng = np.random.default_rng(471)
    X = rng.normal(size=(96, 4)).astype(np.float32)
    if objective == "MultiClass":
        y = np.argmax(X[:, :3], axis=1).astype(np.float32)
        base_score = [0.17, -0.31, 0.23]
    else:
        y = 2.7 * X[:, 0] - 1.3 * X[:, 1] + 0.21 * X[:, 2]
        if objective == "LogLoss":
            y = (y > 0).astype(np.float32)
        base_score = [-0.34948291]
    pool = ctboost.Pool(
        X,
        y,
        weight=np.linspace(0.3, 1.7, 96, dtype=np.float32),
    )
    params = {
        "objective": objective,
        "learning_rate": 0.123456789,
        "max_depth": 2,
        "min_data_in_leaf": 1000 if root_only else 0,
        "alpha": 1.0,
        "random_seed": 19,
        "base_score": base_score,
    }
    if objective == "MultiClass":
        params.update(num_classes=3, multi_strategy=multi_strategy)
    full = ctboost.train(pool, params, num_boost_round=7, eval_set=pool)
    initial = ctboost.train(pool, params, num_boost_round=3, eval_set=pool)
    continued = ctboost.train(
        pool, params, num_boost_round=4, eval_set=pool, init_model=initial
    )

    # Identical rows/labels/weights expose an arithmetic mismatch directly,
    # before a different gradient can change any later tree's split decision.
    np.testing.assert_array_equal(full.loss_history, full.eval_loss_history)
    np.testing.assert_array_equal(continued.loss_history, continued.eval_loss_history)
    np.testing.assert_array_equal(continued.loss_history, full.loss_history)
    np.testing.assert_array_equal(continued.predict(pool), full.predict(pool))
    assert (
        continued._handle.export_state()["trees"]
        == full._handle.export_state()["trees"]
    )
