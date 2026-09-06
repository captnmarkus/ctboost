import os

import numpy as np

import ctboost


def test_build_info_smoke():
    info = ctboost.build_info()

    assert info["version"] == ctboost.__version__
    assert info["package_version"] == ctboost.__version__
    assert ctboost._core.build_info()["version"] == ctboost.__version__
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


def test_sklearn_classifier_smoke():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.1],
            [0.1, 0.8],
            [0.9, 0.2],
            [0.8, 0.9],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    clf = ctboost.CTBoostClassifier(
        iterations=10,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
        task_type="CPU",
    )
    clf.fit(X, y)

    probabilities = clf.predict_proba(X)
    predictions = clf.predict(X)

    assert probabilities.shape == (X.shape[0], 2)
    assert predictions.shape == (X.shape[0],)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)


def test_optional_learning_controls_survive_packaging_and_state_roundtrip():
    # Exercise the new native entry points on every released Python/platform
    # wheel, including CPU training in CUDA-enabled wheels.
    import pickle

    axis = np.repeat(np.arange(3, dtype=np.float32), 30)
    X = np.column_stack([axis, np.tile(np.arange(30, dtype=np.float32), 3)])
    model = ctboost.CTBoostClassifier(
        iterations=3, max_depth=2, task_type="CPU", multi_strategy="multi_output_tree",
        multiclass_feature_test="joint", multiclass_leaf_solver="full",
        leaf_estimation_iterations=3, feature_test="grouped",
    ).fit(X, axis.astype(int))
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (90, 3)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict_proba(X), probabilities)
    assert restored.get_booster().multiclass_feature_test == "joint"
    assert restored.get_booster().multiclass_leaf_solver == "full"

    binary = ctboost.train(X, {
        "objective": "LogLoss", "iterations": 2, "task_type": "CPU",
        "leaf_estimation_backtracking": True, "leaf_estimation_iterations": 3,
    }, label=(axis > 0).astype(np.float32))
    assert binary.leaf_estimation_backtracking is True
    assert np.isfinite(binary.predict(X)).all()
