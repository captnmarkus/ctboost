import numpy as np
import pytest
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


def test_tree_controls_limit_leaf_growth_and_split_gain():
    X = np.linspace(-2.0, 2.0, 64, dtype=np.float32).reshape(-1, 1)
    y = np.where(X[:, 0] > 0.0, 2.0, -2.0).astype(np.float32)
    pool = ctboost.Pool(X, y)

    limited = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.3,
            "max_depth": 4,
            "alpha": 1.0,
            "lambda": 1.0,
            "max_leaves": 2,
        },
        num_boost_round=1,
    )
    limited_state = limited._handle.export_state()
    limited_leaves = sum(1 for node in limited_state["trees"][0]["nodes"] if node["is_leaf"])
    assert limited_leaves <= 2

    blocked = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.3,
            "max_depth": 4,
            "alpha": 1.0,
            "lambda": 1.0,
            "gamma": 1e12,
            "min_data_in_leaf": 40,
        },
        num_boost_round=1,
    )
    blocked_state = blocked._handle.export_state()
    blocked_leaves = sum(1 for node in blocked_state["trees"][0]["nodes"] if node["is_leaf"])
    assert blocked_leaves == 1

    bounded = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.3,
            "max_depth": 4,
            "alpha": 1.0,
            "lambda": 1.0,
            "max_leaf_weight": 0.25,
            "min_samples_split": 100,
        },
        num_boost_round=1,
    )
    bounded_state = bounded._handle.export_state()
    bounded_leaves = [node["leaf_weight"] for node in bounded_state["trees"][0]["nodes"] if node["is_leaf"]]
    assert max(abs(weight) for weight in bounded_leaves) <= 0.25 + 1e-6
    assert len(bounded_leaves) == 1


def test_feature_subsampling_is_seeded_and_deterministic():
    X, y = make_regression(
        n_samples=160,
        n_features=10,
        n_informative=6,
        noise=0.2,
        random_state=11,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    pool = ctboost.Pool(X, y)

    params = {
        "objective": "RMSE",
        "learning_rate": 0.15,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda": 1.0,
        "colsample_bytree": 0.5,
        "random_seed": 23,
    }
    booster_a = ctboost.train(pool, params, num_boost_round=8)
    booster_b = ctboost.train(pool, params, num_boost_round=8)
    preds_a = booster_a.predict(pool)
    preds_b = booster_b.predict(pool)
    np.testing.assert_allclose(preds_a, preds_b, rtol=1e-6, atol=1e-6)

    booster_c = ctboost.train(
        pool,
        {**params, "random_seed": 99},
        num_boost_round=8,
    )
    preds_c = booster_c.predict(pool)
    assert not np.allclose(preds_a, preds_c)


def test_row_subsampling_is_seeded_and_random_forest_mode_trains():
    X, y = make_regression(
        n_samples=180,
        n_features=8,
        n_informative=5,
        noise=0.3,
        random_state=19,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    pool = ctboost.Pool(X, y)

    params = {
        "objective": "RMSE",
        "learning_rate": 0.1,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda": 1.0,
        "subsample": 0.7,
        "bootstrap_type": "Bernoulli",
        "boosting_type": "RandomForest",
        "colsample_bytree": 0.75,
        "random_seed": 29,
    }
    booster_a = ctboost.train(pool, params, num_boost_round=10)
    booster_b = ctboost.train(pool, params, num_boost_round=10)

    preds_a = booster_a.predict(pool)
    preds_b = booster_b.predict(pool)
    np.testing.assert_allclose(preds_a, preds_b, rtol=1e-6, atol=1e-6)
    assert np.mean((preds_a - y) ** 2) < np.mean(y**2)


def test_bayesian_bootstrap_with_bagging_temperature_is_seeded_and_trains():
    X, y = make_regression(
        n_samples=180,
        n_features=8,
        n_informative=5,
        noise=0.3,
        random_state=31,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    pool = ctboost.Pool(X, y)

    params = {
        "objective": "RMSE",
        "learning_rate": 0.1,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda": 1.0,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.5,
        "random_seed": 41,
    }
    booster_a = ctboost.train(pool, params, num_boost_round=10)
    booster_b = ctboost.train(pool, params, num_boost_round=10)

    preds_a = booster_a.predict(pool)
    preds_b = booster_b.predict(pool)
    np.testing.assert_allclose(preds_a, preds_b, rtol=1e-6, atol=1e-6)
    assert np.mean((preds_a - y) ** 2) < np.mean(y**2)


def test_dart_boosting_is_seeded_and_trains():
    X, y = make_regression(
        n_samples=180,
        n_features=8,
        n_informative=5,
        noise=0.25,
        random_state=23,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    pool = ctboost.Pool(X, y)

    params = {
        "objective": "RMSE",
        "learning_rate": 0.1,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda": 1.0,
        "boosting_type": "DART",
        "drop_rate": 0.2,
        "skip_drop": 0.0,
        "max_drop": 2,
        "random_seed": 37,
    }
    booster_a = ctboost.train(pool, params, num_boost_round=12)
    booster_b = ctboost.train(pool, params, num_boost_round=12)
    preds_a = booster_a.predict(pool)
    preds_b = booster_b.predict(pool)

    np.testing.assert_allclose(preds_a, preds_b, rtol=1e-6, atol=1e-6)
    assert np.mean((preds_a - y) ** 2) < np.mean(y**2)


def test_feature_weights_can_disable_a_feature():
    x = np.linspace(-2.0, 2.0, 192, dtype=np.float32)
    X = np.column_stack([x, np.sin(7.0 * x), np.cos(5.0 * x)]).astype(np.float32)
    y = np.where(x > 0.0, 3.0, -3.0).astype(np.float32)
    pool = ctboost.Pool(X, y)

    unrestricted = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=4,
    )
    restricted = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "feature_weights": [0.0, 1.0, 1.0],
        },
        num_boost_round=4,
    )

    restricted_state = restricted._handle.export_state()
    assert all(
        node["split_feature_id"] != 0
        for tree in restricted_state["trees"]
        for node in tree["nodes"]
        if not node["is_leaf"]
    )
    unrestricted_loss = np.mean((unrestricted.predict(pool) - y) ** 2)
    restricted_loss = np.mean((restricted.predict(pool) - y) ** 2)
    assert restricted_loss > unrestricted_loss


def test_leafwise_grow_policy_can_choose_the_better_branch_under_leaf_budget():
    signal = np.repeat(np.array([-1.0, 1.0], dtype=np.float32), 128)
    detail = np.tile(np.repeat(np.array([-1.0, 1.0], dtype=np.float32), 64), 2)
    X = np.column_stack([signal, detail]).astype(np.float32)
    y = np.empty(signal.shape[0], dtype=np.float32)
    y[(signal < 0.0) & (detail < 0.0)] = -3.0
    y[(signal < 0.0) & (detail > 0.0)] = -1.0
    y[(signal > 0.0) & (detail < 0.0)] = 1.0
    y[(signal > 0.0) & (detail > 0.0)] = 9.0
    pool = ctboost.Pool(X, y)

    depthwise = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.3,
            "max_depth": 4,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "max_leaves": 3,
            "grow_policy": "DepthWise",
        },
        num_boost_round=1,
    )
    leafwise = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.3,
            "max_depth": 4,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "max_leaves": 3,
            "grow_policy": "LeafWise",
        },
        num_boost_round=1,
    )

    depthwise_nodes = depthwise._handle.export_state()["trees"][0]["nodes"]
    leafwise_nodes = leafwise._handle.export_state()["trees"][0]["nodes"]
    assert depthwise_nodes[0]["split_feature_id"] == 0
    assert leafwise_nodes[0]["split_feature_id"] == 0
    assert depthwise_nodes[depthwise_nodes[0]["left_child"]]["is_leaf"] is False
    assert depthwise_nodes[depthwise_nodes[0]["right_child"]]["is_leaf"] is True
    assert leafwise_nodes[leafwise_nodes[0]["right_child"]]["is_leaf"] is False
    assert leafwise_nodes[leafwise_nodes[0]["left_child"]]["is_leaf"] is True
    assert np.mean((leafwise.predict(pool) - y) ** 2) < np.mean((depthwise.predict(pool) - y) ** 2)


def test_first_feature_use_penalties_can_steer_first_split():
    rng = np.random.default_rng(79)
    base = rng.normal(size=160).astype(np.float32)
    X = np.column_stack([base, base + 0.01 * rng.normal(size=160)]).astype(np.float32)
    y = base.astype(np.float32)
    pool = ctboost.Pool(X, y)

    booster = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "first_feature_use_penalties": [3.0, 0.0],
            "random_seed": 53,
        },
        num_boost_round=1,
    )
    state = booster._handle.export_state()
    assert state["trees"][0]["nodes"][0]["split_feature_id"] == 1


def test_random_strength_is_seeded():
    rng = np.random.default_rng(83)
    base = rng.normal(size=160).astype(np.float32)
    X = np.column_stack(
        [
            base + 0.01 * rng.normal(size=160),
            base + 0.02 * rng.normal(size=160),
            base + 0.03 * rng.normal(size=160),
        ]
    ).astype(np.float32)
    y = base.astype(np.float32)
    pool = ctboost.Pool(X, y)

    params = {
        "objective": "RMSE",
        "learning_rate": 0.2,
        "max_depth": 2,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "random_strength": 50.0,
        "random_seed": 53,
    }
    booster_a = ctboost.train(pool, params, num_boost_round=4)
    booster_b = ctboost.train(pool, params, num_boost_round=4)
    booster_c = ctboost.train(pool, {**params, "random_seed": 61}, num_boost_round=4)
    state_a = booster_a._handle.export_state()
    state_b = booster_b._handle.export_state()
    state_c = booster_c._handle.export_state()

    assert state_a["trees"][0]["nodes"][0]["split_feature_id"] == state_b["trees"][0]["nodes"][0]["split_feature_id"]
    assert state_a["trees"][0]["nodes"][0]["split_feature_id"] != state_c["trees"][0]["nodes"][0]["split_feature_id"]


def test_poisson_and_tweedie_regression_train_with_finite_predictions():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(128, 4)).astype(np.float32)
    linear = 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]
    y_count = np.exp(linear).astype(np.float32)
    y_tweedie = (np.exp(linear) + 0.1).astype(np.float32)

    poisson = ctboost.CTBoostRegressor(
        iterations=12,
        learning_rate=0.2,
        max_depth=2,
        loss_function="Poisson",
    )
    poisson.fit(X, y_count)
    poisson_pred = poisson.predict(X)
    assert np.all(np.isfinite(poisson_pred))

    tweedie = ctboost.CTBoostRegressor(
        iterations=12,
        learning_rate=0.15,
        max_depth=2,
        loss_function="Tweedie",
        tweedie_variance_power=1.4,
    )
    tweedie.fit(X, y_tweedie)
    tweedie_pred = tweedie.predict(X)
    assert np.all(np.isfinite(tweedie_pred))


def test_survival_regression_improves_concordance():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(220, 4)).astype(np.float32)
    risk = 0.8 * X[:, 0] - 0.5 * X[:, 1]
    event_time = np.exp(1.1 - risk) + 0.05 * rng.random(X.shape[0])
    censor_time = np.quantile(event_time, 0.7)
    observed = event_time <= censor_time
    signed_time = np.where(observed, event_time, -np.minimum(event_time, censor_time)).astype(np.float32)

    booster = ctboost.train(
        ctboost.Pool(X, signed_time),
        {
            "objective": "Cox",
            "eval_metric": "CIndex",
            "learning_rate": 0.15,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=18,
    )
    predictions = booster.predict(X)
    assert np.all(np.isfinite(predictions))

    comparable = 0.0
    concordant = 0.0
    observed_time = np.abs(signed_time)
    for i in range(X.shape[0]):
        if signed_time[i] <= 0.0:
            continue
        for j in range(X.shape[0]):
            if observed_time[j] <= observed_time[i]:
                continue
            comparable += 1.0
            if predictions[i] > predictions[j]:
                concordant += 1.0
            elif predictions[i] == predictions[j]:
                concordant += 0.5
    assert comparable > 0.0
    assert concordant / comparable > 0.6


def test_monotone_constraints_preserve_prediction_order():
    rng = np.random.default_rng(13)
    X = np.linspace(-3.0, 3.0, 160, dtype=np.float32).reshape(-1, 1)
    y = (2.0 * X[:, 0] + rng.normal(scale=0.4, size=X.shape[0])).astype(np.float32)
    pool = ctboost.Pool(X, y)

    booster = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 3,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "monotone_constraints": [1],
        },
        num_boost_round=16,
    )
    predictions = booster.predict(pool)
    assert np.all(np.diff(predictions) >= -1e-6)


def test_interaction_constraints_limit_features_per_path():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(192, 4)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]).astype(np.float32)
    pool = ctboost.Pool(X, y)

    booster = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 3,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "interaction_constraints": [[0, 1], [2, 3]],
        },
        num_boost_round=6,
    )

    state = booster._handle.export_state()
    allowed_groups = [set(group) for group in state["interaction_constraints"]]

    def walk_paths(nodes, node_index=0, path=None):
        if path is None:
            path = []
        node = nodes[node_index]
        if node["is_leaf"]:
            yield list(path)
            return
        feature_id = node["split_feature_id"]
        next_path = path + [feature_id]
        yield from walk_paths(nodes, node["left_child"], next_path)
        yield from walk_paths(nodes, node["right_child"], next_path)

    for tree in state["trees"]:
        for path in walk_paths(tree["nodes"]):
            constrained_features = {feature for feature in path if any(feature in group for group in allowed_groups)}
            if not constrained_features:
                continue
            assert any(constrained_features.issubset(group) for group in allowed_groups)


def test_gpu_training_supports_leafwise_regularization_and_constraints_when_available():
    if not ctboost.build_info()["cuda_enabled"]:
        pytest.skip("CUDA support is not compiled into this build")

    rng = np.random.default_rng(97)
    X = rng.normal(size=(160, 4)).astype(np.float32)
    y = (1.5 * X[:, 0] + 0.4 * X[:, 1] * X[:, 2] - 0.2 * X[:, 3]).astype(np.float32)
    pool = ctboost.Pool(X, y)

    try:
        booster = ctboost.train(
            pool,
            {
                "objective": "RMSE",
                "learning_rate": 0.15,
                "max_depth": 3,
                "alpha": 1.0,
                "lambda_l2": 1.0,
                "task_type": "GPU",
                "devices": "0",
                "feature_weights": [1.0, 0.75, 1.0, 0.5],
                "first_feature_use_penalties": [0.0, 0.0, 0.2, 0.0],
                "random_strength": 0.3,
                "grow_policy": "LeafWise",
                "max_leaves": 4,
                "monotone_constraints": [1, 0, 0, 0],
                "interaction_constraints": [[0, 1, 2], [3]],
            },
            num_boost_round=6,
        )
    except RuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable for GPU-control smoke test: {exc}")

    predictions = booster.predict(pool)
    assert predictions.shape == (X.shape[0],)
    assert np.all(np.isfinite(predictions))
