import numpy as np
import pytest
from sklearn.datasets import make_regression
import ctboost

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
