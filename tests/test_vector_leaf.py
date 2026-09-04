"""Vector storage must preserve CTBoost's existing conditional tree learner."""

import copy
import json
import pickle

import numpy as np
import pytest
from sklearn.base import clone

import ctboost


def _data():
    rng = np.random.default_rng(418)
    X = rng.normal(size=(180, 5)).astype(np.float32)
    y = np.argmax(np.column_stack([X[:, 0] - X[:, 1], X[:, 1], -X[:, 0]]), axis=1)
    return X, y.astype(np.float32)


def _params(**overrides):
    return {
        "objective": "MultiClass",
        "num_classes": 3,
        "max_depth": 3,
        "alpha": 1.0,
        "learning_rate": 0.15,
        "random_seed": 11,
        **overrides,
    }


def _topology(tree):
    return [
        {k: v for k, v in node.items() if k not in {"leaf_weight", "leaf_weights"}}
        for node in tree["nodes"]
    ]


@pytest.mark.parametrize(
    "controls",
    [
        {},
        {"alpha": 0.05},
        {"bootstrap_type": "Bernoulli", "subsample": 0.8, "colsample_bytree": 0.8},
        {"boosting_type": "DART", "drop_rate": 1.0, "skip_drop": 0.0, "max_drop": 1},
        {"grow_policy": "LeafWise", "max_leaves": 4},
        {
            "feature_weights": [0.0, 1.0, 1.0, 1.0, 1.0],
            "interaction_constraints": [[1, 2], [3, 4]],
        },
    ],
)
def test_vector_leaf_preserves_topology_predictions_and_conditional_controls(controls):
    X, y = _data()
    weight = np.linspace(0.1, 2.0, len(y), dtype=np.float32)
    pool = ctboost.Pool(X, y, weight=weight)
    scalar = ctboost.train(pool, _params(**controls), num_boost_round=8)
    vector = ctboost.train(
        pool, _params(multi_strategy="multi_output_tree", **controls), num_boost_round=8
    )
    scalar_trees = scalar._handle.export_state()["trees"]
    vector_trees = vector._handle.export_state()["trees"]
    assert len(scalar_trees) == 24
    assert len(vector_trees) == vector.num_iterations_trained == 8
    for iteration, tree in enumerate(vector_trees):
        for output in range(3):
            scalar_tree = scalar_trees[iteration * 3 + output]
            assert _topology(tree) == _topology(scalar_tree)
            np.testing.assert_array_equal(
                [node["leaf_weights"][output] for node in tree["nodes"]],
                [node["leaf_weight"] for node in scalar_tree["nodes"]],
            )
    np.testing.assert_array_equal(vector.predict(X), scalar.predict(X))
    np.testing.assert_array_equal(vector.loss_history, scalar.loss_history)
    np.testing.assert_allclose(vector.feature_importances_, scalar.feature_importances_)
    np.testing.assert_array_equal(
        vector.predict_leaf_index(X), scalar.predict_leaf_index(X)[:, ::3]
    )
    for iteration, prediction in enumerate(vector.staged_predict(X), start=1):
        np.testing.assert_array_equal(
            prediction, scalar.predict(X, num_iteration=iteration)
        )
        assert vector.predict_leaf_index(X, num_iteration=iteration).shape == (
            len(X),
            iteration,
        )
    np.testing.assert_allclose(
        vector.predict_contrib(X), scalar.predict_contrib(X), atol=1e-6
    )
    np.testing.assert_allclose(
        vector.predict_contrib(X).sum(axis=-1), vector.predict(X), atol=1e-6
    )


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_vector_leaf_preserves_categorical_missing_sparse_and_baseline_routing(
    nan_mode,
):
    sparse = pytest.importorskip("scipy.sparse")
    X, y = _data()
    X[:, 0] = np.round(X[:, 0])
    X[::9, 1] = np.nan
    baseline = np.tile(np.asarray([0.2, -0.1, 0.4], dtype=np.float32), (len(X), 1))
    pool = ctboost.Pool(X, y, cat_features=[0], baseline=baseline)
    params = _params(nan_mode=nan_mode)
    scalar = ctboost.train(pool, params, num_boost_round=5)
    vector = ctboost.train(
        pool, {**params, "multi_strategy": "multi_output_tree"}, num_boost_round=5
    )
    np.testing.assert_array_equal(vector.predict(pool), scalar.predict(pool))
    sparse_pool = ctboost.Pool(
        sparse.csr_matrix(X), cat_features=[0], baseline=baseline
    )
    np.testing.assert_array_equal(vector.predict(sparse_pool), scalar.predict(pool))
    np.testing.assert_allclose(
        vector.predict_contrib(pool).sum(axis=-1), vector.predict(pool), atol=1e-6
    )


@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
def test_multiclass_final_leaf_values_respect_max_leaf_weight(strategy):
    X = np.repeat(np.eye(3, dtype=np.float32), 20, axis=0)
    y = np.repeat(np.arange(3), 20).astype(np.float32)
    model = ctboost.train(
        X,
        _params(multi_strategy=strategy, max_leaf_weight=0.01),
        label=y,
        num_boost_round=1,
    )
    for tree in model._handle.export_state()["trees"]:
        for node in tree["nodes"]:
            if node["is_leaf"]:
                weights = node.get("leaf_weights") or [node["leaf_weight"]]
                assert max(abs(v) for v in weights) <= 0.01000001


def test_vector_leaf_warm_start_snapshot_and_versioned_persistence(tmp_path):
    X, y = _data()
    pool = ctboost.Pool(X, y)
    params = _params(multi_strategy="multi_output_tree")
    full = ctboost.train(pool, params, num_boost_round=7)
    partial = ctboost.train(pool, params, num_boost_round=3)
    for suffix in ["json", "pkl"]:
        path = tmp_path / ("vector." + suffix)
        partial.save_model(path)
        loaded = ctboost.load_model(path)
        assert loaded.multi_strategy == "multi_output_tree"
        resumed = ctboost.train(pool, {}, num_boost_round=4, init_model=loaded)
        np.testing.assert_array_equal(resumed.predict(X), full.predict(X))
        snapshot = ctboost.train(
            pool, params, num_boost_round=7, resume_from_snapshot=path
        )
        np.testing.assert_array_equal(snapshot.predict(X), full.predict(X))
    assert json.loads((tmp_path / "vector.json").read_text())["schema_version"] == 2
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(full)).predict(X), full.predict(X)
    )
    with pytest.raises(ValueError, match="multi_strategy"):
        ctboost.train(
            pool,
            {"multi_strategy": "one_output_per_tree"},
            num_boost_round=1,
            init_model=partial,
        )
    scalar = ctboost.train(pool, _params(), num_boost_round=2)
    legacy_state = dict(scalar._handle.export_state())
    legacy_state.pop("multi_strategy", None)
    legacy_state.pop("format_version", None)
    legacy = ctboost.Booster(ctboost._core.GradientBooster.from_state(legacy_state))
    np.testing.assert_array_equal(legacy.predict(X), scalar.predict(X))
    scalar.save_model(tmp_path / "scalar.json")
    assert json.loads((tmp_path / "scalar.json").read_text())["schema_version"] == 1


@pytest.mark.parametrize("python_surface", [False, True])
def test_vector_leaf_early_stopping_prunes_physical_trees(python_surface):
    # A fixed prediction (no splits, balanced labels and fixed margins) has a
    # constant validation loss, so only the first round should be retained.
    X = np.zeros((90, 2), dtype=np.float32)
    pool = ctboost.Pool(X, np.tile(np.arange(3), 30).astype(np.float32))
    params = _params(multi_strategy="multi_output_tree", boost_from_average=False)
    kwargs = {"callbacks": [lambda env: False]} if python_surface else {}
    model = ctboost.train(
        pool,
        params,
        num_boost_round=10,
        eval_set=pool,
        early_stopping_rounds=2,
        **kwargs,
    )
    assert model.num_iterations_trained == model.best_iteration + 1 == 1
    assert len(model._handle.export_state()["trees"]) == 1
    assert model.predict_leaf_index(X).shape == (len(X), 1)
    assert len(model.learning_rate_history) == 1


def test_vector_classifier_clone_pipeline_and_estimator_persistence(tmp_path):
    pd = pytest.importorskip("pandas")
    X, y = _data()
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])
    frame["category"] = np.where(X[:, 0] > 0, "positive", "negative")
    model = ctboost.CTBoostClassifier(
        iterations=6,
        max_depth=2,
        alpha=1.0,
        multi_strategy="multi_output_tree",
        cat_features=["category"],
    )
    assert clone(model).multi_strategy == "multi_output_tree"
    model.fit(frame, y.astype(int))
    assert model.get_booster().multi_strategy == "multi_output_tree"
    assert model.apply(frame).shape == (len(frame), 6)
    np.testing.assert_allclose(model.predict_proba(frame).sum(axis=1), 1.0, atol=1e-6)
    path = tmp_path / "classifier.json"
    model.save_model(path)
    assert json.loads(path.read_text())["schema_version"] == 2
    loaded = ctboost.CTBoostClassifier.load_model(path)
    np.testing.assert_array_equal(
        loaded.predict_proba(frame), model.predict_proba(frame)
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"multi_strategy": "unknown"}, "multi_strategy"),
        ({"objective": "RMSE", "num_classes": 1}, "multiclass|MultiClass"),
        ({"task_type": "GPU"}, "CPU"),
        ({"distributed_world_size": 2}, "distributed|single"),
    ],
)
def test_vector_leaf_rejects_unsupported_configuration(overrides, match):
    X, y = _data()
    params = {**_params(multi_strategy="multi_output_tree"), **overrides}
    with pytest.raises((ValueError, RuntimeError), match=match):
        ctboost.train(X, params, label=y, num_boost_round=1)


@pytest.mark.parametrize(
    "corruption",
    ["width", "nonfinite", "missing_version", "future_version", "strategy"],
)
def test_vector_state_rejects_inconsistent_layout(corruption):
    X, y = _data()
    model = ctboost.train(
        X, _params(multi_strategy="multi_output_tree"), label=y, num_boost_round=1
    )
    state = copy.deepcopy(model._handle.export_state())
    node = next(n for n in state["trees"][0]["nodes"] if n["is_leaf"])
    if corruption == "width":
        node["leaf_weights"] = [1.0]
    elif corruption == "nonfinite":
        node["leaf_weights"][0] = float("nan")
    elif corruption == "missing_version":
        state.pop("format_version")
    elif corruption == "future_version":
        state["format_version"] = 99
    else:
        state["multi_strategy"] = "one_output_per_tree"
    with pytest.raises(
        (ValueError, RuntimeError),
        match="dimension|leaf_weights|vector|format_version|tree count",
    ):
        ctboost._core.GradientBooster.from_state(state)


def test_vector_leaf_custom_objective_and_schedule_match_scalar():
    X, y = _data()

    def derivatives(prediction, label):
        shifted = prediction - prediction.max(axis=1, keepdims=True)
        prob = np.exp(shifted)
        prob /= prob.sum(axis=1, keepdims=True)
        target = np.eye(3)[label.astype(int)]
        return prob - target, np.maximum(2 * prob * (1 - prob), 1e-6)

    kwargs = {
        "label": y,
        "num_boost_round": 4,
        "obj": derivatives,
        "learning_rate_schedule": [0.2, 0.15, 0.1, 0.05],
    }
    scalar = ctboost.train(X, _params(), **kwargs)
    vector = ctboost.train(X, _params(multi_strategy="multi_output_tree"), **kwargs)
    np.testing.assert_array_equal(vector.predict(X), scalar.predict(X))
    assert vector.learning_rate_history == scalar.learning_rate_history


@pytest.mark.parametrize(
    "strategy,multiplier", [("one_output_per_tree", 3), ("multi_output_tree", 1)]
)
def test_empty_leaf_indices_preserve_physical_tree_dimension(strategy, multiplier):
    X, y = _data()
    model = ctboost.train(
        X, _params(multi_strategy=strategy), label=y, num_boost_round=4
    )
    assert model.predict_leaf_index(X[:0]).shape == (0, 4 * multiplier)
    assert model.predict_leaf_index(X[:0], num_iteration=2).shape == (0, 2 * multiplier)


def test_external_memory_vector_dart_and_warm_start_match_scalar(tmp_path):
    X, y = _data()
    controls = {
        "boosting_type": "DART",
        "drop_rate": 1.0,
        "skip_drop": 0.0,
        "external_memory": True,
        "external_memory_dir": str(tmp_path / "histograms"),
    }
    scalar = ctboost.train(X, _params(**controls), label=y, num_boost_round=4)
    partial = ctboost.train(
        X,
        _params(multi_strategy="multi_output_tree", **controls),
        label=y,
        num_boost_round=2,
    )
    vector = ctboost.train(X, {}, label=y, num_boost_round=2, init_model=partial)
    np.testing.assert_array_equal(vector.predict(X), scalar.predict(X))
