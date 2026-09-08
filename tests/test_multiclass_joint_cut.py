"""Joint cut scoring must not replace conditional feature selection."""

import copy
import pickle

import numpy as np
import pytest
from sklearn.base import clone

import ctboost


def _hidden_signal():
    x = np.repeat([0.0, 1.0], 200).astype(np.float32)[:, None]
    y = np.concatenate([np.tile([0, 1], 100), np.tile([0, 2], 100)])
    return x, y.astype(np.float32)


def _data():
    rng = np.random.default_rng(20260908)
    x = rng.normal(size=(240, 5)).astype(np.float32)
    y = np.argmax(np.column_stack([x[:, 0], x[:, 1], -x[:, 0] - x[:, 1]]), axis=1)
    return x, y.astype(np.float32)


def _params(**kwargs):
    return dict(
        objective="MultiClass",
        num_classes=3,
        max_depth=2,
        alpha=0.05,
        lambda_l2=1.0,
        learning_rate=0.2,
        random_seed=19,
        multiclass_split_score="joint",
        **kwargs,
    )


def _fit(x, y, rounds=3, **kwargs):
    return ctboost.train(x, _params(**kwargs), label=y, num_boost_round=rounds)


def _trees(model):
    return model._handle.export_state()["trees"]


def _bits_equal(left, right):
    np.testing.assert_array_equal(
        np.asarray(left, dtype=np.float32).view(np.uint32),
        np.asarray(right, dtype=np.float32).view(np.uint32),
    )


def _topology(tree):
    return [
        {k: v for k, v in node.items() if k not in {"leaf_weight", "leaf_weights"}}
        for node in tree["nodes"]
    ]


def test_joint_cut_recovers_positive_all_class_gain_after_joint_feature_acceptance():
    x, y = _hidden_signal()
    params = _params(multiclass_feature_test="joint")
    scalar = ctboost.train(
        x, {**params, "multiclass_split_score": "scalar"}, label=y, num_boost_round=1
    )
    joint = ctboost.train(x, params, label=y, num_boost_round=1)
    assert _trees(scalar)[0]["nodes"][0]["is_leaf"]
    assert not _trees(joint)[0]["nodes"][0]["is_leaf"]
    assert _trees(joint)[0]["nodes"][0]["split_feature_id"] == 0
    # Class 0: G_left=G_right=0. Classes 1/2: |G_child|=50,
    # H_child=200*.25*.75=37.5. No half factor in CTBoost's gain.
    expected_gain = 4 * 50**2 / (37.5 + 1)
    assert _trees(joint)[0]["feature_importances"][0] == pytest.approx(
        expected_gain, rel=2e-6
    )
    assert joint.loss_history[-1] < scalar.loss_history[-1]


def test_joint_cut_keeps_conditional_rejection_and_feature_winner():
    x, y = _hidden_signal()
    # The unchanged single-coordinate conditional test cannot see this signal.
    rejected = _fit(x, y, rounds=1)
    assert _trees(rejected)[0]["nodes"][0]["is_leaf"]
    # The binary feature has the lower scalar conditional p-value; the second
    # feature perfectly separates all classes and has a larger joint gain.
    y = np.tile([0, 0, 1, 2], 60).astype(np.float32)
    x = np.column_stack([y == 0, y]).astype(np.float32)
    joint = _fit(x, y, rounds=1)
    params = {**_params(), "multiclass_split_score": "scalar"}
    scalar = ctboost.train(x, params, label=y, num_boost_round=1)
    assert _trees(joint)[0]["nodes"][0]["split_feature_id"] == 0
    assert _trees(scalar)[0]["nodes"][0]["split_feature_id"] == 0


@pytest.mark.parametrize(
    "control",
    [
        {},
        {"multiclass_feature_test": "joint"},
        {"feature_test": "grouped", "feature_test_adjustment": "bonferroni"},
        {"bootstrap_type": "Bernoulli", "subsample": 0.8, "colsample_bytree": 0.8},
        {"bootstrap_type": "Bayesian", "bagging_temperature": 0.5},
        {"grow_policy": "LeafWise", "max_leaves": 4},
        {"boosting_type": "DART", "drop_rate": 0.8, "skip_drop": 0.0},
        {"multiclass_leaf_solver": "full", "leaf_estimation_iterations": 2},
    ],
)
def test_joint_cut_scalar_vector_storage_and_prefix_parity(control):
    x, y = _data()
    scalar = _fit(x, y, **control)
    vector = _fit(x, y, multi_strategy="multi_output_tree", **control)
    for index, tree in enumerate(_trees(vector)):
        for k in range(3):
            scalar_tree = _trees(scalar)[index * 3 + k]
            assert _topology(tree) == _topology(scalar_tree)
            _bits_equal(
                [node["leaf_weights"][k] for node in tree["nodes"]],
                [node["leaf_weight"] for node in scalar_tree["nodes"]],
            )
    for rounds in (1, 2, 3):
        _bits_equal(
            scalar.predict(x, num_iteration=rounds),
            vector.predict(x, num_iteration=rounds),
        )
    _bits_equal(scalar.loss_history, vector.loss_history)


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_joint_cut_categorical_missing_weight_and_sparse_parity(nan_mode):
    sparse = pytest.importorskip("scipy.sparse")
    x, y = _data()
    x[:, 0] = np.round(x[:, 0])
    x[::11, 1] = np.nan
    weights = np.linspace(0.25, 1.75, len(y), dtype=np.float32)
    baseline = np.tile(np.array([0.1, -0.2, 0.3], dtype=np.float32), (len(y), 1))
    pool = ctboost.Pool(x, y, weight=weights, cat_features=[0], baseline=baseline)
    params = _params(nan_mode=nan_mode)
    scalar = ctboost.train(pool, params, num_boost_round=4)
    vector = ctboost.train(
        pool, {**params, "multi_strategy": "multi_output_tree"}, num_boost_round=4
    )
    _bits_equal(scalar.predict(pool), vector.predict(pool))
    query = ctboost.Pool(sparse.csr_matrix(x), cat_features=[0], baseline=baseline)
    _bits_equal(scalar.predict(pool), vector.predict(query))
    assert any(
        node["is_categorical_split"]
        for tree in _trees(scalar)
        for node in tree["nodes"]
    )
    with pytest.raises(ValueError, match="integer frequency weights"):
        ctboost.train(
            pool, {**params, "multiclass_feature_test": "joint"}, num_boost_round=1
        )


@pytest.mark.parametrize(
    "guard", [{"min_data_in_leaf": 201}, {"min_child_weight": 51}, {"gamma": 260}]
)
def test_joint_cut_keeps_child_and_gain_guards(guard):
    x, y = _hidden_signal()
    model = _fit(x, y, rounds=1, multiclass_feature_test="joint", **guard)
    assert _trees(model)[0]["nodes"][0]["is_leaf"]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"multiclass_split_score": "unknown"}, "multiclass_split_score"),
        ({"objective": "RMSE", "num_classes": 1}, "3-32"),
        ({"objective": "LogLoss", "num_classes": 2}, "3-32"),
        ({"num_classes": 33}, "3-32"),
        ({"task_type": "GPU"}, "CPU"),
        ({"distributed_world_size": 2}, "CPU"),
        ({"monotone_constraints": [0]}, "constraints"),
        ({"interaction_constraints": [[0]]}, "constraints"),
        ({"feature_weights": [1.0]}, "feature weights"),
        ({"first_feature_use_penalties": [0.0]}, "first-use"),
        ({"random_strength": 0.1}, "random_strength"),
        ({"max_leaf_weight": 0.1}, "max_leaf_weight"),
    ],
)
def test_joint_cut_rejects_unsupported_native_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ctboost._core.GradientBooster(**{**_params(), **kwargs})


@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
def test_joint_cut_state_pickle_export_resume_and_config_guard(tmp_path, strategy):
    x, y = _data()
    params = _params(multi_strategy=strategy)
    first = ctboost.train(x, params, label=y, num_boost_round=2)
    path = tmp_path / "model.json"
    first.save_model(path)
    loaded = ctboost.Booster.load_model(path)
    assert loaded.multiclass_split_score == "joint"
    predictor_path = tmp_path / "predictor.json"
    first.export_model(predictor_path, export_format="json_predictor")
    predictor = ctboost.load_exported_predictor(predictor_path)
    np.testing.assert_allclose(
        predictor.predict(x), first.predict(x), rtol=1e-6, atol=1e-6
    )
    _bits_equal(first.predict(x), loaded.predict(x))
    _bits_equal(first.predict(x), pickle.loads(pickle.dumps(first)).predict(x))
    resumed = ctboost.train(x, params, label=y, num_boost_round=2, init_model=loaded)
    full = ctboost.train(x, params, label=y, num_boost_round=4)
    _bits_equal(full.predict(x), resumed.predict(x))
    with pytest.raises(ValueError, match="multiclass_split_score"):
        ctboost.train(
            x,
            {**params, "multiclass_split_score": "scalar"},
            label=y,
            num_boost_round=4,
            resume_from_snapshot=path,
        )


def test_scalar_default_and_old_state_remain_equivalent():
    x, y = _data()
    params = _params()
    del params["multiclass_split_score"]
    default = ctboost.train(x, params, label=y, num_boost_round=3)
    explicit = ctboost.train(
        x, {**params, "multiclass_split_score": "scalar"}, label=y, num_boost_round=3
    )
    assert default._handle.export_state() == explicit._handle.export_state()
    state = copy.deepcopy(default._handle.export_state())
    del state["multiclass_split_score"]
    legacy = ctboost._core.GradientBooster.from_state(state)
    assert legacy.multiclass_split_score() == "scalar"
    _bits_equal(default.predict(x), ctboost.Booster(legacy).predict(x))


def test_joint_cut_sklearn_parameter_clone_and_fit():
    x, y = _data()
    model = ctboost.CTBoostClassifier(iterations=3, multiclass_split_score="joint")
    copied = clone(model)
    assert copied.get_params()["multiclass_split_score"] == "joint"
    copied.fit(x, y)
    assert copied.get_booster().multiclass_split_score == "joint"


def test_joint_cut_rejects_custom_objective_before_callback():
    x, y = _data()
    calls = []

    def custom(predictions, labels):
        calls.append(True)
        return np.zeros_like(predictions), np.ones_like(predictions)

    model = ctboost.CTBoostClassifier(
        iterations=1, loss_function=custom, multiclass_split_score="joint"
    )
    with pytest.raises(ValueError, match="built-in objective"):
        model.fit(x, y)
    assert calls == []


@pytest.mark.parametrize("python_surface", [False, True])
def test_joint_cut_snapshot_and_validation_paths(tmp_path, python_surface):
    x, y = _data()
    pool = ctboost.Pool(x, y)
    params = _params()
    callbacks = [lambda info: False] if python_surface else None
    full = ctboost.train(
        pool, params, num_boost_round=4, eval_set=pool, callbacks=callbacks
    )
    first = ctboost.train(
        pool, params, num_boost_round=2, eval_set=pool, callbacks=callbacks
    )
    path = tmp_path / "snapshot.json"
    first.save_model(path)
    resumed = ctboost.train(
        pool,
        params,
        num_boost_round=4,
        eval_set=pool,
        resume_from_snapshot=path,
        callbacks=callbacks,
    )
    _bits_equal(full.predict(x), resumed.predict(x))
    _bits_equal(full.eval_loss_history, resumed.eval_loss_history)


@pytest.mark.parametrize("regularization", [0.0, 1.0, 5.0])
def test_joint_numeric_cut_matches_weighted_all_class_brute_force(regularization):
    rng = np.random.default_rng(17)
    x = np.repeat(np.arange(6, dtype=np.float32), 50)[:, None]
    y = np.concatenate(
        [
            rng.choice(3, size=50, p=p)
            for p in [
                (0.8, 0.1, 0.1),
                (0.6, 0.2, 0.2),
                (0.1, 0.8, 0.1),
                (0.1, 0.3, 0.6),
                (0.2, 0.1, 0.7),
                (0.1, 0.6, 0.3),
            ]
        ]
    )
    weights = rng.uniform(0.25, 1.75, len(y)).astype(np.float32)
    pool = ctboost.Pool(x, y.astype(np.float32), weight=weights)
    params = {
        **_params(),
        "lambda_l2": regularization,
        "max_depth": 1,
        "alpha": 1.0,
        "boost_from_average": False,
        "base_score": [0.0] * 3,
    }
    model = ctboost.train(pool, params, num_boost_round=1)
    # The native softmax objective rounds responses to float before aggregation.
    gradients = (
        (np.full((len(y), 3), 1 / 3) - np.eye(3)[y]).astype(np.float32).astype(float)
    )
    hessians = np.full_like(gradients, np.float32(2 / 9))
    gradients *= weights[:, None]
    hessians *= weights[:, None]
    total_g, total_h = gradients.sum(axis=0), hessians.sum(axis=0)
    parent = np.sum(total_g**2 / (total_h + regularization))
    gains = []
    for threshold in range(5):
        left = x[:, 0] <= threshold
        gl, hl = gradients[left].sum(axis=0), hessians[left].sum(axis=0)
        gr, hr = total_g - gl, total_h - hl
        gains.append(
            np.sum(gl**2 / (hl + regularization) + gr**2 / (hr + regularization))
            - parent
        )
    tree = _trees(model)[0]
    assert tree["feature_importances"][0] == pytest.approx(max(gains), rel=1e-7)
    leaves = model.predict_leaf_index(x)[:, 0]
    np.testing.assert_array_equal(leaves == leaves[0], x[:, 0] <= np.argmax(gains))


@pytest.mark.parametrize("classes", [3, 32])
def test_joint_cut_supported_class_boundaries(classes):
    y = np.repeat(np.arange(classes), 4).astype(np.float32)
    x = y[:, None].copy()
    params = {**_params(), "num_classes": classes, "alpha": 1.0}
    model = ctboost.train(x, params, label=y, num_boost_round=1)
    assert model.predict(x).shape == (len(y), classes)
    assert np.isfinite(model.predict(x)).all()


def test_joint_cut_supports_wide_numeric_bins():
    x = np.arange(512, dtype=np.float32)[:, None]
    y = np.concatenate([np.zeros(300), np.ones(100), np.full(112, 2)]).astype(
        np.float32
    )
    params = {**_params(), "max_bins": 512, "max_depth": 1, "alpha": 1.0}
    model = ctboost.train(x, params, label=y, num_boost_round=1)
    root = _trees(model)[0]["nodes"][0]
    assert not root["is_leaf"]
    assert root["split_bin_index"] > 255
