import numpy as np
import pytest

import ctboost
import ctboost.training.api as training_api
from ctboost.training._train_config import _normalize_training_config


def _regression_data():
    rng = np.random.default_rng(202)
    X = np.asfortranarray(rng.normal(size=(180, 5)).astype(np.float32))
    y = (1.5 * X[:, 0] - 0.75 * X[:, 1] + 0.1 * X[:, 2]).astype(np.float32)
    return X, y


def test_custom_eval_name_uses_native_training_path_and_preserves_history(monkeypatch):
    X, y = _regression_data()

    def fail_python_surface(**_kwargs):
        raise AssertionError("a custom eval name must not select the per-round Python training path")

    monkeypatch.setattr(training_api, "_train_with_python_surface", fail_python_surface)
    booster = ctboost.train(
        X[:130],
        {
            "iterations": 8,
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "random_seed": 9,
        },
        label=y[:130],
        eval_set=(X[130:], y[130:]),
        eval_names=["holdout"],
    )

    assert booster.num_iterations_trained == 8
    assert set(booster.evals_result_) == {"learn", "holdout"}
    np.testing.assert_allclose(
        booster.evals_result_["holdout"]["RMSE"],
        booster.eval_loss_history,
        rtol=0.0,
        atol=0.0,
    )


def test_native_continuation_merges_python_eval_history_and_refreshes_best_iteration(monkeypatch):
    X, y = _regression_data()
    initial = ctboost.train(
        X[:32],
        {"objective": "RMSE", "alpha": 1.0, "random_seed": 7},
        label=y[:32],
        num_boost_round=3,
        eval_set=(X[32:], y[32:]),
        eval_names=["holdout"],
        callbacks=[lambda _env: False],
    )
    initial_history = list(initial.evals_result_["holdout"]["RMSE"])
    initial_learning_rates = list(initial.learning_rate_history)
    initial._training_metadata["learning_rate_history"] = initial_learning_rates + [99.0, 99.0]

    def fail_python_surface(**_kwargs):
        raise AssertionError("native continuation unexpectedly used the Python loop")

    monkeypatch.setattr(training_api, "_train_with_python_surface", fail_python_surface)
    continued = ctboost.train(
        X[:32],
        {"objective": "RMSE", "alpha": 1.0, "random_seed": 7},
        label=y[:32],
        num_boost_round=2,
        eval_set=(X[32:], y[32:]),
        eval_names=["holdout"],
        init_model=initial,
    )

    continued_history = continued.evals_result_["holdout"]["RMSE"]
    assert continued.num_iterations_trained == 5
    assert continued.best_iteration == continued._handle.best_iteration()
    assert len(continued_history) == 5
    assert continued_history[:3] == pytest.approx(initial_history)
    assert continued.eval_loss_history == pytest.approx(continued_history)
    assert len(continued.evals_result_["learn"]["loss"]) == 5
    assert len(continued.learning_rate_history) == 5
    assert continued.learning_rate_history[:3] == pytest.approx(initial_learning_rates)
    assert continued.learning_rate_history[3:] == pytest.approx([continued.learning_rate] * 2)


def test_eval_names_rejects_reserved_learn_dataset_name():
    X, y = _regression_data()
    with pytest.raises(ValueError, match="reserved training dataset name 'learn'"):
        ctboost.train(
            X[:32],
            {"objective": "RMSE", "iterations": 2},
            label=y[:32],
            eval_set=(X[32:], y[32:]),
            eval_names=["learn"],
        )


def test_train_rejects_unlabeled_training_pool_early():
    X, _ = _regression_data()
    with pytest.raises(ValueError, match="^training pool requires labels$"):
        ctboost.train(ctboost.Pool(X), {"objective": "RMSE", "iterations": 2})


def test_train_rejects_unlabeled_eval_pool_early():
    X, y = _regression_data()
    with pytest.raises(ValueError, match="^eval pool requires labels$"):
        ctboost.train(
            ctboost.Pool(X[:130], y[:130]),
            {"objective": "RMSE", "iterations": 2},
            eval_set=ctboost.Pool(X[130:]),
        )


def test_prepared_data_with_custom_eval_name_remains_reusable_on_native_path(monkeypatch):
    X, y = _regression_data()
    params = {"iterations": 4, "max_depth": 2, "alpha": 1.0, "random_seed": 5}
    prepared = ctboost.prepare_training_data(
        X[:130],
        params,
        label=y[:130],
        eval_set=(X[130:], y[130:]),
        eval_names=["holdout"],
    )

    def fail_python_surface(**_kwargs):
        raise AssertionError("prepared data with one native metric should stay on the native path")

    monkeypatch.setattr(training_api, "_train_with_python_surface", fail_python_surface)
    first = ctboost.train(prepared, params)
    second = ctboost.train(prepared, params)

    np.testing.assert_allclose(first.predict(X), second.predict(X), rtol=0.0, atol=0.0)
    assert "holdout" in first.evals_result_
    assert "holdout" in second.evals_result_


@pytest.mark.parametrize(
    ("alias", "canonical_name", "value", "expected"),
    [
        ("n_estimators", "iterations", 3, 3),
        ("num_trees", "iterations", 4, 4),
        ("eta", "learning_rate", 0.25, 0.25),
        ("depth", "max_depth", 2, 2),
        ("reg_lambda", "lambda_l2", 2.5, 2.5),
        ("l2_leaf_reg", "lambda_l2", 3.5, 3.5),
        ("lambda", "lambda_l2", 4.5, 4.5),
        ("random_state", "random_seed", 13, 13),
        ("seed", "random_seed", 17, 17),
        ("max_bin", "max_bins", 32, 32),
    ],
)
def test_low_level_training_aliases_resolve_to_native_parameters(
    alias,
    canonical_name,
    value,
    expected,
):
    X, y = _regression_data()
    params = {
        "iterations": 2,
        "learning_rate": 0.1,
        "max_depth": 1,
        "lambda_l2": 1.0,
        "max_bins": 64,
        "alpha": 1.0,
    }
    params.pop(canonical_name, None)
    params[alias] = value

    booster = ctboost.train(X, params, label=y)
    if canonical_name == "iterations":
        resolved = booster.num_iterations_trained
    else:
        resolved = getattr(booster._handle, canonical_name)()

    if isinstance(expected, float):
        assert resolved == pytest.approx(expected)
    else:
        assert resolved == expected


@pytest.mark.parametrize(
    "params",
    [
        {"iterations": 2, "n_estimators": 2},
        {"n_estimators": 2, "num_trees": 2},
        {"learning_rate": 0.1, "eta": 0.1},
        {"max_depth": 2, "depth": 2},
        {"lambda_l2": 1.0, "reg_lambda": 1.0},
        {"reg_lambda": 1.0, "l2_leaf_reg": 1.0},
        {"random_seed": 1, "random_state": 1},
        {"random_state": 1, "seed": 1},
        {"max_bins": 32, "max_bin": 32},
    ],
)
def test_low_level_training_alias_conflicts_are_rejected(params):
    X, y = _regression_data()
    with pytest.raises(ValueError, match="cannot be used together"):
        ctboost.train(X, params, label=y)


def test_unknown_training_parameter_has_close_name_hint():
    X, y = _regression_data()
    with pytest.raises(
        ValueError,
        match=r"unknown training parameter: 'learnig_rate'; did you mean 'learning_rate'\?",
    ):
        ctboost.train(X, {"learnig_rate": 0.2}, label=y)


def test_auxiliary_training_parameter_names_remain_supported():
    auxiliary_params = {
        # Feature preprocessing.
        "cat_features": None,
        "ordered_ctr": False,
        "one_hot_max_size": 0,
        "max_cat_to_onehot": 0,
        "max_cat_threshold": 0,
        "categorical_combinations": None,
        "pairwise_categorical_combinations": False,
        "simple_ctr": None,
        "combinations_ctr": None,
        "per_feature_ctr": None,
        "text_features": None,
        "text_hash_dim": 64,
        "embedding_features": None,
        "embedding_stats": ("mean", "std"),
        "ctr_prior_strength": 1.0,
        # Evaluation and weighting.
        "eval_metric": "RMSE",
        "class_weights": None,
        "auto_class_weights": None,
        "scale_pos_weight": None,
        # Native and eval-set external memory.
        "external_memory": False,
        "external_memory_dir": None,
        "eval_external_memory": False,
        "eval_external_memory_dir": None,
        # Distributed orchestration.
        "distributed_world_size": 1,
        "distributed_rank": 0,
        "distributed_root": None,
        "distributed_run_id": "default",
        "distributed_timeout": 600.0,
    }

    assert _normalize_training_config(auxiliary_params) == auxiliary_params
