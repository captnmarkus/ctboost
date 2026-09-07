"""Native early stopping retains the best model when its round budget expires."""

from __future__ import annotations

import numpy as np
import pytest

import ctboost


def _data(problem_type):
    X = np.linspace(-1, 1, 768, dtype=np.float32).reshape(-1, 1)
    if problem_type == "regression":
        y = X[:, 0].copy()
        validation_y = -y
    elif problem_type == "binary":
        y = (X[:, 0] > 0).astype(np.int64)
        validation_y = 1 - y
    else:
        y = np.repeat(np.arange(3), len(X) // 3)
        validation_y = (y + 1) % 3
    return X, y, validation_y


def _fit(problem_type, *, boosting_type, multi_strategy, iterations, **fit_kwargs):
    X, y, validation_y = _data(problem_type)
    model_cls = (
        ctboost.CTBoostRegressor
        if problem_type == "regression"
        else ctboost.CTBoostClassifier
    )
    return model_cls(
        iterations=iterations,
        max_depth=2,
        max_bins=16,
        learning_rate=0.3,
        random_seed=4,
        boosting_type=boosting_type,
        multi_strategy=multi_strategy,
        drop_rate=0.5,
        skip_drop=0.0,
        verbose=False,
    ).fit(X, y, eval_set=(X, validation_y), **fit_kwargs)


@pytest.mark.parametrize(
    ("problem_type", "boosting_type", "multi_strategy"),
    [
        ("regression", "Plain", "one_output_per_tree"),
        ("regression", "DART", "one_output_per_tree"),
        ("binary", "Plain", "one_output_per_tree"),
        ("multiclass", "Plain", "one_output_per_tree"),
        ("multiclass", "Plain", "multi_output_tree"),
        ("multiclass", "DART", "one_output_per_tree"),
        ("multiclass", "DART", "multi_output_tree"),
    ],
)
def test_native_budget_end_returns_the_actual_best_ensemble(
    problem_type, boosting_type, multi_strategy
):
    options = {
        "problem_type": problem_type,
        "boosting_type": boosting_type,
        "multi_strategy": multi_strategy,
    }
    native = _fit(**options, iterations=6, early_stopping_rounds=20)
    callback = _fit(
        **options,
        iterations=6,
        early_stopping_rounds=20,
        callbacks=[lambda env: False],
    )
    assert native.best_iteration_ == callback.best_iteration_ == 0
    assert native.get_booster().num_iterations_trained == 1
    best_round = _fit(**options, iterations=1)
    X, _, _ = _data(problem_type)
    expected = best_round.get_booster().predict(X)
    np.testing.assert_allclose(
        native.get_booster().predict(X), expected, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        native.get_booster().predict(X),
        callback.get_booster().predict(X),
        rtol=1e-6,
        atol=1e-6,
    )
    assert native.get_booster().eval_loss_history == pytest.approx(
        best_round.get_booster().eval_loss_history
    )


@pytest.mark.parametrize("boosting_type", ["Plain", "DART"])
def test_no_early_stopping_keeps_all_rounds_despite_earlier_validation_best(
    boosting_type,
):
    options = {
        "problem_type": "regression",
        "boosting_type": boosting_type,
        "multi_strategy": "one_output_per_tree",
        "iterations": 6,
    }
    native = _fit(**options)
    callback = _fit(**options, callbacks=[lambda env: False])
    assert native.best_iteration_ == callback.best_iteration_ == 0
    assert native.get_booster().num_iterations_trained == 6
    assert callback.get_booster().num_iterations_trained == 6
    X, _, _ = _data("regression")
    np.testing.assert_allclose(
        native.predict(X), callback.predict(X), rtol=1e-6, atol=1e-6
    )


def test_native_dart_warm_start_retains_previously_best_ensemble():
    options = {
        "problem_type": "regression",
        "boosting_type": "DART",
        "multi_strategy": "one_output_per_tree",
    }
    initial = _fit(**options, iterations=1, early_stopping_rounds=20)
    continued = _fit(
        **options, iterations=5, early_stopping_rounds=20, init_model=initial
    )
    assert continued.best_iteration_ == 0
    assert continued.get_booster().num_iterations_trained == 1
    X, _, _ = _data("regression")
    np.testing.assert_allclose(continued.predict(X), initial.predict(X), rtol=0, atol=0)


@pytest.mark.parametrize("patience", [20, 2])
@pytest.mark.parametrize("use_callback", [False, True], ids=["native", "callback"])
def test_untrimmed_dart_resume_selects_a_recoverable_best_ensemble(
    patience, use_callback
):
    options = {
        "problem_type": "regression",
        "boosting_type": "DART",
        "multi_strategy": "one_output_per_tree",
    }
    initial = _fit(**options, iterations=6)
    initial_rounds = initial.get_booster().num_iterations_trained
    assert initial_rounds == 6
    assert initial.best_iteration_ == 0
    X, _, validation_y = _data("regression")
    initial_predictions = initial.predict(X).copy()
    initial_error = np.sqrt(np.mean((initial_predictions - validation_y) ** 2))
    callbacks = {"callbacks": [lambda env: False]} if use_callback else {}
    continued = _fit(
        **options,
        iterations=5,
        early_stopping_rounds=patience,
        init_model=initial,
        **callbacks,
    )
    predictions = continued.predict(X)
    actual_error = np.sqrt(np.mean((predictions - validation_y) ** 2))
    assert continued.best_iteration_ >= initial_rounds - 1
    assert (
        continued.get_booster().num_iterations_trained == continued.best_iteration_ + 1
    )
    assert actual_error <= initial_error + 1e-6
    assert actual_error == pytest.approx(
        continued.get_booster().eval_loss_history[continued.best_iteration_],
        rel=1e-6,
        abs=1e-6,
    )
    if continued.best_iteration_ == initial_rounds - 1:
        np.testing.assert_array_equal(predictions, initial_predictions)


@pytest.mark.parametrize("patience", [20, 2])
@pytest.mark.parametrize("use_callback", [False, True], ids=["native", "callback"])
@pytest.mark.parametrize("initial_callback", [False, True], ids=["native-init", "callback-init"])
def test_untrimmed_dart_resume_reevaluates_changed_validation(
    patience, use_callback, initial_callback
):
    initial = _fit(
        "regression",
        boosting_type="DART",
        multi_strategy="one_output_per_tree",
        iterations=6,
        **({"callbacks": [lambda env: False]} if initial_callback else {}),
    )
    assert initial.best_iteration_ == 0
    initial_rounds = initial.get_booster().num_iterations_trained
    assert initial_rounds == 6
    X, y, _ = _data("regression")
    # The full supplied ensemble is perfect on this new validation target.
    # Its historical validation score cannot identify that recoverable optimum.
    validation_y = initial.predict(X).copy()
    callbacks = {"callbacks": [lambda env: False]} if use_callback else {}
    continued = ctboost.CTBoostRegressor(
        **{**initial.get_params(), "iterations": 5}
    ).fit(
        X,
        y,
        eval_set=(X, validation_y),
        early_stopping_rounds=patience,
        init_model=initial,
        **callbacks,
    )
    assert continued.best_iteration_ == initial_rounds - 1
    assert continued.get_booster().num_iterations_trained == initial_rounds
    np.testing.assert_array_equal(continued.predict(X), validation_y)
    assert continued.get_booster().eval_loss_history[continued.best_iteration_] == (
        pytest.approx(0.0, abs=1e-7)
    )
    # A further resume must not rediscover the inaccessible old historical best.
    # Alternate paths to check native state and wrapper metadata agree.
    again = ctboost.CTBoostRegressor(
        **{**initial.get_params(), "iterations": 2}
    ).fit(
        X,
        y,
        eval_set=(X, validation_y),
        early_stopping_rounds=20,
        init_model=continued,
        **({} if use_callback else {"callbacks": [lambda env: False]}),
    )
    assert again.best_iteration_ == initial_rounds - 1
    assert again.get_booster().num_iterations_trained == initial_rounds
    np.testing.assert_array_equal(again.predict(X), validation_y)
    assert again.get_booster().eval_loss_history[again.best_iteration_] == (
        pytest.approx(0.0, abs=1e-7)
    )
