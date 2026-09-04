import importlib.util
import json
from pathlib import Path
import os
import socket
import subprocess
import sys
import threading
import time
import textwrap
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
import ctboost
import ctboost._core as _core
from ctboost.distributed import (
    DistributedCollectiveServer,
    distributed_tcp_request,
)

def test_training_snapshot_resume_matches_explicit_warm_start(tmp_path: Path):
    X, y = make_regression(
        n_samples=180,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=43,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    params = {
        "objective": "RMSE",
        "learning_rate": 0.15,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "random_seed": 19,
    }
    snapshot_path = tmp_path / "resume_snapshot.ctb"
    partial = ctboost.train(
        ctboost.Pool(X, y),
        params,
        num_boost_round=7,
        snapshot_path=snapshot_path,
        snapshot_interval=2,
    )
    assert snapshot_path.exists()
    explicit_resume = ctboost.train(
        ctboost.Pool(X, y),
        params,
        num_boost_round=11,
        init_model=ctboost.load_model(snapshot_path),
        snapshot_path=tmp_path / "explicit_resume_snapshot.ctb",
    )
    resumed = ctboost.train(
        ctboost.Pool(X, y),
        params,
        num_boost_round=18,
        snapshot_path=snapshot_path,
        resume_from_snapshot=True,
    )

    assert partial.num_iterations_trained == 7
    assert resumed.num_iterations_trained == 18
    np.testing.assert_allclose(resumed.predict(X), explicit_resume.predict(X), rtol=1e-6, atol=1e-6)

def test_resume_from_snapshot_rejects_config_drift(tmp_path: Path):
    X, y = make_regression(
        n_samples=180,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=49,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    params = {
        "objective": "RMSE",
        "learning_rate": 0.15,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "random_seed": 17,
    }
    snapshot_path = tmp_path / "strict_resume_snapshot.ctb"
    ctboost.train(
        ctboost.Pool(X, y),
        params,
        num_boost_round=6,
        snapshot_path=snapshot_path,
        snapshot_interval=1,
    )

    with pytest.raises(ValueError, match="Use init_model"):
        ctboost.train(
            ctboost.Pool(X, y),
            {**params, "max_depth": 4},
            num_boost_round=12,
            snapshot_path=snapshot_path,
            resume_from_snapshot=True,
        )


@pytest.mark.parametrize("changed_contract", ["config", "schema", "schedule"])
def test_completed_snapshot_still_validates_resume_contract(tmp_path: Path, changed_contract):
    X = np.arange(80, dtype=np.float32).reshape(40, 2)
    y = X[:, 0]
    pool = ctboost.Pool(X, y, feature_names=["first", "second"])
    params = {"objective": "RMSE", "alpha": 1.0, "max_depth": 2}
    schedule = [0.2, 0.1]
    model = ctboost.train(pool, params, num_boost_round=2, learning_rate_schedule=schedule)
    snapshot_path = tmp_path / "completed.ctb"
    model.save_model(snapshot_path)

    if changed_contract == "config":
        params = {**params, "max_depth": 3}
    elif changed_contract == "schema":
        pool = ctboost.Pool(X, y, feature_names=["renamed", "second"])
    else:
        schedule = [0.3, 0.1]

    with pytest.raises(ValueError, match="resume_from_snapshot"):
        ctboost.train(
            pool,
            params,
            num_boost_round=2,
            learning_rate_schedule=schedule,
            resume_from_snapshot=snapshot_path,
        )


def test_completed_snapshot_returns_unchanged_model_after_validation(tmp_path: Path):
    X = np.arange(80, dtype=np.float32).reshape(40, 2)
    pool = ctboost.Pool(X, X[:, 0])
    params = {"objective": "RMSE", "alpha": 1.0, "max_depth": 2}
    model = ctboost.train(pool, params, num_boost_round=2)
    snapshot_path = tmp_path / "completed.ctb"
    model.save_model(snapshot_path)

    resumed = ctboost.train(pool, params, num_boost_round=2, resume_from_snapshot=snapshot_path)

    assert resumed.num_iterations_trained == model.num_iterations_trained
    assert resumed.evals_result_ == model.evals_result_
    np.testing.assert_array_equal(resumed.predict(pool), model.predict(pool))

def test_estimator_resume_from_snapshot_matches_explicit_warm_start(tmp_path: Path):
    X, y = make_regression(
        n_samples=180,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=47,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    snapshot_path = tmp_path / "estimator_snapshot.ctb"
    partial = ctboost.CTBoostRegressor(
        iterations=8,
        learning_rate=0.15,
        max_depth=3,
        alpha=1.0,
        lambda_l2=1.0,
        random_seed=23,
    )
    partial.fit(X, y, snapshot_path=snapshot_path, snapshot_interval=2)
    assert snapshot_path.exists()

    explicit_resume = ctboost.CTBoostRegressor(
        iterations=12,
        learning_rate=0.15,
        max_depth=3,
        alpha=1.0,
        lambda_l2=1.0,
        random_seed=23,
    )
    explicit_resume.fit(
        X,
        y,
        init_model=ctboost.load_model(snapshot_path),
        snapshot_path=tmp_path / "explicit_estimator_resume_snapshot.ctb",
    )

    resumed = ctboost.CTBoostRegressor(
        iterations=20,
        learning_rate=0.15,
        max_depth=3,
        alpha=1.0,
        lambda_l2=1.0,
        random_seed=23,
    )
    resumed.fit(X, y, snapshot_path=snapshot_path, resume_from_snapshot=True)

    assert resumed._booster.num_iterations_trained == 20
    np.testing.assert_allclose(
        resumed.predict(X),
        explicit_resume.predict(X),
        rtol=1e-6,
        atol=1e-6,
    )


def _assert_exact_training_match(reference, resumed, prediction_data):
    np.testing.assert_array_equal(reference.predict(prediction_data), resumed.predict(prediction_data))
    assert reference.loss_history == resumed.loss_history
    assert reference.eval_loss_history == resumed.eval_loss_history
    assert reference.evals_result_ == resumed.evals_result_
    assert reference.learning_rate_history == resumed.learning_rate_history
    reference_state = dict(reference._handle.export_state())
    resumed_state = dict(resumed._handle.export_state())
    assert reference_state["trees"] == resumed_state["trees"]
    assert reference_state["tree_learning_rates"] == resumed_state["tree_learning_rates"]
    assert reference_state["rng_state"] == resumed_state["rng_state"]


@pytest.mark.parametrize(
    "boosting_params",
    [
        {},
        {
            "boosting_type": "DART",
            "drop_rate": 0.35,
            "skip_drop": 0.0,
            "max_drop": 3,
        },
    ],
    ids=["gradient-boosting", "dart"],
)
def test_snapshot_resume_is_exact_for_stochastic_training_and_eval_history(
    tmp_path: Path,
    boosting_params,
):
    X, y = make_regression(
        n_samples=220,
        n_features=9,
        n_informative=6,
        noise=0.2,
        random_state=123,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    train_pool = ctboost.Pool(X[:170], y[:170])
    validation_pool = ctboost.Pool(X[170:], y[170:])
    params = {
        "objective": "RMSE",
        "learning_rate": 0.13,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "subsample": 0.73,
        "bootstrap_type": "Bernoulli",
        "colsample_bytree": 0.66,
        "random_strength": 2.0,
        "random_seed": 19,
        **boosting_params,
    }

    reference = ctboost.train(
        train_pool,
        params,
        num_boost_round=16,
        eval_set=validation_pool,
        snapshot_path=tmp_path / "uninterrupted.ctb",
        snapshot_interval=2,
    )
    resume_path = tmp_path / "interrupted.ctb"
    ctboost.train(
        train_pool,
        params,
        num_boost_round=6,
        eval_set=validation_pool,
        snapshot_path=resume_path,
        snapshot_interval=2,
    )
    resumed = ctboost.train(
        train_pool,
        params,
        num_boost_round=16,
        eval_set=validation_pool,
        snapshot_path=resume_path,
        snapshot_interval=2,
        resume_from_snapshot=True,
    )

    _assert_exact_training_match(reference, resumed, X)


def test_snapshot_resume_is_exact_with_learning_rate_schedule(tmp_path: Path):
    X, y = make_regression(
        n_samples=180,
        n_features=7,
        n_informative=5,
        noise=0.2,
        random_state=131,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    train_pool = ctboost.Pool(X[:140], y[:140])
    validation_pool = ctboost.Pool(X[140:], y[140:])
    schedule = [0.25] * 3 + [0.12] * 4 + [0.06] * 5
    params = {
        "objective": "RMSE",
        "learning_rate": schedule[0],
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "subsample": 0.8,
        "bootstrap_type": "Bernoulli",
        "colsample_bytree": 0.7,
        "random_seed": 29,
    }

    reference = ctboost.train(
        train_pool,
        params,
        num_boost_round=len(schedule),
        eval_set=validation_pool,
        learning_rate_schedule=schedule,
        snapshot_path=tmp_path / "scheduled_uninterrupted.ctb",
    )
    resume_path = tmp_path / "scheduled_interrupted.ctb"
    ctboost.train(
        train_pool,
        params,
        num_boost_round=5,
        eval_set=validation_pool,
        learning_rate_schedule=schedule,
        snapshot_path=resume_path,
    )
    resumed = ctboost.train(
        train_pool,
        params,
        num_boost_round=len(schedule),
        eval_set=validation_pool,
        learning_rate_schedule=schedule,
        snapshot_path=resume_path,
        resume_from_snapshot=True,
    )

    _assert_exact_training_match(reference, resumed, X)
    assert resumed.learning_rate_history == schedule


def test_periodic_snapshot_can_resume_exactly_after_training_is_interrupted(tmp_path: Path):
    X, y = make_regression(
        n_samples=160,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=139,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    pool = ctboost.Pool(X, y)
    params = {
        "objective": "RMSE",
        "learning_rate": 0.14,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "subsample": 0.75,
        "bootstrap_type": "Bernoulli",
        "colsample_bytree": 0.7,
        "random_seed": 37,
    }
    reference = ctboost.train(
        pool,
        params,
        num_boost_round=12,
        snapshot_path=tmp_path / "periodic_uninterrupted.ctb",
        snapshot_interval=2,
    )
    resume_path = tmp_path / "periodic_interrupted.ctb"

    def stop_after_fifth_iteration(env):
        return env.iteration == 4

    interrupted = ctboost.train(
        pool,
        params,
        num_boost_round=12,
        callbacks=[stop_after_fifth_iteration],
        snapshot_path=resume_path,
        snapshot_interval=2,
    )
    assert interrupted.num_iterations_trained == 5
    # The stop callback runs before the checkpoint callback, so the durable
    # snapshot is the most recent completed interval (four iterations).
    assert ctboost.load_model(resume_path).num_iterations_trained == 4

    resumed = ctboost.train(
        pool,
        params,
        num_boost_round=12,
        snapshot_path=resume_path,
        snapshot_interval=2,
        resume_from_snapshot=True,
    )
    _assert_exact_training_match(reference, resumed, X)


def test_snapshot_resume_reuses_the_fitted_feature_pipeline_exactly(tmp_path: Path):
    rng = np.random.default_rng(137)
    row_count = 120
    data = np.empty((row_count, 3), dtype=object)
    data[:, 0] = rng.choice(["berlin", "oslo", "rome"], size=row_count)
    data[:, 1] = rng.normal(size=row_count).astype(np.float32)
    data[:, 2] = np.where(data[:, 0] == "berlin", "red quick fox", "blue slow fox")
    label = (
        data[:, 1].astype(np.float32)
        + 0.7 * (data[:, 0] == "berlin").astype(np.float32)
    ).astype(np.float32)
    params = {
        "objective": "RMSE",
        "learning_rate": 0.15,
        "max_depth": 2,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "cat_features": [0],
        "ordered_ctr": True,
        "text_features": [2],
        "text_hash_dim": 8,
        "subsample": 0.8,
        "bootstrap_type": "Bernoulli",
        "colsample_bytree": 0.75,
        "random_seed": 31,
    }

    reference = ctboost.train(
        data,
        params,
        label=label,
        num_boost_round=12,
        snapshot_path=tmp_path / "pipeline_uninterrupted.ctb",
    )
    resume_path = tmp_path / "pipeline_interrupted.ctb"
    ctboost.train(
        data,
        params,
        label=label,
        num_boost_round=5,
        snapshot_path=resume_path,
    )
    resumed = ctboost.train(
        data,
        params,
        label=label,
        num_boost_round=12,
        snapshot_path=resume_path,
        resume_from_snapshot=True,
    )

    _assert_exact_training_match(reference, resumed, data)
    assert resumed._feature_pipeline is not None
    assert resumed._feature_pipeline.to_state() == reference._feature_pipeline.to_state()
