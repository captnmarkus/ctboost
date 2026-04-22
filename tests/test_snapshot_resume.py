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
    explicit_resume.fit(X, y, init_model=ctboost.load_model(snapshot_path))

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
