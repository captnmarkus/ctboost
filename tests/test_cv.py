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
from types import SimpleNamespace
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
import ctboost
import ctboost._core as _core
from ctboost.distributed import (
    DistributedCollectiveServer,
    distributed_tcp_request,
)

def test_cv_returns_fold_aggregates():
    X, y = make_classification(
        n_samples=180,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=19,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    results = ctboost.cv(
        ctboost.Pool(X, y),
        {
            "objective": "Logloss",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=10,
        nfold=3,
        random_state=7,
    )

    expected_keys = {
        "iterations",
        "train_loss_mean",
        "train_loss_std",
        "valid_loss_mean",
        "valid_loss_std",
        "best_iteration_mean",
        "best_iteration_std",
    }
    assert expected_keys == set(results)
    assert results["iterations"].shape == results["train_loss_mean"].shape
    assert results["iterations"].shape == results["valid_loss_mean"].shape
    assert np.all(np.isfinite(results["train_loss_mean"]))
    assert np.all(np.isfinite(results["valid_loss_mean"]))
    assert np.isfinite(results["best_iteration_mean"])


@pytest.mark.parametrize("eval_metric,choose", [("RMSE", np.argmin), ("AUC", np.argmax)])
def test_cv_selects_shared_round_from_fold_validation_metrics(eval_metric, choose):
    from sklearn.model_selection import KFold

    X, y = make_classification(n_samples=96, n_features=5, random_state=19)
    params = {"objective": "Logloss", "eval_metric": [eval_metric, "Logloss"], "max_depth": 2}
    result = ctboost.cv(
        ctboost.Pool(X, y), params, num_boost_round=8, nfold=3,
        stratified=False, shuffle=False, select_num_boost_round=True,
    )
    histories = []
    for train_ids, valid_ids in KFold(n_splits=3).split(X):
        fold = ctboost.train(
            ctboost.Pool(X[train_ids], y[train_ids]), params, num_boost_round=8,
            eval_set=ctboost.Pool(X[valid_ids], y[valid_ids]),
        )
        histories.append(fold.eval_loss_history)
    means = np.mean(histories, axis=0)
    chosen = int(choose(means))
    assert result["best_num_boost_round"] == chosen + 1
    assert result["best_valid_score"] == pytest.approx(means[chosen])
    refit = ctboost.train(ctboost.Pool(X, y), params, num_boost_round=result["best_num_boost_round"])
    assert refit.num_iterations_trained == chosen + 1


@pytest.mark.parametrize(
    "histories, expected",
    [
        ([[1.0, 2.0, 2.0, 8.0, 9.0], [2.0, 1.5, 1.5, np.nan]], (2, 1.75)),
        ([[1.0, 1.0 + 1e-9], [2.0, 2.0]], (2, 1.5 + 0.5e-9)),
        ([[np.nan, 1.0], [2.0, np.nan]], None),
    ],
)
def test_cv_selection_requires_every_fold_and_keeps_earliest_tie(monkeypatch, histories, expected):
    from ctboost.training import api

    histories = iter(histories)

    def fake_train(pool, params, *, eval_set, **kwargs):
        assert set(pool.label).isdisjoint(eval_set.label)
        scores = next(histories)
        return SimpleNamespace(loss_history=[0] * len(scores), eval_loss_history=scores, best_iteration=0)

    monkeypatch.setattr(api, "train", fake_train)
    pool = ctboost.Pool(np.arange(8).reshape(-1, 1), np.arange(8))
    params = {"eval_metric": ctboost.make_eval_metric(lambda pred, label: 0.0, higher_is_better=True)}
    if expected is None:
        with pytest.raises(ValueError, match="finite validation score in every fold"):
            ctboost.cv(pool, params, nfold=2, select_num_boost_round=True)
    else:
        result = ctboost.cv(pool, params, nfold=2, select_num_boost_round=True)
        assert result["best_num_boost_round"] == expected[0]
        assert result["best_valid_score"] == expected[1]
