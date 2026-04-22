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

def test_low_level_train_accepts_raw_feature_pipeline_and_persists_it(tmp_path: Path):
    rng = np.random.default_rng(17)
    row_count = 96
    X = np.empty((row_count, 4), dtype=object)
    X[:, 0] = rng.choice(["red", "green", "blue"], size=row_count)
    X[:, 1] = rng.normal(size=row_count).astype(np.float32)
    X[:, 2] = np.where(X[:, 1].astype(np.float32) > 0.0, "warm fast fox", "cold slow fox")
    X[:, 3] = [np.asarray([value, value * 0.5, -value], dtype=np.float32) for value in X[:, 1].astype(np.float32)]
    y = (
        0.8 * X[:, 1].astype(np.float32)
        + (X[:, 0] == "red").astype(np.float32)
        + 0.1 * (X[:, 1].astype(np.float32) > 0.0).astype(np.float32)
    ).astype(np.float32)

    booster = ctboost.train(
        X,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 3,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "cat_features": [0],
            "ordered_ctr": True,
            "text_features": [2],
            "embedding_features": [3],
        },
        label=y,
        num_boost_round=10,
    )

    prediction = booster.predict(X)
    assert prediction.shape == (row_count,)
    assert np.all(np.isfinite(prediction))

    model_path = tmp_path / "raw_booster.json"
    booster.save_model(model_path)
    restored = ctboost.load_model(model_path)
    np.testing.assert_allclose(restored.predict(X), prediction, rtol=1e-6, atol=1e-6)

def test_low_level_train_persists_per_feature_ctr_configuration_with_combination_keys(tmp_path: Path):
    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(23)
    row_count = 72
    frame = pd.DataFrame(
        {
            "city": rng.choice(["berlin", "rome", "oslo"], size=row_count),
            "segment": rng.choice(["retail", "pro", "edu", "other"], size=row_count),
            "value": rng.normal(size=row_count).astype(np.float32),
        }
    )
    label = (
        0.6 * frame["value"].to_numpy(dtype=np.float32)
        + (frame["city"] == "berlin").to_numpy(dtype=np.float32)
        + 0.2 * (frame["segment"] == "pro").to_numpy(dtype=np.float32)
    ).astype(np.float32)

    booster = ctboost.train(
        frame,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 3,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "cat_features": ["city", "segment"],
            "one_hot_max_size": 0,
            "simple_ctr": ["Frequency"],
            "per_feature_ctr": {
                "city": ["Mean"],
                ("city", "segment"): ["Frequency"],
            },
            "categorical_combinations": [["city", "segment"]],
        },
        label=label,
        num_boost_round=8,
    )

    prediction = booster.predict(frame)
    model_path = tmp_path / "booster_with_ctr_config.json"
    booster.save_model(model_path)
    restored = ctboost.load_model(model_path)
    np.testing.assert_allclose(restored.predict(frame), prediction, rtol=1e-6, atol=1e-6)

def test_prepare_training_data_reuses_raw_feature_pipeline_and_eval_sets():
    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(23)
    row_count = 96
    frame = pd.DataFrame(
        {
            "city": rng.choice(["berlin", "rome", "oslo"], size=row_count),
            "headline": rng.choice(["red quick fox", "blue slow fox"], size=row_count),
            "value": rng.normal(size=row_count).astype(np.float32),
        }
    )
    label = (
        0.8 * frame["value"].to_numpy(dtype=np.float32)
        + (frame["city"] == "berlin").to_numpy(dtype=np.float32)
        + 0.15 * (frame["headline"] == "red quick fox").to_numpy(dtype=np.float32)
    ).astype(np.float32)

    params = {
        "objective": "RMSE",
        "learning_rate": 0.2,
        "max_depth": 2,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "random_seed": 11,
        "cat_features": ["city"],
        "ordered_ctr": True,
        "text_features": ["headline"],
        "text_hash_dim": 16,
    }

    prepared = ctboost.prepare_training_data(
        frame.iloc[:64],
        params,
        label=label[:64],
        eval_set=[(frame.iloc[64:], label[64:])],
        eval_names=["holdout"],
    )
    assert prepared.feature_pipeline is not None
    assert prepared.eval_names == ["holdout"]

    prepared_booster = ctboost.train(
        prepared,
        params,
        num_boost_round=10,
        early_stopping_rounds=4,
    )
    repeated_booster = ctboost.train(
        prepared,
        params,
        num_boost_round=10,
        early_stopping_rounds=4,
    )
    raw_booster = ctboost.train(
        frame.iloc[:64],
        params,
        label=label[:64],
        num_boost_round=10,
        eval_set=[(frame.iloc[64:], label[64:])],
        eval_names=["holdout"],
        early_stopping_rounds=4,
    )

    np.testing.assert_allclose(
        prepared_booster.predict(frame),
        raw_booster.predict(frame),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        repeated_booster.predict(frame),
        raw_booster.predict(frame),
        rtol=1e-6,
        atol=1e-6,
    )
    assert prepared_booster.best_iteration == raw_booster.best_iteration
    assert repeated_booster.evals_result_ == raw_booster.evals_result_

def test_booster_get_borders_round_trips_into_feature_border_training():
    X, y = make_regression(
        n_samples=96,
        n_features=4,
        n_informative=3,
        noise=0.1,
        random_state=7,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    initial = ctboost.train(
        X,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "max_bins": 32,
            "max_bin_by_feature": {0: 5, 1: 7},
            "border_selection_method": "Uniform",
            "nan_mode_by_feature": {2: "Max"},
        },
        label=y,
        num_boost_round=6,
    )

    exported = initial.get_borders()
    assert exported is not None
    assert len(exported["feature_borders"]) == X.shape[1]
    assert exported["nan_mode_by_feature"][2] == "Max"

    replay = ctboost.train(
        X,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "feature_borders": exported["feature_borders"],
            "nan_mode_by_feature": exported["nan_mode_by_feature"],
        },
        label=y,
        num_boost_round=6,
    )

    np.testing.assert_allclose(
        replay.predict(X),
        initial.predict(X),
        rtol=1e-6,
        atol=1e-6,
    )

def test_prepare_pool_and_train_support_external_memory(tmp_path: Path):
    X, y = make_regression(
        n_samples=144,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=23,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    prepared = ctboost.prepare_pool(
        X,
        y,
        external_memory=True,
        external_memory_dir=tmp_path / "prepared_pool",
    )
    assert getattr(prepared, "_external_memory_backing", None) is not None
    assert (tmp_path / "prepared_pool" / "data.npy").exists()

    booster = ctboost.train(
        X,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "external_memory": True,
            "external_memory_dir": str(tmp_path / "train_pool"),
        },
        label=y,
        num_boost_round=8,
    )
    baseline = ctboost.train(
        X,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "random_seed": 0,
        },
        label=y,
        num_boost_round=8,
    )
    prediction = booster.predict(X)
    assert prediction.shape == (X.shape[0],)
    assert np.all(np.isfinite(prediction))
    assert (tmp_path / "train_pool" / "data.npy").exists()
    np.testing.assert_allclose(prediction, baseline.predict(X), rtol=1e-6, atol=1e-6)
