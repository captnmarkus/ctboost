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

def test_booster_export_model_generates_standalone_python_predictor(tmp_path: Path):
    X, y = make_regression(
        n_samples=96,
        n_features=5,
        n_informative=4,
        noise=0.1,
        random_state=29,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    booster = ctboost.train(
        X,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        label=y,
        num_boost_round=8,
    )

    export_path = tmp_path / "standalone_predictor.py"
    booster.export_model(export_path)

    spec = importlib.util.spec_from_file_location("ctboost_standalone_predictor", export_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    standalone_pred = np.asarray(module.predict(X), dtype=np.float32)
    np.testing.assert_allclose(standalone_pred, booster.predict(X), rtol=1e-6, atol=1e-6)
    single_prediction = float(module.predict(X[0]))
    np.testing.assert_allclose(single_prediction, booster.predict(X[:1])[0], rtol=1e-6, atol=1e-6)

def test_booster_export_model_generates_json_predictor(tmp_path: Path):
    X, y = make_regression(
        n_samples=96,
        n_features=5,
        n_informative=4,
        noise=0.1,
        random_state=31,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    booster = ctboost.train(
        X,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        label=y,
        num_boost_round=8,
    )

    export_path = tmp_path / "predictor.json"
    booster.export_model(export_path, export_format="json_predictor")

    predictor = ctboost.load_exported_predictor(export_path)
    exported_pred = np.asarray(predictor.predict(X), dtype=np.float32)
    np.testing.assert_allclose(exported_pred, booster.predict(X), rtol=1e-6, atol=1e-6)

def test_estimator_export_model_matches_predict_proba(tmp_path: Path):
    X, y = make_classification(
        n_samples=160,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=37,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    clf = ctboost.CTBoostClassifier(
        iterations=12,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    clf.fit(X, y)

    export_path = tmp_path / "standalone_classifier.py"
    clf.export_model(export_path)

    spec = importlib.util.spec_from_file_location("ctboost_standalone_classifier", export_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    standalone_proba = np.asarray(module.predict_proba(X), dtype=np.float32)
    np.testing.assert_allclose(standalone_proba, clf.predict_proba(X), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(module.predict_class(X), dtype=np.int32), clf.predict(X))
