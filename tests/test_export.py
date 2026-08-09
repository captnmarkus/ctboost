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


def test_booster_export_model_generates_cpp17_c_api(tmp_path: Path):
    X, y = make_regression(
        n_samples=64,
        n_features=4,
        n_informative=3,
        random_state=33,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    booster = ctboost.train(
        X,
        {"objective": "RMSE", "max_depth": 2, "alpha": 1.0},
        label=y,
        num_boost_round=4,
    )

    export_path = tmp_path / "ctboost_model.cpp"
    booster.export_model(export_path)
    source = export_path.read_text(encoding="utf-8")

    assert "ctboost_predict(" in source
    assert "ctboost_inference_manifest_json" in source
    assert "static constexpr Node kNodes[]" in source
    assert '"kind":"standalone_cpp"' in source
    assert "rows > output_size / kPredictionDimension" in source
    assert "rows > std::numeric_limits<std::size_t>::max() / columns" in source


def test_onnx_export_matches_native_regression_and_binary_probabilities(tmp_path: Path):
    pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")

    X, y = make_classification(
        n_samples=96,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=35,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    model = ctboost.CTBoostClassifier(
        iterations=7,
        learning_rate=0.2,
        max_depth=3,
        alpha=1.0,
        random_seed=5,
    ).fit(X, y)

    export_path = tmp_path / "classifier.onnx"
    model.export_model(export_path)
    session = onnxruntime.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
    raw, probabilities = session.run(None, {"features": X})

    np.testing.assert_allclose(raw, model.get_booster().predict(X), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(probabilities, model.predict_proba(X), rtol=2e-5, atol=2e-5)


def test_onnx_export_matches_native_missing_value_routes(tmp_path: Path):
    pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(39)
    X = rng.normal(size=(80, 4)).astype(np.float32)
    X[::7, 1] = np.nan
    y = (np.nan_to_num(X[:, 1]) + X[:, 2]).astype(np.float32)
    booster = ctboost.train(
        X,
        {"objective": "RMSE", "max_depth": 3, "alpha": 1.0, "nan_mode": "Max"},
        label=y,
        num_boost_round=6,
    )

    export_path = tmp_path / "regressor.onnx"
    booster.export_model(export_path)
    session = onnxruntime.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
    (raw,) = session.run(None, {"features": X})

    np.testing.assert_allclose(raw, booster.predict(X), rtol=2e-5, atol=2e-5)


def test_onnx_export_matches_native_categorical_bin_routes(tmp_path: Path):
    pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    X = np.asarray(
        [[0.0, -1.0], [2.0, 0.0], [5.0, 1.0], [0.0, 2.0], [2.0, 3.0], [5.0, 4.0]]
        * 10,
        dtype=np.float32,
    )
    y = (X[:, 0] == 5.0).astype(np.float32)
    pool = ctboost.Pool(X, y, cat_features=[0])
    booster = ctboost.train(
        pool,
        {"objective": "RMSE", "max_depth": 2, "alpha": 1.0},
        num_boost_round=5,
    )
    prediction_data = np.asarray(
        [[0.0, 0.5], [2.0, 0.5], [5.0, 0.5], [3.0, 0.5]], dtype=np.float32
    )
    prediction_pool = ctboost.Pool(prediction_data, cat_features=[0])

    export_path = tmp_path / "categorical.onnx"
    booster.export_model(export_path)
    session = onnxruntime.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
    (raw,) = session.run(None, {"features": prediction_data})

    np.testing.assert_allclose(raw, booster.predict(prediction_pool), rtol=2e-5, atol=2e-5)

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

    with pytest.raises(ValueError, match="class_labels size"):
        clf.get_booster().export_model(
            tmp_path / "invalid-labels.json",
            export_format="json_predictor",
            class_labels=["only-one-label"],
        )
