"""Portable scoring and explanation contracts for shared vector trees."""

import ctypes
import importlib.util
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from math import factorial
from pathlib import Path

import numpy as np
import pytest

import ctboost
from ctboost.export_runtime import ExportedPredictor


def _python_module(path):
    spec = importlib.util.spec_from_file_location("standalone_vector_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _probabilities(booster, X):
    raw = booster.predict(X).astype(np.float64)
    values = np.exp(raw - raw.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


@pytest.fixture(scope="module")
def vector_model():
    rng = np.random.default_rng(572)
    X = rng.normal(size=(96, 3)).astype(np.float32)
    X[:, 2] = rng.integers(0, 3, size=X.shape[0])
    y = np.argmax(np.column_stack([X[:, 0], X[:, 1], X[:, 2] - 1]), axis=1)
    X[::9, 0] = np.nan
    pool = ctboost.Pool(X, y.astype(np.float32), cat_features=[2])
    booster = ctboost.train(
        pool,
        {
            "objective": "MultiClass",
            "num_classes": 3,
            "multi_strategy": "multi_output_tree",
            "max_depth": 2,
            "alpha": 1.0,
            "random_seed": 9,
            "nan_mode": "Min",
        },
        num_boost_round=4,
        learning_rate_schedule=[0.3, 0.17, 0.09, 0.04],
    )
    return booster, X


def test_compact_vector_json_and_python_exports_match_native(vector_model, tmp_path):
    booster, X = vector_model
    json_path = tmp_path / "vector.json"
    python_path = tmp_path / "vector.py"
    booster.export_model(json_path)
    booster.export_model(python_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["format_version"] == 2
    assert payload["multi_strategy"] == "multi_output_tree"
    assert len(payload["trees"]) == booster.num_iterations_trained == 4
    assert len(payload["tree_learning_rates"]) == 4
    manifest = ctboost.load_inference_manifest(json_path)
    assert manifest["schema_version"] == 2
    assert manifest["model"]["tree_count"] == 4
    assert manifest["model"]["iteration_count"] == 4
    assert manifest["model"]["trees_per_iteration"] == 1
    assert manifest["artifact"]["tree_representation"] == "shared_topology_vector_leaves"
    for predictor in (ctboost.load_exported_predictor(json_path), _python_module(python_path)):
        np.testing.assert_allclose(predictor.predict_raw(X), booster.predict(X), atol=1e-6)
        np.testing.assert_allclose(predictor.predict_raw(X[0]), booster.predict(X[:1])[0], atol=1e-6)
        np.testing.assert_allclose(predictor.predict_proba(X), _probabilities(booster, X), atol=1e-6)
        np.testing.assert_array_equal(predictor.predict_class(X), np.argmax(booster.predict(X), axis=1))
        assert predictor.predict_raw([]) == []


@pytest.mark.parametrize("corruption", ["version", "dimension", "strategy", "nonfinite", "rates"])
def test_vector_predictor_rejects_incompatible_payloads(vector_model, tmp_path, corruption):
    booster, _ = vector_model
    path = tmp_path / "vector.json"
    booster.export_model(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("inference_manifest")
    if corruption == "version":
        payload["format_version"] = 1
    elif corruption == "dimension":
        payload["trees"][0]["nodes"][0]["leaf_weights"].pop()
    elif corruption == "strategy":
        payload.pop("multi_strategy")
    elif corruption == "nonfinite":
        payload["trees"][0]["nodes"][0]["leaf_weights"][0] = float("nan")
    else:
        payload["tree_learning_rates"].pop()
    with pytest.raises(ValueError, match="vector"):
        ExportedPredictor(payload)


def test_vector_explanations_reconstruct_predictions_and_match_coalitions(vector_model):
    booster, X = vector_model
    foreground, background = X[:2], X[20:24]
    for iteration in (1, 2, None):
        shap = booster.predict_shap(foreground, background, num_iteration=iteration)
        interactions = booster.predict_shap_interactions(foreground, background, num_iteration=iteration)
        raw = booster.predict(foreground, num_iteration=iteration)
        assert shap.shape == (2, 3, 4)
        assert interactions.shape == (2, 3, 4, 4)
        np.testing.assert_allclose(shap.sum(axis=2), raw, atol=1e-6)
        np.testing.assert_allclose(interactions.sum(axis=3), shap, atol=1e-6)
        influence, coverage = booster.calc_leaf_influence(
            foreground, X, num_iteration=iteration, return_coverage=True
        )
        np.testing.assert_allclose(influence.sum(axis=2), raw - booster.base_score, atol=1e-6)
        np.testing.assert_allclose(coverage, 1.0)

    coalition_values = {}
    for mask in range(8):
        hybrid = background.copy()
        for feature in range(3):
            if mask & (1 << feature):
                hybrid[:, feature] = foreground[0, feature]
        coalition_values[mask] = booster.predict(hybrid).mean(axis=0)
    expected = np.zeros((3, 4))
    expected[:, -1] = coalition_values[0]
    for feature in range(3):
        for mask in range(8):
            if mask & (1 << feature):
                continue
            size = bin(mask).count("1")
            scale = factorial(size) * factorial(2 - size) / factorial(3)
            expected[:, feature] += scale * (
                coalition_values[mask | (1 << feature)] - coalition_values[mask]
            )
    np.testing.assert_allclose(shap[0], expected, atol=1e-6)


def test_vector_tree_plots_show_all_leaf_outputs(vector_model):
    booster, _ = vector_model
    dot = booster.tree_to_dot(tree_index=3)
    assert "weights=[" in dot
    assert "contributions=[" in dot
    state = booster._handle.export_state()
    leaf = next(node for node in state["trees"][3]["nodes"] if node["is_leaf"])
    expected = ", ".join(f"{value * state['tree_learning_rates'][3]:.6g}" for value in leaf["leaf_weights"])
    assert f"contributions=[{expected}]" in dot
    with pytest.raises(IndexError, match="out of range"):
        booster.tree_to_dot(tree_index=4)
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = booster.plot_tree(tree_index=3)
    assert any(text.get_text().startswith("leaf\n[") for text in ax.texts)
    plt.close(ax.figure)


def _compile_predictor(path, tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("cl")
    if compiler is None:
        if sys.platform == "win32" and shutil.which("cmake"):
            build = tmp_path / "cpp-build"
            source = Path(__file__).parent / "cpp_export_smoke"
            configured = subprocess.run(
                ["cmake", "-S", str(source), "-B", str(build), "-G", "Visual Studio 17 2022",
                 "-A", "x64", f"-DCTBOOST_GENERATED_SOURCE={path}"],
                capture_output=True, text=True, check=False,
            )
            if configured.returncode == 0:
                subprocess.run(
                    ["cmake", "--build", str(build), "--config", "Release"],
                    check=True, capture_output=True, text=True,
                )
                return ctypes.CDLL(str(build / "Release" / "ctboost_generated_export.dll"))
        pytest.skip("C++ compiler unavailable")
    library = tmp_path / ("predictor.dll" if sys.platform == "win32" else "predictor.so")
    if Path(compiler).stem.lower() == "cl":
        command = [compiler, "/nologo", "/LD", "/EHsc", "/std:c++17", str(path), f"/Fe:{library}"]
    else:
        command = [compiler, "-std=c++17", "-shared", "-fPIC", str(path), "-o", str(library)]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    return ctypes.CDLL(str(library))


def _cpp_predictions(library, X, dimensions):
    rows = np.ascontiguousarray(X, dtype=np.float32)
    predictions = np.empty((rows.shape[0], dimensions), dtype=np.float32)
    float_pointer = ctypes.POINTER(ctypes.c_float)
    predict = library.ctboost_predict
    predict.argtypes = [float_pointer, ctypes.c_size_t, ctypes.c_size_t, float_pointer, ctypes.c_size_t]
    predict.restype = ctypes.c_int
    status = predict(
        rows.ctypes.data_as(float_pointer), rows.shape[0], rows.shape[1],
        predictions.ctypes.data_as(float_pointer), predictions.size,
    )
    assert status == 0
    return predictions


def test_vector_cpp_export_expands_only_at_boundary(vector_model, tmp_path):
    booster, X = vector_model
    state_before = deepcopy(booster._handle.export_state())
    path = tmp_path / "vector.cpp"
    booster.export_model(path)
    source = path.read_text(encoding="utf-8")
    assert '"tree_representation":"expanded_scalar_trees"' in source
    assert '"exported_tree_count":12' in source
    assert booster._handle.export_state() == state_before
    library = _compile_predictor(path, tmp_path)
    np.testing.assert_allclose(_cpp_predictions(library, X, 3), booster.predict(X), atol=1e-6)


def test_cpp_export_integral_float_literals_compile_and_predict(tmp_path):
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    booster = ctboost.train(X, {"objective": "RMSE", "alpha": 1.0}, label=np.ones(10), num_boost_round=2)
    path = tmp_path / "scalar.cpp"
    booster.export_model(path)
    library = _compile_predictor(path, tmp_path)
    np.testing.assert_allclose(_cpp_predictions(library, X, 1)[:, 0], booster.predict(X), atol=1e-6)


def test_vector_onnx_export_matches_native(vector_model, tmp_path):
    onnx = pytest.importorskip("onnx")
    runtime = pytest.importorskip("onnxruntime")
    booster, X = vector_model
    path = tmp_path / "vector.onnx"
    booster.export_model(path)
    metadata = {entry.key: entry.value for entry in onnx.load(str(path)).metadata_props}
    manifest = json.loads(metadata["ctboost.inference_manifest"])
    assert manifest["artifact"]["tree_representation"] == "expanded_scalar_trees"
    session = runtime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = dict(zip([output.name for output in session.get_outputs()], session.run(None, {"features": X})))
    np.testing.assert_allclose(outputs["raw_predictions"], booster.predict(X), atol=1e-6)
    np.testing.assert_allclose(outputs["probabilities"], _probabilities(booster, X), atol=1e-6)


def test_standalone_python_predictors_respect_forbidden_missing_values(tmp_path):
    X = np.arange(32, dtype=np.float32).reshape(16, 2)
    booster = ctboost.train(
        X, {"objective": "RMSE", "nan_mode": "Forbidden", "alpha": 1.0},
        label=X[:, 0], num_boost_round=2,
    )
    paths = (tmp_path / "scalar.json", tmp_path / "scalar.py")
    for path in paths:
        booster.export_model(path)
    for predictor in (ctboost.load_exported_predictor(paths[0]), _python_module(paths[1])):
        for missing in (None, float("nan")):
            with pytest.raises(ValueError, match="Forbidden"):
                predictor.predict_raw([missing, 1.0])
