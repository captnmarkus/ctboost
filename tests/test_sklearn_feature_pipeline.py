import os
from pathlib import Path
import socket
import subprocess
import sys
import textwrap
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
import ctboost

def test_feature_pipeline_handles_ctr_text_and_embedding_columns(tmp_path):
    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(41)
    category = np.where(rng.random(96) > 0.5, "alpha", "beta")
    text = np.where(category == "alpha", "red quick fox", "blue slow fox")
    embedding = [np.asarray([float(index % 3), float(index % 5), float(index % 7)], dtype=np.float32) for index in range(96)]
    numeric = rng.normal(size=96).astype(np.float32)
    target = (
        (category == "alpha").astype(np.float32) * 1.5
        + 0.25 * numeric
        + np.asarray([values[0] - 0.1 * values[1] for values in embedding], dtype=np.float32)
    )

    frame = pd.DataFrame(
        {
            "category": category,
            "text": text,
            "embedding": embedding,
            "numeric": numeric,
        }
    )

    reg = ctboost.CTBoostRegressor(
        iterations=18,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
        ordered_ctr=True,
        cat_features=["category"],
        categorical_combinations=[["category", "text"]],
        text_features=["text"],
        text_hash_dim=16,
        embedding_features=["embedding"],
        embedding_stats=("mean", "std", "l2"),
    )
    reg.fit(frame, target)

    prediction = reg.predict(frame)
    assert prediction.shape == (frame.shape[0],)
    assert np.all(np.isfinite(prediction))
    assert reg.n_features_in_ > frame.shape[1]

    model_path = tmp_path / "pipeline_regressor.ctb"
    reg.save_model(model_path)
    restored = ctboost.CTBoostRegressor.load_model(model_path)
    restored_prediction = restored.predict(frame)
    np.testing.assert_allclose(prediction, restored_prediction, rtol=1e-6, atol=1e-6)

def test_feature_pipeline_supports_one_hot_threshold_category_bucketing_and_per_source_ctrs():
    pd = pytest.importorskip("pandas")

    frame = pd.DataFrame(
        {
            "small_cat": ["a", "b", "a", "c", "b", "c"],
            "large_cat": ["x0", "x1", "x2", "x3", "x4", "x5"],
            "numeric": np.asarray([0.2, 0.1, -0.4, 0.7, -0.2, 0.5], dtype=np.float32),
        }
    )
    target = np.asarray([1.0, 0.0, 1.0, 0.2, 0.0, 0.5], dtype=np.float32)

    pipeline = ctboost.FeaturePipeline(
        cat_features=["small_cat", "large_cat"],
        one_hot_max_size=3,
        max_cat_threshold=3,
        simple_ctr=["Frequency"],
        per_feature_ctr={"large_cat": ["Mean"]},
    )
    transformed, cat_features, feature_names = pipeline.fit_transform_array(frame, target)

    assert transformed.shape[0] == frame.shape[0]
    assert "small_cat_is_a" in feature_names
    assert "small_cat_is_b" in feature_names
    assert "small_cat_is_c" in feature_names
    assert "large_cat_ctr" in feature_names
    assert "large_cat_freq_ctr" not in feature_names

    large_cat_column = transformed[:, cat_features[0]]
    assert np.unique(large_cat_column[~np.isnan(large_cat_column)]).size <= 3

    expected_transformed, expected_cat_features, expected_feature_names = pipeline.transform_array(frame)
    restored = ctboost.FeaturePipeline.from_state(pipeline.to_state())
    restored_transformed, restored_cat_features, restored_feature_names = restored.transform_array(frame)
    np.testing.assert_allclose(expected_transformed, restored_transformed, rtol=1e-6, atol=1e-6)
    assert restored_cat_features == cat_features
    assert restored_feature_names == expected_feature_names

def test_regressor_accepts_named_quantization_controls_and_exports_borders():
    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(19)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=96).astype(np.float32),
            "b": rng.normal(size=96).astype(np.float32),
            "c": rng.normal(size=96).astype(np.float32),
        }
    )
    target = (1.2 * frame["a"] - 0.4 * frame["b"] + 0.1 * frame["c"]).to_numpy(dtype=np.float32)
    frame.loc[::7, "c"] = np.nan

    reg = ctboost.CTBoostRegressor(
        iterations=10,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
        max_bins=48,
        max_bin_by_feature={"a": 5},
        border_selection_method="Uniform",
        nan_mode_by_feature={"c": "Max"},
        feature_borders={"b": [-0.5, 0.0, 0.5]},
    )
    reg.fit(frame, target)

    borders = reg._booster.get_borders()
    assert borders is not None
    assert borders["nan_mode_by_feature"][2] == "Max"
    np.testing.assert_allclose(
        np.asarray(borders["feature_borders"][1], dtype=np.float32),
        np.array([-0.5, 0.0, 0.5], dtype=np.float32),
    )
