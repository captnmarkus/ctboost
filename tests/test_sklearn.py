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


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_classification_data():
    X, y = make_classification(
        n_samples=192,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=13,
    )
    return X.astype(np.float32), y.astype(np.float32)


def test_classifier_predict_proba_and_feature_importances():
    X, y = _make_classification_data()

    clf = ctboost.CTBoostClassifier(
        iterations=18,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    clf.fit(X, y)

    probabilities = clf.predict_proba(X)
    predictions = clf.predict(X)
    importances = clf.feature_importances_

    assert probabilities.shape == (X.shape[0], 2)
    assert predictions.shape == (X.shape[0],)
    assert importances.shape == (X.shape[1],)
    assert np.all(importances >= 0.0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(importances.sum(), 100.0, rtol=1e-5, atol=1e-5)


def test_regressor_feature_importances_match_feature_count():
    X, y = make_regression(
        n_samples=128,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=21,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    reg = ctboost.CTBoostRegressor(
        iterations=12,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    reg.fit(X, y)

    assert reg.predict(X).shape == (X.shape[0],)
    assert reg.feature_importances_.shape == (X.shape[1],)


def test_regressor_accepts_learning_rate_schedule():
    X, y = make_regression(
        n_samples=128,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=27,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    schedule = [0.2, 0.2, 0.1, 0.1, 0.05, 0.05]

    reg = ctboost.CTBoostRegressor(
        iterations=len(schedule),
        learning_rate=schedule[0],
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
        random_seed=17,
    )
    reg.fit(X, y, learning_rate_schedule=schedule)

    assert reg._booster.learning_rate_history == pytest.approx(schedule)
    assert reg.predict(X).shape == (X.shape[0],)


def test_regressor_early_stopping_uses_eval_set():
    X, y = make_regression(
        n_samples=256,
        n_features=8,
        n_informative=5,
        noise=20.0,
        random_state=29,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X_train, X_val = X[:96], X[96:]
    y_train = y[:96]
    y_val = -y[96:]

    reg = ctboost.CTBoostRegressor(
        iterations=500,
        learning_rate=0.3,
        max_depth=4,
        alpha=1.0,
        lambda_l2=0.5,
    )
    reg.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=15,
    )

    predictions = reg.predict(X_val)

    assert reg.best_iteration_ + 1 < 500
    assert len(reg._booster.loss_history) == reg.best_iteration_ + 1
    assert predictions.shape == (X_val.shape[0],)


def test_regressor_distributed_fit_matches_central_estimator(tmp_path: Path):
    X, y = make_regression(
        n_samples=128,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=91,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    shard_indices = [np.arange(0, 64), np.arange(64, 128)]
    np.save(tmp_path / "X_full.npy", X)
    for rank, indices in enumerate(shard_indices):
        np.save(tmp_path / f"X_shard_{rank}.npy", X[indices])
        np.save(tmp_path / f"y_shard_{rank}.npy", y[indices])

    port = _find_free_tcp_port()
    worker_script = tmp_path / "sklearn_dist_worker.py"
    worker_script.write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            import sys
            import numpy as np
            import ctboost

            rank = int(sys.argv[1])
            root = Path(sys.argv[2])
            port = int(sys.argv[3])
            X = np.load(root / f"X_shard_{rank}.npy")
            y = np.load(root / f"y_shard_{rank}.npy")
            X_full = np.load(root / "X_full.npy")

            reg = ctboost.CTBoostRegressor(
                iterations=10,
                learning_rate=0.2,
                max_depth=2,
                alpha=1.0,
                lambda_l2=1.0,
                random_seed=7,
                distributed_world_size=2,
                distributed_rank=rank,
                distributed_root=f"tcp://127.0.0.1:{port}",
                distributed_run_id="sklearn-case",
                distributed_timeout=120.0,
            )
            reg.fit(X, y)
            np.save(root / f"sk_pred_{rank}.npy", reg.predict(X_full))
            """,
        ),
        encoding="utf-8",
    )

    worker_env = os.environ.copy()
    worker_env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + worker_env.get("PYTHONPATH", "")
    worker_one = subprocess.Popen(
        [sys.executable, str(worker_script), "1", str(tmp_path), str(port)],
        env=worker_env,
    )
    worker_zero = subprocess.Popen(
        [sys.executable, str(worker_script), "0", str(tmp_path), str(port)],
        env=worker_env,
    )
    assert worker_one.wait(timeout=180) == 0
    assert worker_zero.wait(timeout=180) == 0

    distributed_pred_0 = np.load(tmp_path / "sk_pred_0.npy")
    distributed_pred_1 = np.load(tmp_path / "sk_pred_1.npy")

    central = ctboost.CTBoostRegressor(
        iterations=10,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
        random_seed=7,
    )
    central.fit(X, y)
    central_pred = central.predict(X)

    np.testing.assert_allclose(distributed_pred_0, central_pred, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(distributed_pred_1, central_pred, rtol=1e-6, atol=1e-6)


def test_classifier_cpu_gpu_parity_or_graceful_cuda_error():
    X, y = _make_classification_data()

    cpu = ctboost.CTBoostClassifier(
        iterations=16,
        learning_rate=0.15,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
        task_type="CPU",
    )
    cpu.fit(X, y)

    if not ctboost.build_info()["cuda_enabled"]:
        gpu = ctboost.CTBoostClassifier(
            iterations=16,
            learning_rate=0.15,
            max_depth=2,
            alpha=1.0,
            lambda_l2=1.0,
            task_type="GPU",
        )
        with pytest.raises(RuntimeError, match="compiled without CUDA"):
            gpu.fit(X, y)
        return

    gpu = ctboost.CTBoostClassifier(
        iterations=16,
        learning_rate=0.15,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
        task_type="GPU",
    )
    try:
        gpu.fit(X, y)
    except RuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable for parity test: {exc}")

    np.testing.assert_allclose(
        gpu.predict_proba(X),
        cpu.predict_proba(X),
        rtol=1e-4,
        atol=1e-4,
    )


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
