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


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_distributed_tcp_request_retries_until_coordinator_is_ready():
    port = _find_free_tcp_port()
    root = f"tcp://127.0.0.1:{port}"
    server = DistributedCollectiveServer("127.0.0.1", port)

    def delayed_start() -> None:
        time.sleep(0.2)
        server.start()

    starter = threading.Thread(target=delayed_start, daemon=True)
    starter.start()
    try:
        response = distributed_tcp_request(root, 5.0, "ping", "__health__", 0, 1, b"")
        assert response == b""
    finally:
        starter.join(timeout=5.0)
        server.stop()


def test_booster_save_load_and_staged_predict_round_trip(tmp_path: Path):
    X, y = make_regression(
        n_samples=160,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=41,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    train_pool = ctboost.Pool(X[:120], y[:120])
    valid_pool = ctboost.Pool(X[120:], y[120:])
    full_pool = ctboost.Pool(X, y)
    booster = ctboost.train(
        train_pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.15,
            "max_depth": 3,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=12,
        eval_set=valid_pool,
    )

    staged_predictions = list(booster.staged_predict(full_pool))

    assert len(staged_predictions) == booster.num_iterations_trained
    np.testing.assert_allclose(staged_predictions[-1], booster.predict(full_pool), rtol=1e-6, atol=1e-6)
    assert booster.eval_loss_history

    model_path = tmp_path / "booster.pkl"
    booster.save_model(model_path)
    restored = ctboost.load_model(model_path)

    np.testing.assert_allclose(restored.predict(full_pool), booster.predict(full_pool), rtol=1e-6, atol=1e-6)
    assert restored.loss_history == booster.loss_history
    assert restored.eval_loss_history == booster.eval_loss_history


def test_booster_json_save_load_round_trip(tmp_path: Path):
    X, y = make_regression(
        n_samples=96,
        n_features=5,
        n_informative=4,
        noise=0.1,
        random_state=29,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    pool = ctboost.Pool(X, y)
    booster = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=8,
    )

    model_path = tmp_path / "booster.json"
    booster.save_model(model_path)
    restored = ctboost.load_model(model_path)

    np.testing.assert_allclose(restored.predict(pool), booster.predict(pool), rtol=1e-6, atol=1e-6)
    assert restored.objective_name == booster.objective_name
    assert restored.eval_metric_name == booster.eval_metric_name


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


def test_distributed_multi_host_training_merges_shards_and_matches_central_fit(tmp_path: Path):
    X, y = make_regression(
        n_samples=96,
        n_features=6,
        n_informative=4,
        noise=0.15,
        random_state=31,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    shard_indices = [np.arange(0, 48), np.arange(48, 96)]
    np.save(tmp_path / "X_full.npy", X)
    for rank, indices in enumerate(shard_indices):
        np.save(tmp_path / f"X_shard_{rank}.npy", X[indices])
        np.save(tmp_path / f"y_shard_{rank}.npy", y[indices])

    worker_script = tmp_path / "distributed_worker.py"
    worker_script.write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            import json
            import sys
            import numpy as np
            import ctboost

            rank = int(sys.argv[1])
            root = Path(sys.argv[2])
            X = np.load(root / f"X_shard_{rank}.npy")
            y = np.load(root / f"y_shard_{rank}.npy")
            X_full = np.load(root / "X_full.npy")

            booster = ctboost.train(
                X,
                {
                    "objective": "RMSE",
                    "learning_rate": 0.2,
                    "max_depth": 2,
                    "alpha": 1.0,
                    "lambda_l2": 1.0,
                    "random_seed": 11,
                    "external_memory": True,
                    "external_memory_dir": str(root / "native_ext"),
                    "distributed_world_size": 2,
                    "distributed_rank": rank,
                    "distributed_root": str(root / "dist_run"),
                    "distributed_run_id": "case-1",
                    "distributed_timeout": 120.0,
                },
                label=y,
                num_boost_round=6,
            )
            np.save(root / f"pred_{rank}.npy", booster.predict(X_full))
            with (root / f"schema_{rank}.json").open("w", encoding="utf-8") as stream:
                json.dump(booster.get_quantization_schema(), stream, sort_keys=True)
            """
        ),
        encoding="utf-8",
    )

    worker_env = os.environ.copy()
    worker_env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + worker_env.get("PYTHONPATH", "")
    worker_one = subprocess.Popen(
        [sys.executable, str(worker_script), "1", str(tmp_path)],
        env=worker_env,
    )
    worker_zero = subprocess.Popen(
        [sys.executable, str(worker_script), "0", str(tmp_path)],
        env=worker_env,
    )
    assert worker_one.wait(timeout=180) == 0
    assert worker_zero.wait(timeout=180) == 0

    distributed_pred_0 = np.load(tmp_path / "pred_0.npy")
    distributed_pred_1 = np.load(tmp_path / "pred_1.npy")
    with (tmp_path / "schema_0.json").open("r", encoding="utf-8") as stream:
        distributed_schema_0 = json.load(stream)
    with (tmp_path / "schema_1.json").open("r", encoding="utf-8") as stream:
        distributed_schema_1 = json.load(stream)
    central = ctboost.train(
        X,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "random_seed": 11,
            "external_memory": True,
            "external_memory_dir": str(tmp_path / "central_ext"),
        },
        label=y,
        num_boost_round=6,
    )
    central_pred = central.predict(X)
    central_schema = central.get_quantization_schema()

    np.testing.assert_allclose(distributed_pred_0, central_pred, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(distributed_pred_1, central_pred, rtol=1e-6, atol=1e-6)
    assert distributed_schema_0 == central_schema
    assert distributed_schema_1 == central_schema


def test_distributed_tcp_training_supports_eval_set_and_init_model(tmp_path: Path):
    X, y = make_regression(
        n_samples=160,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=67,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_train, X_eval = X[:96], X[96:]
    y_train, y_eval = y[:96], y[96:]

    init_model = ctboost.train(
        X_train,
        {
            "objective": "RMSE",
            "learning_rate": 0.18,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "random_seed": 23,
        },
        label=y_train,
        num_boost_round=3,
    )
    init_model_path = tmp_path / "init_model.ctb"
    init_model.save_model(init_model_path)

    shard_indices = [np.arange(0, 48), np.arange(48, 96)]
    eval_indices = [np.arange(0, 32), np.arange(32, 64)]
    np.save(tmp_path / "X_full.npy", X)
    for rank, indices in enumerate(shard_indices):
        np.save(tmp_path / f"X_train_{rank}.npy", X_train[indices])
        np.save(tmp_path / f"y_train_{rank}.npy", y_train[indices])
        np.save(tmp_path / f"X_eval_{rank}.npy", X_eval[eval_indices[rank]])
        np.save(tmp_path / f"y_eval_{rank}.npy", y_eval[eval_indices[rank]])

    port = _find_free_tcp_port()
    worker_script = tmp_path / "distributed_eval_worker.py"
    worker_script.write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            import json
            import sys
            import numpy as np
            import ctboost

            rank = int(sys.argv[1])
            root = Path(sys.argv[2])
            port = int(sys.argv[3])
            X_train = np.load(root / f"X_train_{rank}.npy")
            y_train = np.load(root / f"y_train_{rank}.npy")
            X_eval = np.load(root / f"X_eval_{rank}.npy")
            y_eval = np.load(root / f"y_eval_{rank}.npy")
            X_full = np.load(root / "X_full.npy")
            init_model = ctboost.load_model(root / "init_model.ctb")

            booster = ctboost.train(
                X_train,
                {
                    "objective": "RMSE",
                    "eval_metric": "RMSE",
                    "learning_rate": 0.18,
                    "max_depth": 2,
                    "alpha": 1.0,
                    "lambda_l2": 1.0,
                    "random_seed": 23,
                    "distributed_world_size": 2,
                    "distributed_rank": rank,
                    "distributed_root": f"tcp://127.0.0.1:{port}",
                    "distributed_run_id": "eval-init-case",
                    "distributed_timeout": 120.0,
                },
                label=y_train,
                num_boost_round=12,
                eval_set=(X_eval, y_eval),
                early_stopping_rounds=4,
                init_model=init_model,
            )
            np.save(root / f"eval_pred_{rank}.npy", booster.predict(X))
            with (root / f"eval_state_{rank}.json").open("w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "loss_history": booster.loss_history,
                        "eval_loss_history": booster.eval_loss_history,
                        "best_iteration": booster.best_iteration,
                    },
                    stream,
                    sort_keys=True,
                )
            """.replace("booster.predict(X)", "booster.predict(X_full)")
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

    distributed_pred_0 = np.load(tmp_path / "eval_pred_0.npy")
    distributed_pred_1 = np.load(tmp_path / "eval_pred_1.npy")
    with (tmp_path / "eval_state_0.json").open("r", encoding="utf-8") as stream:
        distributed_state_0 = json.load(stream)
    with (tmp_path / "eval_state_1.json").open("r", encoding="utf-8") as stream:
        distributed_state_1 = json.load(stream)

    central = ctboost.train(
        X_train,
        {
            "objective": "RMSE",
            "eval_metric": "RMSE",
            "learning_rate": 0.18,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "random_seed": 23,
        },
        label=y_train,
        num_boost_round=12,
        eval_set=(X_eval, y_eval),
        early_stopping_rounds=4,
        init_model=ctboost.load_model(init_model_path),
    )
    central_pred = central.predict(X)

    np.testing.assert_allclose(distributed_pred_0, central_pred, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(distributed_pred_1, central_pred, rtol=1e-6, atol=1e-6)
    assert distributed_state_0["best_iteration"] == central.best_iteration
    assert distributed_state_1["best_iteration"] == central.best_iteration
    np.testing.assert_allclose(
        np.asarray(distributed_state_0["loss_history"], dtype=np.float64),
        np.asarray(central.loss_history, dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(distributed_state_0["eval_loss_history"], dtype=np.float64),
        np.asarray(central.eval_loss_history, dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(distributed_state_1["eval_loss_history"], dtype=np.float64),
        np.asarray(central.eval_loss_history, dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_distributed_tcp_ranking_training_matches_central_fit(tmp_path: Path):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(48, 5)).astype(np.float32)
    y = np.tile(np.asarray([3.0, 2.0, 1.0, 0.0], dtype=np.float32), 12)
    group_id = np.repeat(np.arange(12, dtype=np.int64), 4)

    shard_indices = [np.arange(0, 24), np.arange(24, 48)]
    np.save(tmp_path / "X_full_rank.npy", X)
    for rank, indices in enumerate(shard_indices):
        np.save(tmp_path / f"X_rank_{rank}.npy", X[indices])
        np.save(tmp_path / f"y_rank_{rank}.npy", y[indices])
        np.save(tmp_path / f"group_rank_{rank}.npy", group_id[indices])

    port = _find_free_tcp_port()
    worker_script = tmp_path / "distributed_rank_worker.py"
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
            X = np.load(root / f"X_rank_{rank}.npy")
            y = np.load(root / f"y_rank_{rank}.npy")
            group_id = np.load(root / f"group_rank_{rank}.npy")
            X_full = np.load(root / "X_full_rank.npy")

            booster = ctboost.train(
                X,
                {
                    "objective": "PairLogit",
                    "eval_metric": "NDCG",
                    "learning_rate": 0.15,
                    "max_depth": 2,
                    "alpha": 1.0,
                    "lambda_l2": 1.0,
                    "random_seed": 9,
                    "distributed_world_size": 2,
                    "distributed_rank": rank,
                    "distributed_root": f"tcp://127.0.0.1:{port}",
                    "distributed_run_id": "ranking-case",
                    "distributed_timeout": 120.0,
                },
                label=y,
                group_id=group_id,
                num_boost_round=8,
            )
            np.save(root / f"rank_pred_{rank}.npy", booster.predict(X_full))
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

    distributed_pred_0 = np.load(tmp_path / "rank_pred_0.npy")
    distributed_pred_1 = np.load(tmp_path / "rank_pred_1.npy")
    central = ctboost.train(
        X,
        {
            "objective": "PairLogit",
            "eval_metric": "NDCG",
            "learning_rate": 0.15,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "random_seed": 9,
        },
        label=y,
        group_id=group_id,
        num_boost_round=8,
    )
    central_pred = central.predict(X)

    np.testing.assert_allclose(distributed_pred_0, central_pred, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(distributed_pred_1, central_pred, rtol=1e-6, atol=1e-6)


def test_booster_state_stores_quantization_schema_once_and_loads_legacy_tree_schema():
    X, y = make_regression(
        n_samples=128,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=13,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    pool = ctboost.Pool(X, y)
    booster = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 3,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=4,
    )

    state = dict(booster._handle.export_state())
    assert "quantization_schema" in state
    for tree_state in state["trees"]:
        assert "num_bins_per_feature" not in tree_state
        assert "cut_offsets" not in tree_state
        assert "cut_values" not in tree_state
        assert "categorical_mask" not in tree_state
        assert "missing_value_mask" not in tree_state
        assert "nan_mode" not in tree_state

    legacy_state = dict(state)
    quantization_schema = dict(legacy_state.pop("quantization_schema"))
    legacy_tree_states = []
    for tree_state in legacy_state["trees"]:
        upgraded_tree_state = dict(tree_state)
        upgraded_tree_state.update(quantization_schema)
        legacy_tree_states.append(upgraded_tree_state)
    legacy_state["trees"] = legacy_tree_states

    restored = ctboost.Booster(_core.GradientBooster.from_state(legacy_state))
    np.testing.assert_allclose(restored.predict(pool), booster.predict(pool), rtol=1e-6, atol=1e-6)


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


def test_classifier_save_load_and_staged_predict_proba(tmp_path: Path):
    X, y = make_classification(
        n_samples=192,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=23,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    clf = ctboost.CTBoostClassifier(
        iterations=14,
        learning_rate=0.15,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    clf.fit(X, y)

    staged_probabilities = list(clf.staged_predict_proba(X))

    assert len(staged_probabilities) == clf._booster.num_iterations_trained
    np.testing.assert_allclose(staged_probabilities[-1], clf.predict_proba(X), rtol=1e-6, atol=1e-6)

    model_path = tmp_path / "classifier.pkl"
    clf.save_model(model_path)
    restored = ctboost.CTBoostClassifier.load_model(model_path)

    np.testing.assert_array_equal(restored.classes_, clf.classes_)
    np.testing.assert_allclose(restored.predict_proba(X), clf.predict_proba(X), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(restored.predict(X), clf.predict(X))


def test_classifier_json_save_load_round_trip(tmp_path: Path):
    X, y = make_classification(
        n_samples=128,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=31,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    clf = ctboost.CTBoostClassifier(
        iterations=10,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    clf.fit(X, y)

    model_path = tmp_path / "classifier.json"
    clf.save_model(model_path)
    restored = ctboost.CTBoostClassifier.load_model(model_path)

    np.testing.assert_array_equal(restored.classes_, clf.classes_)
    np.testing.assert_allclose(restored.predict_proba(X), clf.predict_proba(X), rtol=1e-6, atol=1e-6)
