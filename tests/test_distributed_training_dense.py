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

from tests.helpers import find_free_tcp_port as _find_free_tcp_port
from tests.helpers import wait_for_tcp_listener as _wait_for_tcp_listener

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
    worker_zero = subprocess.Popen(
        [sys.executable, str(worker_script), "0", str(tmp_path), str(port)],
        env=worker_env,
    )
    _wait_for_tcp_listener(port)
    worker_one = subprocess.Popen(
        [sys.executable, str(worker_script), "1", str(tmp_path), str(port)],
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
