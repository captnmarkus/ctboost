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
