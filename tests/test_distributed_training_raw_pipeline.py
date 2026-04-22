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

def test_distributed_tcp_training_fits_raw_feature_pipeline_across_ranks(tmp_path: Path):
    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(79)
    frame = pd.DataFrame(
        {
            "city": rng.choice(["berlin", "rome", "oslo"], size=72),
            "text": rng.choice(["red quick fox", "blue slow fox"], size=72),
            "value": rng.normal(size=72).astype(np.float32),
        }
    )
    target = (
        0.8 * frame["value"].to_numpy(dtype=np.float32)
        + (frame["city"] == "berlin").to_numpy(dtype=np.float32)
        + 0.15 * (frame["text"] == "red quick fox").to_numpy(dtype=np.float32)
    ).astype(np.float32)

    shard_indices = [np.arange(0, 36), np.arange(36, 72)]
    frame.to_pickle(tmp_path / "frame_full.pkl")
    np.save(tmp_path / "target_full.npy", target)
    for rank, indices in enumerate(shard_indices):
        frame.iloc[indices].to_pickle(tmp_path / f"frame_train_{rank}.pkl")
        np.save(tmp_path / f"target_train_{rank}.npy", target[indices])

    port = _find_free_tcp_port()
    worker_script = tmp_path / "distributed_raw_pipeline_worker.py"
    worker_script.write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            import sys
            import numpy as np
            import pandas as pd
            import ctboost

            rank = int(sys.argv[1])
            root = Path(sys.argv[2])
            port = int(sys.argv[3])

            frame_train = pd.read_pickle(root / f"frame_train_{rank}.pkl")
            target_train = np.load(root / f"target_train_{rank}.npy")
            frame_full = pd.read_pickle(root / "frame_full.pkl")

            booster = ctboost.train(
                frame_train,
                {
                    "objective": "RMSE",
                    "learning_rate": 0.2,
                    "max_depth": 2,
                    "alpha": 1.0,
                    "lambda_l2": 1.0,
                    "random_seed": 11,
                    "cat_features": ["city"],
                    "ordered_ctr": True,
                    "text_features": ["text"],
                    "text_hash_dim": 16,
                    "distributed_world_size": 2,
                    "distributed_rank": rank,
                    "distributed_root": f"tcp://127.0.0.1:{port}",
                    "distributed_run_id": "raw-pipeline-case",
                    "distributed_timeout": 120.0,
                },
                label=target_train,
                num_boost_round=8,
            )
            np.save(root / f"raw_pipeline_pred_{rank}.npy", booster.predict(frame_full))
            """
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

    distributed_pred_0 = np.load(tmp_path / "raw_pipeline_pred_0.npy")
    distributed_pred_1 = np.load(tmp_path / "raw_pipeline_pred_1.npy")

    central = ctboost.train(
        frame,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "random_seed": 11,
            "cat_features": ["city"],
            "ordered_ctr": True,
            "text_features": ["text"],
            "text_hash_dim": 16,
        },
        label=target,
        num_boost_round=8,
    )
    central_pred = central.predict(frame)

    np.testing.assert_allclose(distributed_pred_0, central_pred, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(distributed_pred_1, central_pred, rtol=1e-6, atol=1e-6)
