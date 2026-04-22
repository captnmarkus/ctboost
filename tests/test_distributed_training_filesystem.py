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

def test_distributed_filesystem_training_supports_advanced_eval_without_tcp(tmp_path: Path):
    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(83)
    frame = pd.DataFrame(
        {
            "city": rng.choice(["berlin", "rome", "oslo"], size=72),
            "text": rng.choice(["red quick fox", "blue slow fox"], size=72),
            "value": rng.normal(size=72).astype(np.float32),
        }
    )
    margin = (
        0.8 * frame["value"].to_numpy(dtype=np.float32)
        + 0.9 * (frame["city"] == "berlin").to_numpy(dtype=np.float32)
        + 0.3 * (frame["text"] == "red quick fox").to_numpy(dtype=np.float32)
    )
    target = (margin > np.median(margin)).astype(np.float32)

    train_indices = [np.arange(0, 24), np.arange(24, 48)]
    eval_indices = [np.arange(48, 60), np.arange(60, 72)]
    frame.to_pickle(tmp_path / "frame_full.pkl")
    np.save(tmp_path / "target_full.npy", target)
    for rank, indices in enumerate(train_indices):
        frame.iloc[indices].to_pickle(tmp_path / f"frame_train_{rank}.pkl")
        np.save(tmp_path / f"target_train_{rank}.npy", target[indices])
    for rank, indices in enumerate(eval_indices):
        frame.iloc[indices].to_pickle(tmp_path / f"frame_eval_{rank}.pkl")
        np.save(tmp_path / f"target_eval_{rank}.npy", target[indices])

    worker_script = tmp_path / "distributed_filesystem_advanced_worker.py"
    worker_script.write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            import json
            import sys
            import numpy as np
            import pandas as pd
            import ctboost

            rank = int(sys.argv[1])
            root = Path(sys.argv[2])
            frame_train = pd.read_pickle(root / f"frame_train_{rank}.pkl")
            target_train = np.load(root / f"target_train_{rank}.npy")
            frame_eval = pd.read_pickle(root / f"frame_eval_{rank}.pkl")
            target_eval = np.load(root / f"target_eval_{rank}.npy")
            frame_full = pd.read_pickle(root / "frame_full.pkl")

            booster = ctboost.train(
                frame_train,
                {
                    "objective": "Logloss",
                    "eval_metric": ["BalancedAccuracy", "AUC"],
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
                    "distributed_root": str(root / "dist_run"),
                    "distributed_run_id": "filesystem-advanced-case",
                    "distributed_timeout": 120.0,
                },
                label=target_train,
                num_boost_round=18,
                eval_set=[(frame_eval, target_eval)],
                eval_names=["holdout"],
                early_stopping_rounds=4,
                callbacks=[
                    ctboost.checkpoint_callback(root / "filesystem_checkpoint.json", interval=2),
                ],
            )
            np.save(root / f"filesystem_pred_{rank}.npy", booster.predict(frame_full))
            with (root / f"filesystem_state_{rank}.json").open("w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "best_iteration": booster.best_iteration,
                        "evals_result": booster.evals_result_,
                    },
                    stream,
                    sort_keys=True,
                )
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

    distributed_pred_0 = np.load(tmp_path / "filesystem_pred_0.npy")
    distributed_pred_1 = np.load(tmp_path / "filesystem_pred_1.npy")
    with (tmp_path / "filesystem_state_0.json").open("r", encoding="utf-8") as stream:
        distributed_state_0 = json.load(stream)
    with (tmp_path / "filesystem_state_1.json").open("r", encoding="utf-8") as stream:
        distributed_state_1 = json.load(stream)
    assert (tmp_path / "filesystem_checkpoint.json").exists()

    central = ctboost.train(
        frame.iloc[:48],
        {
            "objective": "Logloss",
            "eval_metric": ["BalancedAccuracy", "AUC"],
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
        label=target[:48],
        num_boost_round=18,
        eval_set=[(frame.iloc[48:], target[48:])],
        eval_names=["holdout"],
        early_stopping_rounds=4,
    )
    central_pred = central.predict(frame)

    np.testing.assert_allclose(distributed_pred_0, central_pred, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(distributed_pred_1, central_pred, rtol=1e-6, atol=1e-6)
    assert distributed_state_0["best_iteration"] == central.best_iteration
    assert distributed_state_1["best_iteration"] == central.best_iteration
    assert distributed_state_0["evals_result"] == central.evals_result_
    assert distributed_state_1["evals_result"] == central.evals_result_
