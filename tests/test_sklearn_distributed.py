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

from tests.helpers import find_free_tcp_port as _find_free_tcp_port
from tests.helpers import make_classification_data as _make_classification_data

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
            root.mkdir(parents=True, exist_ok=True)
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
