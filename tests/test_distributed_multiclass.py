"""Multiclass workers must share structure targets and final class weights."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import ctboost
from tests.helpers import (
    authenticated_tcp_root,
    find_free_tcp_port,
    wait_for_tcp_listener,
)


@pytest.mark.parametrize("backend", ["filesystem", "tcp"])
@pytest.mark.parametrize("max_depth", [0, 2])
def test_distributed_multiclass_skewed_shards_match_central_fit(tmp_path, backend, max_depth):
    # The workers have different highest-variance class gradients. Unequal
    # shard weights also require the structure moments to share a global
    # denominator; local leaf fitting previously returned different models.
    labels = [
        np.repeat(np.arange(3), [40, 8, 2]).astype(np.float32),
        np.repeat(np.arange(3), [8, 27, 15]).astype(np.float32),
    ]
    features = [np.eye(3, dtype=np.float32)[label.astype(int)] for label in labels]
    weights = [np.full(len(label), rank + 1, dtype=np.float32)
               for rank, label in enumerate(labels)]
    X = np.concatenate(features)
    y = np.concatenate(labels)
    weight = np.concatenate(weights)
    params = {"objective": "MultiClass", "num_classes": 3, "learning_rate": 0.2,
              "max_depth": max_depth, "alpha": 0.05, "lambda_l2": 1.0,
              "random_seed": 41, "boost_from_average": False}
    np.save(tmp_path / "X_full.npy", X)
    (tmp_path / "params.json").write_text(json.dumps(params), encoding="utf-8")
    for rank in range(2):
        np.savez(tmp_path / f"shard_{rank}.npz", X=features[rank],
                 y=labels[rank], weight=weights[rank])

    worker = tmp_path / "worker.py"
    worker.write_text(textwrap.dedent("""
        import json
        from pathlib import Path
        import sys
        import numpy as np
        import ctboost

        rank, directory, coordinator = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
        shard = np.load(directory / f"shard_{rank}.npz")
        params = json.loads((directory / "params.json").read_text(encoding="utf-8"))
        params.update(distributed_world_size=2, distributed_rank=rank,
                      distributed_root=coordinator, distributed_run_id="multiclass",
                      distributed_timeout=30.0)
        pool = ctboost.Pool(shard["X"], shard["y"], weight=shard["weight"])
        model = ctboost.train(pool, params, num_boost_round=3)
        X = np.load(directory / "X_full.npy")
        np.save(directory / f"prediction_{rank}.npy", model.predict(X))
        (directory / f"state_{rank}.json").write_text(
            json.dumps(model._handle.export_state()), encoding="utf-8")
    """), encoding="utf-8")
    port = find_free_tcp_port() if backend == "tcp" else None
    coordinator = authenticated_tcp_root(port) if port is not None else str(tmp_path / "coordinator")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + environment.get("PYTHONPATH", "")
    workers = []
    try:
        for rank in range(2):
            workers.append(subprocess.Popen(
                [sys.executable, str(worker), str(rank), str(tmp_path), coordinator],
                env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ))
            if rank == 0 and port is not None:
                wait_for_tcp_listener(port, timeout=20.0)
        for process in workers:
            stdout, stderr = process.communicate(timeout=60)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in workers:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)

    central = ctboost.train(ctboost.Pool(X, y, weight=weight), params, num_boost_round=3)
    expected = central.predict(X)
    central_trees = central._handle.export_state()["trees"]
    if max_depth > 0:
        assert not central_trees[0]["nodes"][0]["is_leaf"]
        assert central_trees[0]["nodes"][0]["split_feature_id"] == 1
    worker_predictions = []
    for rank in range(2):
        actual = np.load(tmp_path / f"prediction_{rank}.npy")
        worker_predictions.append(actual)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        state = json.loads((tmp_path / f"state_{rank}.json").read_text(encoding="utf-8"))
        for tree, expected_tree in zip(state["trees"], central_trees):
            assert len(tree["nodes"]) == len(expected_tree["nodes"])
            for node, expected_node in zip(tree["nodes"], expected_tree["nodes"]):
                assert {key: value for key, value in node.items() if key != "leaf_weight"} == {
                    key: value for key, value in expected_node.items() if key != "leaf_weight"
                }
                np.testing.assert_allclose(node["leaf_weight"], expected_node["leaf_weight"],
                                           rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(worker_predictions[0], worker_predictions[1])
