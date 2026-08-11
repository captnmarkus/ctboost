import pickle
import threading
from pathlib import Path

import numpy as np

from ctboost.training import _distributed_pool


def test_distributed_pickle_is_published_only_after_serialization(
    tmp_path: Path, monkeypatch
):
    target = tmp_path / "feature_pipeline_rank_00000.pkl"
    payload = {"data": np.arange(12, dtype=np.float32).reshape(4, 3)}
    serialization_complete = threading.Event()
    allow_publication = threading.Event()
    errors = []
    original_dump = pickle.dump

    def blocked_dump(value, stream, *, protocol):
        original_dump(value, stream, protocol=protocol)
        serialization_complete.set()
        if not allow_publication.wait(timeout=10):
            raise TimeoutError("test did not release atomic pickle publication")

    monkeypatch.setattr(_distributed_pool.pickle, "dump", blocked_dump)

    def publish():
        try:
            _distributed_pool._atomic_pickle_dump(target, payload)
        except Exception as exc:  # pragma: no cover - surfaced by the assertion
            errors.append(exc)

    worker = threading.Thread(target=publish)
    worker.start()
    assert serialization_complete.wait(timeout=10)
    assert not target.exists()
    assert len(list(tmp_path.glob(".*.tmp"))) == 1

    allow_publication.set()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert errors == []
    assert not list(tmp_path.glob(".*.tmp"))
    with target.open("rb") as stream:
        restored = pickle.load(stream)
    np.testing.assert_array_equal(restored["data"], payload["data"])
