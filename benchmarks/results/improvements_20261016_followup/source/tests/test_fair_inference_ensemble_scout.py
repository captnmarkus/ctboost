"""Synthetic latency-runner checks; no timers, fitting, or real-data access."""
import copy
import hashlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from benchmarks import fair_inference_ensemble_scout as runner
from benchmarks import fixed_seed_ensemble as ensemble


class RowModel:
    num_iterations_trained = 3

    def __init__(self, offset):
        self.offset = offset
        self.inplace_delta = 0.
        self.calls = 0

    def predict(self, frame, **kwargs):
        self.calls += 1
        return np.asarray(frame["value"], dtype=np.float32) + self.offset

    def inplace_predict(self, frame, **kwargs):
        return self.predict(frame, **kwargs) + self.inplace_delta

    def num_boosted_rounds(self):
        return 3


def payload(library="xgboost"):
    return ensemble.make_payload([
        {"seed": seed, "model_id": f"toy/{library}/seed{seed}", "model_sha256": str(index)*64,
         "payload": {"library": library, "problem": "binary", "schema": {}, "model": RowModel(float(index))}}
        for index, seed in enumerate((47, 947), 1)])


def test_every_frozen_batch_checks_xgb_components_and_ensemble(monkeypatch):
    pd = pytest.importorskip("pandas")
    monkeypatch.setitem(sys.modules, "xgboost", types.SimpleNamespace(DMatrix=lambda frame, **kwargs: frame))
    model = payload()
    source = pd.DataFrame({"value": np.array([-2., .5, 3.], dtype=np.float32)})
    expected = ensemble.predict(model, source)
    checked_rows = []
    original = ensemble.verify_xgb_inplace
    def verify(model, frame):
        checked_rows.append(len(frame))
        return original(model, frame)
    monkeypatch.setattr(ensemble, "verify_xgb_inplace", verify)
    for count in (1, 64, 1000):
        positions = np.arange(count) % len(source)
        raw = source.iloc[positions].reset_index(drop=True)
        normalized, actual, guard = runner.check_batch(model, raw, expected[positions])
        assert len(normalized) == count and guard["bit_exact"]
        runner.assert_bits(actual, expected[positions])
    assert checked_rows == [1, 64, 1000]


def test_inplace_difference_blocks_batch_before_timing(monkeypatch):
    pd = pytest.importorskip("pandas")
    monkeypatch.setitem(sys.modules, "xgboost", types.SimpleNamespace(DMatrix=lambda frame, **kwargs: frame))
    model = payload()
    model["components"][1]["payload"]["model"].inplace_delta = .125
    with pytest.raises(AssertionError):
        runner.check_batch(model, pd.DataFrame({"value": [1., 2.]}))


def test_each_timed_path_retains_two_complete_model_calls():
    pd = pytest.importorskip("pandas")
    model = payload("ctboost")
    frame = pd.DataFrame({"value": [1., 2.]})
    native = [record["payload"]["model"] for record in model["components"]]
    ensemble.predict(model, frame, xgb_inplace=True)
    assert [item.calls for item in native] == [1, 1]
    normalized, _ = runner.accuracy.normalize(frame, model["schema"])
    ensemble.predict_normalized(model, normalized, xgb_inplace=True)
    assert [item.calls for item in native] == [2, 2]


def test_component_file_hash_and_metadata_are_checked(tmp_path):
    model = payload("ctboost")
    sources = []
    for record in model["components"]:
        path = tmp_path / f"{record['seed']}.pkl"
        path.write_bytes(f"synthetic-{record['seed']}".encode())
        record["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append({"dataset": "toy", "library": "ctboost", "seed": record["seed"],
                        "model_path": str(path), "model_sha256": record["model_sha256"], "model_bytes": path.stat().st_size})
    selected = {"seed_pair": [47, 947], "components": copy.deepcopy(sources)}
    result = runner.component_metadata(model, selected, sources, "toy")
    assert [item["retained_rounds"] for item in result] == [3, 3]
    assert [item["seed"] for item in result] == [47, 947]
    Path(sources[1]["model_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="component identity"):
        runner.component_metadata(model, selected, sources, "toy")


def test_exact_check_rejects_dtype_change():
    with pytest.raises(ValueError, match="dtype"):
        runner.assert_bits(np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float64))


def test_frozen_dependencies_require_both_math_helpers_and_normalizer(tmp_path):
    dependencies = {str(Path(module.__file__).resolve()): runner.accuracy.digest(module.__file__)
                    for module in (runner.accuracy, runner.ensemble, runner.fixed_temperature)}
    source = tmp_path / "extra.py"
    source.write_bytes(b"frozen")
    dependencies[str(source)] = runner.accuracy.digest(source)
    assert runner.checked_dependencies({"dependencies": dependencies}) == dependencies
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="dependency changed"):
        runner.checked_dependencies({"dependencies": dependencies})
    dependencies.pop(str(source))
    dependencies.pop(str(Path(runner.ensemble.__file__).resolve()))
    with pytest.raises(ValueError, match="exact helper"):
        runner.checked_dependencies({"dependencies": dependencies})
