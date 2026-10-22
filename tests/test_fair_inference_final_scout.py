"""Synthetic guards for final latency receipts; no timers or model fitting."""
import copy
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from benchmarks import fair_inference_final_scout as runner


class RowModel:
    num_iterations_trained = 3

    def __init__(self):
        self.calls = 0
        self.inplace_delta = 0.

    def predict(self, frame, **kwargs):
        self.calls += 1
        return np.asarray(frame["value"], dtype=np.float32)

    def inplace_predict(self, frame, **kwargs):
        return self.predict(frame, **kwargs) + self.inplace_delta


def payload(library="xgboost"):
    return {"library": library, "problem": "binary", "schema": {}, "model": RowModel()}


def test_all_batches_guard_inplace_and_match_saved_rows(monkeypatch):
    pd = pytest.importorskip("pandas")
    monkeypatch.setitem(sys.modules, "xgboost", types.SimpleNamespace(DMatrix=lambda frame, **kwargs: frame))
    model = payload()
    source = pd.DataFrame({"value": np.array([-2., .5, 3.], dtype=np.float32)})
    expected = runner.predict_raw(model, source)
    for count in (1, 64, 1000):
        positions = np.arange(count) % len(source)
        raw = source.iloc[positions].reset_index(drop=True)
        normalized, actual, guard = runner.check_batch(model, raw, expected[positions])
        assert len(normalized) == count and guard["bit_exact"] and guard["rows"] == count
        runner.assert_bits(actual, expected[positions])


def test_inplace_difference_rejected_before_timing(monkeypatch):
    pd = pytest.importorskip("pandas")
    monkeypatch.setitem(sys.modules, "xgboost", types.SimpleNamespace(DMatrix=lambda frame, **kwargs: frame))
    model = payload()
    model["model"].inplace_delta = .125
    with pytest.raises(AssertionError):
        runner.check_batch(model, pd.DataFrame({"value": [1., 2.]}))


def test_each_path_uses_one_model_call_and_temperature_once(monkeypatch):
    pd = pytest.importorskip("pandas")
    model = payload("ctboost")
    frame = pd.DataFrame({"value": [2., 4.]})
    temperatures = []
    def temperature(record, margins):
        temperatures.append(record)
        return margins / 2.
    monkeypatch.setattr(runner.fixed_temperature, "apply_temperature", temperature)
    np.testing.assert_array_equal(runner.predict_raw(model, frame), [1., 2.])
    np.testing.assert_array_equal(runner.predict_normalized(model, frame), [1., 2.])
    assert model["model"].calls == 2 and temperatures == [model, model]


def test_quality_seed_never_substitutes_multi_seed_mean():
    metadata = runner.quality_metadata({"errors": {"47": .2, "947": .4, "mean": .3}}, 47)
    assert metadata["development_error"] == .2
    assert metadata["development_mean_error"] == .3
    assert metadata["development_seed"] == 47
    with pytest.raises(ValueError, match="seed"):
        runner.quality_metadata({"errors": {"mean": .3}}, 47)
    with pytest.raises(ValueError, match="seed"):
        runner.quality_metadata({"error": .2, "errors": {"947": .4, "mean": .3}}, 47)


def test_prediction_hash_resolves_exact_seed_and_rejects_conflicts():
    primary, companion = "a" * 64, "b" * 64
    assert runner.prediction_hash({"predictions_sha256": primary}, 47) == ("predictions_sha256", primary)
    assert runner.prediction_hash({"predictions137_sha256": companion}, 137) == ("predictions137_sha256", companion)
    assert runner.prediction_hash({"predictions_sha256": companion, "predictions137_sha256": companion}, 137)[1] == companion
    with pytest.raises(ValueError, match="Conflicting"):
        runner.prediction_hash({"predictions_sha256": primary, "predictions137_sha256": companion}, 137)
    with pytest.raises(ValueError, match="Ambiguous"):
        runner.prediction_hash({"predictions_sha256": primary, "predictions137_sha256": companion}, 47)
    with pytest.raises(ValueError, match="loaded seed"):
        runner.prediction_hash({"predictions947_sha256": companion}, 137)


def test_reference_binds_sources_runtime_batches_and_unique_records():
    common = {"runner_sha256": "a" * 64, "runtime": {"native": "b" * 64}, "benchmark_batches": [1, 64, 1000]}
    receipt = {**common, "check_only": True,
               "records": [{"dataset": "toy", "rows": 1, "library": "ctboost"}]}
    assert len(runner.verify_reference(receipt, common)) == 1
    for key, changed in (("check_only", False), ("benchmark_batches", [1]), ("runtime", {})):
        altered = {**receipt, key: changed}
        with pytest.raises(ValueError, match="checked reference"):
            runner.verify_reference(altered, common)
    duplicate = copy.deepcopy(receipt)
    duplicate["records"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        runner.verify_reference(duplicate, common)


def test_single_payload_rejects_ensemble_or_task_mismatch():
    model = payload()
    runner.validate_payload(model, "xgboost", "binary")
    with pytest.raises(ValueError, match="single-model"):
        runner.validate_payload({**model, "components": []}, "xgboost", "binary")
    with pytest.raises(ValueError, match="single-model"):
        runner.validate_payload(model, "ctboost", "binary")


def test_dependency_map_requires_current_normalizer_and_temperature(tmp_path):
    dependencies = {str(Path(module.__file__).resolve()): runner.accuracy.digest(module.__file__)
                    for module in (runner.accuracy, runner.fixed_temperature)}
    assert runner.checked_dependencies({"dependencies": dependencies}) == dependencies
    dependencies.pop(str(Path(runner.fixed_temperature.__file__).resolve()))
    with pytest.raises(ValueError, match="exact helper"):
        runner.checked_dependencies({"dependencies": dependencies})
