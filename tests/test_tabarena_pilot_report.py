"""Publication staging preserves failures and rejects partial/tampered evidence."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from benchmarks.tabarena import pilot_report as report
from benchmarks.tabarena.pilot_evaluation import evaluate_pilot, expected_pilot_jobs


@pytest.fixture
def completed_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    protocol = report.read_json(root / "benchmarks/tabarena/pilot_0159_v1.json")
    protocol["datasets"] = [next(d for d in protocol["datasets"] if d["problem_type"] == family)
                            for family in ("binary", "regression", "multiclass")]
    protocol["compute"]["expected_fit_count"] = 16
    source = tmp_path / "source"
    source_file = source / "benchmarks/tabarena/pilot_evaluation.py"
    source_file.parent.mkdir(parents=True)
    shutil.copyfile(root / "benchmarks/tabarena/pilot_evaluation.py", source_file)
    run = tmp_path / "run"
    run.mkdir()
    protocol_path, calibration_path = tmp_path / "protocol.json", tmp_path / "calibration.json"
    report.write_json(protocol_path, protocol)
    protocol_hash = report.sha256(protocol_path)
    calibration = {"hardware": {"platform": "fixture", "physical_cores": 8, "logical_cpus": 16}}
    report.write_json(calibration_path, calibration)
    runtime = {"native_extension_sha256": "a" * 64, "environment_sha256": "b" * 64,
               "packages": {"ctboost": "0.1.59"},
               "source_sha256": {"benchmarks/tabarena/pilot_evaluation.py": report.sha256(source_file)}}
    resources = {"workers": 8, "threads_per_worker": 2, "blas_openmp_threads": 1,
                 "fit_time_limit_seconds": 300, "hard_fit_grace_seconds": 30,
                 "memory_per_worker_bytes": 2 * 1024**3, "memory_budget_bytes": 16 * 1024**3}
    execution = {"protocol_sha256": protocol_hash, "calibration_sha256": report.canonical_hash(calibration),
                 "resources": resources, "resource_contract_sha256": report.canonical_hash(resources),
                 "runtime": runtime, "public_wheel": {"sha256": "c" * 64},
                 "prepared": {"datasets": [{"dataset": d["dataset_name"], "rows_outer_train": 16}
                                            for d in protocol["datasets"]]}}
    report.write_json(run / "execution.json", execution)
    report.write_json(run / "progress.json", {"active_blocks": [], "queued_blocks": 0})
    records = []
    for job in expected_pilot_jobs(protocol):
        destination = run / "fits" / job["dataset"] / f"inner{job['inner_fold']}" / f"{job['variant']}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        validation = np.arange(job["inner_fold"] * 2, job["inner_fold"] * 2 + 2, dtype=np.int64)
        training = np.setdiff1d(np.arange(16), validation)
        digest = hashlib.sha256()
        for indices in (training, validation):
            array = np.asarray(indices, dtype="<i8")
            digest.update(len(array).to_bytes(8, "little"))
            digest.update(array.tobytes())
        prediction_path = destination.with_suffix(".npz")
        np.savez_compressed(prediction_path, predictions=np.array([0.1, 0.9]), labels=np.array([0, 1]),
                            validation_indices=validation)
        record = {**job, "status": "ok", "metric_error": 0.5, "fit_seconds": 10.0,
                  "peak_rss_bytes": 100_000_000, "deadline_stopped": False,
                  "protocol_sha256": protocol_hash, "split_sha256": digest.hexdigest(),
                  "preprocessing_sha256": "d" * 64,
                  "resource_contract_sha256": execution["resource_contract_sha256"],
                  "execution_runtime_sha256": report.canonical_hash(runtime),
                  "prediction_sha256": report.sha256(prediction_path), "rows_validation": 2,
                  "retained_boosting_rounds": 10, "secondary_metrics": {}}
        records.append(record)
        report.write_json(destination, record)
    report.write_json(run / "decision.json", evaluate_pilot(protocol, records, protocol_sha256=protocol_hash))
    return {"run_root": run, "protocol_path": protocol_path, "calibration_path": calibration_path,
            "source_root": source}, records


def test_export_is_deterministic_and_checksums_cover_readme_and_raw_records(tmp_path, completed_run):
    arguments, records = completed_run
    outputs = [tmp_path / "publish-a", tmp_path / "publish-b"]
    for output in outputs:
        summary = report.export_pilot_report(output=output, **arguments)
        assert summary["records"] == 16
        assert summary["prediction_archives"] == 16
        assert not summary["full_hpo_justified"]
    assert (outputs[0] / "SHA256SUMS").read_bytes() == (outputs[1] / "SHA256SUMS").read_bytes()
    sums = (outputs[0] / "SHA256SUMS").read_text().splitlines()
    assert f"{report.sha256(outputs[0] / 'README.md')}  README.md" in sums
    for line in sums:
        digest, relative = line.split("  ", 1)
        assert report.sha256(outputs[0] / relative) == digest
    assert not list(outputs[0].rglob("*.pkl"))
    for path in (arguments["run_root"] / "fits").glob("*/inner*/*.json"):
        assert path.read_bytes() == (outputs[0] / path.relative_to(arguments["run_root"])).read_bytes()


@pytest.mark.parametrize("change", ["missing_decision", "active", "missing_fit", "duplicate_fit", "prediction", "source", "decision"])
def test_partial_or_tampered_runs_fail_before_creating_publication(tmp_path, completed_run, change, monkeypatch):
    arguments, _ = completed_run
    run = arguments["run_root"]
    if change == "missing_decision":
        (run / "decision.json").unlink()
    elif change == "active":
        report.write_json(run / "progress.json", {"active_blocks": [{"pid": 12}], "queued_blocks": 0})
    elif change == "missing_fit":
        next((run / "fits").glob("*/inner*/*.json")).unlink()
    elif change == "duplicate_fit":
        source = next((run / "fits").glob("*/inner*/*.json"))
        shutil.copyfile(source, source.with_name("duplicate.json"))
    elif change == "prediction":
        next((run / "fits").glob("*/inner*/*.npz")).write_bytes(b"corrupted")
    elif change == "source":
        (arguments["source_root"] / "benchmarks/tabarena/pilot_evaluation.py").write_text("changed")
    elif change == "decision":
        decision = report.read_json(run / "decision.json")
        decision["full_hpo_justified"] = True
        report.write_json(run / "decision.json", decision)
    if change in {"missing_decision", "active", "missing_fit", "duplicate_fit"}:
        monkeypatch.setattr(report, "evaluate_pilot", lambda *_args, **_kwargs: pytest.fail("Partial runs must not be evaluated"))
    output = tmp_path / "publication"
    with pytest.raises(ValueError):
        report.export_pilot_report(output=output, **arguments)
    assert not output.exists()


def test_failure_is_preserved_with_undefined_arm_mean(tmp_path, completed_run):
    arguments, records = completed_run
    failed = next(r for r in records if r["variant"] == "backtracking_3" and r["problem_type"] == "binary")
    failed.update(status="failed", metric_error=None, error="deliberate fixture failure")
    destination = arguments["run_root"] / "fits" / failed["dataset"] / f"inner{failed['inner_fold']}" / "backtracking_3.json"
    report.write_json(destination, failed)
    protocol = report.read_json(arguments["protocol_path"])
    decision = evaluate_pilot(protocol, records, protocol_sha256=report.sha256(arguments["protocol_path"]))
    report.write_json(arguments["run_root"] / "decision.json", decision)
    output = tmp_path / "publication"
    summary = report.export_pilot_report(output=output, **arguments)
    assert summary["prediction_archives"] == 15
    rows = report.read_json(output / "dataset_arm_results.json")
    row = next(r for r in rows if r["dataset"] == failed["dataset"] and r["variant"] == "backtracking_3")
    assert row["successful_fits"] == 1
    assert row["mean_validation_error"] is None
    assert row["relative_error_improvement"] is None
    assert "failed: 1" in (output / "README.md").read_text(encoding="utf-8")
