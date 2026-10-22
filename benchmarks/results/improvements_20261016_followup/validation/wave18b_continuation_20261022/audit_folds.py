"""Read-only independent completion audit; writes only a fresh audit receipt."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(r"C:\apps\ctboost")
ARCHIVE = ROOT / "benchmarks/results/improvements_20261016_followup"
RECOVERY = ARCHIVE / "accuracy_wave18b_recovery"
HERE = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def check_hash(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch: {path}")
    return actual

def main():
    out = HERE / "fold_terminal_audit.json"
    if out.exists():
        raise ValueError("Preserve existing audit")
    recovery = read(RECOVERY / "recovery_plan.json")
    plan = read(RECOVERY / "plan.json")
    completion = read(RECOVERY / "fold_completion.json")
    terminal = read(RECOVERY / "recovery_completion.json")
    policy_path = ARCHIVE / "inference/final_runner_preparation/selection_policy_v4.json"
    policy = read(policy_path)
    check_hash(policy_path, "7d8e71259b2dd6cb1e47b568e6899c3612a3fd6913e1da1703ec6124444168c2")
    check_hash(RECOVERY / "plan.json", policy["candidate_policy"]["plan_sha256"])
    check_hash(RECOVERY / "recovery_plan.json", "2f4faaa1f896f20d12a528da3fb93a36f88c4e23b3a610eaa61624e54aaab57e")
    check_hash(RECOVERY / "fold_manifest.json", plan["fold_manifest_sha256"])
    for mapping in (recovery["original_files_sha256"], recovery["preserved_prediction_sha256"], policy["recovery_binding"], plan["dependencies"]):
        for path, expected in mapping.items():
            check_hash(path, expected)
    for relative, expected in recovery["copied_files_sha256"].items():
        check_hash(RECOVERY / relative, expected)
    for relative, expected in plan["runtime"]["ctboost_python_sha256"].items():
        check_hash(Path(recovery["arguments"]["package_root"]) / relative, expected)
    native = list((Path(recovery["arguments"]["package_root"]) / "ctboost").glob("_core.cp312-win_amd64.pyd"))
    assert len(native) == 1
    check_hash(native[0], plan["runtime"]["ctboost_native_sha256"])
    assert recovery["resources"] == plan["resources"]
    assert recovery["reused_verified_global_indices"] == list(range(297))
    assert recovery["fresh_global_indices"] == list(range(297, 540))
    assert completion["all_fits_successful"] is True
    assert terminal["all_fits_successful"] is True and terminal["selection_or_development_run"] is False
    assert terminal["reused_verified_jobs"] == 297 and terminal["extra_original_interrupted_native_fit_count"] == 1
    for counts, total in ((completion["completion_counts"], 540), (terminal["new_process_counts"], 243)):
        assert all(counts[k] == total for k in ("planned_jobs", "native_fits_started", "native_fits_completed", "fully_verified_jobs"))
    jobs = plan["jobs"]
    assert len(jobs) == 36 and len({(j["dataset"], j["library"]) for j in jobs}) == 36
    identities = [dict(dataset=j["dataset"], library=j["library"], profile_index=p, fold=f, requested_rounds=n, seed=47)
                  for j in jobs for p,n in enumerate(((j["rounds"]+1)//2,j["rounds"],2*j["rounds"])) for f in range(5)]
    paths = [RECOVERY / "fold_jobs" / f"{i:04d}.json" for i in range(540)]
    assert completion["records"] == [str(path) for path in paths]
    assert sorted((RECOVERY / "fold_jobs").glob("*.json")) == paths
    receipts = {}
    peaks, walls = [], []
    for i, (path, identity) in enumerate(zip(paths, identities, strict=True)):
        record = read(path)
        resource_path = RECOVERY / "fold_job_logs" / f"{i:04d}-resource.json"
        headroom_path = RECOVERY / "fold_job_logs" / f"{i:04d}-headroom.json"
        resource, headroom = read(resource_path), read(headroom_path)
        assert all(record.get(k) == v for k,v in identity.items()), i
        assert all(record.get(k) is True for k in ("native_fit_started", "native_fit_completed", "reload_bit_exact")), i
        assert record["status"] == "ok" and record.get("finished_at") and math.isfinite(record["validation_error"]), i
        assert record["validation_labels_received"] is False and record["early_stopping"] is False, i
        assert resource["exit_code"] == 0 and resource["resource_failure"] is None and resource["cleanup_complete"] is True, i
        assert resource["killed_processes"] == [] and headroom["sufficient"] is True, i
        assert 0 < record["retained_rounds"] <= record["requested_rounds"], i
        assert record["library"] == "lightgbm" or record["retained_rounds"] == record["requested_rounds"], i
        prediction = Path(record["prediction_path"])
        expected_parent = Path(recovery["arguments"]["source_artifacts"] if i < 297 else recovery["arguments"]["artifacts"]) / "fold_predictions"
        assert prediction.parent == expected_parent and prediction.name == f"{i:04d}.npz", i
        check_hash(prediction, record["prediction_sha256"])
        for evidence in (path, resource_path, headroom_path):
            receipts[str(evidence.relative_to(ARCHIVE))] = digest(evidence)
        peaks.append(resource["peak_rss_bytes"])
        walls.append(resource["wall_seconds"])
    absent = ("selected_cv.json", "frozen_selections.json", "development.json", "union_jobs", "union_job_logs")
    assert all(not (RECOVERY / name).exists() for name in absent)
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "fold_stage_complete_integrity_passed_remaining_planned_stages_required",
        "audit_script_sha256": digest(__file__),
        "plan_sha256": digest(RECOVERY / "plan.json"),
        "fold_manifest_sha256": digest(RECOVERY / "fold_manifest.json"),
        "recovery_plan_sha256": digest(RECOVERY / "recovery_plan.json"),
        "fold_completion_sha256": digest(RECOVERY / "fold_completion.json"),
        "recovery_completion_sha256": digest(RECOVERY / "recovery_completion.json"),
        "timing_policy_v4_sha256": digest(policy_path),
        "original_files_and_incident_checked": len(recovery["original_files_sha256"]),
        "copied_files_byte_hash_checked": len(recovery["copied_files_sha256"]),
        "plan_dependency_files_checked": len(plan["dependencies"]),
        "runtime_python_files_checked": len(plan["runtime"]["ctboost_python_sha256"]),
        "scientific_runtime_native_sha256": digest(native[0]),
        "fold_jobs_verified": 540, "reused_verified_jobs": 297, "newly_executed_jobs": 243,
        "preserved_extra_interrupted_native_fit": 1,
        "cumulative_native_fits_before_unions": 3071,
        "cumulative_native_fits_if_all36_unions_finish": 3107,
        "earlier_diagnostic_replays_separate": 8,
        "prediction_archives_hash_only_checked": 540,
        "peak_owned_process_rss_bytes": max(peaks), "maximum_worker_wall_seconds": max(walls),
        "new_suffix_peak_rss_bytes": max(peaks[297:]), "new_suffix_maximum_worker_wall_seconds": max(walls[297:]),
        "frozen_resources": plan["resources"],
        "primary_timing_eligible": False, "fallback_timing_eligible": False,
        "reason": "All540 fold jobs succeeded. There is no numerical/resource failure, and no complete36-model union policy or development receipt yet. Continue unchanged select,36 union fits,development; then only original conditional677/official gates. Do not rerun folds or select fallback.",
        "remaining_stages": ["select", "run_union", "development", "conditional seed677 only if primary gate passes", "official outer only if original primary and two-seed mean gates pass"],
        "review_limits": "Independent standard-library metadata and streaming hash audit; no array decoding, model loading, scoring, fitting, process lookup or management. Runtime binary and Python hashes checked; full import/dependency-version parity must still pass unchanged main.checked before every scientific stage.",
        "fold_evidence_sha256": receipts,
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"audit": str(out), "sha256": digest(out), "status": report["status"], "folds": 540, "peak_rss": max(peaks), "maximum_seconds": max(walls)}))

if __name__ == "__main__":
    main()
