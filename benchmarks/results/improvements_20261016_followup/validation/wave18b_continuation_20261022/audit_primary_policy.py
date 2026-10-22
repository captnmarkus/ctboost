"""Verify the completed primary policy and recompute its gate from saved errors.

Does not decode model/prediction payloads or access confirmation/official data.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(r"C:\apps\ctboost")
ARCHIVE = ROOT / "benchmarks/results/improvements_20261016_followup"
OUT = ARCHIVE / "accuracy_wave18b_recovery"
ARTIFACTS = ROOT / ".tmp/improvements-20261016/followup/accuracy_wave18b_recovery"
HERE = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    destination = HERE / "primary_policy_audit.json"
    if destination.exists():
        raise ValueError("Preserve prior audits")
    plan, choices, frozen, dev = [read(OUT / name) for name in ("plan.json", "selected_cv.json", "frozen_selections.json", "development.json")]
    fold_audit = read(HERE / "fold_terminal_audit.json")
    for path, expected in fold_audit["fold_evidence_sha256"].items():
        assert digest(ARCHIVE / path) == expected, path
    recovery = read(OUT / "recovery_plan.json")
    for path, expected in recovery["original_files_sha256"].items():
        assert digest(path) == expected, path
    for path, expected in plan["dependencies"].items():
        assert digest(path) == expected, path
    plan_hash = digest(OUT / "plan.json")
    selected_hash = digest(OUT / "selected_cv.json")
    frozen_hash = digest(OUT / "frozen_selections.json")
    assert plan_hash == fold_audit["plan_sha256"]
    assert selected_hash == "0c1d68453a4bbf6a457f052aa9ab9b79eb432088df940a235ed5e50a8d3e6223"
    assert frozen["plan_sha256"] == dev["plan_sha256"] == plan_hash
    assert frozen["selected_cv_sha256"] == selected_hash and dev["frozen_selections_sha256"] == frozen_hash
    assert frozen["all_fits_successful"] is True
    assert frozen["jobs"] == [c["job"] for c in choices["choices"]]
    assert all(frozen["union_completion_counts"][k] == 36 for k in ("planned_jobs", "native_fits_started", "native_fits_completed", "fully_verified_jobs"))
    for document in (plan, frozen, dev):
        assert document["official_test_scored"] is False
    expected = {(case["dataset_name"], lib) for case in plan["cases"] for lib in ("ctboost","xgboost","lightgbm")}
    assert len(expected) == 36 and len(frozen["selections"]) == 36
    assert {(m["dataset"],m["library"]) for m in frozen["selections"]} == expected
    manifest, peak, walls = {}, [], []
    union_paths = sorted((OUT / "union_jobs").glob("*.json"))
    assert [p.name for p in union_paths] == [f"{i:04d}.json" for i in range(36)]
    for i,(job,selection,path) in enumerate(zip(frozen["jobs"], frozen["selections"], union_paths, strict=True)):
        record = read(path)
        resource_path = OUT / "union_job_logs" / f"{i:04d}-resource.json"
        headroom_path = OUT / "union_job_logs" / f"{i:04d}-headroom.json"
        resource, headroom = read(resource_path), read(headroom_path)
        assert record["status"] == "ok" and record["selection"] == selection
        assert record["native_fit_started"] is True and record["native_fit_completed"] is True and record.get("finished_at")
        assert record["dataset"] == job["dataset"] and record["library"] == job["library"] and record["seed"] == 47
        assert selection["fixed_job"] == job and selection["requested_rounds"] == job["rounds"]
        assert resource["exit_code"] == 0 and resource["resource_failure"] is None and resource["cleanup_complete"] is True
        assert resource["killed_processes"] == [] and headroom["sufficient"] is True
        assert resource["peak_rss_bytes"] <= plan["resources"]["maximum_job_rss_bytes"]
        assert resource["wall_seconds"] < plan["resources"]["maximum_job_seconds"]
        model = Path(selection["model_path"])
        assert model == ARTIFACTS / "models" / job["dataset"] / (job["library"] + ".pkl")
        assert model.stat().st_size == selection["model_bytes"] and digest(model) == selection["model_sha256"]
        for evidence in (path, resource_path, headroom_path, model):
            manifest[str(evidence)] = digest(evidence)
        peak.append(resource["peak_rss_bytes"])
        walls.append(resource["wall_seconds"])
    assert len(dev["results"]) == 12
    assert {r["dataset"] for r in dev["results"]} == {c["dataset_name"] for c in plan["cases"]}
    for row in dev["results"]:
        assert set(row["models"]) == {"ctboost", "xgboost", "lightgbm"}
        for library, value in row["models"].items():
            assert math.isfinite(value["error"])
            selected = next(m for m in frozen["selections"] if (m["dataset"],m["library"]) == (row["dataset"], library))
            assert value["selection"] == selected
            prediction = ARTIFACTS / "development" / row["dataset"] / (library+".npz")
            assert digest(prediction) == value["predictions_sha256"]
            manifest[str(prediction)] = value["predictions_sha256"]
    summaries = []
    for comparator in ("xgboost", "lightgbm"):
        for family in ("all", "binary", "regression", "multiclass"):
            rows = [row for row in dev["results"] if family == "all" or row["problem_type"] == family]
            gains = [(r["models"][comparator]["error"] - r["models"]["ctboost"]["error"])/max(r["models"][comparator]["error"],1e-12) for r in rows]
            summary = dict(comparator=comparator, family=family, datasets=len(rows),median_relative_error_gain=statistics.median(gains),wins=sum(g > 1e-12 for g in gains),ties=sum(abs(g)<=1e-12 for g in gains),worst_relative_regression=max(0,-min(gains)))
            assert summary == next(s for s in dev["summary"] if (s["comparator"],s["family"]) == (comparator,family))
            summaries.append(summary)
    passed = all(s["median_relative_error_gain"] >= 0 and s["wins"]+s["ties"] >= 6 for s in summaries if s["family"] == "all")
    assert passed == dev["development_target_met"]
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(), "status": "complete_primary_policy_integrity_passed",
        "audit_script_sha256": digest(__file__), "fold_audit_sha256": digest(HERE / "fold_terminal_audit.json"),
        "source_sha256": {name:digest(OUT / name) for name in ("plan.json","fold_manifest.json","recovery_plan.json","recovery_completion.json","fold_completion.json","selected_cv.json","frozen_selections.json","development.json")},
        "fold_evidence_unchanged": True, "original_interrupted_evidence_unchanged": True, "frozen_dependencies_unchanged": True,
        "verified_fold_jobs": 540, "verified_union_jobs": 36, "verified_saved_models": 36, "verified_saved_development_predictions": 36,
        "cumulative_quality_native_fits": 3107, "earlier_diagnostic_replays_separate": 8,
        "peak_union_rss_bytes": max(peak), "max_union_wall_seconds": max(walls),
        "development_target_met": passed, "development_summary": summaries,
        "primary_timing_policy_selected": "accuracy_wave18b_recovery", "fallback_permitted": False,
        "timing_ready_subject_to_root_exclusivity_and_reference_checks": not passed,
        "remaining_quality_work": "Run unchanged fixed677 replication and evaluate original mean gate before timing" if passed else "None: primary gate failed, so conditional677 and official assessment are not permitted",
        "scope": "Frozen original scientific runtime c5ea; fresh format5 arithmetic is not measured by this accuracy study. Final inference uses independently parity-checked04ad runtime under prospective v4 policy. This audit uses stored error scalars and byte hashes, not re-evaluation or selection. No raw arrays or labels are exposed.",
        "evidence_sha256": manifest,
    }
    destination.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"receipt":str(destination),"sha256":digest(destination),"primary_target_met":passed,"summary":[s for s in summaries if s["family"]=="all"]}))

if __name__ == "__main__":
    main()
