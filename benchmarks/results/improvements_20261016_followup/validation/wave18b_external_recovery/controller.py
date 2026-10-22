"""Incident-specific recovery of externally terminated wave18b, not a new search.

The scientific plan, fold manifest, workers, criteria and limits stay byte-identical.
Only verified prefix receipts are copied; the interrupted global index and every
unstarted index are dispatched afresh. No selection or scoring happens here.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
MAIN_PATH = HERE.parents[3] / "fair_accuracy_current_round_cv.py"
TEST_PATH = HERE / "test_controller.py"
PROTOCOL = "wave18b_external_interruption_recovery_20261016"
FIRST_NEW_INDEX, TOTAL = 297, 540
ORIGINAL_PLAN_SHA256 = "d368d34f9245e27d818f105b70163474878f9a6fdfd0f720a1204440b0ca09d2"
INCIDENT_SHA256 = "156d9ce25a987c10f1a7ad345224539d2d5c5274b5d6b980c8a57cdfba6519a9"


def load_main():
    spec = importlib.util.spec_from_file_location("incident_current_cv", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main = load_main()
base = main.base


def identity_matches(record, identity):
    if any(record.get(key) != value for key, value in identity.items()):
        raise ValueError("Frozen global fit identity changed")


def validate_prefix_record(record, identity, resource, headroom):
    identity_matches(record, identity)
    if (record.get("status") != "ok" or record.get("native_fit_started") is not True
            or record.get("native_fit_completed") is not True or record.get("reload_bit_exact") is not True
            or not record.get("finished_at") or resource.get("exit_code") != 0
            or resource.get("resource_failure") is not None or resource.get("cleanup_complete") is not True
            or headroom.get("sufficient") is not True):
        raise ValueError("Only fully verified successful prefix jobs can be reused")


def validate_interruption(incident, record, identity, resource):
    identity_matches(record, identity)
    affected = incident.get("affected_worker", {})
    if (incident.get("status") != "confirmed_agent_process_ownership_error"
            or affected.get("job_index") != FIRST_NEW_INDEX
            or any(affected.get(key) != identity[key] for key in
                   ("dataset", "library", "profile_index", "fold", "requested_rounds"))
            or record.get("status") != "failed" or record.get("native_fit_completed") is not True
            or resource.get("exit_code") != 4294967295 or resource.get("resource_failure") is not None
            or resource.get("killed_processes") != [] or resource.get("cleanup_complete") is not True
            or affected.get("owned_process_identities") != resource.get("owned_processes")
            or not any(row["pid"] == 7488 and row["create_time"] == 1792167981.537644
                       for row in resource["owned_processes"])):
        raise ValueError("The exact externally interrupted worker must be documented")


def source_args(args):
    result = copy.copy(args)
    result.output, result.artifacts = args.source, args.source_artifacts
    return result


def path_arguments(args):
    return {name: str(getattr(args, name)) for name in
            ("output", "artifacts", "package_root", "dependencies", "previous", "source", "source_artifacts", "incident")}


def validate_source(args):
    if (args.source.name != "accuracy_wave18b" or args.output.name != "accuracy_wave18b_recovery"
            or args.source_artifacts.name != "accuracy_wave18b" or args.artifacts.name != "accuracy_wave18b_recovery"
            or args.output.parent != args.source.parent or args.artifacts.parent != args.source_artifacts.parent):
        raise ValueError("Use the exact sibling original and recovery directories")
    plan = main.checked(source_args(args))
    if base.digest(args.source / "plan.json") != ORIGINAL_PLAN_SHA256:
        raise ValueError("Recovery is restricted to the original wave18b plan")
    if base.digest(args.incident) != INCIDENT_SHA256:
        raise ValueError("Incident receipt changed")
    incident = base.read(args.incident)
    for relative, expected in incident["closed_original_evidence"].items():
        if base.digest(args.source.parent / relative) != expected:
            raise ValueError("Original incident evidence changed")
    for forbidden in ("selected_cv.json", "frozen_selections.json", "development.json", "recovery_started.json"):
        if (args.source / forbidden).exists():
            raise ValueError("No original selection/scoring/recovery may have occurred")
    completion = base.read(args.source / "fold_completion.json")
    expected_counts = {"planned_jobs": TOTAL, "native_fits_started": FIRST_NEW_INDEX + 1,
                       "native_fits_completed": FIRST_NEW_INDEX + 1, "fully_verified_jobs": FIRST_NEW_INDEX}
    if (completion.get("all_fits_successful") is not False or completion.get("plan_sha256") != ORIGINAL_PLAN_SHA256
            or any(completion["completion_counts"].get(key) != value for key, value in expected_counts.items())):
        raise ValueError("Unexpected original completion boundary")
    identities = main.support.enumerate_fit_jobs(plan["jobs"])
    if len(identities) != TOTAL or completion["records"] != [str(main.job_path(source_args(args), "fold_job", i)) for i in range(TOTAL)]:
        raise ValueError("Original global job record set changed")
    copies = [Path("plan.json"), Path("fold_manifest.json")]
    predictions = {}
    for index, identity in enumerate(identities):
        relative = Path("fold_jobs") / f"{index:04d}.json"
        record = base.read(args.source / relative)
        identity_matches(record, identity)
        if index < FIRST_NEW_INDEX:
            log_paths = [Path("fold_job_logs") / f"{index:04d}{suffix}" for suffix in
                         ("-resource.json", "-headroom.json", ".txt")]
            resource, headroom = [base.read(args.source / path) for path in log_paths[:2]]
            validate_prefix_record(record, identity, resource, headroom)
            prediction = Path(record["prediction_path"]).resolve()
            if (not prediction.is_relative_to(args.source_artifacts / "fold_predictions")
                    or base.digest(prediction) != record["prediction_sha256"]):
                raise ValueError("A verified prediction artifact changed or escaped its source directory")
            predictions[str(prediction)] = record["prediction_sha256"]
            copies.extend([relative, *log_paths])
        elif index == FIRST_NEW_INDEX:
            resource = base.read(args.source / "fold_job_logs" / f"{index:04d}-resource.json")
            validate_interruption(incident, record, identity, resource)
        elif record.get("status") != "failed" or record.get("failure") != "not_started_after_terminal_failure":
            raise ValueError("Only the exact unstarted suffix may follow the interrupted job")
    return plan, copies, predictions


def prepare(args):
    import psutil
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Use fresh recovery directories; preserve every prior attempt")
    plan, copies, predictions = validate_source(args)
    source_hashes = {str(path): base.digest(path) for path in sorted(args.source.rglob("*")) if path.is_file()}
    source_hashes[str(args.incident)] = base.digest(args.incident)
    quiet = {"recorded_at": base.now(), "available_bytes": psutil.virtual_memory().available,
             "required_available_bytes": plan["resources"]["maximum_job_rss_bytes"] + plan["resources"]["headroom_reserve_bytes"],
             "scope": "Informational preparation snapshot; unchanged bounded headroom gate runs before every worker."}
    args.output.mkdir(parents=True)
    args.artifacts.mkdir(parents=True)
    copied = {}
    for relative in copies:
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.source / relative, destination)
        expected = base.digest(args.source / relative)
        if base.digest(destination) != expected:
            raise ValueError("Recovery copy is not byte-identical")
        copied[str(relative)] = expected
    recovery = {"protocol": PROTOCOL, "created_at": base.now(), "script_sha256": base.digest(__file__),
        "test_sha256": base.digest(TEST_PATH), "main_sha256": base.digest(MAIN_PATH), "arguments": path_arguments(args),
        "original_plan_sha256": ORIGINAL_PLAN_SHA256, "original_files_sha256": source_hashes,
        "copied_files_sha256": copied, "preserved_prediction_sha256": predictions, "resources": plan["resources"],
        "quiet_memory_snapshot": quiet, "reused_verified_global_indices": list(range(FIRST_NEW_INDEX)),
        "fresh_global_indices": list(range(FIRST_NEW_INDEX, TOTAL)), "extra_interrupted_native_fit_count": 1,
        "interim_quality_values_used": False,
        "scope": "Continue the same frozen experiment after confirmed external operator termination. Existing JSON receipts are decoded only to verify identities/status/provenance; their quality values are not inspected, aggregated or selected. No new configuration, fold, seed, criterion, limit, or fallback is introduced.",
        "subsequent_stages": "After all540 verified records, use the unchanged main select/run_union/development and original conditional677/outer gates. No original failed receipt is changed."}
    base.write(args.output / "recovery_plan.json", recovery)
    checked(args)
    print({"recovery_plan_sha256": base.digest(args.output / "recovery_plan.json"),
           "scientific_plan_sha256": base.digest(args.output / "plan.json"), "reused_verified": FIRST_NEW_INDEX,
           "fresh_fold_jobs": TOTAL - FIRST_NEW_INDEX, "mandatory_union_jobs": 36}, flush=True)


def checked(args):
    recovery = base.read(args.output / "recovery_plan.json")
    if (recovery.get("protocol") != PROTOCOL or recovery.get("script_sha256") != base.digest(__file__)
            or recovery.get("test_sha256") != base.digest(TEST_PATH) or recovery.get("main_sha256") != base.digest(MAIN_PATH)
            or recovery.get("arguments") != path_arguments(args) or recovery.get("original_plan_sha256") != ORIGINAL_PLAN_SHA256
            or recovery.get("reused_verified_global_indices") != list(range(FIRST_NEW_INDEX))
            or recovery.get("fresh_global_indices") != list(range(FIRST_NEW_INDEX, TOTAL))):
        raise ValueError("Frozen recovery controller or paths changed")
    for path, expected in recovery["original_files_sha256"].items():
        if base.digest(path) != expected:
            raise ValueError("Original evidence changed after recovery freeze")
    for relative, expected in recovery["copied_files_sha256"].items():
        if base.digest(args.output / relative) != expected:
            raise ValueError("Copied verified evidence changed after recovery freeze")
    for path, expected in recovery["preserved_prediction_sha256"].items():
        if base.digest(path) != expected:
            raise ValueError("A reused prediction archive changed")
    plan = main.checked(args)
    if base.digest(args.output / "plan.json") != ORIGINAL_PLAN_SHA256 or plan["resources"] != recovery["resources"]:
        raise ValueError("Scientific plan or resources changed")
    return plan, recovery


def worker_command(args, index):
    if type(index) is not int or not FIRST_NEW_INDEX <= index < TOTAL:
        raise ValueError("Recovery can dispatch only the frozen suffix global indices")
    command = [sys.executable, "-I", str(MAIN_PATH), "fold_job"]
    for name in ("output", "artifacts", "package_root", "dependencies", "previous"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    return command + ["--job-index", str(index)]


def run(args):
    import psutil
    plan, recovery = checked(args)
    marker = args.output / "recovery_started.json"
    if marker.exists() or (args.output / "fold_completion.json").exists():
        raise ValueError("Preserve every started recovery; no implicit resume")
    identities = main.support.enumerate_fit_jobs(plan["jobs"])
    for index in range(FIRST_NEW_INDEX, TOTAL):
        if main.job_path(args, "fold_job", index).exists():
            raise ValueError("Fresh suffix paths must not exist before dispatch")
    base.write(marker, {"started_at": base.now(), "recovery_plan_sha256": base.digest(args.output / "recovery_plan.json")})
    records = [base.read(main.job_path(args, "fold_job", index)) for index in range(FIRST_NEW_INDEX)]
    logs, terminal = args.output / "fold_job_logs", None
    for index in range(FIRST_NEW_INDEX, TOTAL):
        identity, path = identities[index], main.job_path(args, "fold_job", index)
        while terminal is None and (args.artifacts / "pause_fits").exists():
            time.sleep(.5)
        if terminal is not None:
            record = {**identity, "status": "failed", "failure": "not_started_after_terminal_failure", "cause": terminal}
            base.write(path, record)
            records.append(record)
            continue
        headroom = main.resources.wait_for_headroom(plan["resources"], available=lambda: psutil.virtual_memory().available)
        base.write(logs / f"{index:04d}-headroom.json", headroom)
        if not headroom["sufficient"]:
            terminal = "insufficient_system_headroom"
            record = {**identity, "status": "failed", "failure": terminal, "headroom": headroom}
        else:
            with (logs / f"{index:04d}.txt").open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(worker_command(args, index), stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                resource = main.cv.watchdog.watch(process, maximum_seconds=plan["resources"]["maximum_job_seconds"],
                    maximum_rss_bytes=plan["resources"]["maximum_job_rss_bytes"])
            base.write(logs / f"{index:04d}-resource.json", resource)
            if not resource["cleanup_complete"]:
                raise RuntimeError("Owned-process cleanup incomplete; no further dispatch")
            record = base.read(path) if path.exists() else {**identity, "status": "failed", "failure": "worker_did_not_complete"}
            identity_matches(record, identity)
            if resource["exit_code"] != 0 or resource["resource_failure"] is not None:
                record.update(status="failed", resource=resource)
            if record["status"] != "ok":
                terminal = "owned_fit_or_resource_failed"
        base.write(path, record)
        records.append(record)
        print({"stage": "recovery_fold_job", "index": index, "status": record["status"]}, flush=True)
    checked(args)
    success = len(records) == TOTAL and all(row["status"] == "ok" for row in records)
    base.write(args.output / "fold_completion.json", {"plan_sha256": ORIGINAL_PLAN_SHA256,
        "recovery_plan_sha256": base.digest(args.output / "recovery_plan.json"), "all_fits_successful": success,
        "completion_counts": main.completion_counts(records, TOTAL),
        "count_scope": "The540 scientific job records include297 reused verified jobs; actual new process counts are in recovery_completion.json.",
        "records": [str(main.job_path(args, "fold_job", index)) for index in range(TOTAL)]})
    base.write(args.output / "recovery_completion.json", {"recovery_plan_sha256": base.digest(args.output / "recovery_plan.json"),
        "finished_at": base.now(), "all_fits_successful": success, "reused_verified_jobs": FIRST_NEW_INDEX,
        "new_process_counts": main.completion_counts(records[FIRST_NEW_INDEX:], TOTAL - FIRST_NEW_INDEX),
        "extra_original_interrupted_native_fit_count": 1, "original_failed_attempt_preserved": True,
        "selection_or_development_run": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "verify"))
    for name in ("output", "artifacts", "package-root", "dependencies", "previous", "source", "source-artifacts", "incident"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    if args.stage == "verify":
        checked(args)
        print({"status": "verified", "recovery_plan_sha256": base.digest(args.output / "recovery_plan.json")})
    else:
        globals()[args.stage](args)
