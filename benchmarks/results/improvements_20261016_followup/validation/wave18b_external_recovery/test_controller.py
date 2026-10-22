"""Synthetic incident-controller checks; no real frames, models, fits or scores."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).with_name("controller.py")
spec = importlib.util.spec_from_file_location("recovery_under_test", PATH)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def identity(index=297):
    return {"dataset": "houses", "library": "xgboost", "profile_index": 2, "fold": index - 295,
            "requested_rounds": 2174, "seed": 47}


def successful_record(index=297):
    return {**identity(index), "status": "ok", "native_fit_started": True, "native_fit_completed": True,
            "reload_bit_exact": True, "finished_at": "synthetic"}


def resource():
    return {"exit_code": 0, "resource_failure": None, "cleanup_complete": True, "killed_processes": []}


def test_verified_prefix_accepted():
    controller.validate_prefix_record(successful_record(), identity(), resource(), {"sufficient": True})


@pytest.mark.parametrize("field,value", [("status", "failed"), ("native_fit_started", False),
    ("native_fit_completed", False), ("reload_bit_exact", False), ("finished_at", None), ("fold", 3)])
def test_unverified_or_wrong_prefix_rejected(field, value):
    row = successful_record()
    row[field] = value
    with pytest.raises(ValueError):
        controller.validate_prefix_record(row, identity(), resource(), {"sufficient": True})


@pytest.mark.parametrize("field,value", [("exit_code", 1), ("resource_failure", "timeout"), ("cleanup_complete", False)])
def test_bad_prefix_resource_rejected(field, value):
    record = resource()
    record[field] = value
    with pytest.raises(ValueError):
        controller.validate_prefix_record(successful_record(), identity(), record, {"sufficient": True})


def incident_fixture():
    owned = [{"pid": 7488, "create_time": 1792167981.537644}]
    record = {**successful_record(), "status": "failed"}
    r = {**resource(), "exit_code": 4294967295, "owned_processes": owned}
    incident = {"status": "confirmed_agent_process_ownership_error",
                "affected_worker": {**identity(), "job_index": 297, "owned_process_identities": owned}}
    return incident, record, r


def test_exact_operator_interruption_accepted():
    incident, record, r = incident_fixture()
    controller.validate_interruption(incident, record, identity(), r)


@pytest.mark.parametrize("mutation", ["wrong_pid", "wrong_time", "wrong_index", "watchdog_kill", "resource_failure", "unconfirmed"])
def test_other_failures_cannot_enter_recovery(mutation):
    incident, record, r = copy.deepcopy(incident_fixture())
    if mutation == "wrong_pid":
        r["owned_processes"][0]["pid"] = 999
    elif mutation == "wrong_time":
        r["owned_processes"][0]["create_time"] += 1
    elif mutation == "wrong_index":
        incident["affected_worker"]["job_index"] = 298
    elif mutation == "watchdog_kill":
        r["killed_processes"] = [7488]
    elif mutation == "resource_failure":
        r["resource_failure"] = "rss"
    else:
        incident["status"] = "unexplained_exit"
    with pytest.raises(ValueError):
        controller.validate_interruption(incident, record, identity(), r)


def args_for(tmp_path):
    output = tmp_path / "accuracy_wave18b_recovery"
    output.mkdir()
    (output / "fold_job_logs").mkdir()
    return SimpleNamespace(output=output, artifacts=tmp_path / "artifacts/accuracy_wave18b_recovery",
        package_root=tmp_path / "runtime", dependencies=tmp_path / "deps", previous=tmp_path / "previous",
        source=tmp_path / "accuracy_wave18b", source_artifacts=tmp_path / "artifacts/accuracy_wave18b",
        incident=tmp_path / "incident.json")


def test_worker_keeps_original_global_index_and_frozen_main(tmp_path):
    args = args_for(tmp_path)
    for index in (297, 539):
        command = controller.worker_command(args, index)
        assert command[2] == str(controller.MAIN_PATH)
        assert command[3] == "fold_job"
        assert command[-2:] == ["--job-index", str(index)]
        assert "select" not in command and "development" not in command
    for index in (-1, 0, 296, 540, 297.0, True):
        with pytest.raises(ValueError):
            controller.worker_command(args, index)


@pytest.mark.parametrize("failure", [None, "worker", "headroom", "cleanup"])
def test_dispatch_only_fresh_suffix_and_terminal_barrier(tmp_path, monkeypatch, failure):
    args = args_for(tmp_path)
    monkeypatch.setattr(controller, "TOTAL", 300)
    plan = {"resources": {"maximum_job_seconds": 1200, "maximum_job_rss_bytes": 8589934592}, "jobs": []}
    monkeypatch.setattr(controller, "checked", lambda _: (plan, {}))
    monkeypatch.setattr(controller.main.support, "enumerate_fit_jobs", lambda _: [identity(i) for i in range(300)])
    controller.base.write(args.output / "recovery_plan.json", {"synthetic": True})
    prefix_bytes = {}
    for index in range(297):
        path = controller.main.job_path(args, "fold_job", index)
        controller.base.write(path, successful_record(index))
        prefix_bytes[index] = path.read_bytes()
    calls, resource_calls = [], []
    monkeypatch.setattr(controller.main.resources, "wait_for_headroom", lambda *a, **k: {"sufficient": failure != "headroom"})

    def popen(command, **kwargs):
        index = int(command[-1])
        calls.append(index)
        row = successful_record(index)
        if failure == "worker":
            row.update(status="failed", error="synthetic")
        controller.base.write(controller.main.job_path(args, "fold_job", index), row)
        return SimpleNamespace(pid=index)

    def watch(process, **kwargs):
        resource_calls.append(kwargs)
        return {**resource(), "cleanup_complete": failure != "cleanup"}

    monkeypatch.setattr(controller.subprocess, "Popen", popen)
    monkeypatch.setattr(controller.main.cv.watchdog, "watch", watch)
    if failure == "cleanup":
        with pytest.raises(RuntimeError, match="cleanup"):
            controller.run(args)
        assert calls == [297]
        assert not (args.output / "fold_completion.json").exists()
    else:
        controller.run(args)
        completion = controller.base.read(args.output / "fold_completion.json")
        assert completion["all_fits_successful"] is (failure is None)
        assert calls == ([297, 298, 299] if failure is None else ([] if failure == "headroom" else [297]))
        fresh = controller.base.read(args.output / "recovery_completion.json")["new_process_counts"]
        assert fresh["native_fits_completed"] == len(calls)
        assert fresh["planned_jobs"] == 3
        if failure is not None:
            assert controller.base.read(controller.main.job_path(args, "fold_job", 298))["failure"] == "not_started_after_terminal_failure"
    for index, original in prefix_bytes.items():
        assert controller.main.job_path(args, "fold_job", index).read_bytes() == original
    assert all(row == {"maximum_seconds": 1200, "maximum_rss_bytes": 8589934592} for row in resource_calls)


def test_existing_suffix_never_overwritten(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    monkeypatch.setattr(controller, "checked", lambda _: ({"jobs": []}, {}))
    monkeypatch.setattr(controller.main.support, "enumerate_fit_jobs", lambda _: [])
    path = controller.main.job_path(args, "fold_job", 297)
    controller.base.write(path, {"preserve": True})
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Fresh suffix"):
        controller.run(args)
    assert path.read_bytes() == original


@pytest.mark.parametrize("target", ["original", "copied", "prediction", "controller", "arguments"])
def test_recovery_hash_guards_reject_mutation(tmp_path, monkeypatch, target):
    args = args_for(tmp_path)
    args.source.mkdir()
    original = args.source / "evidence.json"
    original.write_text("original", encoding="utf-8")
    prediction = tmp_path / "prediction.npz"
    prediction.write_bytes(b"synthetic prediction bytes")
    plan_path = args.output / "plan.json"
    controller.base.write(plan_path, {"resources": {"fixed": True}})
    plan_sha = controller.base.digest(plan_path)
    monkeypatch.setattr(controller, "ORIGINAL_PLAN_SHA256", plan_sha)
    monkeypatch.setattr(controller.main, "checked", lambda _: {"resources": {"fixed": True}})
    copied = args.output / "copied.json"
    copied.write_text("copied", encoding="utf-8")
    frozen = {"protocol": controller.PROTOCOL, "script_sha256": controller.base.digest(controller.__file__),
        "test_sha256": controller.base.digest(controller.TEST_PATH), "main_sha256": controller.base.digest(controller.MAIN_PATH),
        "arguments": controller.path_arguments(args), "original_plan_sha256": plan_sha,
        "reused_verified_global_indices": list(range(297)), "fresh_global_indices": list(range(297, 540)),
        "original_files_sha256": {str(original): controller.base.digest(original)},
        "copied_files_sha256": {"plan.json": plan_sha, "copied.json": controller.base.digest(copied)},
        "preserved_prediction_sha256": {str(prediction): controller.base.digest(prediction)}, "resources": {"fixed": True}}
    controller.base.write(args.output / "recovery_plan.json", frozen)
    controller.checked(args)
    if target in ("original", "copied", "prediction"):
        {"original": original, "copied": copied, "prediction": prediction}[target].write_bytes(b"changed")
    elif target == "controller":
        frozen["script_sha256"] = "0" * 64
        controller.base.write(args.output / "recovery_plan.json", frozen)
    else:
        args.previous = tmp_path / "wrong-parent"
    with pytest.raises(ValueError):
        controller.checked(args)


def test_unchanged_selector_refuses_incomplete_recovery(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    monkeypatch.setattr(controller.main, "checked", lambda _: {})
    controller.base.write(args.output / "plan.json", {"synthetic": True})
    controller.base.write(args.output / "fold_completion.json", {
        "plan_sha256": controller.base.digest(args.output / "plan.json"), "all_fits_successful": False})
    with pytest.raises(ValueError, match="All540"):
        controller.main.select(args)
    assert not (args.output / "selected_cv.json").exists()
