"""Synthetic scheduling/recovery tests; no official data, models, or fit calls."""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest

psutil = pytest.importorskip("psutil")

from benchmarks.tabarena import local_hpo_controller as controller

GIB = controller.GIB


def parent(index=0, dataset="SYNTHETIC_DATASET"):
    return {
        "dataset": dataset,
        "task_id": index + 1000,
        "config_index": index,
        "config_name": f"SYNTHETIC_CONFIG_{index}",
        "problem_type": "binary",
        "repeat": 0,
        "fold": 0,
        "child_seeds": list(range(index * 8, index * 8 + 8)),
    }


@pytest.fixture
def make_controller(tmp_path, monkeypatch):
    project = tmp_path / "EXPLICIT_SYNTHETIC_PROJECT"
    script = project / "benchmarks/tabarena/local_hpo.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "# Synthetic fixture; this file performs no model fitting.\n", encoding="utf-8"
    )
    monkeypatch.setattr(controller, "ROOT", project)
    decision, protocol = project / "decision.json", project / "protocol.json"
    controller.write_json(decision, {"synthetic": True})
    controller.write_json(protocol, {"synthetic": True})
    validations = []

    def validate_result(raw, expected_parent, plan_hash):
        result = controller.read_json(raw)
        assert result == {"synthetic_parent": expected_parent, "plan_sha256": plan_hash}
        validations.append(str(raw))
        return {"raw_sha256": controller.file_hash(raw), "synthetic_only": True}

    worker = SimpleNamespace(validate_result=validate_result)

    def make(parents=None, output=None, max_workers=2):
        output = output or project / "SYNTHETIC_RESULTS"
        plan = {"parents": parents or [parent()]}
        if not (output / "plan.json").exists():
            controller.write_json(output / "plan.json", plan)
        return controller.Controller(
            output=output,
            plan=plan,
            worker=worker,
            decision=decision,
            protocol=protocol,
            python=sys.executable,
            max_workers=max_workers,
            poll_seconds=0.1,
        )

    make.script, make.validations = script, validations
    return make


def artifact(run, p, *, status="complete", raw=True):
    manifest_path, raw_path, _ = controller.parent_paths(run.output, p)
    if raw:
        controller.write_json(
            raw_path, {"synthetic_parent": p, "plan_sha256": run.plan_hash}
        )
    manifest = {
        "parent": p,
        "plan_sha256": run.plan_hash,
        "status": status,
        "elapsed_seconds": 2,
        "peak_rss_bytes": 600_000_000,
    }
    if raw:
        manifest["validation"] = {"raw_sha256": controller.file_hash(raw_path)}
    controller.write_json(manifest_path, manifest)
    return manifest_path, raw_path


def test_slots_are_disjoint_and_limited_to_available_cpus():
    policy = controller.allocation_policy(range(16), 8)
    assert policy["cpu_slots"] == [[i, i + 8] for i in range(8)]
    assert len({cpu for slot in policy["cpu_slots"] for cpu in slot}) == 16
    assert controller.allocation_policy([3, 5, 9, 13, 20], 8)["cpu_slots"] == [
        [3, 9],
        [5, 13],
    ]


@pytest.mark.parametrize("workers", [0, 9, True])
def test_invalid_worker_count(workers):
    with pytest.raises(ValueError):
        controller.allocation_policy(range(16), workers)


def test_cpu_allocation_needs_two_cpus():
    with pytest.raises(ValueError, match="Two"):
        controller.allocation_policy([1], 1)


def test_reservation_is_conservative_for_unknown_and_completed_datasets():
    policy = controller.allocation_policy(range(16), 8)
    assert controller.reservation_bytes("a", {}, policy) == 8 * GIB
    assert controller.reservation_bytes("a", {"a": GIB // 3}, policy) == GIB
    assert controller.reservation_bytes("a", {"a": 2 * GIB}, policy) == 3 * GIB
    assert controller.reservation_bytes("a", {"a": 8 * GIB}, policy) == 12 * GIB


def test_dispatch_accounts_for_unconsumed_reservations_and_free_reserve():
    policy = controller.allocation_policy(range(16), 8)
    active = [{"reservation_bytes": 8 * GIB, "rss_bytes": 2 * GIB}]
    assert not controller.can_dispatch(17 * GIB, 32 * GIB, 8 * GIB, active, policy)
    assert controller.can_dispatch(18 * GIB, 32 * GIB, 8 * GIB, active, policy)
    assert not controller.can_dispatch(40 * GIB, 16 * GIB, 8 * GIB, active, policy)
    assert not controller.can_dispatch(4 * GIB, 32 * GIB, GIB, [], policy)


def test_dispatch_does_not_double_count_rss():
    policy = controller.allocation_policy(range(16), 8)
    active = [{"reservation_bytes": 3 * GIB, "rss_bytes": 3 * GIB}]
    assert controller.can_dispatch(5 * GIB, 32 * GIB, GIB, active, policy)


def test_running_import_rss_does_not_establish_completed_dataset_peak(make_controller):
    run = make_controller()
    record = {"status": "running", "peak_rss_bytes": 100_000_000}
    run._observe(parent(), record)
    assert not run.state["observed_dataset_peak_bytes"]
    record["status"] = "complete"
    run._observe(parent(), record)
    assert run.state["observed_dataset_peak_bytes"][parent()["dataset"]] == 100_000_000


def test_dispatch_scans_past_unknown_high_memory_jobs(make_controller, monkeypatch):
    parents = [parent(0, "unknown"), parent(1, "known"), parent(2, "known")]
    run = make_controller(parents)
    run.state["observed_dataset_peak_bytes"]["known"] = GIB // 3
    monkeypatch.setattr(
        controller.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=6 * GIB, total=32 * GIB),
    )
    launched = []

    def launch(key, slot, reservation):
        launched.append((key, slot, reservation))
        run.state["parents"][key].update(
            status="running", slot=slot, reservation_bytes=reservation, rss_bytes=0
        )

    monkeypatch.setattr(run, "launch", launch)
    run.dispatch()
    assert [entry[0] for entry in launched] == [
        controller.parent_key(p) for p in parents[1:]
    ]
    assert run.state["parents"][controller.parent_key(parents[0])]["status"] == "queued"


def test_pause_prevents_new_dispatch(make_controller, monkeypatch):
    run = make_controller()
    (run.output / "PAUSE").touch()
    monkeypatch.setattr(run, "launch", lambda *_: pytest.fail("dispatch while paused"))
    run.dispatch()
    assert run.progress()["paused"]


def test_sleep_inhibition_restores_on_exception_and_drain(monkeypatch):
    calls = []
    monkeypatch.setattr(controller, "_set_execution_state", calls.append)
    with pytest.raises(RuntimeError), controller.SleepInhibitor(
        enabled=True
    ) as inhibitor:
        inhibitor.set_active(True)
        inhibitor.set_active(True)
        inhibitor.set_active(False)
        inhibitor.set_active(True)
        raise RuntimeError("synthetic shutdown")
    assert calls == [0x80000001, 0x80000000, 0x80000001, 0x80000000]


def test_non_windows_sleep_hook_is_not_called(monkeypatch):
    monkeypatch.setattr(
        controller, "_set_execution_state", lambda *_: pytest.fail("Win32 hook")
    )
    with controller.SleepInhibitor(enabled=False) as inhibitor:
        inhibitor.set_active(True)


class FakeProcess:
    def __init__(self, pid=73, created=100, command=None):
        self.pid, self.created = pid, created
        self.command = command or ["python", "synthetic_worker"]
        self.signals = []
        self.child_processes = []
        self.alive = True

    def create_time(self):
        return self.created

    def cmdline(self):
        return self.command

    def is_running(self):
        return self.alive

    def status(self):
        return psutil.STATUS_RUNNING

    def children(self, recursive=False):
        assert recursive
        return self.child_processes

    def suspend(self):
        self.signals.append("suspend")

    def kill(self):
        self.signals.append("kill")
        self.alive = False


@pytest.mark.parametrize("change", ["create_time", "command"])
def test_pid_reuse_or_command_mismatch_is_never_signalled(monkeypatch, change):
    process = FakeProcess()
    identity = controller.process_identity(process)
    if change == "create_time":
        process.created += 1
    else:
        process.command = ["unrelated", "work"]
    monkeypatch.setattr(controller.psutil, "Process", lambda _: process)
    monkeypatch.setattr(
        controller.psutil, "wait_procs", lambda *_args, **_kwargs: ([], [])
    )
    assert controller.terminate_owned_tree(identity) == []
    assert process.signals == []


def test_owned_descendants_stopped_but_reused_descendant_not_signalled(monkeypatch):
    root, child, reused = FakeProcess(), FakeProcess(74), FakeProcess(75)
    reused_identity = controller.process_identity(reused)
    reused.created += 2
    root.child_processes = [child]
    processes = {process.pid: process for process in (root, child, reused)}
    monkeypatch.setattr(controller.psutil, "Process", processes.__getitem__)
    monkeypatch.setattr(
        controller.psutil, "wait_procs", lambda values, **_: (values, [])
    )
    killed = controller.terminate_owned_tree(
        controller.process_identity(root), [reused_identity]
    )
    assert set(killed) == {73, 74}
    assert root.signals == ["suspend", "kill"]
    assert child.signals == ["kill"]
    assert reused.signals == []


def test_controller_lock_is_exclusive_and_released(tmp_path):
    with controller.controller_lock(tmp_path), pytest.raises(
        (OSError, BlockingIOError)
    ), controller.controller_lock(tmp_path):
        pytest.fail("second controller acquired the lock")
    with controller.controller_lock(tmp_path):
        pass


def test_execution_policy_is_frozen_before_any_launch(make_controller):
    run = make_controller()
    manifest = controller.read_json(run.output / "controller_execution.json")
    assert manifest["controller_source_sha256"] == controller.file_hash(
        controller.__file__
    )
    assert manifest["allocation_policy"]["timeout_seconds"] == 4500
    assert manifest["allocation_policy"]["physical_free_reserve_bytes"] == 4 * GIB
    with pytest.raises(ValueError, match="Frozen controller"):
        make_controller(max_workers=1)


def test_resume_imports_valid_completed_artifact_without_fitting(make_controller):
    run = make_controller()
    manifest, raw = artifact(run, parent())
    before = (manifest.read_bytes(), raw.read_bytes())
    run.reconcile()
    key = controller.parent_key(parent())
    assert run.state["parents"][key]["status"] == "complete"
    assert (manifest.read_bytes(), raw.read_bytes()) == before
    restored = make_controller()
    restored.reconcile()
    assert len(make_controller.validations) == 2


def test_completed_raw_modified_on_restart_is_refused(make_controller):
    run = make_controller()
    _, raw = artifact(run, parent())
    run.reconcile()
    raw.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        make_controller().reconcile()


@pytest.mark.parametrize(
    "status", ["failed", "interrupted", "resource_pressure", "timed_out"]
)
def test_terminal_failures_are_preserved_and_never_requeued(make_controller, status):
    run = make_controller()
    key = controller.parent_key(parent())
    manifest, _ = artifact(run, parent(), status="failed", raw=False)
    before = manifest.read_bytes()
    run.state["parents"][key] = {"status": status, "error": "synthetic failure"}
    run.save()
    restored = make_controller()
    restored.reconcile()
    assert restored.state["parents"][key] == run.state["parents"][key]
    assert manifest.read_bytes() == before


def test_started_worker_without_raw_becomes_terminal_interrupted(
    make_controller, monkeypatch
):
    run = make_controller()
    key = controller.parent_key(parent())
    manifest, _ = artifact(run, parent(), status="running", raw=False)
    before = manifest.read_bytes()
    identity = {"pid": 900, "create_time": 100, "command": ["synthetic"]}
    run.state["parents"][key] = {"status": "running", "identity": identity}
    stopped = []
    monkeypatch.setattr(
        controller,
        "terminate_owned_tree",
        lambda identity, *_: stopped.append(identity),
    )
    run.reconcile()
    assert stopped == [identity]
    assert run.state["parents"][key]["status"] == "interrupted"
    assert manifest.read_bytes() == before
    with pytest.raises(ValueError, match="Only an unstarted"):
        run.launch(key, 0, GIB)


def test_launch_claim_recovers_crash_before_pid_state_write(
    make_controller, monkeypatch
):
    run = make_controller()
    key = controller.parent_key(parent())
    claim = run.output / "claim.json"
    identity = {
        "pid": 900,
        "create_time": 100,
        "command": [sys.executable, "synthetic", "unique-token"],
    }
    controller.write_json(claim, {"token": "unique-token", "identity": identity})
    run.state["parents"][key] = {
        "status": "launching",
        "claim_path": str(claim),
        "launch_token": "unique-token",
        "command": identity["command"],
        "launched_epoch": 99,
    }
    stopped = []
    monkeypatch.setattr(
        controller,
        "terminate_owned_tree",
        lambda identity, *_: stopped.append(identity),
    )
    run.reconcile()
    assert stopped == [identity]
    assert run.state["parents"][key]["status"] == "interrupted"


def test_foreign_launch_claim_is_refused(make_controller):
    run = make_controller()
    claim = run.output / "claim.json"
    controller.write_json(claim, {"token": "foreign", "identity": {}})
    with pytest.raises(ValueError, match="launch claim"):
        run._claimed_identity(
            {
                "claim_path": str(claim),
                "launch_token": "owned",
                "command": ["python"],
                "launched_epoch": 100,
            }
        )


def test_started_worker_with_valid_raw_is_completed_without_retry(
    make_controller, monkeypatch
):
    run = make_controller()
    key = controller.parent_key(parent())
    artifact(run, parent(), status="running")
    run.state["parents"][key] = {"status": "running", "identity": None}
    monkeypatch.setattr(controller, "terminate_owned_tree", lambda *_: [])
    run.reconcile()
    assert run.state["parents"][key]["status"] == "complete"


def test_failed_worker_even_with_raw_remains_failed(make_controller):
    run = make_controller()
    key = controller.parent_key(parent())
    artifact(run, parent(), status="failed")
    run.reconcile()
    assert run.state["parents"][key]["status"] == "failed"
    assert not make_controller.validations


def test_resource_failure_artifact_is_preserved(make_controller):
    run = make_controller()
    key = controller.parent_key(parent())
    manifest, _ = artifact(run, parent(), status="running", raw=False)
    failure = manifest.with_name("resource_failure.json")
    failure.write_text('{"synthetic_limit": true}', encoding="utf-8")
    before = failure.read_bytes()
    run.reconcile()
    assert run.state["parents"][key]["status"] == "resource_limit"
    assert failure.read_bytes() == before


def test_raw_without_worker_provenance_is_terminal_invalid(make_controller):
    run = make_controller()
    manifest, _ = artifact(run, parent())
    manifest.unlink()
    run.reconcile()
    assert (
        run.state["parents"][controller.parent_key(parent())]["status"]
        == "invalid_result"
    )


def test_timeout_stops_owned_worker_and_never_retries(make_controller, monkeypatch):
    run = make_controller()
    key = controller.parent_key(parent())
    run.state["parents"][key] = {
        "status": "running",
        "identity": {"synthetic": True},
        "launched_epoch": time.time() - 4501,
        "launched_monotonic": time.monotonic() - 4501,
    }
    monkeypatch.setattr(controller, "owned_process", lambda _: object())
    monkeypatch.setattr(controller, "sample_owned_tree", lambda _: (GIB, []))
    stopped = []
    monkeypatch.setattr(
        controller,
        "terminate_owned_tree",
        lambda identity, *_: stopped.append(identity),
    )
    run.poll()
    assert stopped == [{"synthetic": True}]
    assert run.state["parents"][key]["status"] == "timed_out"


def test_memory_pressure_stops_only_newest_owned_parent(make_controller, monkeypatch):
    run = make_controller([parent(0), parent(1)])
    for index, (key, record) in enumerate(run.state["parents"].items()):
        record.update(
            status="running",
            identity={"synthetic": index},
            launched_epoch=time.time() + index,
            launched_monotonic=time.monotonic(),
        )
    monkeypatch.setattr(controller, "owned_process", lambda _: object())
    monkeypatch.setattr(controller, "sample_owned_tree", lambda _: (GIB, []))
    monkeypatch.setattr(
        controller.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=3 * GIB, total=32 * GIB),
    )
    stopped = []
    monkeypatch.setattr(
        controller,
        "terminate_owned_tree",
        lambda identity, *_: stopped.append(identity),
    )
    run.poll()
    assert stopped == [{"synthetic": 1}]
    assert run.state["parents"][controller.parent_key(parent(0))]["status"] == "running"
    assert (
        run.state["parents"][controller.parent_key(parent(1))]["status"]
        == "resource_pressure"
    )


def test_synthetic_worker_launch_cache_validation_and_resume(make_controller):
    """A real child process writes synthetic JSON only; no model libraries or fitting."""
    make_controller.script.write_text(
        """
import argparse, hashlib, json, time
from pathlib import Path
import psutil
parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--output", type=Path)
parser.add_argument("--dataset")
parser.add_argument("--config-index", type=int)
parser.add_argument("--affinity")
args, _ = parser.parse_known_args()
psutil.Process().cpu_affinity([int(value) for value in args.affinity.split(",")])
plan_path = args.output / "plan.json"
plan = json.loads(plan_path.read_text())
parent = next(p for p in plan["parents"] if p["config_index"] == args.config_index)
plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
key = f"{parent['config_name']}/{parent['task_id']}/0_0"
raw = args.output / "data" / key / "results.pkl"
raw.parent.mkdir(parents=True)
time.sleep(0.2)
raw.write_text(json.dumps({"synthetic_parent": parent, "plan_sha256": plan_hash}))
manifest = args.output / "artifacts" / key / "manifest.json"
manifest.parent.mkdir(parents=True)
manifest.write_text(json.dumps({
    "parent": parent, "plan_sha256": plan_hash, "status": "complete",
    "peak_rss_bytes": psutil.Process().memory_info().rss, "elapsed_seconds": 0.2,
    "validation": {"raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()},
    "synthetic_affinity": psutil.Process().cpu_affinity(),
}))
""",
        encoding="utf-8",
    )
    run = make_controller(max_workers=1)
    key = controller.parent_key(parent())
    run.launch(key, 0, GIB)
    try:
        deadline = time.monotonic() + 15
        while (
            run.state["parents"][key]["status"] in controller.ACTIVE
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)
            run.poll()
        assert run.state["parents"][key]["status"] == "complete"
        manifest, raw, directory = controller.parent_paths(run.output, parent())
        assert controller.read_json(manifest)["synthetic_affinity"] == sorted(
            run.policy["cpu_slots"][0]
        )
        permit = controller.read_json(directory / "permit.json")
        assert permit["identity"] == run.state["parents"][key]["identity"]
        preserved = (
            manifest.read_bytes(),
            raw.read_bytes(),
            (directory / "worker.log").read_bytes(),
        )
        resumed = make_controller(max_workers=1)
        resumed.reconcile()
        assert resumed.state["parents"][key]["status"] == "complete"
        assert preserved == (
            manifest.read_bytes(),
            raw.read_bytes(),
            (directory / "worker.log").read_bytes(),
        )
    finally:
        controller.terminate_owned_tree(run.state["parents"][key].get("identity"))


def test_main_revalidates_existing_plan_once_before_controller(
    make_controller, monkeypatch
):
    run = make_controller()
    calls = []
    worker = SimpleNamespace(
        _configure_process=lambda *_: None,
        bootstrap_imports=lambda: None,
        build_plan=lambda **kwargs: calls.append(kwargs) or {"resources": "fixed"},
        resource_contract=lambda *_: "fixed",
    )
    monkeypatch.setattr(controller.importlib.util, "module_from_spec", lambda _: worker)
    monkeypatch.setattr(
        controller.importlib.util,
        "spec_from_file_location",
        lambda *_: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: None)),
    )
    monkeypatch.setattr(
        controller, "Controller", lambda **_: SimpleNamespace(run=lambda: 0)
    )
    assert (
        controller.main(
            [
                "--output",
                str(run.output),
                "--decision",
                str(run.decision),
                "--protocol",
                str(run.protocol),
            ]
        )
        == 0
    )
    assert len(calls) == 1
    assert calls[0]["num_cpus"] == 2 and calls[0]["memory_limit_gb"] == 8


def test_main_refuses_missing_plan(tmp_path):
    with pytest.raises(ValueError, match="immutable HPO plan"):
        controller.main(
            ["--output", str(tmp_path), "--decision", "unused", "--protocol", "unused"]
        )
