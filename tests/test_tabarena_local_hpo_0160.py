"""Explicit synthetic controller fixtures; no official data or model fitting."""

from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("psutil")

from benchmarks.tabarena import local_hpo_0160 as local

base = local.base
GIB = base.GIB
RUNTIME = {"explicit_synthetic_fixture": True}
REGISTRATION = {"receipt": "EXPLICIT_SYNTHETIC_REGISTRATION"}
RESOURCES = {"num_cpus": 2, "num_gpus": 0, "memory_limit_gb": 8.0}


def parent(index=0, owner="local", outcome="complete"):
    return {"parent_id": f"EXPLICIT_SYNTHETIC_PARENT_{index}", "ordinal": index,
            "owner": owner, "dataset": f"SYNTHETIC_DATASET_{index}", "task_id": 1000 + index,
            "config_index": index, "config_name": f"SYNTHETIC_CONFIG_{index}",
            "problem_type": "binary", "repeat": 0, "fold": 0,
            "child_seeds": list(range(index * 8, index * 8 + 8)), "synthetic_outcome": outcome}


@pytest.fixture
def factory(tmp_path, monkeypatch):
    root = tmp_path / "EXPLICIT_SYNTHETIC_FIXTURE"
    root.mkdir()
    script = root / "synthetic_worker.py"
    script.write_text('''# No model training or network access: writes synthetic controller evidence only.
import argparse, hashlib, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("command")
for name in ["plan", "output", "parent-id", "host", "wheel", "registration", "affinity"]:
    p.add_argument("--" + name)
a = p.parse_args()
plan = json.loads(Path(a.plan).read_text())
assert plan["explicit_synthetic_fixture"] is True
parent = next(row for row in plan["parents"] if row["parent_id"] == a.parent_id)
assert parent["owner"] == a.host == "local"
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
runtime = {"explicit_synthetic_fixture": True}
key = Path(parent["config_name"]) / str(parent["task_id"]) / "0_0"
raw = Path(a.output) / "data" / key / "results.pkl"
manifest = {"parent": parent, "plan_sha256": digest(plan), "status": parent["synthetic_outcome"],
    "host": a.host, "resources": plan["resources"], "runtime": runtime,
    "registration": {"receipt": "EXPLICIT_SYNTHETIC_REGISTRATION"}, "pid": os.getpid(),
    "elapsed_seconds": 0.01, "peak_rss_bytes": 104857600}
if manifest["status"] == "complete":
    write(raw, {"parent": parent, "plan_sha256": digest(plan), "runtime_sha256": digest(runtime)})
    manifest["validation"] = {"raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
else:
    manifest["error"] = "Deliberate synthetic failure; never retry"
write(Path(a.output) / "artifacts" / key / "manifest.json", manifest)
raise SystemExit(0 if manifest["status"] == "complete" else 1)
''', encoding="utf-8")
    wheel, receipt = root / "synthetic.whl", root / "registration.json"
    wheel.write_bytes(b"EXPLICIT_SYNTHETIC_WHEEL_NOT_INSTALLABLE")
    base.write_json(receipt, REGISTRATION)
    calls = []

    def validate_plan(plan):
        assert plan["explicit_synthetic_fixture"] is True
        calls.append("plan")
        return plan

    def validate_result(raw, expected, digest):
        result = base.read_json(raw)
        assert result["parent"] == expected and result["plan_sha256"] == digest
        return {"raw_sha256": base.file_hash(raw), "runtime_sha256": result["runtime_sha256"]}

    worker = SimpleNamespace(
        __file__=str(script), RESOURCES=RESOURCES, validate_plan=validate_plan,
        read_json=base.read_json, plan_hash=base.json_hash, source_hash=base.file_hash,
        verify_registration=lambda *_: REGISTRATION,
        runtime_provenance=lambda *_: RUNTIME, validate_parent_result=validate_result,
    )
    monkeypatch.setattr(base.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=30 * GIB, total=32 * GIB))
    monkeypatch.setattr(base, "_set_execution_state", lambda *_: None)

    def make(parents=None, max_workers=1):
        plan_path = root / "plan.json"
        if not plan_path.exists():
            base.write_json(plan_path, {"explicit_synthetic_fixture": True,
                            "resources": RESOURCES, "parents": parents or [parent()]})
        return local.Controller(output=root / "output", plan_path=plan_path,
                                wheel=wheel, registration=receipt, worker=worker,
                                max_workers=max_workers, poll_seconds=.1)
    make.worker, make.root, make.calls = worker, root, calls
    return make


def artifact(run, p, **overrides):
    manifest, raw, _ = base.parent_paths(run.output, p)
    base.write_json(raw, {"parent": p, "plan_sha256": run.plan_hash,
                         "runtime_sha256": run.runtime_hash})
    payload = {"parent": p, "plan_sha256": run.plan_hash, "status": "complete",
               "host": "local", "resources": RESOURCES, "runtime": RUNTIME,
               "registration": REGISTRATION, "elapsed_seconds": 2, "peak_rss_bytes": GIB // 2,
               "validation": {"raw_sha256": base.file_hash(raw)}, **overrides}
    base.write_json(manifest, payload)
    return manifest, raw


def test_dispatch_owns_only_fixed_local_assignments(factory):
    run = factory([parent(0), parent(1, owner="kaggle"), parent(2)])
    assert [p["parent_id"] for p in run.parents.values()] == [parent(0)["parent_id"], parent(2)["parent_id"]]
    assert run.plan_hash == base.json_hash(run.plan)
    assert run.plan_hash != base.file_hash(run.plan_path)
    assert factory.calls == ["plan"]
    assert run.policy["unseen_dataset_reservation_bytes"] == 8 * GIB
    assert run.policy["physical_free_reserve_bytes"] == 4 * GIB
    assert run.policy["observed_peak_multiplier"] == 1.5


def test_worker_command_binds_local_identity_receipt_wheel_and_affinity(factory):
    run = factory()
    command = run._worker_command(parent(), 0, "claim", "permit", "token")
    assert command[:3] == [str(sys.executable), "-I", str(local.Path(local.__file__).resolve())]
    assert command[command.index("--host") + 1] == "local"
    assert command[command.index("--parent-id") + 1] == parent()["parent_id"]
    assert command[command.index("--wheel") + 1] == str(run.wheel)
    assert command[command.index("--registration") + 1] == str(run.registration)
    assert command[command.index("--affinity") + 1] == ",".join(map(str, run.policy["cpu_slots"][0]))
    assert "--decision" not in command


def test_canonical_plan_hash_allows_only_whitespace_reformat_on_resume(factory):
    first = factory()
    first.plan_path.write_text(json.dumps(first.plan, separators=(",", ":")), encoding="utf-8")
    assert factory().plan_hash == first.plan_hash
    changed = copy.deepcopy(first.plan)
    changed["parents"][0]["owner"] = "kaggle"
    base.write_json(first.plan_path, changed)
    with pytest.raises(ValueError, match="local parent"):
        factory()


def test_registration_failure_prevents_controller_state_or_launch(factory):
    def refuse(*_):
        raise ValueError("Unregistered synthetic plan")
    factory.worker.verify_registration = refuse
    with pytest.raises(ValueError, match="Unregistered"):
        factory()
    assert not (factory.root / "output/controller_state.json").exists()


@pytest.mark.parametrize("field", ["wheel", "runtime", "registration", "worker"])
def test_execution_provenance_drift_refuses_resume(factory, field):
    run = factory()
    if field == "wheel":
        run.wheel.write_bytes(b"changed")
    elif field == "runtime":
        factory.worker.runtime_provenance = lambda *_: {"changed": True}
    elif field == "registration":
        factory.worker.verify_registration = lambda *_: {"changed": True}
    else:
        run.worker_script.write_text("# changed synthetic source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen local controller"):
        factory()


@pytest.mark.parametrize("change", ["runtime", "host", "resources", "registration", "status", "raw"])
def test_result_requires_complete_local_attested_runtime(factory, change):
    run = factory()
    manifest, raw = artifact(run, parent())
    assert run._validate_completed(parent(), base.read_json(manifest), raw)["runtime_sha256"] == run.runtime_hash
    data = base.read_json(manifest)
    if change == "raw":
        raw.write_bytes(b"changed")
    else:
        data[change] = "invalid"
    with pytest.raises(ValueError, match="Completed parent"):
        run._validate_completed(parent(), data, raw)


def test_resource_timeout_preserves_terminal_evidence(factory):
    run = factory()
    manifest, _ = artifact(run, parent(), status="running")
    evidence = {"status": "timeout", "limit_seconds": 4500}
    base.write_json(manifest.with_name("resource_failure.json"), evidence)
    key = base.parent_key(parent())
    run._finish_from_artifacts(key, fallback="failed", exit_code=4)
    assert run.state["parents"][key]["status"] == "timed_out"
    assert run.state["parents"][key]["resource_failure"] == evidence


def test_raw_runtime_digest_must_match_manifest_attestation(factory):
    run = factory()
    manifest, raw = artifact(run, parent())
    result = base.read_json(raw)
    result["runtime_sha256"] = "0" * 64
    base.write_json(raw, result)
    data = base.read_json(manifest)
    data["validation"]["raw_sha256"] = base.file_hash(raw)
    with pytest.raises(ValueError, match="Raw parent runtime"):
        run._validate_completed(parent(), data, raw)


def test_terminal_launch_failure_does_not_block_independent_assignments(factory, monkeypatch):
    run = factory([parent(0), parent(1)])
    visited = []

    def launch(self, key, slot, reservation):
        visited.append(key)
        self.state["parents"][key].update(status="launch_failed")
        raise OSError("Deliberate synthetic launch failure")

    monkeypatch.setattr(base.Controller, "launch", launch)
    run.dispatch()
    assert visited == list(run.parents)
    assert all("Deliberate synthetic" in r["error"] for r in run.state["parents"].values())
    run.dispatch()
    assert visited == list(run.parents)  # Terminal attempts are never dispatched again.


def test_completed_cache_verified_and_reused_without_launch(factory):
    run = factory()
    artifact(run, parent())
    run.reconcile()
    assert run.state["parents"][base.parent_key(parent())]["status"] == "complete"
    resumed = factory()
    resumed.reconcile()
    assert resumed.progress()["counts"]["complete"] == 1


def test_synthetic_process_failure_continues_next_parent_and_never_retries(factory):
    parents = [parent(0, outcome="failed"), parent(1), parent(2, owner="kaggle")]
    run = factory(parents)
    with base.controller_lock(run.output):
        assert run.run() == 1
    records = run.state["parents"]
    failed, completed = records[base.parent_key(parents[0])], records[base.parent_key(parents[1])]
    assert failed["status"] == "failed" and "Deliberate synthetic" in failed["worker_error"]
    assert completed["status"] == "complete"
    assert not base.parent_paths(run.output, parents[2])[2].exists()
    identity = completed["identity"]
    resumed = factory()
    with base.controller_lock(resumed.output):
        assert resumed.run() == 1
    assert resumed.state["parents"][base.parent_key(parents[1])]["identity"] == identity
