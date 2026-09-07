"""Synthetic full-split controller receipts; no subprocesses or model fitting."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("psutil")

from benchmarks.tabarena import local_hpo_full_0160 as local

base = local.base
RESOURCES = {"num_cpus": 2, "memory_limit_gb": 8.0, "num_gpus": 0}
RUNTIME = {"synthetic_public_wheel": True}
REGISTRATION = {"synthetic_registered_full_plan": True}


def parent(repeat=0, fold=0, config=0, owner="local"):
    return {
        "parent_id": f"openml-123-r{repeat}-f{fold}-c{config:03d}",
        "task_id": 123,
        "dataset": "SYNTHETIC",
        "config_name": f"CTBoost_c{config}",
        "config_index": config,
        "repeat": repeat,
        "fold": fold,
        "owner": owner,
        "child_seeds": list(range(config * 8, config * 8 + 8)),
    }


@pytest.fixture
def factory(tmp_path, monkeypatch):
    worker_path = tmp_path / "synthetic_worker.py"
    worker_path.write_text(
        "# Fixture only; this script must never run.\n", encoding="utf-8"
    )
    wheel, registration = tmp_path / "synthetic.whl", tmp_path / "registration.json"
    wheel.write_bytes(b"SYNTHETIC_NOT_INSTALLABLE")
    base.write_json(registration, REGISTRATION)

    def validate(raw, expected_parent, plan_hash):
        result = base.read_json(raw)
        assert result["parent"] == expected_parent
        assert result["plan_sha256"] == plan_hash
        return {
            "raw_sha256": base.file_hash(raw),
            "runtime_sha256": result["runtime_sha256"],
            "outer_split": result["outer_split"],
        }

    worker = SimpleNamespace(
        __file__=str(worker_path),
        RESOURCES=RESOURCES,
        validate_plan=lambda plan: plan,
        read_json=base.read_json,
        plan_hash=base.json_hash,
        source_hash=base.file_hash,
        verify_registration=lambda *_: REGISTRATION,
        runtime_provenance=lambda *_: RUNTIME,
        validate_parent_result=validate,
    )
    allocation = base.allocation_policy
    monkeypatch.setattr(
        base,
        "allocation_policy",
        lambda _allowed, count: allocation(list(range(16)), count),
    )
    monkeypatch.setattr(
        base.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=32 * base.GIB, total=32 * base.GIB),
    )

    def make(parents=None):
        plan_path = tmp_path / "plan.json"
        if not plan_path.exists():
            base.write_json(
                plan_path,
                {
                    "resources": RESOURCES,
                    "parents": parents or [parent()],
                    "synthetic": True,
                },
            )
        return local.Controller(
            output=tmp_path / "output",
            plan_path=plan_path,
            wheel=wheel,
            registration=registration,
            worker=worker,
            max_workers=8,
            poll_seconds=0.1,
        )

    make.worker = worker
    return make


def artifact(run, split, status="complete"):
    manifest, raw, _directory = base.parent_paths(run.output, split)
    record = {
        "parent": split,
        "plan_sha256": run.plan_hash,
        "status": status,
        "host": "local",
        "resources": RESOURCES,
        "runtime": RUNTIME,
        "registration": REGISTRATION,
        "elapsed_seconds": 1,
        "peak_rss_bytes": 1024,
        "outer_split": {"synthetic_indices_sha256": "a" * 64},
    }
    if status == "complete":
        base.write_json(
            raw,
            {
                "parent": split,
                "plan_sha256": run.plan_hash,
                "runtime_sha256": run.runtime_hash,
                "outer_split": record["outer_split"],
            },
        )
        record["validation"] = {"raw_sha256": base.file_hash(raw)}
    else:
        record["error"] = "Synthetic terminal failure"
    base.write_json(manifest, record)
    return manifest, raw


def test_split_paths_are_unique_and_lite_zero_zero_remains_compatible(tmp_path):
    splits = [
        parent(repeat, fold, config)
        for config in range(2)
        for repeat in range(3)
        for fold in range(7)
    ]
    keys = [base.parent_key(split) for split in splits]
    assert len(set(keys)) == 42
    for split in splits:
        manifest, raw, receipt = base.parent_paths(tmp_path, split)
        suffix = f"{split['repeat']}_{split['fold']}"
        assert manifest.parent.name == raw.parent.name == receipt.name == suffix
    assert base.parent_key({"config_name": "Old", "task_id": 123}) == "Old/123/0_0"
    assert base.parent_key(parent()) == "CTBoost_c0/123/0_0"


@pytest.mark.parametrize("invalid", [-1, True, "1", "../1", 1.5])
def test_invalid_split_identity_is_refused(invalid):
    with pytest.raises(ValueError, match="repeat and fold"):
        base.parent_key({**parent(), "fold": invalid})


def test_full_command_keeps_exact_outer_split_identity_and_resource_policy(factory):
    split = parent(2, 6, 25)
    run = factory([split, parent(0, 0, owner="kaggle")])
    assert list(run.parents) == ["CTBoost_c25/123/2_6"]
    command = run._worker_command(split, 7, "claim", "permit", "token")
    assert command[:3] == [
        str(Path(sys.executable).resolve()),
        "-I",
        str(Path(local.__file__).resolve()),
    ]
    assert command[command.index("--parent-id") + 1] == "openml-123-r2-f6-c025"
    assert command[command.index("--host") + 1] == "local"
    assert command[command.index("--affinity") + 1] == "7,15"
    assert run.parents[base.parent_key(split)]["child_seeds"] == list(range(200, 208))
    assert run.policy["max_workers"] == 8
    assert run.policy["num_cpus_per_parent"] == 2
    assert run.policy["memory_limit_bytes_per_parent"] == 8 * base.GIB
    assert run.policy["physical_free_reserve_bytes"] == 4 * base.GIB


def test_restart_distinguishes_completed_failed_interrupted_and_unstarted_splits(
    factory, monkeypatch
):
    splits = [parent(0, 0), parent(0, 1), parent(1, 0), parent(1, 1)]
    run = factory(splits)
    artifact(run, splits[0])
    artifact(run, splits[1], status="failed")
    run.state["parents"][base.parent_key(splits[2])]["status"] = "running"
    run.save()
    run.reconcile()
    expected = ["complete", "failed", "interrupted", "queued"]
    assert [
        run.state["parents"][base.parent_key(split)]["status"] for split in splits
    ] == expected
    before = {path: path.read_bytes() for path in run.output.rglob("results.pkl")}
    resumed = factory()
    resumed.reconcile()
    assert [
        resumed.state["parents"][base.parent_key(split)]["status"] for split in splits
    ] == expected
    visited = []

    def launch(self, key, _slot, _reservation):
        visited.append(key)
        self.state["parents"][key]["status"] = "failed"

    monkeypatch.setattr(base.Controller, "launch", launch)
    resumed.dispatch()
    resumed.dispatch()
    assert visited == [base.parent_key(splits[3])]
    assert {path: path.read_bytes() for path in before} == before


def test_other_fold_artifact_cannot_satisfy_completion(factory):
    splits = [parent(0, 0), parent(0, 1)]
    run = factory(splits)
    manifest, raw = artifact(run, splits[0])
    with pytest.raises(ValueError, match="Completed parent identity"):
        run._validate_completed(splits[1], base.read_json(manifest), raw)
    run.reconcile()
    assert run.state["parents"][base.parent_key(splits[1])]["status"] == "queued"


@pytest.mark.parametrize("missing", [False, True])
def test_prefit_outer_split_receipt_must_match_validated_raw(factory, missing):
    split = parent(2, 6)
    run = factory([split])
    manifest, raw = artifact(run, split)
    record = base.read_json(manifest)
    if missing:
        del record["outer_split"]
    else:
        record["outer_split"]["synthetic_indices_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="outer-split receipt differs"):
        run._validate_completed(split, record, raw)


@pytest.mark.parametrize(
    "field",
    ["wheel", "runtime", "registration", "worker", "controller", "delegated", "base"],
)
def test_execution_freeze_rejects_provenance_drift(factory, field):
    run = factory()
    if field == "wheel":
        run.wheel.write_bytes(b"changed")
    elif field == "runtime":
        factory.worker.runtime_provenance = lambda *_: {"changed": True}
    elif field == "registration":
        factory.worker.verify_registration = lambda *_: {"changed": True}
    else:
        changed_path = {
            "worker": run.worker_script,
            "controller": Path(local.__file__),
            "delegated": local._LEGACY_PATH,
            "base": local.legacy._BASE_PATH,
        }[field].resolve()
        original = factory.worker.source_hash
        factory.worker.source_hash = lambda path: (
            "changed" if Path(path).resolve() == changed_path else original(path)
        )
    with pytest.raises(ValueError, match="Frozen full local controller"):
        factory()


def test_changed_repeat_in_registered_plan_refuses_resume(factory):
    run = factory()
    plan = copy.deepcopy(run.plan)
    plan["parents"][0]["repeat"] = 1
    base.write_json(run.plan_path, plan)
    with pytest.raises(ValueError, match="Frozen full local controller"):
        factory()
