"""Offline coordinator tests: no model fitting or subprocess launches."""

import hashlib
import importlib.util
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks/tabarena/worker_throughput.py"
)
SPEC = importlib.util.spec_from_file_location("throughput_harness_test", MODULE_PATH)
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def test_claims_are_exclusive_and_do_not_retry_started_jobs(tmp_path):
    (tmp_path / "claims").mkdir()
    jobs = [{"job_id": index} for index in range(32)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        claimed = list(
            pool.map(lambda slot: harness.claim(tmp_path, jobs, slot), range(32))
        )
    assert sorted(job["job_id"] for job in claimed) == list(range(32))
    assert harness.claim(tmp_path, jobs, 0) is None
    assert len(list((tmp_path / "claims").glob("*.json"))) == 32


@pytest.mark.parametrize("reason", ["unpaused", "running", "launching", "stale"])
def test_official_guard_refuses_unsafe_or_stale_receipts(tmp_path, reason):
    pause = tmp_path / "PAUSE"
    pause.touch()
    progress = {"paused": True, "active": [], "counts": {"running": 0, "launching": 0}}
    if reason == "unpaused":
        pause.unlink()
    if reason in {"running", "launching"}:
        progress["counts"][reason] = 1
    path = tmp_path / "progress.json"
    harness.write(path, progress)
    if reason == "stale":
        os.utime(path, (time.time() - 60, time.time() - 60))
    with pytest.raises(RuntimeError, match="fully drained"):
        harness.require_drained(tmp_path)


def test_drained_guard_is_read_only(tmp_path):
    pause = tmp_path / "PAUSE"
    pause.write_bytes(b"root owns this pause")
    harness.write(
        tmp_path / "progress.json",
        {"paused": True, "active": [], "counts": {"running": 0}},
    )
    harness.require_drained(tmp_path)
    assert pause.read_bytes() == b"root owns this pause"


def test_layouts_distinguish_current_pairs_and_actual_siblings():
    topology = {
        "allowed_logical_cpus": list(range(16)),
        "cores": [
            {
                "core_index": index,
                "processors": [
                    {"logical_cpu": cpu, "group": 0}
                    for cpu in (index * 2, index * 2 + 1)
                ],
            }
            for index in range(8)
        ],
    }
    layouts = harness.layouts(topology)
    assert layouts["A"]["slots"] == [[index, index + 8] for index in range(8)]
    assert layouts["B"]["slots"] == [[index] for index in range(16)]
    assert layouts["C"]["slots"] == [[index * 2, index * 2 + 1] for index in range(8)]
    assert all(
        len(set(row["physical_cores"])) == 1 for row in layouts["C"]["core_membership"]
    )


def test_ownership_requires_matching_creation_time_and_command():
    pytest.importorskip("psutil")
    command = [str(MODULE_PATH), "--worker", "arm", "--slot", "2"]
    identity = {"create_time": 123.0, "command": command}
    process = SimpleNamespace(create_time=lambda: 123.0, cmdline=lambda: command)
    assert harness.owns(process, identity)
    assert not harness.owns(process, {**identity, "create_time": 122.0})
    assert not harness.owns(process, {**identity, "command": command[:-1] + ["3"]})


def evidence(
    tmp_path,
    *,
    bad_predictions=False,
    cold_regression=False,
    second_steady_regression=False,
):
    np = pytest.importorskip("numpy")
    job = {"job_id": 0, "profile": "synthetic", "model_seed": 160000}
    plan = {
        "jobs": [job],
        "runtime": {"spec_sha256": "spec"},
        "layouts": {
            "A": {"histogram_threads": 2},
            "B": {"histogram_threads": 1},
            "C": {"histogram_threads": 2},
        },
    }
    arms = []
    for index, name in enumerate(harness.ORDER):
        values = np.full(256, 0.2 + (0.01 if bad_predictions and index == 1 else 0.0))
        row = {
            **job,
            "status": "complete",
            "is_warmup": False,
            "histogram_threads": plan["layouts"][name]["histogram_threads"],
            "histogram_threads_env": str(plan["layouts"][name]["histogram_threads"]),
            "iterations_trained": 128,
            "spec_sha256": "spec",
            "data_sha256": "data",
            "prediction_values": values.tolist(),
            "prediction_shape": [256],
            "prediction_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }
        harness.write(tmp_path / f"{index:02d}-{name}" / "results/000.json", row)
        rate = [100, 114, 111, 112, 115, 101][index]
        if second_steady_regression and index == 3:
            rate = 109
        cold_rate = 90 if cold_regression and index in (2, 3) else rate - 5
        arms.append(
            {
                "index": index,
                "layout": name,
                "minimum_system_free_bytes": 5 * harness.GIB,
                "steady_seconds": 3600 / rate,
                "steady_fits_per_hour": rate,
                "total_fits_per_hour": cold_rate,
            }
        )
    return plan, arms


def test_summary_prefers_sibling_pairs_within_five_percent(tmp_path):
    plan, arms = evidence(tmp_path)
    result = harness.summarize(tmp_path, plan, arms)
    assert result["selected_layout"] == "C"
    assert result["comparison"]["B"]["passes"]
    assert result["comparison"]["C"]["passes"]
    assert result["prediction_bitwise_equal"]
    assert result["automatic_policy_change"] is False


@pytest.mark.parametrize("variation", ["cold_regression", "second_steady_regression"])
def test_candidate_must_pass_both_orders_and_include_startup(tmp_path, variation):
    plan, arms = evidence(tmp_path, **{variation: True})
    result = harness.summarize(tmp_path, plan, arms)
    assert result["selected_layout"] == "B"
    assert not result["comparison"]["C"]["passes"]


def test_prediction_failure_preserves_current_policy(tmp_path):
    plan, arms = evidence(tmp_path, bad_predictions=True)
    result = harness.summarize(tmp_path, plan, arms)
    assert not result["prediction_allclose"]
    assert result["selected_layout"] == "A"


def test_incomplete_iteration_evidence_is_rejected(tmp_path):
    plan, arms = evidence(tmp_path)
    path = tmp_path / "02-C/results/000.json"
    row = harness.read(path)
    harness.write(path, {**row, "iterations_trained": 127})
    with pytest.raises(ValueError, match="workload or prediction"):
        harness.summarize(tmp_path, plan, arms)


@pytest.mark.parametrize(
    "change",
    [{"status": "failed"}, {"is_warmup": True}, {"histogram_threads_env": "1"}],
)
def test_summary_rejects_wrong_execution_controls(tmp_path, change):
    plan, arms = evidence(tmp_path)
    path = tmp_path / "02-C/results/000.json"
    harness.write(path, {**harness.read(path), **change})
    with pytest.raises(ValueError, match="workload or prediction"):
        harness.summarize(tmp_path, plan, arms)


def test_sibling_preference_uses_geometric_mean(tmp_path):
    plan, arms = evidence(tmp_path)
    for arm, rate in zip(arms, [80, 100, 185, 185, 400, 80]):
        arm.update(
            steady_fits_per_hour=rate,
            steady_seconds=3600 / rate,
            total_fits_per_hour=rate - 5,
        )
    result = harness.summarize(tmp_path, plan, arms)
    assert result["comparison"]["B"]["geometric_mean_steady_fits_per_hour"] == 200
    assert result["selected_layout"] == "B"


def test_worker_prepares_and_warms_once_before_shared_start_barrier(
    tmp_path, monkeypatch
):
    arm = tmp_path / "00-A"
    (arm / "claims").mkdir(parents=True)
    jobs = [{"job_id": 0}, {"job_id": 1}]
    plan = {
        "source_sha256": {"test": "frozen"},
        "runtime": {"test": True},
        "jobs": jobs,
        "layouts": {"A": {"slots": [[0, 8]], "histogram_threads": 2}},
    }
    harness.write(tmp_path / "spec.json", plan)
    harness.write(arm / "arm.json", {"layout": "A"})
    affinity, events = [], []

    def cpu_affinity(value=None):
        if value is not None:
            affinity[:] = value
        return affinity

    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process=lambda: SimpleNamespace(cpu_affinity=cpu_affinity)),
    )
    monkeypatch.setattr(harness, "source_hashes", lambda: plan["source_sha256"])

    def prepare():
        events.append("prepare")
        return "prepared"

    def fit(job, prepared, *, histogram_threads):
        assert prepared == "prepared" and histogram_threads == 2
        if job["job_id"] >= 0:
            assert (arm / "START").is_file()
        events.append(job["job_id"])
        return {**job, "iterations_trained": 4 if job["job_id"] == -1 else 128}

    fake_workload = SimpleNamespace(
        configure_threads=lambda threads: events.append(("threads", threads)),
        runtime_provenance=lambda: plan["runtime"],
        prepare_profiles=prepare,
        fit_job=fit,
        warmup_job=lambda: {"job_id": -1},
    )
    monkeypatch.setattr(harness, "helper", lambda name: fake_workload)

    def release_barrier(_seconds):
        assert (arm / "ready/00.json").is_file()
        assert list((arm / "claims").iterdir()) == []
        harness.write(arm / "START", {"test": "released"})

    monkeypatch.setattr(harness.time, "sleep", release_barrier)
    harness.worker_main(arm, 0)
    assert events == [("threads", 2), "prepare", -1, 0, 1]
    assert harness.read(arm / "ready/00.json")["warmup"]["iterations_trained"] == 4
    assert len(list((arm / "results").glob("*.json"))) == 2


def test_monitor_accepts_clean_exit_between_poll_and_process_lookup(
    tmp_path, monkeypatch
):
    class Gone(Exception):
        pass

    state = {"gone": False, "race_polled": False}
    command = []

    def poll():
        if state["gone"] and not state["race_polled"]:
            state["race_polled"] = True
            return None
        return 0 if state["gone"] else None

    process = SimpleNamespace(pid=123, returncode=0, poll=poll)

    def lookup(_pid):
        if state["gone"]:
            raise Gone
        return SimpleNamespace(
            create_time=lambda: 123.0,
            cmdline=lambda: command,
            children=lambda **kwargs: [],
            memory_info=lambda: SimpleNamespace(rss=1024),
        )

    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            Process=lookup,
            NoSuchProcess=Gone,
            AccessDenied=PermissionError,
            virtual_memory=lambda: SimpleNamespace(available=5 * harness.GIB),
        ),
    )
    arm = tmp_path / "00-A"

    def launch(args, **kwargs):
        command[:] = args
        assert args[1:3] == ["-I", "-B"]
        harness.write(arm / "ready/00.json", {})
        return process

    monkeypatch.setattr(harness.subprocess, "Popen", launch)

    def finish(_seconds):
        assert (arm / "START").is_file()
        harness.write(
            arm / "results/000.json",
            {"job_id": 0, "finished_monotonic": time.perf_counter()},
        )
        state["gone"] = True

    monkeypatch.setattr(harness.time, "sleep", finish)
    plan = {
        "jobs": [{"job_id": 0}],
        "layouts": {"A": {"slots": [[0, 8]]}},
        "guards": {
            "physical_free_bytes": 4 * harness.GIB,
            "startup_seconds": 120,
            "arm_seconds": 1200,
            "sampling_seconds": 0.5,
        },
    }
    result = harness.run_arm(tmp_path, 0, plan, None)
    assert result["status"] == "complete"
    assert state["race_polled"]
