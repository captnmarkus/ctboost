"""Local runner recovery must not repeat started fits or trust mutable inputs."""

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil")

from benchmarks.tabarena import local_pilot as runner
from benchmarks.tabarena.pilot_evaluation import expected_pilot_jobs


def test_atomic_checkpoint_survives_transient_windows_reader_lock(tmp_path, monkeypatch):
    destination = tmp_path / "checkpoint.json"
    destination.write_text('{"previous":true}')
    original_replace = Path.replace
    attempts = []

    def temporarily_locked(source, target):
        attempts.append(1)
        if len(attempts) <= 2:
            assert json.loads(destination.read_text()) == {"previous": True}
            raise PermissionError("Windows destination is briefly open by a reader")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", temporarily_locked)
    runner.write_json(destination, {"complete": True})
    assert json.loads(destination.read_text()) == {"complete": True}
    assert len(attempts) == 3
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.fixture
def pilot_protocol():
    path = Path(__file__).resolve().parents[1] / "benchmarks/tabarena/pilot_0159_v1.json"
    return json.loads(path.read_text()), hashlib.sha256(path.read_bytes()).hexdigest()


def interrupted_state(tmp_path, pilot_protocol):
    protocol, protocol_hash = pilot_protocol
    job = expected_pilot_jobs(protocol)[0]
    return {
        **job, "phase": "fit", "status": "failed", "pid": 7654321,
        "process_create_time": 123.5, "protocol_sha256": protocol_hash,
        "split_sha256": "a" * 64, "preprocessing_sha256": "b" * 64,
        "resource_contract_sha256": "c" * 64, "deadline_stopped": False,
        "fit_started_monotonic": 123,
        "result_path": str(runner.record_path(tmp_path, job["dataset"], job["inner_fold"], job["variant"])),
    }


def test_recovery_records_started_fit_once_and_keeps_unstarted_jobs_unrun(tmp_path, pilot_protocol, monkeypatch):
    state = interrupted_state(tmp_path, pilot_protocol)
    runner.write_json(tmp_path / "active/slot0.json", state)
    monkeypatch.setattr(runner, "matching_owned_worker", lambda *_: None)
    runner.reconcile_active_journals(tmp_path, *pilot_protocol)
    records = runner.load_records(tmp_path)
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["interrupted"] is True
    assert records[0]["job_id"] == state["job_id"]
    assert len(list((tmp_path / "journal").glob("*.json"))) == 1
    runner.reconcile_active_journals(tmp_path, *pilot_protocol)
    assert runner.load_records(tmp_path) == records
    assert len(list((tmp_path / "journal").glob("*.json"))) == 1


def test_recovery_never_overwrites_a_completed_result(tmp_path, pilot_protocol):
    state = interrupted_state(tmp_path, pilot_protocol)
    destination = Path(state["result_path"])
    complete = {"status": "ok", "metric_error": 0.1, "evidence": "must remain unchanged"}
    runner.write_json(destination, complete)
    before = destination.read_bytes()
    runner.preserve_interrupted_fit(state, tmp_path, *pilot_protocol, "stopped")
    assert destination.read_bytes() == before


def test_recovery_rejects_foreign_result_path(tmp_path, pilot_protocol):
    state = interrupted_state(tmp_path, pilot_protocol)
    state["result_path"] = str(tmp_path.parent / "unrelated.json")
    with pytest.raises(ValueError, match="outside"):
        runner.preserve_interrupted_fit(state, tmp_path, *pilot_protocol, "stopped")
    assert not (tmp_path.parent / "unrelated.json").exists()


def test_recovery_rereads_journal_after_stopping_a_live_worker(tmp_path, pilot_protocol, monkeypatch):
    fit_state = interrupted_state(tmp_path, pilot_protocol)
    path = tmp_path / "active/slot0.json"
    runner.write_json(path, {"phase": "between_fits", "pid": 7654321, "process_create_time": 123.5})
    owned = object()
    monkeypatch.setattr(runner, "matching_owned_worker", lambda *_: owned)

    def kill(process):
        assert process is owned
        runner.write_json(path, fit_state)

    monkeypatch.setattr(runner, "kill_worker", kill)
    runner.reconcile_active_journals(tmp_path, *pilot_protocol)
    assert runner.load_records(tmp_path)[0]["job_id"] == fit_state["job_id"]


class FakeProcess:
    def __init__(self, created, command):
        self.created = created
        self.command = command

    def create_time(self):
        return self.created

    def cmdline(self):
        return self.command


def test_process_ownership_requires_creation_time_script_stage_and_output(tmp_path, monkeypatch):
    command = ["python.exe", "-I", str(Path(runner.__file__).resolve()), "worker", "--output", str(tmp_path)]
    process = FakeProcess(123.5, command)
    monkeypatch.setattr(psutil, "Process", lambda _pid: process)
    assert runner.matching_owned_worker({"pid": 10, "process_create_time": 123.5}, tmp_path) is process
    assert runner.matching_owned_worker({"pid": 10, "process_create_time": 122}, tmp_path) is None
    for foreign in (["python.exe", "user_script.py"], command[:3] + ["run"] + command[4:], command[:-1] + [str(tmp_path / "other")]):
        process.command = foreign
        with pytest.raises(RuntimeError, match="ownership"):
            runner.matching_owned_worker({"pid": 10, "process_create_time": 123.5}, tmp_path)


def test_missing_creation_time_never_authorizes_killing_a_live_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(psutil, "pid_exists", lambda _pid: True)
    with pytest.raises(RuntimeError, match="creation-time"):
        runner.matching_owned_worker({"pid": 10}, tmp_path)


def test_worker_verifies_inputs_against_frozen_manifest_not_rewritten_sidecar(tmp_path):
    path = tmp_path / "data.pkl"
    path.write_bytes(b"original training inputs")
    prepared = {"dataset": "example", "protocol_sha256": "a" * 64,
                "data_sha256": runner.file_hash(path), "splits": {"0": {"sha256": "b" * 64}}}
    execution = {"protocol_sha256": "a" * 64, "prepared": {"datasets": [copy.deepcopy(prepared)]}}
    runner.verify_prepared_inputs(execution, "a" * 64, prepared, path, "example")
    path.write_bytes(b"changed training inputs")
    with pytest.raises(ValueError, match="data changed"):
        runner.verify_prepared_inputs(execution, "a" * 64, prepared, path, "example")
    prepared["data_sha256"] = runner.file_hash(path)
    with pytest.raises(ValueError, match="manifest changed"):
        runner.verify_prepared_inputs(execution, "a" * 64, prepared, path, "example")


def test_memory_sampler_captures_rss_synchronously_even_for_instant_work(monkeypatch):
    class SampleProcess:
        def memory_info(self):
            return type("Memory", (), {"rss": 123456})()

    monkeypatch.setattr(psutil, "Process", SampleProcess)
    with runner.MemorySampler() as sample:
        assert sample.peak == 123456
    assert sample.peak == 123456


def test_worker_checks_real_native_and_source_identity_before_preprocessing(tmp_path, monkeypatch):
    execution = {"resources": {}, "affinity": [[0]], "runtime": {"native_extension_sha256": "old"}}
    runner.write_json(tmp_path / "execution.json", execution)
    args = type("Args", (), {"output": tmp_path, "slot": 0, "dataset": "example", "fold": 0})()

    class WorkerProcess:
        def cpu_affinity(self, _affinity):
            return None

        def create_time(self):
            return 123.5

    monkeypatch.setattr(psutil, "Process", WorkerProcess)
    monkeypatch.setattr(runner, "bootstrap_imports", lambda: (None, None))
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"native_extension_sha256": "new"})
    with pytest.raises(ValueError, match="native extension"):
        runner.run_block(args)


def test_public_wheel_must_match_the_installed_native_extension(tmp_path):
    output = tmp_path / "pilot"
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel = wheel_dir / "ctboost-0.1.59-cp312-cp312-win_amd64.whl"
    native = b"verified native extension bytes"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("ctboost/_core.cp312-win_amd64.pyd", native)
    report = runner.public_wheel_provenance(output, hashlib.sha256(native).hexdigest())
    assert report["filename"] == wheel.name
    assert report["sha256"] == runner.file_hash(wheel)
    with pytest.raises(ValueError, match="matching"):
        runner.public_wheel_provenance(output, "0" * 64)
