"""Real-process checks of resource accounting and owned descendant cleanup."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


def watchdog():
    source = Path(__file__).resolve().parents[1] / "benchmarks/owned_process_watchdog.py"
    spec = importlib.util.spec_from_file_location("tested_owned_watchdog", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spawn(code):
    return subprocess.Popen([sys.executable, "-I", "-c", code], stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def identity_alive(record):
    try:
        proc = psutil.Process(record["pid"])
        return proc.create_time() == record["create_time"] and proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def parent_code(child_path, allocation):
    child = ("import json, pathlib, psutil, time; "
             f"data = bytearray({allocation}); "
             f"pathlib.Path({str(child_path)!r}).write_text(json.dumps(dict(pid=psutil.Process().pid, create_time=psutil.Process().create_time()))); "
             "time.sleep(30)")
    return ("import subprocess, sys; "
            f"child = subprocess.Popen([sys.executable, '-I', '-c', {child!r}]); child.wait()")


def cleanup(process):
    try:
        owner = psutil.Process(process.pid)
        for child in reversed(owner.children(recursive=True)):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        if owner.is_running():
            owner.kill()
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=5)


def test_child_memory_counts_and_unrelated_process_survives(tmp_path):
    child_path = tmp_path / "child.json"
    unrelated = spawn("import time; time.sleep(30)")
    process = spawn(parent_code(child_path, 96 * 1024 ** 2))
    try:
        receipt = watchdog().watch(process, maximum_seconds=10, maximum_rss_bytes=80 * 1024 ** 2, poll_seconds=.02)
        assert receipt["resource_failure"] == "job_tree_memory_limit"
        assert receipt["peak_rss_bytes"] > 80 * 1024 ** 2
        assert len(receipt["owned_processes"]) >= 2
        assert unrelated.poll() is None
        assert all(not identity_alive(record) for record in receipt["owned_processes"])
        if child_path.exists():
            assert not identity_alive(json.loads(child_path.read_text()))
    finally:
        cleanup(process)
        cleanup(unrelated)


def test_timeout_kills_owned_child_and_launcher(tmp_path):
    child_path = tmp_path / "child.json"
    process = spawn(parent_code(child_path, 0))
    try:
        receipt = watchdog().watch(process, maximum_seconds=2, maximum_rss_bytes=1024 ** 3, poll_seconds=.02)
        assert child_path.exists(), "Child must have started before the timeout"
        assert receipt["resource_failure"] == "job_timeout"
        assert len(receipt["killed_processes"]) >= 1
        assert not identity_alive(json.loads(child_path.read_text()))
        assert all(not identity_alive(record) for record in receipt["owned_processes"])
    finally:
        cleanup(process)


def test_already_exited_process_is_safe():
    process = spawn("pass")
    process.wait(timeout=5)
    receipt = watchdog().watch(process, maximum_seconds=1, maximum_rss_bytes=1024 ** 3)
    assert receipt["exit_code"] == 0
    assert receipt["resource_failure"] is None
    assert receipt["killed_processes"] == []


def test_missing_process_during_initial_lookup_is_safe(monkeypatch):
    module = watchdog()
    process = spawn("pass")
    process.wait(timeout=5)
    def vanished(pid):
        raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(module.psutil, "Process", vanished)
    receipt = module.watch(process, maximum_seconds=1, maximum_rss_bytes=1024 ** 3)
    assert receipt["exit_code"] == 0 and receipt["owned_processes"] == []


def test_denied_memory_inspection_still_kills_known_owner(monkeypatch):
    module = watchdog()
    process = spawn("import time; time.sleep(30)")
    original = psutil.Process.memory_info
    def denied_memory(proc):
        if proc.pid == process.pid:
            raise psutil.AccessDenied(proc.pid)
        return original(proc)
    monkeypatch.setattr(psutil.Process, "memory_info", denied_memory)
    try:
        receipt = module.watch(process, maximum_seconds=3, maximum_rss_bytes=1024 ** 3, poll_seconds=.02)
        assert receipt["resource_failure"] == "job_monitor_access_denied"
        assert receipt["cleanup_complete"] is True
        assert process.poll() is not None
        assert all(not identity_alive(record) for record in receipt["owned_processes"])
    finally:
        cleanup(process)
