"""Run the gated local HPO plan with owned processes and resumable bookkeeping.

Create ``plan.json`` with local_hpo.py first, then run this file with the same
pinned environment's ``python -I``. Creating ``OUTPUT/PAUSE`` drains active
workers without starting more. Removing it allows dispatch to continue. Started
parents are never automatically retried, including after a controller restart.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import runpy
import statistics
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3
ACTIVE = {"launching", "running"}
TERMINAL = {
    "complete",
    "failed",
    "interrupted",
    "timed_out",
    "resource_limit",
    "resource_pressure",
    "invalid_result",
    "launch_failed",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    deadline = time.monotonic() + 5
    try:
        while True:
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
    finally:
        temporary.unlink(missing_ok=True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@contextmanager
def controller_lock(output):
    """An OS lock is released even when its owning controller crashes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def parent_key(parent):
    return f"{parent['config_name']}/{parent['task_id']}/0_0"


def parent_paths(output, parent):
    key = parent_key(parent)
    return (
        Path(output) / "artifacts" / key / "manifest.json",
        Path(output) / "data" / key / "results.pkl",
        Path(output) / "controller" / key,
    )


def process_identity(process):
    """A PID alone never proves ownership: its creation time and command must match."""
    return {
        "pid": process.pid,
        "create_time": process.create_time(),
        "command": process.cmdline(),
    }


def owned_process(identity):
    if not isinstance(identity, dict) or not identity.get("command"):
        return None
    try:
        process = psutil.Process(identity["pid"])
        if (
            abs(process.create_time() - identity["create_time"]) > 1e-6
            or process.cmdline() != identity["command"]
            or not process.is_running()
            or process.status() == psutil.STATUS_ZOMBIE
        ):
            return None
        return process
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None


def same_command(actual, expected):
    return (
        bool(actual)
        and bool(expected)
        and os.path.normcase(os.path.realpath(actual[0]))
        == os.path.normcase(os.path.realpath(expected[0]))
        and actual[1:] == expected[1:]
    )


def claim_matches_command(identity, record):
    command = identity.get("command", [])
    expected = record.get("command", [])
    executables = record.get("allowed_executables", expected[:1])
    return any(
        same_command(command, [executable, *expected[1:]]) for executable in executables
    )


def sample_owned_tree(identity):
    process = owned_process(identity)
    if process is None:
        return 0, []
    children = []
    rss = 0
    for child in [process, *process.children(recursive=True)]:
        try:
            child_identity = process_identity(child)
            if owned_process(child_identity) is not None:
                rss += child.memory_info().rss
                if child.pid != process.pid:
                    children.append(child_identity)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return rss, children


def terminate_owned_tree(identity, descendants=()):
    """Stop only verified owned processes, rechecking identities before each signal."""
    process = owned_process(identity)
    targets = list(descendants)
    if process is not None:
        # Prevent the worker from launching descendants while its tree is collected.
        try:
            process.suspend()
            targets.extend(
                process_identity(p) for p in process.children(recursive=True)
            )
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    if identity is not None:
        targets.append(identity)
    seen = set()
    signalled = []
    for target in reversed(targets):
        key = (target["pid"], target["create_time"])
        if key in seen:
            continue
        seen.add(key)
        current = owned_process(target)
        if current is not None:
            try:
                current.kill()
                signalled.append(current)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
    _, alive = psutil.wait_procs(signalled, timeout=5)
    if alive:
        raise RuntimeError("An owned worker could not be stopped; dispatch is refused")
    return [process.pid for process in signalled]


def allocation_policy(allowed_cpus, max_workers):
    if isinstance(max_workers, bool) or not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be between 1 and 8")
    cpus = sorted(set(allowed_cpus))
    count = min(max_workers, len(cpus) // 2)
    if not count:
        raise ValueError("Two available logical CPUs are required per parent")
    # Pair corresponding halves of the available CPU list, without assuming a
    # vendor-specific SMT numbering scheme. Every slot is disjoint.
    selected = cpus[: count * 2]
    return {
        "policy_version": 1,
        "num_cpus_per_parent": 2,
        "memory_limit_bytes_per_parent": 8 * GIB,
        "max_workers": count,
        "cpu_slots": [[selected[i], selected[i + count]] for i in range(count)],
        "affinity_policy": "pair sorted selected logical CPUs across equal halves",
        "reservation_floor_bytes": GIB,
        "unseen_dataset_reservation_bytes": 8 * GIB,
        "observed_peak_multiplier": 1.5,
        "physical_free_reserve_bytes": 4 * GIB,
        "timeout_seconds": 3600 + 15 * 60,
        "pressure_action": "stop newest owned parent, terminal resource_pressure",
        "retry_policy": "never retry any started parent automatically",
        "dispatch_order": "frozen plan order among memory-admissible parents",
        "pause_file": "PAUSE",
    }


def reservation_bytes(dataset, observed_peaks, policy):
    peak = observed_peaks.get(dataset)
    if peak is None or peak <= 0:
        return policy["unseen_dataset_reservation_bytes"]
    return max(
        policy["reservation_floor_bytes"],
        math.ceil(peak * policy["observed_peak_multiplier"]),
    )


def can_dispatch(available, total, candidate_reservation, active, policy):
    """Available RAM already excludes RSS, so reserve only each worker's growth."""
    reserve = policy["physical_free_reserve_bytes"]
    commitments = [
        max(job["reservation_bytes"], job.get("rss_bytes", 0)) for job in active
    ]
    growth = sum(
        max(0, committed - job.get("rss_bytes", 0))
        for committed, job in zip(commitments, active)
    )
    return (
        available >= reserve + growth + candidate_reservation
        and total >= reserve + sum(commitments) + candidate_reservation
    )


def _set_execution_state(flags):
    import ctypes

    if not ctypes.windll.kernel32.SetThreadExecutionState(flags):
        raise OSError("Windows refused the process-lifetime sleep inhibition request")


class SleepInhibitor:
    """Prevent system sleep only while workers are active, without changing power plans."""

    def __init__(self, enabled=None):
        self.enabled = os.name == "nt" if enabled is None else enabled
        self.active = False

    def set_active(self, active):
        active = bool(active)
        if self.enabled and active != self.active:
            _set_execution_state(0x80000001 if active else 0x80000000)
            self.active = active

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.set_active(False)


class Controller:
    def __init__(
        self,
        *,
        output,
        plan,
        worker,
        decision,
        protocol,
        python=None,
        max_workers=8,
        poll_seconds=5,
    ):
        self.output = Path(output).resolve()
        self.plan = plan
        self.worker = worker
        self.decision = Path(decision).resolve()
        self.protocol = Path(protocol).resolve()
        self.python = str(Path(python or sys.executable).resolve())
        self.worker_script = ROOT / "benchmarks/tabarena/local_hpo.py"
        self.plan_hash = file_hash(self.output / "plan.json")
        self.policy = allocation_policy(psutil.Process().cpu_affinity(), max_workers)
        if not math.isfinite(poll_seconds) or not 0.1 <= poll_seconds <= 30:
            raise ValueError("poll_seconds must be between 0.1 and 30")
        self.poll_seconds = poll_seconds
        self.processes = {}
        self.state_path = self.output / "controller_state.json"
        self.parents = {parent_key(parent): parent for parent in plan["parents"]}
        if len(self.parents) != len(plan["parents"]):
            raise ValueError("Duplicate parent identities in the frozen plan")
        self._freeze_execution()
        if self.state_path.exists():
            self.state = read_json(self.state_path)
            if (
                self.state.get("plan_sha256") != self.plan_hash
                or set(self.state.get("parents", {})) != set(self.parents)
                or any(
                    item.get("status") not in ACTIVE | TERMINAL | {"queued"}
                    for item in self.state["parents"].values()
                )
            ):
                raise ValueError("Controller state does not match the frozen plan")
        else:
            self.state = {
                "schema_version": 1,
                "plan_sha256": self.plan_hash,
                "parents": {key: {"status": "queued"} for key in self.parents},
                "observed_dataset_peak_bytes": {},
            }
        self.save()

    def _freeze_execution(self):
        path = self.output / "controller_execution.json"
        binding = {
            "schema_version": 1,
            "plan_sha256": self.plan_hash,
            "controller_source_sha256": file_hash(__file__),
            "worker_source_sha256": file_hash(self.worker_script),
            "python": self.python,
            "python_sha256": file_hash(self.python),
            "base_python": str(Path(sys._base_executable).resolve()),
            "base_python_sha256": file_hash(sys._base_executable),
            "psutil_version": psutil.__version__,
            "decision_path": str(self.decision),
            "decision_sha256": file_hash(self.decision),
            "protocol_path": str(self.protocol),
            "protocol_sha256": file_hash(self.protocol),
            "allocation_policy": self.policy,
            "allocation_policy_sha256": json_hash(self.policy),
            "poll_seconds": self.poll_seconds,
        }
        if path.exists():
            existing = read_json(path)
            if {k: v for k, v in existing.items() if k != "created_at"} != binding:
                raise ValueError("Frozen controller execution policy or source changed")
        else:
            write_json(path, {**binding, "created_at": now()})

    def save(self):
        self.state["updated_at"] = now()
        write_json(self.state_path, self.state)

    def _observe(self, parent, record, manifest=None):
        if record.get("status") not in TERMINAL:
            return  # Import-time RSS is not an observed completed-fit peak.
        peak = max(
            record.get("peak_rss_bytes", 0),
            (manifest or {}).get("peak_rss_bytes", 0),
        )
        if peak > 0:
            history = self.state["observed_dataset_peak_bytes"]
            history[parent["dataset"]] = max(history.get(parent["dataset"], 0), peak)

    def _validate_completed(self, parent, manifest, raw):
        if (
            manifest.get("parent") != parent
            or manifest.get("plan_sha256") != self.plan_hash
            or manifest.get("validation", {}).get("raw_sha256")
            not in (None, file_hash(raw))
        ):
            raise ValueError("Worker checkpoint identity or completed raw hash changed")
        return self.worker.validate_result(raw, parent, self.plan_hash)

    def _finish_from_artifacts(self, key, *, fallback, exit_code=None):
        parent, record = self.parents[key], self.state["parents"][key]
        manifest_path, raw, _ = parent_paths(self.output, parent)
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        resource_failure = manifest_path.with_name("resource_failure.json")
        outcome = fallback
        validation = None
        error = None
        if manifest and (
            manifest.get("parent") != parent
            or manifest.get("plan_sha256") != self.plan_hash
        ):
            raise ValueError(f"Foreign worker artifact for {key}; leaving it untouched")
        if resource_failure.exists():
            outcome = "resource_limit"
        elif manifest.get("status") == "failed":
            outcome = "failed"
        elif raw.is_file() and manifest:
            try:
                validation = self._validate_completed(parent, manifest, raw)
                outcome = "complete"
            except Exception as exc:  # noqa: BLE001 - invalid cache evidence is terminal and preserved.
                outcome, error = "invalid_result", f"{type(exc).__name__}: {exc}"
        elif raw.exists() or manifest.get("status") == "complete":
            outcome, error = (
                "invalid_result",
                "Raw result or matching worker provenance is missing",
            )
        record.update(
            status=outcome, finished_at=now(), exit_code=exit_code, rss_bytes=0
        )
        if validation is not None:
            record["validation"] = validation
        if error is not None:
            record["error"] = error
        if manifest.get("elapsed_seconds") is not None:
            record["elapsed_seconds"] = manifest["elapsed_seconds"]
        elif "launched_epoch" in record:
            record["elapsed_seconds"] = max(0, time.time() - record["launched_epoch"])
        self._observe(parent, record, manifest)

    def _claimed_identity(self, record):
        identity = record.get("identity")
        if identity is not None:
            return identity
        claim_path = record.get("claim_path")
        if not claim_path or not Path(claim_path).is_file():
            return None
        claim = read_json(claim_path)
        identity = claim.get("identity", {})
        if (
            claim.get("token") != record.get("launch_token")
            or not claim_matches_command(identity, record)
            or identity.get("create_time", 0) < record["launched_epoch"] - 2
        ):
            raise ValueError("Worker launch claim does not match its owned command")
        return identity

    def _stop(self, record):
        descendants = list(record.get("descendants", []))
        if record.get("launcher_identity") is not None:
            descendants.append(record["launcher_identity"])
        return terminate_owned_tree(self._claimed_identity(record), descendants)

    def reconcile(self):
        """Validate completions; started parents become terminal, never queued again."""
        for key, parent in self.parents.items():
            record = self.state["parents"][key]
            manifest_path, raw, directory = parent_paths(self.output, parent)
            if record["status"] in ACTIVE:
                self._stop(record)
                self._finish_from_artifacts(key, fallback="interrupted")
            elif record["status"] == "complete":
                if not manifest_path.is_file() or not raw.is_file():
                    raise ValueError(f"Completed parent lost required artifacts: {key}")
                validated = self._validate_completed(
                    parent, read_json(manifest_path), raw
                )
                if (
                    record.get("validation", {}).get("raw_sha256")
                    != validated["raw_sha256"]
                ):
                    raise ValueError(f"Completed controller result changed: {key}")
            elif record["status"] == "queued" and (
                manifest_path.exists() or raw.parent.exists() or directory.exists()
            ):
                # A missing controller state must not turn an existing started fit
                # into a fresh fit. Unknown live workers are never signalled.
                if manifest_path.exists():
                    manifest = read_json(manifest_path)
                    pid = manifest.get("pid")
                    if (
                        pid
                        and psutil.pid_exists(pid)
                        and manifest.get("status") in {"preparing", "running"}
                    ):
                        raise RuntimeError(
                            "An unowned started worker exists; dispatch is refused"
                        )
                self._finish_from_artifacts(key, fallback="interrupted")
            self._observe(parent, record)
        self.save()

    def _worker_command(self, parent, slot, claim, permit, token):
        worker_args = [
            str(self.worker_script.resolve()),
            "worker",
            "--decision",
            str(self.decision),
            "--protocol",
            str(self.protocol),
            "--output",
            str(self.output),
            "--dataset",
            parent["dataset"],
            "--config-index",
            str(parent["config_index"]),
            "--num-cpus",
            "2",
            "--memory-limit-gb",
            "8",
            "--affinity",
            ",".join(str(cpu) for cpu in self.policy["cpu_slots"][slot]),
        ]
        return [
            self.python,
            "-I",
            str(Path(__file__).resolve()),
            "_worker",
            "--claim",
            str(claim),
            "--permit",
            str(permit),
            "--token",
            token,
            "--",
            *worker_args,
        ]

    def launch(self, key, slot, reservation):
        parent, record = self.parents[key], self.state["parents"][key]
        if record["status"] != "queued":
            raise ValueError("Only an unstarted queued parent may be launched")
        manifest, raw, directory = parent_paths(self.output, parent)
        if manifest.exists() or raw.parent.exists() or directory.exists():
            raise ValueError("Existing parent artifacts forbid a new launch")
        directory.mkdir(parents=True)
        claim, permit = directory / "claim.json", directory / "permit.json"
        token = uuid.uuid4().hex
        command = self._worker_command(parent, slot, claim, permit, token)
        record.update(
            status="launching",
            launched_at=now(),
            launched_epoch=time.time(),
            launched_monotonic=time.monotonic(),
            launch_token=token,
            claim_path=str(claim),
            command=command,
            allowed_executables=[
                self.python,
                str(Path(sys._base_executable).resolve()),
            ],
            slot=slot,
            affinity=self.policy["cpu_slots"][slot],
            reservation_bytes=reservation,
            rss_bytes=0,
            peak_rss_bytes=0,
        )
        self.save()  # Persist intent before creating a process.
        process = None
        try:
            options = (
                {"creationflags": subprocess.CREATE_NO_WINDOW}
                if os.name == "nt"
                else {}
            )
            with (directory / "worker.log").open("xb") as log:
                process = subprocess.Popen(
                    command,
                    cwd=str(ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    **options,
                )
            launcher_identity = process_identity(psutil.Process(process.pid))
            if not same_command(launcher_identity["command"], command):
                raise RuntimeError(
                    "Launched worker command differs from the frozen command"
                )
            record["launcher_identity"] = launcher_identity
            self.save()
            deadline = time.monotonic() + 20
            while not claim.is_file():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(
                        "The owned worker did not provide its launch claim"
                    )
                time.sleep(0.05)
            identity = self._claimed_identity(record)
            claimed = owned_process(identity)
            if claimed is None:
                raise RuntimeError("The claiming worker is no longer alive")
            if not any(
                p.pid == launcher_identity["pid"]
                and process_identity(p) == launcher_identity
                for p in [claimed, *claimed.parents()]
            ):
                raise RuntimeError(
                    "The worker claim is not a child of our owned launcher"
                )
            record.update(identity=identity, status="running")
            self.processes[key] = process
            self.save()  # The worker cannot fit before this durable ownership record.
            write_json(permit, {"token": token, "identity": identity})
        except BaseException:
            if process is not None:
                # Invalid claims must not prevent cleanup of a separately
                # verified launcher and its descendants.
                terminate_owned_tree(record.get("launcher_identity"))
            record.update(status="launch_failed", finished_at=now())
            self.save()
            raise

    def poll(self):
        for key, record in self.state["parents"].items():
            if record["status"] != "running":
                continue
            process = self.processes.get(key)
            exit_code = process.poll() if process is not None else None
            identity = record["identity"]
            if exit_code is not None or owned_process(identity) is None:
                self._stop(record)
                self._finish_from_artifacts(key, fallback="failed", exit_code=exit_code)
                continue
            rss, descendants = sample_owned_tree(identity)
            record.update(
                rss_bytes=rss,
                descendants=descendants,
                peak_rss_bytes=max(record.get("peak_rss_bytes", 0), rss),
            )
            self._observe(self.parents[key], record)
            if (
                time.monotonic() - record["launched_monotonic"]
                > self.policy["timeout_seconds"]
            ):
                self._stop(record)
                self._finish_from_artifacts(key, fallback="timed_out")
        active = [r for r in self.state["parents"].values() if r["status"] in ACTIVE]
        memory = psutil.virtual_memory()
        if active and memory.available < self.policy["physical_free_reserve_bytes"]:
            newest_key = max(
                (
                    key
                    for key, r in self.state["parents"].items()
                    if r["status"] in ACTIVE
                ),
                key=lambda key: self.state["parents"][key]["launched_epoch"],
            )
            record = self.state["parents"][newest_key]
            record["pressure_available_bytes"] = memory.available
            self._stop(record)
            self._finish_from_artifacts(newest_key, fallback="resource_pressure")
        self.save()

    def dispatch(self):
        if (self.output / self.policy["pause_file"]).exists():
            return
        for key, parent in self.parents.items():
            record = self.state["parents"][key]
            if record["status"] != "queued":
                continue
            active = [
                r for r in self.state["parents"].values() if r["status"] in ACTIVE
            ]
            free_slots = set(range(self.policy["max_workers"])) - {
                r["slot"] for r in active
            }
            if not free_slots:
                break
            history = self.state["observed_dataset_peak_bytes"]
            # Grow reservations when a currently running sibling establishes a
            # larger footprint; never lower a running worker's reservation.
            for active_key, active_record in self.state["parents"].items():
                if active_record["status"] in ACTIVE:
                    active_record["reservation_bytes"] = max(
                        active_record["reservation_bytes"],
                        reservation_bytes(
                            self.parents[active_key]["dataset"], history, self.policy
                        ),
                        math.ceil(
                            active_record.get("peak_rss_bytes", 0)
                            * self.policy["observed_peak_multiplier"]
                        ),
                    )
            reservation = reservation_bytes(parent["dataset"], history, self.policy)
            memory = psutil.virtual_memory()
            if can_dispatch(
                memory.available, memory.total, reservation, active, self.policy
            ):
                self.launch(key, min(free_slots), reservation)

    def progress(self):
        counts = {status: 0 for status in sorted(ACTIVE | TERMINAL | {"queued"})}
        active = []
        durations = []
        for key, record in self.state["parents"].items():
            counts[record["status"]] += 1
            if record["status"] in ACTIVE:
                active.append(
                    {
                        "parent": key,
                        **{
                            k: record.get(k)
                            for k in (
                                "identity",
                                "affinity",
                                "rss_bytes",
                                "peak_rss_bytes",
                                "reservation_bytes",
                                "launched_at",
                            )
                        },
                    }
                )
            if record["status"] == "complete" and record.get("elapsed_seconds", 0) > 0:
                durations.append(record["elapsed_seconds"])
        remaining = counts["queued"] + len(active)
        eta = None
        if durations and remaining:
            eta = statistics.median(durations) * remaining / max(1, len(active))
        memory = psutil.virtual_memory()
        progress = {
            "updated_at": now(),
            "plan_sha256": self.plan_hash,
            "counts": counts,
            "total": len(self.parents),
            "active": active,
            "terminal_failures": sum(counts[s] for s in TERMINAL - {"complete"}),
            "physical_available_bytes": memory.available,
            "active_rss_bytes": sum(row["rss_bytes"] or 0 for row in active),
            "eta_seconds": eta,
            "eta_basis": "median completed parent wall time / currently active workers; approximate",
            "paused": (self.output / self.policy["pause_file"]).exists(),
        }
        write_json(self.output / "progress.json", progress)
        return progress

    def run(self):
        self.reconcile()
        with SleepInhibitor() as sleep_inhibitor:
            try:
                while True:
                    self.poll()
                    self.dispatch()
                    progress = self.progress()
                    sleep_inhibitor.set_active(bool(progress["active"]))
                    print(
                        json.dumps(
                            {
                                k: progress[k]
                                for k in (
                                    "updated_at",
                                    "counts",
                                    "physical_available_bytes",
                                    "active_rss_bytes",
                                    "eta_seconds",
                                    "paused",
                                )
                            }
                        ),
                        flush=True,
                    )
                    if not progress["active"] and not progress["counts"]["queued"]:
                        return 0 if progress["terminal_failures"] == 0 else 1
                    time.sleep(self.poll_seconds)
            finally:
                # Controlled shutdown preserves every started parent as terminal.
                for key, record in self.state["parents"].items():
                    if record["status"] in ACTIVE:
                        self._stop(record)
                        self._finish_from_artifacts(key, fallback="interrupted")
                self.save()
                self.progress()


def _worker_main(argv):
    """Internal launch handshake; run the real worker in this same owned PID."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", type=Path, required=True)
    parser.add_argument("--permit", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    worker_args = args.worker_args
    if worker_args and worker_args[0] == "--":
        worker_args = worker_args[1:]
    if not worker_args:
        raise ValueError("Missing worker script")
    identity = process_identity(psutil.Process())
    write_json(args.claim, {"token": args.token, "identity": identity})
    deadline = time.monotonic() + 30
    while not args.permit.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Controller never granted the durable launch permit")
        time.sleep(0.05)
    if read_json(args.permit) != {"token": args.token, "identity": identity}:
        raise ValueError("Launch permit does not match this worker process")
    sys.argv = worker_args
    runpy.run_path(worker_args[0], run_name="__main__")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_worker":
        return _worker_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not (output / "plan.json").is_file():
        raise ValueError("Create the immutable HPO plan before starting its controller")
    spec = importlib.util.spec_from_file_location(
        "_ctboost_local_hpo", ROOT / "benchmarks/tabarena/local_hpo.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    worker._configure_process(2, None)
    worker.bootstrap_imports()
    with controller_lock(output):
        # Recomputes the complete pilot gate and plan once at controller startup.
        plan = worker.build_plan(
            decision_path=args.decision,
            protocol_path=args.protocol,
            output=output,
            num_cpus=2,
            memory_limit_gb=8,
        )
        if plan["resources"] != worker.resource_contract(2, 8):
            raise ValueError("This controller requires fixed 2 CPU / 8 GiB parents")
        return Controller(
            output=output,
            plan=plan,
            worker=worker,
            decision=args.decision,
            protocol=args.protocol,
            max_workers=args.max_workers,
            poll_seconds=args.poll_seconds,
        ).run()


if __name__ == "__main__":
    raise SystemExit(main())
