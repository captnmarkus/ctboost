"""Resource guard for a Popen child, including Windows venv interpreter children."""

from __future__ import annotations

import time

import psutil


def watch(process, *, maximum_seconds, maximum_rss_bytes, poll_seconds=.1):
    """Wait for an owned process tree; kill only identities observed beneath it.

    RSS is the sum across live owned processes (shared pages may be counted more
    than once). A PID must still have its recorded creation time before every
    inspection or kill. A vanished launcher is safe, including an immediate exit.
    """
    started = time.monotonic()
    owned = {}
    peak = 0
    failure = None
    killed = []

    def remember(proc):
        try:
            identity = (proc.pid, proc.create_time())
            owned.setdefault(identity, proc)
        except psutil.NoSuchProcess:
            pass

    try:
        remember(psutil.Process(process.pid))
    except psutil.NoSuchProcess:
        pass

    def live(proc, identity, strict=True):
        try:
            if proc.create_time() != identity[1] or not proc.is_running():
                return False
            try:
                return proc.status() != psutil.STATUS_ZOMBIE
            except psutil.AccessDenied:
                if strict:
                    raise
                return True
        except psutil.NoSuchProcess:
            return False

    def sample(strict=True):
        # Discover from every previously observed live ancestor, so an observed
        # interpreter remains owned if its launcher has already exited.
        for identity, proc in list(owned.items()):
            if live(proc, identity, strict):
                try:
                    for child in proc.children(recursive=True):
                        remember(child)
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied:
                    if strict:
                        raise
        alive, rss = [], 0
        for identity, proc in list(owned.items()):
            if live(proc, identity, strict):
                try:
                    rss += proc.memory_info().rss
                    alive.append((identity, proc))
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied:
                    if strict:
                        raise
                    alive.append((identity, proc))
        return alive, rss

    try:
        while True:
            alive, rss = sample()
            peak = max(peak, rss)
            exited = process.poll() is not None
            if exited and not alive:
                break
            if exited and alive:
                # MSVC leaves PDB/telemetry servers alive after a completed
                # build. Close only already observed owned identities, once
                # every remaining owned process is one of these two daemons.
                try:
                    compiler_daemons = all(proc.name().lower() in {"mspdbsrv.exe", "vctip.exe"}
                                           for _, proc in alive)
                except psutil.NoSuchProcess:
                    compiler_daemons = False
                if compiler_daemons:
                    for identity, proc in reversed(alive):
                        try:
                            if live(proc, identity):
                                proc.kill()
                                if identity not in killed:
                                    killed.append(identity)
                        except psutil.NoSuchProcess:
                            pass
                    continue
            if rss > maximum_rss_bytes:
                failure = "job_tree_memory_limit"
                break
            if time.monotonic() - started > maximum_seconds:
                failure = "job_timeout"
                break
            time.sleep(poll_seconds)
    except psutil.AccessDenied:
        failure = "job_monitor_access_denied"
    finally:
        if failure is not None:
            # Repeat discovery and child-first termination to close ordinary
            # child-spawn/exit races. Never target a reused PID or another tree.
            deadline = time.monotonic() + 10
            while True:
                alive, _ = sample(strict=False)
                for identity, proc in reversed(alive):
                    try:
                        if live(proc, identity, strict=False):
                            proc.kill()
                            if identity not in killed:
                                killed.append(identity)
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.AccessDenied:
                        # Try every other remembered identity before giving up;
                        # the caller will stop dispatch if cleanup cannot finish.
                        pass
                process.poll()
                alive, _ = sample(strict=False)
                if not alive and process.poll() is not None:
                    break
                if time.monotonic() >= deadline:
                    return {"exit_code": process.poll(), "resource_failure": failure,
                            "peak_rss_bytes": peak, "wall_seconds": time.monotonic() - started,
                            "cleanup_complete": False,
                            "remaining_processes": [{"pid": i[0], "create_time": i[1]} for i, _ in alive],
                            "owned_processes": [{"pid": pid, "create_time": created} for pid, created in owned],
                            "killed_processes": [{"pid": pid, "create_time": created} for pid, created in killed]}
                time.sleep(poll_seconds)
    return {"exit_code": process.wait(timeout=5), "resource_failure": failure, "cleanup_complete": True,
            "peak_rss_bytes": peak, "rss_scope": "sum of observed live owned process tree",
            "wall_seconds": time.monotonic() - started,
            "owned_processes": [{"pid": pid, "create_time": created} for pid, created in owned],
            "killed_processes": [{"pid": pid, "create_time": created} for pid, created in killed]}
