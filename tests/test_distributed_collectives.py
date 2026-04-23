import importlib.util
import json
from pathlib import Path
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
import textwrap
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
import ctboost
import ctboost._core as _core
from ctboost.distributed import (
    DistributedCollectiveServer,
    distributed_tcp_request,
)
from ctboost.distributed.tcp import _read_exact, _read_line

from tests.helpers import find_free_tcp_port as _find_free_tcp_port
from tests.helpers import wait_for_tcp_listener as _wait_for_tcp_listener

def test_distributed_tcp_request_retries_until_coordinator_is_ready():
    port = _find_free_tcp_port()
    root = f"tcp://127.0.0.1:{port}"
    server = DistributedCollectiveServer("127.0.0.1", port)

    def delayed_start() -> None:
        time.sleep(0.2)
        server.start()

    starter = threading.Thread(target=delayed_start, daemon=True)
    starter.start()
    try:
        response = distributed_tcp_request(root, 5.0, "ping", "__health__", 0, 1, b"")
        assert response == b""
    finally:
        starter.join(timeout=5.0)
        server.stop()

def test_distributed_tcp_request_waits_for_slow_coordinator_response():
    port = _find_free_tcp_port()
    root = f"tcp://127.0.0.1:{port}"

    class SlowServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    class SlowHandler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            header = _read_line(self.rfile).split("\t")
            assert len(header) == 5
            _read_exact(self.rfile, int(header[-1]))
            time.sleep(1.5)
            self.wfile.write(b"ok\t4\nslow")

    server = SlowServer(("127.0.0.1", port), SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        started = time.monotonic()
        response = distributed_tcp_request(root, 5.0, "ping", "__slow__", 0, 1, b"")
        assert response == b"slow"
        assert time.monotonic() - started >= 1.5
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

def test_distributed_collective_context_waits_for_all_ranks_before_shutdown(tmp_path: Path):
    port = _find_free_tcp_port()
    worker_script = tmp_path / "distributed_barrier_worker.py"
    worker_script.write_text(
        textwrap.dedent(
            """
            import json
            import sys
            import time

            from ctboost.distributed import distributed_tcp_request
            from ctboost.training import _distributed_collective_context

            rank = int(sys.argv[1])
            port = int(sys.argv[2])
            delay = float(sys.argv[3])
            distributed = {
                "backend": "tcp",
                "root": f"tcp://127.0.0.1:{port}",
                "host": "127.0.0.1",
                "port": port,
                "rank": rank,
                "world_size": 2,
                "run_id": "barrier-case",
                "timeout": 30.0,
            }
            with _distributed_collective_context(distributed):
                if delay > 0.0:
                    time.sleep(delay)
                distributed_tcp_request(
                    distributed["root"],
                    distributed["timeout"],
                    "ping",
                    "__health__",
                    rank,
                    1,
                    b"",
                )
            print(json.dumps({"rank": rank, "ok": True}))
            """
        ),
        encoding="utf-8",
    )

    worker_env = os.environ.copy()
    worker_env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + worker_env.get("PYTHONPATH", "")
    worker_zero = subprocess.Popen(
        [sys.executable, str(worker_script), "0", str(port), "0.0"],
        env=worker_env,
    )
    _wait_for_tcp_listener(port)
    worker_one = subprocess.Popen(
        [sys.executable, str(worker_script), "1", str(port), "0.5"],
        env=worker_env,
    )
    assert worker_one.wait(timeout=60) == 0
    assert worker_zero.wait(timeout=60) == 0
