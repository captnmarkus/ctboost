"""Coordinator server for ctboost.distributed."""

from __future__ import annotations

import socketserver
import threading
from typing import Callable, Dict, List, Optional

from .merge import build_schema_collect_response
from .payloads import (
    gather_payloads,
    sum_gpu_snapshot_payloads,
    sum_node_hist_payloads,
)
from .tcp import _read_exact, _read_line


class _CollectiveState:
    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.payloads: Dict[int, bytes] = {}
        self.response: Optional[bytes] = None
        self.completed_ranks: set[int] = set()
        self.condition = threading.Condition()


class _ThreadingCollectiveTcpServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class DistributedCollectiveServer:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        schema_builder: Optional[Callable[[List[bytes]], bytes]] = None,
    ) -> None:
        self._schema_builder = schema_builder
        self._states: Dict[tuple[str, str], _CollectiveState] = {}
        self._response_sent: Dict[tuple[str, str], set[int]] = {}
        self._states_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._stopped = threading.Event()
        self._server = _ThreadingCollectiveTcpServer((host, port), self._make_handler())
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        finally:
            self._stopped.set()

    def _make_handler(self):
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                try:
                    header = _read_line(self.rfile).split("\t")
                except ConnectionError:
                    return
                if len(header) != 5:
                    raise RuntimeError("invalid distributed coordinator request header")
                op, key, raw_rank, raw_world, raw_payload_size = header
                rank = int(raw_rank)
                world_size = int(raw_world)
                response = owner._dispatch(
                    op,
                    key,
                    rank,
                    world_size,
                    _read_exact(self.rfile, int(raw_payload_size)),
                )
                self.wfile.write(f"ok\t{len(response)}\n".encode("utf-8"))
                if response:
                    self.wfile.write(response)
                self.wfile.flush()
                owner._mark_response_sent(op, key, rank, world_size)

        return Handler

    def _mark_response_sent(self, op: str, key: str, rank: int, world_size: int) -> None:
        if op != "barrier" or not key.endswith("/__shutdown__"):
            return

        state_key = (op, key)
        with self._states_lock:
            sent_ranks = self._response_sent.setdefault(state_key, set())
            sent_ranks.add(rank)
            if len(sent_ranks) < world_size:
                return
            self._response_sent.pop(state_key, None)

        threading.Thread(target=self.stop, daemon=True).start()

    def _dispatch(self, op: str, key: str, rank: int, world_size: int, payload: bytes) -> bytes:
        if op == "ping":
            return b""
        state_key = (op, key)
        with self._states_lock:
            state = self._states.get(state_key)
            if state is None:
                state = _CollectiveState(world_size)
                self._states[state_key] = state
        with state.condition:
            if state.world_size != world_size:
                raise RuntimeError("distributed coordinator world_size mismatch")
            state.payloads[rank] = payload
            if state.response is None and len(state.payloads) == state.world_size:
                ordered_payloads = [state.payloads[index] for index in range(state.world_size)]
                if op == "node_hist_reduce":
                    state.response = sum_node_hist_payloads(ordered_payloads)
                elif op == "gpu_snapshot_reduce":
                    state.response = sum_gpu_snapshot_payloads(ordered_payloads)
                elif op == "broadcast":
                    state.response = state.payloads.get(0, b"")
                elif op == "allgather":
                    state.response = gather_payloads(ordered_payloads)
                elif op == "barrier":
                    state.response = b""
                elif op == "schema_collect":
                    if self._schema_builder is None:
                        raise RuntimeError("schema_collect requested without a schema builder")
                    state.response = self._schema_builder(ordered_payloads)
                else:
                    raise RuntimeError(f"unsupported distributed coordinator op {op!r}")
                state.condition.notify_all()
            while state.response is None:
                state.condition.wait()
            response = state.response
            state.completed_ranks.add(rank)
            if len(state.completed_ranks) == state.world_size:
                with self._states_lock:
                    self._states.pop(state_key, None)
            return response

    def start(self) -> None:
        self._thread.start()

    def wait(self, timeout: Optional[float] = None) -> bool:
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def stop(self) -> None:
        with self._stop_lock:
            if self._stopped.is_set():
                self._server.server_close()
                return
            self._server.shutdown()
            self._server.server_close()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=5.0)
        self._stopped.set()


def run_distributed_collective_server(host: str, port: int) -> None:
    server = DistributedCollectiveServer(host, port, schema_builder=build_schema_collect_response)
    server.start()
    try:
        while not server.wait(timeout=3600.0):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
