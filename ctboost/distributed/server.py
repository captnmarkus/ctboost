"""Coordinator server for ctboost.distributed."""

from __future__ import annotations

import hmac
import re
import socket
import socketserver
import threading
import time
from typing import Callable, Dict, List, Optional

from .merge import build_schema_collect_response
from .payloads import (
    gather_payloads,
    sum_gpu_snapshot_payloads,
    sum_node_hist_payloads,
)
from .tcp import (
    MAX_ERROR_BYTES,
    MAX_HEADER_BYTES,
    MAX_KEY_BYTES,
    MAX_PAYLOAD_BYTES,
    MAX_WORLD_SIZE,
    PROTOCOL_VERSION,
    _ALLOWED_OPS,
    _AUTH_TOKEN_RE,
    _read_exact,
    _read_line,
)


_MAX_ACTIVE_STATES = 10_000
_MAX_ACTIVE_PAYLOAD_BYTES = MAX_PAYLOAD_BYTES
_COMPLETED_STATE_GRACE_SECONDS = 5.0
_SAFE_ERROR_RE = re.compile(r"[^\x20-\x7e]+")


class _CollectiveState:
    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.payloads: Dict[int, bytes] = {}
        self.response: Optional[bytes] = None
        self.error: Optional[Exception] = None
        self.completed_ranks: set[int] = set()
        self.expiry_scheduled = False
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
        auth_token: Optional[str] = None,
        expected_run_id: Optional[str] = None,
        expected_world_size: Optional[int] = None,
        max_payload_bytes: int = MAX_PAYLOAD_BYTES,
        max_active_payload_bytes: int = _MAX_ACTIVE_PAYLOAD_BYTES,
        request_timeout: float = 30.0,
        collective_timeout: float = 600.0,
    ) -> None:
        resolved_host = str(host).strip()
        if not resolved_host or resolved_host in {"0.0.0.0", "::"}:
            raise ValueError("distributed coordinator requires a concrete bind host")
        resolved_token = None if auth_token is None else str(auth_token).strip().lower()
        if resolved_token is None:
            raise ValueError("distributed coordinator requires an authentication token")
        if resolved_token is not None and _AUTH_TOKEN_RE.fullmatch(resolved_token) is None:
            raise ValueError(
                "distributed tcp auth token must contain exactly 64 hexadecimal characters"
            )
        resolved_run_id = None if expected_run_id is None else str(expected_run_id)
        if resolved_run_id is not None and (
            not resolved_run_id
            or "\t" in resolved_run_id
            or "\n" in resolved_run_id
            or len(resolved_run_id.encode("utf-8")) > MAX_KEY_BYTES
        ):
            raise ValueError("distributed run id is empty or exceeds protocol limits")
        resolved_world_size = (
            None if expected_world_size is None else int(expected_world_size)
        )
        if resolved_world_size is not None and not (
            1 <= resolved_world_size <= MAX_WORLD_SIZE
        ):
            raise ValueError(f"distributed world_size must be in [1, {MAX_WORLD_SIZE}]")
        resolved_max_payload = int(max_payload_bytes)
        if resolved_max_payload <= 0 or resolved_max_payload > MAX_PAYLOAD_BYTES:
            raise ValueError("distributed max payload size is outside the allowed range")
        resolved_max_active_payload = int(max_active_payload_bytes)
        if resolved_max_active_payload <= 0:
            raise ValueError("distributed active payload limit must be positive")
        resolved_timeout = float(request_timeout)
        if resolved_timeout <= 0.0:
            raise ValueError("distributed request timeout must be positive")
        resolved_collective_timeout = float(collective_timeout)
        if resolved_collective_timeout <= 0.0:
            raise ValueError("distributed collective timeout must be positive")

        self._schema_builder = schema_builder
        self._auth_token = resolved_token
        self._expected_run_id = resolved_run_id
        self._expected_world_size = resolved_world_size
        self._max_payload_bytes = resolved_max_payload
        self._max_active_payload_bytes = resolved_max_active_payload
        self._active_payload_bytes = 0
        self._request_timeout = resolved_timeout
        self._collective_timeout = resolved_collective_timeout
        self._states: Dict[tuple[str, str], _CollectiveState] = {}
        self._response_sent: Dict[tuple[str, str], set[int]] = {}
        self._states_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._stopped = threading.Event()
        self._server = _ThreadingCollectiveTcpServer(
            (resolved_host, int(port)), self._make_handler()
        )
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        finally:
            self._stopped.set()

    @staticmethod
    def _safe_error_message(error: Exception) -> bytes:
        message = _SAFE_ERROR_RE.sub(" ", str(error)).strip()
        if not message:
            message = "distributed coordinator rejected the request"
        return message.encode("utf-8", errors="replace")[:MAX_ERROR_BYTES]

    @staticmethod
    def _write_response(stream, status: str, payload: bytes) -> None:
        stream.write(
            f"{PROTOCOL_VERSION}\t{status}\t{len(payload)}\n".encode("utf-8")
        )
        if payload:
            stream.write(payload)
        stream.flush()

    def _validate_key(self, key: str) -> None:
        if (
            not key
            or "\t" in key
            or "\n" in key
            or len(key.encode("utf-8")) > MAX_KEY_BYTES
        ):
            raise ValueError("distributed coordinator key is empty or exceeds protocol limits")
        if self._expected_run_id is not None and not (
            key == self._expected_run_id or key.startswith(f"{self._expected_run_id}/")
        ):
            raise ValueError("distributed coordinator key does not match the active run")

    def _parse_request_header(self, header: str) -> tuple[str, str, int, int, int]:
        fields = header.split("\t")
        if len(fields) != 7 or fields[0] != PROTOCOL_VERSION:
            raise PermissionError("distributed coordinator authentication failed")
        # Authentication is deliberately checked before parsing attacker-controlled
        # numeric fields or reading a declared payload body.
        if not hmac.compare_digest(fields[1].lower(), self._auth_token):
            raise PermissionError("distributed coordinator authentication failed")
        _, _, op, key, raw_rank, raw_world, raw_payload_size = fields

        if op not in _ALLOWED_OPS:
            raise ValueError("unsupported distributed coordinator operation")
        self._validate_key(key)
        numeric_fields = (raw_rank, raw_world, raw_payload_size)
        if any(
            not field or any(character < "0" or character > "9" for character in field)
            for field in numeric_fields
        ):
            raise ValueError("distributed coordinator numeric header field is invalid")
        rank = int(raw_rank)
        world_size = int(raw_world)
        payload_size = int(raw_payload_size)
        if world_size <= 0 or world_size > MAX_WORLD_SIZE:
            raise ValueError(f"distributed world_size must be in [1, {MAX_WORLD_SIZE}]")
        if self._expected_world_size is not None and world_size != self._expected_world_size:
            raise ValueError("distributed coordinator world_size does not match the active run")
        if rank < 0 or rank >= world_size:
            raise ValueError("distributed rank must be in [0, world_size)")
        if payload_size < 0 or payload_size > self._max_payload_bytes:
            raise ValueError("distributed request payload exceeds the allowed size")
        if op in {"ping", "barrier"} and payload_size != 0:
            raise ValueError(f"distributed {op} requests must not contain a payload")
        if op == "broadcast" and rank != 0 and payload_size != 0:
            raise ValueError("non-root distributed broadcast requests must not contain a payload")
        return op, key, rank, world_size, payload_size

    def _make_handler(self):
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                try:
                    self.request.settimeout(owner._request_timeout)
                    header = _read_line(self.rfile, max_bytes=MAX_HEADER_BYTES)
                    (
                        op,
                        key,
                        rank,
                        world_size,
                        payload_size,
                    ) = owner._parse_request_header(header)
                    payload = _read_exact(
                        self.rfile,
                        payload_size,
                        max_size=owner._max_payload_bytes,
                    )
                    response = owner._dispatch(op, key, rank, world_size, payload)
                    if len(response) > owner._max_payload_bytes:
                        raise ValueError("distributed response payload exceeds the allowed size")
                    owner._write_response(self.wfile, "ok", response)
                    owner._mark_response_sent(op, key, rank, world_size)
                except (ConnectionError, BrokenPipeError, socket.timeout):
                    return
                except Exception as exc:
                    try:
                        owner._write_response(
                            self.wfile,
                            "error",
                            owner._safe_error_message(exc),
                        )
                    except (ConnectionError, BrokenPipeError, OSError):
                        return

        return Handler

    def _mark_response_sent(self, op: str, key: str, rank: int, world_size: int) -> None:
        state_key = (op, key)
        with self._states_lock:
            state = self._states.get(state_key)
        if state is None:
            return
        with state.condition:
            state.completed_ranks.add(rank)
            complete = len(state.completed_ranks) == state.world_size
        if not complete:
            return

        if op == "barrier" and key.endswith("/__shutdown__"):
            with self._states_lock:
                sent_ranks = self._response_sent.setdefault(state_key, set())
                sent_ranks.update(state.completed_ranks)
                if len(sent_ranks) < world_size:
                    return
                self._response_sent.pop(state_key, None)
            threading.Thread(target=self.stop, daemon=True).start()
            return

        self._schedule_state_expiry(state_key, state)

    def _expire_state(
        self,
        state_key: tuple[str, str],
        expected_state: _CollectiveState,
    ) -> None:
        # Serialize removal with dispatch accounting. A handler may already
        # hold a state pointer when its grace-period timer fires.
        with expected_state.condition:
            with self._states_lock:
                state = self._states.get(state_key)
                if state is not expected_state:
                    return
                released = sum(len(payload) for payload in state.payloads.values())
                if state.response is not None:
                    released += len(state.response)
                self._active_payload_bytes = max(0, self._active_payload_bytes - released)
                self._states.pop(state_key, None)

    def _schedule_state_expiry(
        self,
        state_key: tuple[str, str],
        state: _CollectiveState,
    ) -> None:
        with state.condition:
            if state.expiry_scheduled:
                return
            state.expiry_scheduled = True
        expiry = threading.Timer(
            _COMPLETED_STATE_GRACE_SECONDS,
            self._expire_state,
            args=(state_key, state),
        )
        expiry.daemon = True
        expiry.start()

    def _preflight_response_size(self, op: str, payloads: List[bytes]) -> None:
        response_size: Optional[int] = None
        if op == "allgather":
            response_size = 8 + 8 * len(payloads) + sum(map(len, payloads))
        elif op == "broadcast":
            response_size = len(payloads[0])
        elif op == "barrier":
            response_size = 0
        elif op in {"node_hist_reduce", "gpu_snapshot_reduce"}:
            # Both reducers preserve the encoded shape of each valid input.
            response_size = len(payloads[0])
        if response_size is not None and response_size > self._max_payload_bytes:
            raise RuntimeError("distributed response payload exceeds the allowed size")
        if response_size is not None:
            with self._states_lock:
                if (
                    self._active_payload_bytes + response_size
                    > self._max_active_payload_bytes
                ):
                    raise RuntimeError("distributed coordinator active payload limit exceeded")

    def _dispatch(self, op: str, key: str, rank: int, world_size: int, payload: bytes) -> bytes:
        if op == "ping":
            return b""
        state_key = (op, key)
        created_state = False
        with self._states_lock:
            state = self._states.get(state_key)
            if state is None:
                if len(self._states) >= _MAX_ACTIVE_STATES:
                    raise RuntimeError("distributed coordinator has too many active operations")
                if self._active_payload_bytes + len(payload) > self._max_active_payload_bytes:
                    raise RuntimeError("distributed coordinator active payload limit exceeded")
                state = _CollectiveState(world_size)
                self._states[state_key] = state
                created_state = True
        with state.condition:
            if state.error is not None:
                raise type(state.error)(str(state.error)) from None
            if state.world_size != world_size:
                state.error = RuntimeError("distributed coordinator world_size mismatch")
                self._schedule_state_expiry(state_key, state)
                state.condition.notify_all()
                raise RuntimeError("distributed coordinator world_size mismatch")
            if rank in state.payloads:
                if state.payloads[rank] != payload:
                    state.error = RuntimeError(
                        "distributed coordinator received a conflicting duplicate rank"
                    )
                    self._schedule_state_expiry(state_key, state)
                    state.condition.notify_all()
                    raise RuntimeError("distributed coordinator received a conflicting duplicate rank")
            else:
                active_limit_exceeded = False
                with self._states_lock:
                    if self._active_payload_bytes + len(payload) > self._max_active_payload_bytes:
                        if created_state and not state.payloads:
                            self._states.pop(state_key, None)
                        active_limit_exceeded = True
                    else:
                        self._active_payload_bytes += len(payload)
                if active_limit_exceeded:
                    if not created_state:
                        state.error = RuntimeError(
                            "distributed coordinator active payload limit exceeded"
                        )
                        self._schedule_state_expiry(state_key, state)
                        state.condition.notify_all()
                    raise RuntimeError("distributed coordinator active payload limit exceeded")
                state.payloads[rank] = payload
            if (
                state.response is None
                and state.error is None
                and len(state.payloads) == state.world_size
            ):
                ordered_payloads = [state.payloads[index] for index in range(state.world_size)]
                try:
                    self._preflight_response_size(op, ordered_payloads)
                    if op == "node_hist_reduce":
                        response = sum_node_hist_payloads(ordered_payloads)
                    elif op == "gpu_snapshot_reduce":
                        response = sum_gpu_snapshot_payloads(ordered_payloads)
                    elif op == "broadcast":
                        response = state.payloads.get(0, b"")
                    elif op == "allgather":
                        response = gather_payloads(ordered_payloads)
                    elif op == "barrier":
                        response = b""
                    elif op == "schema_collect":
                        if self._schema_builder is None:
                            raise RuntimeError("schema_collect requested without a schema builder")
                        response = self._schema_builder(ordered_payloads)
                    else:  # The protocol validator rejects this before state allocation.
                        raise RuntimeError("unsupported distributed coordinator operation")
                    if len(response) > self._max_payload_bytes:
                        raise RuntimeError("distributed response payload exceeds the allowed size")
                    with self._states_lock:
                        if (
                            self._active_payload_bytes + len(response)
                            > self._max_active_payload_bytes
                        ):
                            raise RuntimeError(
                                "distributed coordinator active payload limit exceeded"
                            )
                        self._active_payload_bytes += len(response)
                    state.response = response
                except Exception as exc:
                    state.error = RuntimeError(
                        self._safe_error_message(exc).decode("utf-8", errors="replace")
                    )
                    self._schedule_state_expiry(state_key, state)
                finally:
                    state.condition.notify_all()
            deadline = time.monotonic() + self._collective_timeout
            while state.response is None and state.error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    state.error = TimeoutError("distributed collective operation timed out")
                    self._schedule_state_expiry(state_key, state)
                    state.condition.notify_all()
                    break
                state.condition.wait(timeout=remaining)
            if state.error is not None:
                raise type(state.error)(str(state.error)) from None
            response = state.response
            if response is None:  # Narrow Optional[bytes] for static type checkers.
                raise RuntimeError("distributed collective operation failed")
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


def run_distributed_collective_server(
    host: str,
    port: int,
    *,
    auth_token: str,
    expected_run_id: str,
    expected_world_size: int,
    max_payload_bytes: int = MAX_PAYLOAD_BYTES,
    max_active_payload_bytes: int = _MAX_ACTIVE_PAYLOAD_BYTES,
    request_timeout: float = 30.0,
    collective_timeout: float = 600.0,
) -> None:
    server = DistributedCollectiveServer(
        host,
        port,
        schema_builder=build_schema_collect_response,
        auth_token=auth_token,
        expected_run_id=expected_run_id,
        expected_world_size=expected_world_size,
        max_payload_bytes=max_payload_bytes,
        max_active_payload_bytes=max_active_payload_bytes,
        request_timeout=request_timeout,
        collective_timeout=collective_timeout,
    )
    server.start()
    try:
        while not server.wait(timeout=3600.0):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
