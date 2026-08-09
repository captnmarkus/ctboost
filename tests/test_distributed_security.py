import pickle
import socket
import socketserver
import threading
import time

import numpy as np
import pytest
from sklearn.base import clone

import ctboost
import ctboost._integration_utils as integration_utils
from ctboost import _core
from ctboost.distributed import (
    DistributedCollectiveServer,
    distributed_tcp_request,
    parse_distributed_root,
    redact_distributed_root,
)
from ctboost.distributed.tcp import MAX_HEADER_BYTES, MAX_PAYLOAD_BYTES, _read_exact, _read_line
from ctboost.training._distributed_config import _normalize_distributed_config
from ctboost.training._train_native import _make_native_booster
from ctboost.training._train_params import _resolve_native_training_params

from tests.helpers import TEST_DISTRIBUTED_AUTH_TOKEN
from tests.helpers import authenticated_tcp_root
from tests.helpers import find_free_tcp_port


_UNPICKLE_PROBES = []


def _record_unpickle_probe():
    _UNPICKLE_PROBES.append(True)


class _PickleProbe:
    def __reduce__(self):
        return _record_unpickle_probe, ()


def _raw_exchange(port: int, header: bytes, payload: bytes = b"") -> tuple[str, bytes]:
    with socket.create_connection(("127.0.0.1", port), timeout=2.0) as connection:
        connection.settimeout(2.0)
        connection.sendall(header + payload)
        stream = connection.makefile("rb", buffering=0)
        response_header = _read_line(stream)
        fields = response_header.split("\t")
        assert len(fields) == 3
        assert fields[0] == "CTB1"
        return fields[1], _read_exact(stream, int(fields[2]), max_size=4096)


def test_endpoint_allocator_binds_a_concrete_host_and_adds_random_auth(monkeypatch):
    real_socket = socket.socket
    bound_addresses = []

    class RecordingSocket:
        def __init__(self, *args, **kwargs):
            self._socket = real_socket(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self._socket.close()

        def bind(self, address):
            bound_addresses.append(address)
            self._socket.bind(address)

        def getsockname(self):
            return self._socket.getsockname()

    monkeypatch.setattr(integration_utils.socket, "socket", RecordingSocket)
    root, _ = integration_utils.allocate_tcp_endpoint("127.0.0.1")

    parsed = parse_distributed_root(root)
    assert bound_addresses == [("127.0.0.1", 0)]
    assert parsed.host == "127.0.0.1"
    assert parsed.auth_token is not None
    assert len(parsed.auth_token) == 64


def test_tcp_roots_fail_closed_without_authentication_or_with_wildcard_hosts():
    port = find_free_tcp_port()
    bare_root = f"tcp://127.0.0.1:{port}"

    with pytest.raises(ValueError, match="authenticated root"):
        distributed_tcp_request(bare_root, 0.1, "ping", "run/health", 0, 1, b"")
    with pytest.raises(ValueError, match="authenticated root"):
        _normalize_distributed_config(
            {
                "distributed_world_size": 2,
                "distributed_rank": 0,
                "distributed_root": bare_root,
            }
        )
    with pytest.raises(ValueError, match="concrete host"):
        parse_distributed_root(f"tcp://0.0.0.0:{port}/auth/{TEST_DISTRIBUTED_AUTH_TOKEN}")
    with pytest.raises(ValueError, match="concrete bind host"):
        DistributedCollectiveServer(
            "0.0.0.0", port, auth_token=TEST_DISTRIBUTED_AUTH_TOKEN
        )
    with pytest.raises(ValueError, match="authentication token"):
        DistributedCollectiveServer("127.0.0.1", port)


def test_authentication_is_checked_before_numeric_fields_body_and_pickle():
    _UNPICKLE_PROBES.clear()
    builder_calls = []
    port = find_free_tcp_port()

    def schema_builder(payloads):
        builder_calls.append(True)
        pickle.loads(payloads[0])
        return b"ok"

    server = DistributedCollectiveServer(
        "127.0.0.1",
        port,
        schema_builder=schema_builder,
        auth_token=TEST_DISTRIBUTED_AUTH_TOKEN,
        expected_run_id="secure-run",
        expected_world_size=1,
        request_timeout=0.2,
    )
    server.start()
    try:
        malicious_payload = pickle.dumps(_PickleProbe())
        wrong_token = "b" * 64
        status, message = _raw_exchange(
            port,
            (
                f"CTB1\t{wrong_token}\tschema_collect\tsecure-run\t"
                f"not-a-rank\tnot-a-world\t{len(malicious_payload)}\n"
            ).encode("utf-8"),
            malicious_payload,
        )
        assert status == "error"
        assert b"authentication failed" in message
        assert builder_calls == []
        assert _UNPICKLE_PROBES == []

        assert distributed_tcp_request(
            authenticated_tcp_root(port),
            2.0,
            "ping",
            "secure-run/__health__",
            0,
            1,
            b"",
        ) == b""
    finally:
        server.stop()


def test_server_rejects_oversized_headers_and_declared_payloads_without_reading_body():
    port = find_free_tcp_port()
    server = DistributedCollectiveServer(
        "127.0.0.1", port, auth_token=TEST_DISTRIBUTED_AUTH_TOKEN
    )
    server.start()
    try:
        status, message = _raw_exchange(port, b"x" * (MAX_HEADER_BYTES + 1))
        assert status == "error"
        assert len(message) <= 4096

        status, message = _raw_exchange(
            port,
            (
                f"CTB1\t{TEST_DISTRIBUTED_AUTH_TOKEN}\tallgather\trun/oversize\t"
                f"0\t1\t{MAX_PAYLOAD_BYTES + 1}\n"
            ).encode("utf-8"),
        )
        assert status == "error"
        assert b"exceeds the allowed size" in message
    finally:
        server.stop()


def test_client_rejects_oversized_response_before_allocating():
    port = find_free_tcp_port()

    class OversizedResponseServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    class OversizedResponseHandler(socketserver.StreamRequestHandler):
        def handle(self):
            fields = _read_line(self.rfile).split("\t")
            _read_exact(self.rfile, int(fields[-1]))
            self.wfile.write(f"CTB1\tok\t{MAX_PAYLOAD_BYTES + 1}\n".encode("ascii"))

    server = OversizedResponseServer(("127.0.0.1", port), OversizedResponseHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(ValueError, match="outside the allowed range"):
            distributed_tcp_request(
                authenticated_tcp_root(port), 2.0, "ping", "run/health", 0, 1, b""
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_collective_retries_are_idempotent_and_conflicts_fail():
    port = find_free_tcp_port()
    root = authenticated_tcp_root(port)
    server = DistributedCollectiveServer(
        "127.0.0.1", port, auth_token=TEST_DISTRIBUTED_AUTH_TOKEN
    )
    server.start()
    try:
        first = distributed_tcp_request(root, 2.0, "allgather", "run/retry", 0, 1, b"same")
        second = distributed_tcp_request(root, 2.0, "allgather", "run/retry", 0, 1, b"same")
        assert second == first
        with pytest.raises(RuntimeError, match="conflicting duplicate rank"):
            distributed_tcp_request(root, 2.0, "allgather", "run/retry", 0, 1, b"different")
    finally:
        server.stop()


def test_terminal_collective_failures_release_state_and_accounting(monkeypatch):
    import ctboost.distributed.server as server_module

    monkeypatch.setattr(server_module, "_COMPLETED_STATE_GRACE_SECONDS", 0.01)
    port = find_free_tcp_port()
    server = DistributedCollectiveServer(
        "127.0.0.1",
        port,
        auth_token=TEST_DISTRIBUTED_AUTH_TOKEN,
        max_active_payload_bytes=1,
        collective_timeout=0.02,
    )
    server.start()
    try:
        with pytest.raises(RuntimeError, match="active payload limit"):
            server._dispatch("allgather", "run/rejected", 0, 1, b"too large")
        assert server._states == {}
        assert server._active_payload_bytes == 0

        with pytest.raises(TimeoutError, match="timed out"):
            server._dispatch("barrier", "run/stuck", 0, 2, b"")
        terminal_state = server._states[("barrier", "run/stuck")]
        with pytest.raises(TimeoutError, match="timed out"):
            server._dispatch("barrier", "run/stuck", 1, 2, b"")
        assert terminal_state.payloads == {0: b""}
        deadline = time.monotonic() + 1.0
        while server._states and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server._states == {}
        assert server._active_payload_bytes == 0
    finally:
        server.stop()


def test_tokens_are_live_clone_parameters_but_absent_from_persisted_state(monkeypatch):
    port = find_free_tcp_port()
    root = authenticated_tcp_root(port)
    bare_root = f"tcp://127.0.0.1:{port}"
    estimator = ctboost.CTBoostRegressor(distributed_root=root)

    assert estimator.get_params(deep=False)["distributed_root"] == root
    assert clone(estimator).distributed_root == root
    assert TEST_DISTRIBUTED_AUTH_TOKEN not in repr(estimator)
    assert TEST_DISTRIBUTED_AUTH_TOKEN.encode("ascii") not in pickle.dumps(estimator)
    assert redact_distributed_root(root) == bare_root
    assert redact_distributed_root(f"{bare_root}/legacy/unvalidated") == bare_root

    native = _core.GradientBooster(distributed_root=root)
    assert native.export_state()["distributed_root"] == bare_root

    pool = ctboost.Pool(
        np.zeros((2, 1), dtype=np.float32),
        label=np.zeros(2, dtype=np.float32),
    )
    native_params = _resolve_native_training_params(
        {
            "distributed_world_size": 2,
            "distributed_rank": 1,
            "distributed_root": root,
            "distributed_run_id": "resume-run",
            "distributed_timeout": 3.0,
        },
        pool,
        init_state=None,
    )
    captured = {}

    class FakeNativeBooster:
        def __init__(self, **kwargs):
            captured["constructor"] = kwargs

        def load_state(self, state):
            captured["state"] = dict(state)

    import ctboost.training._train_native as train_native_module

    monkeypatch.setattr(train_native_module._core, "GradientBooster", FakeNativeBooster)
    _make_native_booster(
        native_params,
        2,
        native_eval_metric="RMSE",
        state={"distributed_root": bare_root},
    )
    assert captured["state"]["distributed_root"] == root
    assert captured["state"]["distributed_rank"] == 1
    assert captured["state"]["distributed_run_id"] == "resume-run"
