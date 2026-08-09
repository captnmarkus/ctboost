"""TCP transport helpers for ctboost.distributed."""

from __future__ import annotations

from dataclasses import dataclass
import re
import secrets
import socket
import time
from typing import Any, Optional


_AUTH_TOKEN_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ALLOWED_OPS = frozenset(
    {
        "allgather",
        "barrier",
        "broadcast",
        "gpu_snapshot_reduce",
        "node_hist_reduce",
        "ping",
        "schema_collect",
    }
)
PROTOCOL_VERSION = "CTB1"
MAX_HEADER_BYTES = 16 * 1024
MAX_KEY_BYTES = 4 * 1024
MAX_WORLD_SIZE = 65_536
MAX_PAYLOAD_BYTES = 1024 * 1024 * 1024
MAX_ERROR_BYTES = 4 * 1024


@dataclass(frozen=True)
class ParsedDistributedRoot:
    backend: str
    root: str
    host: Optional[str] = None
    port: Optional[int] = None
    auth_token: Optional[str] = None


def new_distributed_auth_token() -> str:
    """Return a cryptographically random token for one TCP training run."""

    return secrets.token_hex(32)


def authenticated_tcp_root(host: str, port: int, auth_token: Optional[str] = None) -> str:
    """Build an authenticated TCP root without exposing a wildcard endpoint."""

    resolved_host = str(host).strip()
    if not resolved_host or resolved_host in {"0.0.0.0", "::"}:
        raise ValueError("distributed tcp root requires a concrete host, not a wildcard address")
    resolved_port = int(port)
    if resolved_port <= 0 or resolved_port > 65535:
        raise ValueError("distributed tcp port must be in [1, 65535]")
    token = str(auth_token or new_distributed_auth_token()).strip().lower()
    if _AUTH_TOKEN_RE.fullmatch(token) is None:
        raise ValueError("distributed tcp auth token must contain exactly 64 hexadecimal characters")
    return f"tcp://{resolved_host}:{resolved_port}/auth/{token}"


def _split_tcp_root(value: str) -> tuple[str, str, Optional[str]]:
    endpoint_with_path = value[len("tcp://") :]
    endpoint, separator, raw_path = endpoint_with_path.partition("/")
    if ":" not in endpoint:
        raise ValueError("distributed tcp root must be formatted like tcp://host:port")
    host, raw_port = endpoint.rsplit(":", 1)
    if not host:
        raise ValueError("distributed tcp root must include a host")
    if host in {"0.0.0.0", "::"}:
        raise ValueError("distributed tcp root requires a concrete host, not a wildcard address")
    if not raw_port or any(character < "0" or character > "9" for character in raw_port):
        raise ValueError("distributed tcp port must be an integer")
    port = int(raw_port)
    if port <= 0 or port > 65535:
        raise ValueError("distributed tcp port must be in [1, 65535]")

    auth_token: Optional[str] = None
    if separator:
        path_parts = [part for part in raw_path.split("/") if part]
        if len(path_parts) == 1:
            candidate = path_parts[0]
        elif len(path_parts) == 2 and path_parts[0] == "auth":
            candidate = path_parts[1]
        else:
            raise ValueError(
                "distributed tcp root path must contain only an authentication token"
            )
        if _AUTH_TOKEN_RE.fullmatch(candidate) is None:
            raise ValueError(
                "distributed tcp auth token must contain exactly 64 hexadecimal characters"
            )
        auth_token = candidate.lower()
    return host, str(port), auth_token


def parse_distributed_root(root: Any) -> ParsedDistributedRoot:
    value = str(root or "")
    if value.startswith("tcp://"):
        host, raw_port, auth_token = _split_tcp_root(value)
        return ParsedDistributedRoot(
            "tcp",
            value,
            host=host,
            port=int(raw_port),
            auth_token=auth_token,
        )
    return ParsedDistributedRoot("filesystem", value)


def redact_distributed_root(root: Any) -> str:
    """Remove an ephemeral TCP authentication token from a persisted display value."""

    value = str(root or "")
    if not value.startswith("tcp://"):
        return value
    # Persistence must not leak a credential even when the original value came
    # from an older release or is otherwise not valid under the current parser.
    endpoint = value[len("tcp://") :].split("/", 1)[0]
    return f"tcp://{endpoint}"


def _read_exact(stream, size: int, *, max_size: int = MAX_PAYLOAD_BYTES) -> bytes:
    resolved_size = int(size)
    if resolved_size < 0 or resolved_size > int(max_size):
        raise ValueError("distributed payload size is outside the allowed range")
    chunks = bytearray()
    while len(chunks) < resolved_size:
        chunk = stream.read(resolved_size - len(chunks))
        if not chunk:
            raise ConnectionError("distributed coordinator connection closed unexpectedly")
        chunks.extend(chunk)
    return bytes(chunks)


def _read_line(stream, *, max_bytes: int = MAX_HEADER_BYTES) -> str:
    line = bytearray()
    while True:
        if len(line) >= int(max_bytes):
            raise ValueError("distributed protocol line exceeds the allowed size")
        char = stream.read(1)
        if not char:
            raise ConnectionError("distributed coordinator connection closed unexpectedly")
        if char == b"\n":
            try:
                return line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("distributed protocol line must be valid UTF-8") from exc
        line.extend(char)


def _validate_outgoing_request(
    op: str,
    key: str,
    rank: int,
    world_size: int,
    payload: bytes,
) -> None:
    if op not in _ALLOWED_OPS:
        raise ValueError(f"unsupported distributed coordinator op {op!r}")
    if not key or "\t" in key or "\n" in key or len(key.encode("utf-8")) > MAX_KEY_BYTES:
        raise ValueError("distributed coordinator key is empty or exceeds protocol limits")
    if world_size <= 0 or world_size > MAX_WORLD_SIZE:
        raise ValueError(f"distributed world_size must be in [1, {MAX_WORLD_SIZE}]")
    if rank < 0 or rank >= world_size:
        raise ValueError("distributed rank must be in [0, world_size)")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("distributed request payload exceeds the allowed size")
    if op in {"ping", "barrier"} and payload:
        raise ValueError(f"distributed {op} requests must not contain a payload")
    if op == "broadcast" and rank != 0 and payload:
        raise ValueError("non-root distributed broadcast requests must not contain a payload")


def _parse_response(stream) -> bytes:
    response_line = _read_line(stream)
    response_header = response_line.split("\t")
    if len(response_header) != 3 or response_header[0] != PROTOCOL_VERSION:
        raise RuntimeError("invalid authenticated distributed coordinator response")
    _, status, raw_size = response_header
    if status not in {"ok", "error"}:
        raise RuntimeError("invalid authenticated distributed coordinator response status")
    if not raw_size or any(character < "0" or character > "9" for character in raw_size):
        raise RuntimeError("invalid distributed coordinator response size")
    response_size = int(raw_size)
    response_limit = MAX_PAYLOAD_BYTES if status == "ok" else MAX_ERROR_BYTES
    response = _read_exact(stream, response_size, max_size=response_limit)
    if status != "ok":
        message = response.decode("utf-8", errors="replace")
        raise RuntimeError(message or "distributed coordinator rejected the request")
    return response


def distributed_tcp_request(
    root: str,
    timeout_seconds: float,
    op: str,
    key: str,
    rank: int,
    world_size: int,
    payload: bytes,
) -> bytes:
    parsed = parse_distributed_root(root)
    if parsed.backend != "tcp" or parsed.host is None or parsed.port is None:
        raise ValueError("distributed tcp request requires a tcp://host:port root")
    if parsed.auth_token is None:
        raise ValueError(
            "distributed tcp requests require an authenticated root ending in "
            "'/auth/<64-hex-token>'"
        )
    resolved_op = str(op)
    resolved_key = str(key)
    resolved_rank = int(rank)
    resolved_world_size = int(world_size)
    resolved_payload = bytes(payload)
    _validate_outgoing_request(
        resolved_op,
        resolved_key,
        resolved_rank,
        resolved_world_size,
        resolved_payload,
    )
    deadline = time.monotonic() + float(timeout_seconds)
    last_error: Optional[BaseException] = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            if last_error is None:
                raise TimeoutError(f"timed out waiting for distributed tcp coordinator at {redact_distributed_root(root)}")
            raise TimeoutError(
                f"timed out waiting for distributed tcp coordinator at {redact_distributed_root(root)}"
            ) from last_error
        connect_timeout = min(max(remaining, 0.05), 1.0)
        try:
            with socket.create_connection((parsed.host, parsed.port), timeout=connect_timeout) as connection:
                response_timeout = max(deadline - time.monotonic(), 0.05)
                connection.settimeout(response_timeout)
                stream = connection.makefile("rwb", buffering=0)
                header = (
                    f"{PROTOCOL_VERSION}\t{parsed.auth_token}\t{resolved_op}\t"
                    f"{resolved_key}\t{resolved_rank}\t{resolved_world_size}\t"
                    f"{len(resolved_payload)}\n"
                ).encode("utf-8")
                if len(header) > MAX_HEADER_BYTES:
                    raise ValueError("distributed request header exceeds the allowed size")
                stream.write(header)
                if resolved_payload:
                    stream.write(resolved_payload)
                return _parse_response(stream)
        except (ConnectionError, OSError) as exc:
            last_error = exc
            time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))


def wait_for_distributed_tcp_coordinator(
    root: str,
    timeout_seconds: float,
    *,
    run_id: Optional[str] = None,
    world_size: int = 1,
) -> None:
    deadline = time.time() + timeout_seconds
    health_key = "__health__" if run_id is None else f"{run_id}/__health__"
    while time.time() < deadline:
        try:
            distributed_tcp_request(
                root,
                min(0.5, timeout_seconds),
                "ping",
                health_key,
                0,
                int(world_size),
                b"",
            )
            return
        except (OSError, RuntimeError, ValueError):
            time.sleep(0.05)
    raise TimeoutError(
        f"timed out waiting for distributed tcp coordinator at {redact_distributed_root(root)}"
    )
