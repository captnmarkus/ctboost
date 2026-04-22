"""TCP transport helpers for ctboost.distributed."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import time
from typing import Any, Optional


@dataclass(frozen=True)
class ParsedDistributedRoot:
    backend: str
    root: str
    host: Optional[str] = None
    port: Optional[int] = None


def parse_distributed_root(root: Any) -> ParsedDistributedRoot:
    value = str(root or "")
    if value.startswith("tcp://"):
        endpoint = value[len("tcp://") :].split("/", 1)[0]
        if ":" not in endpoint:
            raise ValueError("distributed tcp root must be formatted like tcp://host:port")
        host, raw_port = endpoint.rsplit(":", 1)
        if not host:
            raise ValueError("distributed tcp root must include a host")
        port = int(raw_port)
        if port <= 0 or port > 65535:
            raise ValueError("distributed tcp port must be in [1, 65535]")
        return ParsedDistributedRoot("tcp", value, host=host, port=port)
    return ParsedDistributedRoot("filesystem", value)


def _read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise ConnectionError("distributed coordinator connection closed unexpectedly")
        chunks.extend(chunk)
    return bytes(chunks)


def _read_line(stream) -> str:
    line = bytearray()
    while True:
        char = stream.read(1)
        if not char:
            raise ConnectionError("distributed coordinator connection closed unexpectedly")
        if char == b"\n":
            return line.decode("utf-8")
        line.extend(char)


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
    deadline = time.monotonic() + float(timeout_seconds)
    last_error: Optional[BaseException] = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            if last_error is None:
                raise TimeoutError(f"timed out waiting for distributed tcp coordinator at {root}")
            raise TimeoutError(f"timed out waiting for distributed tcp coordinator at {root}") from last_error
        attempt_timeout = min(max(remaining, 0.05), 1.0)
        try:
            with socket.create_connection((parsed.host, parsed.port), timeout=attempt_timeout) as connection:
                connection.settimeout(attempt_timeout)
                stream = connection.makefile("rwb", buffering=0)
                header = f"{op}\t{key}\t{rank}\t{world_size}\t{len(payload)}\n".encode("utf-8")
                stream.write(header)
                if payload:
                    stream.write(payload)
                response_header = _read_line(stream).split("\t", 1)
                if len(response_header) != 2:
                    raise RuntimeError("invalid distributed coordinator response")
                status, raw_size = response_header
                if status != "ok":
                    raise RuntimeError(raw_size)
                return _read_exact(stream, int(raw_size))
        except (ConnectionError, OSError) as exc:
            last_error = exc
            time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))


def wait_for_distributed_tcp_coordinator(root: str, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            distributed_tcp_request(root, min(0.5, timeout_seconds), "ping", "__health__", 0, 1, b"")
            return
        except (OSError, RuntimeError, ValueError):
            time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for distributed tcp coordinator at {root}")
