"""Distributed coordination helpers for CTBoost."""

from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
import socket
import socketserver
import struct
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from . import _core
from .core import Pool


@dataclass(frozen=True)
class ParsedDistributedRoot:
    backend: str
    root: str
    host: Optional[str] = None
    port: Optional[int] = None


def parse_distributed_root(root: Any) -> ParsedDistributedRoot:
    value = str(root or "")
    if value.startswith("tcp://"):
        endpoint = value[len("tcp://") :]
        endpoint = endpoint.split("/", 1)[0]
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
            raise TimeoutError(
                f"timed out waiting for distributed tcp coordinator at {root}"
            ) from last_error
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
                response_size = int(raw_size)
                return _read_exact(stream, response_size)
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


def _parse_node_hist_payload(payload: bytes) -> Dict[str, Any]:
    view = memoryview(payload)
    offset = 0
    feature_count, sample_count, sample_weight_sum, total_gradient, total_hessian, gradient_square_sum = (
        struct.unpack_from("<QQdddd", view, offset)
    )
    offset += struct.calcsize("<QQdddd")
    features = []
    for _ in range(feature_count):
        (bin_count,) = struct.unpack_from("<Q", view, offset)
        offset += struct.calcsize("<Q")
        grad = np.frombuffer(view[offset : offset + bin_count * 8], dtype="<f8").copy()
        offset += bin_count * 8
        hess = np.frombuffer(view[offset : offset + bin_count * 8], dtype="<f8").copy()
        offset += bin_count * 8
        weight = np.frombuffer(view[offset : offset + bin_count * 8], dtype="<f8").copy()
        offset += bin_count * 8
        features.append((grad, hess, weight))
    return {
        "sample_count": int(sample_count),
        "sample_weight_sum": float(sample_weight_sum),
        "total_gradient": float(total_gradient),
        "total_hessian": float(total_hessian),
        "gradient_square_sum": float(gradient_square_sum),
        "features": features,
    }


def _encode_node_hist_payload(parsed: Dict[str, Any]) -> bytes:
    feature_count = len(parsed["features"])
    chunks = [
        struct.pack(
            "<QQdddd",
            feature_count,
            int(parsed["sample_count"]),
            float(parsed["sample_weight_sum"]),
            float(parsed["total_gradient"]),
            float(parsed["total_hessian"]),
            float(parsed["gradient_square_sum"]),
        )
    ]
    for grad, hess, weight in parsed["features"]:
        grad = np.asarray(grad, dtype=np.float64)
        hess = np.asarray(hess, dtype=np.float64)
        weight = np.asarray(weight, dtype=np.float64)
        chunks.append(struct.pack("<Q", int(grad.shape[0])))
        chunks.append(grad.astype("<f8", copy=False).tobytes(order="C"))
        chunks.append(hess.astype("<f8", copy=False).tobytes(order="C"))
        chunks.append(weight.astype("<f8", copy=False).tobytes(order="C"))
    return b"".join(chunks)


def sum_node_hist_payloads(payloads: Iterable[bytes]) -> bytes:
    iterator = iter(payloads)
    first = _parse_node_hist_payload(next(iterator))
    result = {
        "sample_count": first["sample_count"],
        "sample_weight_sum": first["sample_weight_sum"],
        "total_gradient": first["total_gradient"],
        "total_hessian": first["total_hessian"],
        "gradient_square_sum": first["gradient_square_sum"],
        "features": [(grad.copy(), hess.copy(), weight.copy()) for grad, hess, weight in first["features"]],
    }
    for payload in iterator:
        parsed = _parse_node_hist_payload(payload)
        result["sample_count"] += parsed["sample_count"]
        result["sample_weight_sum"] += parsed["sample_weight_sum"]
        result["total_gradient"] += parsed["total_gradient"]
        result["total_hessian"] += parsed["total_hessian"]
        result["gradient_square_sum"] += parsed["gradient_square_sum"]
        if len(parsed["features"]) != len(result["features"]):
            raise ValueError("distributed node histogram feature counts must match")
        for index, (grad, hess, weight) in enumerate(parsed["features"]):
            result_grad, result_hess, result_weight = result["features"][index]
            if grad.shape != result_grad.shape:
                raise ValueError("distributed node histogram bin counts must match")
            result_grad += grad
            result_hess += hess
            result_weight += weight
    return _encode_node_hist_payload(result)


def _parse_gpu_snapshot_payload(payload: bytes) -> Dict[str, Any]:
    view = memoryview(payload)
    offset = 0
    total_bins, sample_count, sample_weight_sum, total_gradient, total_hessian, gradient_square_sum = (
        struct.unpack_from("<QQdddd", view, offset)
    )
    offset += struct.calcsize("<QQdddd")
    gradient_sums = np.frombuffer(view[offset : offset + total_bins * 4], dtype="<f4").copy()
    offset += total_bins * 4
    hessian_sums = np.frombuffer(view[offset : offset + total_bins * 4], dtype="<f4").copy()
    offset += total_bins * 4
    weight_sums = np.frombuffer(view[offset : offset + total_bins * 4], dtype="<f4").copy()
    return {
        "sample_count": int(sample_count),
        "sample_weight_sum": float(sample_weight_sum),
        "total_gradient": float(total_gradient),
        "total_hessian": float(total_hessian),
        "gradient_square_sum": float(gradient_square_sum),
        "gradient_sums": gradient_sums,
        "hessian_sums": hessian_sums,
        "weight_sums": weight_sums,
    }


def _encode_gpu_snapshot_payload(parsed: Dict[str, Any]) -> bytes:
    gradient_sums = np.asarray(parsed["gradient_sums"], dtype=np.float32)
    hessian_sums = np.asarray(parsed["hessian_sums"], dtype=np.float32)
    weight_sums = np.asarray(parsed["weight_sums"], dtype=np.float32)
    total_bins = int(gradient_sums.shape[0])
    return b"".join(
        [
            struct.pack(
                "<QQdddd",
                total_bins,
                int(parsed["sample_count"]),
                float(parsed["sample_weight_sum"]),
                float(parsed["total_gradient"]),
                float(parsed["total_hessian"]),
                float(parsed["gradient_square_sum"]),
            ),
            gradient_sums.astype("<f4", copy=False).tobytes(order="C"),
            hessian_sums.astype("<f4", copy=False).tobytes(order="C"),
            weight_sums.astype("<f4", copy=False).tobytes(order="C"),
        ]
    )


def sum_gpu_snapshot_payloads(payloads: Iterable[bytes]) -> bytes:
    iterator = iter(payloads)
    first = _parse_gpu_snapshot_payload(next(iterator))
    result = {
        "sample_count": first["sample_count"],
        "sample_weight_sum": first["sample_weight_sum"],
        "total_gradient": first["total_gradient"],
        "total_hessian": first["total_hessian"],
        "gradient_square_sum": first["gradient_square_sum"],
        "gradient_sums": first["gradient_sums"].copy(),
        "hessian_sums": first["hessian_sums"].copy(),
        "weight_sums": first["weight_sums"].copy(),
    }
    for payload in iterator:
        parsed = _parse_gpu_snapshot_payload(payload)
        if parsed["gradient_sums"].shape != result["gradient_sums"].shape:
            raise ValueError("distributed gpu snapshot bin counts must match")
        result["sample_count"] += parsed["sample_count"]
        result["sample_weight_sum"] += parsed["sample_weight_sum"]
        result["total_gradient"] += parsed["total_gradient"]
        result["total_hessian"] += parsed["total_hessian"]
        result["gradient_square_sum"] += parsed["gradient_square_sum"]
        result["gradient_sums"] += parsed["gradient_sums"]
        result["hessian_sums"] += parsed["hessian_sums"]
        result["weight_sums"] += parsed["weight_sums"]
    return _encode_gpu_snapshot_payload(result)


def _validate_group_shards(group_arrays: List[np.ndarray]) -> None:
    seen_groups = set()
    for shard_groups in group_arrays:
        unique_groups = {int(value) for value in np.unique(np.asarray(shard_groups, dtype=np.int64))}
        overlap = seen_groups.intersection(unique_groups)
        if overlap:
            raise ValueError(
                "distributed ranking/group training requires each group_id to live on exactly one rank"
            )
        seen_groups.update(unique_groups)


def _merge_dense_payloads(shards: List[Dict[str, Any]]) -> Pool:
    data = np.asfortranarray(
        np.concatenate([np.asarray(shard["data"], dtype=np.float32) for shard in shards], axis=0),
        dtype=np.float32,
    )
    label = np.concatenate([np.asarray(shard["label"], dtype=np.float32) for shard in shards], axis=0)
    weight = np.concatenate([np.asarray(shard["weight"], dtype=np.float32) for shard in shards], axis=0)
    has_group_id = any(shard["group_id"] is not None for shard in shards)
    if has_group_id and any(shard["group_id"] is None for shard in shards):
        raise ValueError("distributed shards must either all include group_id or all omit it")
    group_id = None
    if has_group_id:
        group_arrays = [np.asarray(shard["group_id"], dtype=np.int64) for shard in shards]
        _validate_group_shards(group_arrays)
        group_id = np.concatenate(group_arrays, axis=0)
    return Pool(
        data=data,
        label=label,
        cat_features=list(shards[0]["cat_features"]),
        weight=weight,
        group_id=group_id,
        feature_names=shards[0]["feature_names"],
        _releasable_feature_storage=True,
    )


def _merge_sparse_payloads(shards: List[Dict[str, Any]]) -> Pool:
    total_rows = sum(int(shard["num_rows"]) for shard in shards)
    num_cols = int(shards[0]["num_cols"])
    total_nnz = sum(int(np.asarray(shard["sparse_data"]).shape[0]) for shard in shards)
    data = np.empty(total_nnz, dtype=np.float32)
    indices = np.empty(total_nnz, dtype=np.int64)
    indptr = np.empty(num_cols + 1, dtype=np.int64)
    label = np.concatenate([np.asarray(shard["label"], dtype=np.float32) for shard in shards], axis=0)
    weight = np.concatenate([np.asarray(shard["weight"], dtype=np.float32) for shard in shards], axis=0)
    has_group_id = any(shard["group_id"] is not None for shard in shards)
    if has_group_id and any(shard["group_id"] is None for shard in shards):
        raise ValueError("distributed shards must either all include group_id or all omit it")
    group_id = None
    if has_group_id:
        group_arrays = [np.asarray(shard["group_id"], dtype=np.int64) for shard in shards]
        _validate_group_shards(group_arrays)
        group_id = np.concatenate(group_arrays, axis=0)

    row_bases = [0]
    for shard in shards[:-1]:
        row_bases.append(row_bases[-1] + int(shard["num_rows"]))

    nnz_offset = 0
    indptr[0] = 0
    for col in range(num_cols):
        for shard_index, shard in enumerate(shards):
            sparse_data = np.asarray(shard["sparse_data"], dtype=np.float32)
            sparse_indices = np.asarray(shard["sparse_indices"], dtype=np.int64)
            sparse_indptr = np.asarray(shard["sparse_indptr"], dtype=np.int64)
            begin = int(sparse_indptr[col])
            end = int(sparse_indptr[col + 1])
            nnz = end - begin
            if nnz <= 0:
                continue
            next_offset = nnz_offset + nnz
            data[nnz_offset:next_offset] = sparse_data[begin:end]
            indices[nnz_offset:next_offset] = sparse_indices[begin:end] + row_bases[shard_index]
            nnz_offset = next_offset
        indptr[col + 1] = nnz_offset

    return Pool.from_csc_components(
        data,
        indices,
        indptr,
        (total_rows, num_cols),
        label,
        cat_features=list(shards[0]["cat_features"]),
        weight=weight,
        group_id=group_id,
        feature_names=shards[0]["feature_names"],
        _releasable_feature_storage=True,
    )


def _merge_payloads(shards: List[Dict[str, Any]]) -> Pool:
    if not shards:
        raise ValueError("distributed training requires at least one shard payload")
    storage_kinds = {str(shard["storage"]) for shard in shards}
    if len(storage_kinds) != 1:
        raise ValueError("distributed shards must agree on storage kind")
    num_cols = int(shards[0]["num_cols"])
    cat_features = list(shards[0]["cat_features"])
    feature_names = shards[0]["feature_names"]
    for shard in shards[1:]:
        if int(shard["num_cols"]) != num_cols:
            raise ValueError("distributed shards must agree on num_cols")
        if list(shard["cat_features"]) != cat_features:
            raise ValueError("distributed shards must agree on cat_features")
        if shard["feature_names"] != feature_names:
            raise ValueError("distributed shards must agree on feature_names")
    if storage_kinds == {"dense"}:
        return _merge_dense_payloads(shards)
    return _merge_sparse_payloads(shards)


def _quantization_schema_from_debug_histogram(hist_state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_bins_per_feature": list(hist_state["num_bins_per_feature"]),
        "cut_offsets": list(hist_state["cut_offsets"]),
        "cut_values": list(hist_state["cut_values"]),
        "categorical_mask": list(hist_state["categorical_mask"]),
        "missing_value_mask": list(hist_state["missing_value_mask"]),
        "nan_mode": int(hist_state["nan_mode"]),
        "nan_modes": list(hist_state["nan_modes"]),
    }


def build_schema_collect_response(payloads: List[bytes]) -> bytes:
    requests = [unpickle_payload(payload) for payload in payloads]
    shards = [request["shard"] for request in requests]
    merged_pool = _merge_payloads(shards)
    schema_request = requests[0]["schema_request"]
    hist_state = _core._debug_build_histogram(
        merged_pool._handle,
        max_bins=int(schema_request["max_bins"]),
        nan_mode=str(schema_request["nan_mode"]),
        max_bin_by_feature=list(schema_request["max_bin_by_feature"]),
        border_selection_method=str(schema_request["border_selection_method"]),
        nan_mode_by_feature=list(schema_request["nan_mode_by_feature"]),
        feature_borders=list(schema_request["feature_borders"]),
        external_memory=bool(schema_request["external_memory"]),
        external_memory_dir=str(schema_request["external_memory_dir"]),
    )
    return json.dumps(_quantization_schema_from_debug_histogram(hist_state), sort_keys=True).encode("utf-8")


def gather_payloads(payloads: Iterable[bytes]) -> bytes:
    payload_list = list(payloads)
    chunks = [struct.pack("<Q", len(payload_list))]
    for payload in payload_list:
        chunks.append(struct.pack("<Q", len(payload)))
        chunks.append(payload)
    return b"".join(chunks)


class _CollectiveState:
    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.payloads: Dict[int, bytes] = {}
        self.response: Optional[bytes] = None
        self.remaining = world_size
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
        self._host = host
        self._port = port
        self._schema_builder = schema_builder
        self._states: Dict[tuple[str, str], _CollectiveState] = {}
        self._states_lock = threading.Lock()
        self._server = _ThreadingCollectiveTcpServer((host, port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

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
                payload_size = int(raw_payload_size)
                payload = _read_exact(self.rfile, payload_size)
                response = owner._dispatch(op, key, rank, world_size, payload)
                self.wfile.write(f"ok\t{len(response)}\n".encode("utf-8"))
                if response:
                    self.wfile.write(response)

        return Handler

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
            state.remaining -= 1
            if state.remaining == 0:
                with self._states_lock:
                    self._states.pop(state_key, None)
            return response

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


def pickle_payload(value: Any) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_payload(payload: bytes) -> Any:
    return pickle.loads(payload)


def run_distributed_collective_server(host: str, port: int) -> None:
    server = DistributedCollectiveServer(host, port, schema_builder=build_schema_collect_response)
    server.start()
    try:
        while True:
            time.sleep(3600.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m ctboost.distributed <host> <port>")
    run_distributed_collective_server(sys.argv[1], int(sys.argv[2]))
