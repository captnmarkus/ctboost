"""Payload encoding helpers for ctboost.distributed."""

from __future__ import annotations

import pickle
import struct
from typing import Any, Dict, Iterable

import numpy as np


def _parse_node_hist_payload(payload: bytes) -> Dict[str, Any]:
    view = memoryview(payload)
    offset = 0
    feature_count, sample_count, sample_weight_sum, total_gradient, total_hessian, gradient_square_sum = struct.unpack_from(
        "<QQdddd",
        view,
        offset,
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
    chunks = [
        struct.pack(
            "<QQdddd",
            len(parsed["features"]),
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
        if len(parsed["features"]) != len(result["features"]):
            raise ValueError("distributed node histogram feature counts must match")
        result["sample_count"] += parsed["sample_count"]
        result["sample_weight_sum"] += parsed["sample_weight_sum"]
        result["total_gradient"] += parsed["total_gradient"]
        result["total_hessian"] += parsed["total_hessian"]
        result["gradient_square_sum"] += parsed["gradient_square_sum"]
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
    total_bins, sample_count, sample_weight_sum, total_gradient, total_hessian, gradient_square_sum = struct.unpack_from(
        "<QQdddd",
        view,
        offset,
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
    return b"".join(
        [
            struct.pack(
                "<QQdddd",
                int(gradient_sums.shape[0]),
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


def gather_payloads(payloads: Iterable[bytes]) -> bytes:
    payload_list = list(payloads)
    chunks = [struct.pack("<Q", len(payload_list))]
    for payload in payload_list:
        chunks.append(struct.pack("<Q", len(payload)))
        chunks.append(payload)
    return b"".join(chunks)


def pickle_payload(value: Any) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_payload(payload: bytes) -> Any:
    return pickle.loads(payload)
