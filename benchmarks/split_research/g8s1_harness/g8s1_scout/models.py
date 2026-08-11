"""Importable paired scout model classes and frozen portfolio generators."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from typing import Any

from .constants import (
    FORBIDDEN_BASE_FIELDS,
    NUM_CONFIGS,
    NUM_RANDOM_CONFIGS,
    P50_SHA256,
    P200_RANDOM_SHA256,
    P201_SHA256,
    TREATMENT_COMMON,
    TREATMENTS,
    canonical_p200_path,
)
from .loader import load_benchmark_module

_adapter = load_benchmark_module("ctboost_model")
CTBoostTabArenaModel = _adapter.CTBoostTabArenaModel
generate_configs_ctboost = _adapter.generate_configs_ctboost


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_base_configs(configs: list[dict[str, Any]], *, label: str) -> None:
    for index, config in enumerate(configs):
        overlap = sorted(FORBIDDEN_BASE_FIELDS.intersection(config))
        if overlap:
            raise RuntimeError(
                f"{label}[{index}] contains frozen treatment fields: {overlap}"
            )
        if "random_seed" in config:
            raise RuntimeError(f"{label}[{index}] hard-codes a model seed")


def _load_canonical_p200() -> list[dict[str, Any]]:
    path = canonical_p200_path()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("sealed canonical P200 document is missing or linked")
    configs = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(configs, list)
        or len(configs) != 200
        or any(not isinstance(config, dict) for config in configs)
    ):
        raise RuntimeError("sealed canonical P200 document has invalid structure")
    return configs


def _float_ulp_distance(left: float, right: float) -> int:
    if not math.isfinite(left) or not math.isfinite(right):
        raise RuntimeError("live adapter P200 contains a non-finite float")

    def ordered_bits(value: float) -> int:
        bits = int.from_bytes(struct.pack(">d", value), byteorder="big")
        return bits ^ ((bits >> 63) * 0x7FFFFFFFFFFFFFFF)

    return abs(ordered_bits(left) - ordered_bits(right))


def _validate_live_adapter_p200(
    canonical: list[dict[str, Any]], live: list[dict[str, Any]]
) -> None:
    if len(live) != len(canonical):
        raise RuntimeError(
            "live adapter P200 cardinality drift: "
            f"expected {len(canonical)}, observed {len(live)}"
        )
    for index, (expected, observed) in enumerate(zip(canonical, live, strict=True)):
        if tuple(observed) != tuple(expected):
            raise RuntimeError(f"live adapter P200[{index}] key/order drift")
        for key, expected_value in expected.items():
            observed_value = observed[key]
            if type(expected_value) is float:
                if type(observed_value) is not float:
                    raise RuntimeError(f"live adapter P200[{index}].{key} type drift")
                distance = _float_ulp_distance(expected_value, observed_value)
                if distance > 1:
                    raise RuntimeError(
                        f"live adapter P200[{index}].{key} drifted by {distance} ULPs"
                    )
            elif (
                type(observed_value) is not type(expected_value)
                or observed_value != expected_value
            ):
                raise RuntimeError(
                    f"live adapter P200[{index}].{key} discrete value drift"
                )


def base_random_p200() -> list[dict[str, Any]]:
    configs = _load_canonical_p200()
    digest = hashlib.sha256(canonical_json_bytes(configs)).hexdigest()
    if digest != P200_RANDOM_SHA256:
        raise RuntimeError(
            f"P200 identity drift: expected {P200_RANDOM_SHA256}, observed {digest}"
        )
    _validate_live_adapter_p200(configs, generate_configs_ctboost(200))
    _validate_base_configs(configs, label="P200")
    return copy.deepcopy(configs)


def base_p201() -> list[dict[str, Any]]:
    configs = [{}, *base_random_p200()]
    digest = hashlib.sha256(canonical_json_bytes(configs)).hexdigest()
    if digest != P201_SHA256:
        raise RuntimeError(
            f"P201 identity drift: expected {P201_SHA256}, observed {digest}"
        )
    _validate_base_configs(configs, label="P201")
    return copy.deepcopy(configs)


def base_p50() -> list[dict[str, Any]]:
    configs = base_p201()[:NUM_CONFIGS]
    if len(configs) != NUM_CONFIGS:
        raise RuntimeError(
            f"P50 cardinality drift: expected {NUM_CONFIGS}, observed {len(configs)}"
        )
    digest = hashlib.sha256(canonical_json_bytes(configs)).hexdigest()
    if digest != P50_SHA256:
        raise RuntimeError(
            f"P50 identity drift: expected {P50_SHA256}, observed {digest}"
        )
    _validate_base_configs(configs, label="P50")
    return copy.deepcopy(configs)


def paired_configs() -> dict[str, list[dict[str, Any]]]:
    base = base_p50()
    paired: dict[str, list[dict[str, Any]]] = {}
    for treatment in ("quadratic", "grouped"):
        values: list[dict[str, Any]] = []
        for config in base:
            effective = copy.deepcopy(config)
            effective.update(TREATMENT_COMMON)
            effective["feature_test"] = treatment
            values.append(effective)
        paired[treatment] = values
    _validate_pairs(paired)
    return paired


def _validate_pairs(paired: dict[str, list[dict[str, Any]]]) -> None:
    if tuple(paired) != ("quadratic", "grouped"):
        raise RuntimeError("paired treatment order drifted")
    for index, (quadratic, grouped) in enumerate(
        zip(paired["quadratic"], paired["grouped"], strict=True)
    ):
        differing = sorted(
            key
            for key in set(quadratic).union(grouped)
            if quadratic.get(key) != grouped.get(key)
        )
        if differing != ["feature_test"]:
            raise RuntimeError(
                f"paired P50[{index}] differs in {differing}, not only feature_test"
            )
        for config, treatment in ((quadratic, "quadratic"), (grouped, "grouped")):
            if config["feature_test"] != treatment:
                raise RuntimeError(f"P50[{index}] has incorrect {treatment} treatment")
            if (
                config["feature_test_bins"] != 8
                or config["feature_test_adjustment"] != "none"
            ):
                raise RuntimeError(
                    f"P50[{index}] changed the frozen grouped-scout controls"
                )


class CTBoostQuadraticScoutV1Model(CTBoostTabArenaModel):
    ag_key = TREATMENTS["quadratic"]["ag_key"]
    ag_name = TREATMENTS["quadratic"]["ag_name"]


class CTBoostGrouped8ScoutV1Model(CTBoostTabArenaModel):
    ag_key = TREATMENTS["grouped"]["ag_key"]
    ag_name = TREATMENTS["grouped"]["ag_name"]


def _random_configs(treatment: str, count: int) -> list[dict[str, Any]]:
    requested = int(count)
    if requested != NUM_RANDOM_CONFIGS:
        raise RuntimeError(
            f"frozen scout requires exactly {NUM_RANDOM_CONFIGS} random configs, got {requested}"
        )
    return copy.deepcopy(paired_configs()[treatment][1:])


def generate_quadratic_configs(count: int) -> list[dict[str, Any]]:
    return _random_configs("quadratic", count)


def generate_grouped_configs(count: int) -> list[dict[str, Any]]:
    return _random_configs("grouped", count)


def build_generators() -> tuple[Any, Any]:
    from tabarena.utils.config_utils import CustomAGConfigGenerator

    paired = paired_configs()
    quadratic = CustomAGConfigGenerator(
        model_cls=CTBoostQuadraticScoutV1Model,
        search_space_func=generate_quadratic_configs,
        manual_configs=[copy.deepcopy(paired["quadratic"][0])],
    )
    grouped = CustomAGConfigGenerator(
        model_cls=CTBoostGrouped8ScoutV1Model,
        search_space_func=generate_grouped_configs,
        manual_configs=[copy.deepcopy(paired["grouped"][0])],
    )
    return quadratic, grouped


__all__ = [
    "CTBoostGrouped8ScoutV1Model",
    "CTBoostQuadraticScoutV1Model",
    "base_p50",
    "base_p201",
    "base_random_p200",
    "build_generators",
    "canonical_json_bytes",
    "generate_grouped_configs",
    "generate_quadratic_configs",
    "paired_configs",
]
