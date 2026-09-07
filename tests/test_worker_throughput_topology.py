"""Validate Windows topology parsing with deliberately different SMT numbering."""

import struct

import pytest

from benchmarks.tabarena.worker_throughput_topology import (
    parse_core_records,
    validate_layout,
)


def core_record(mask, *, group=0, pointer_size=8):
    size = 32 + pointer_size + 8
    record = bytearray(size)
    struct.pack_into("<II", record, 0, 0, size)
    record[8] = int(mask.bit_count() > 1)
    struct.pack_into("<H", record, 30, 1)
    record[32:32 + pointer_size] = mask.to_bytes(pointer_size, "little")
    struct.pack_into("<H", record, 32 + pointer_size, group)
    return bytes(record)


@pytest.mark.parametrize("pointer_size", [4, 8])
def test_actual_sibling_mapping_does_not_assume_split_halves(pointer_size):
    data = core_record(3, pointer_size=pointer_size) + core_record(12, pointer_size=pointer_size)
    cores = parse_core_records(data, pointer_size=pointer_size)
    topology = {"cores": cores, "allowed_logical_cpus": [0, 1, 2, 3]}
    assert validate_layout(topology, [[0, 2], [1, 3]]) == [
        {"slot": 0, "logical_cpus": [0, 2], "physical_cores": [0, 1]},
        {"slot": 1, "logical_cpus": [1, 3], "physical_cores": [0, 1]},
    ]
    assert [row["physical_cores"] for row in validate_layout(topology, [[i] for i in range(4)])] == [
        [0], [0], [1], [1]
    ]


def test_distinct_processor_groups_preserve_their_identity():
    cores = parse_core_records(core_record(3) + core_record(3, group=1))
    assert cores[1]["processors"] == [{"group": 1, "logical_cpu": 0},
                                      {"group": 1, "logical_cpu": 1}]


@pytest.mark.parametrize("data", [b"", b"\0" * 7, core_record(3)[:-1],
                                    core_record(0), core_record(3) + core_record(2)])
def test_invalid_or_overlapping_windows_records_are_rejected(data):
    with pytest.raises(ValueError):
        parse_core_records(data)


@pytest.mark.parametrize("slots", [[], [[0, 1], [1, 2]], [[0, 1]], [[0, 1], [2, 4]],
                                     [[0, 1], [], [2, 3]], [[False, 1], [2, 3]]])
def test_layout_must_be_a_complete_disjoint_partition(slots):
    topology = {"cores": parse_core_records(core_record(3) + core_record(12)),
                "allowed_logical_cpus": [0, 1, 2, 3]}
    with pytest.raises(ValueError, match="exactly once"):
        validate_layout(topology, slots)
