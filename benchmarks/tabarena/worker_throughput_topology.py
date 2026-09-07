"""Read Windows CPU topology and validate the two local throughput layouts.

This reports processor-core relationships, without changing process affinity
or assuming how Windows numbers simultaneous multithreading siblings.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import struct
import sys
from pathlib import Path


def parse_core_records(data, *, pointer_size=struct.calcsize("P")):
    """Parse RelationProcessorCore records returned by Windows, including groups."""
    if pointer_size not in (4, 8):
        raise ValueError("Unsupported Windows affinity mask size")
    cores, seen, offset = [], set(), 0
    while offset < len(data):
        if len(data) - offset < 32:
            raise ValueError("Truncated processor-core record")
        relation, size = struct.unpack_from("<II", data, offset)
        groups = struct.unpack_from("<H", data, offset + 30)[0]
        if relation != 0 or not groups or size != 32 + groups * (pointer_size + 8):
            raise ValueError("Invalid processor-core relationship or record size")
        if offset + size > len(data):
            raise ValueError("Truncated processor-core group masks")
        processors = []
        for index in range(groups):
            start = offset + 32 + index * (pointer_size + 8)
            mask = int.from_bytes(data[start:start + pointer_size], "little")
            group = struct.unpack_from("<H", data, start + pointer_size)[0]
            if not mask:
                raise ValueError("Empty processor-core affinity mask")
            for cpu in range(pointer_size * 8):
                if mask & (1 << cpu):
                    if (group, cpu) in seen:
                        raise ValueError("Logical processor belongs to multiple cores")
                    seen.add((group, cpu))
                    processors.append({"group": group, "logical_cpu": cpu})
        cores.append({"core_index": len(cores), "smt": bool(data[offset + 8] & 1),
                      "efficiency_class": data[offset + 9], "processors": processors})
        offset += size
    if not cores:
        raise ValueError("No processor-core relationships returned")
    return cores


def collect_topology():
    if sys.platform != "win32":
        raise RuntimeError("This topology reader requires Windows")
    import psutil

    api = ctypes.WinDLL("kernel32", use_last_error=True).GetLogicalProcessorInformationEx
    api.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    api.restype = ctypes.c_int
    size = ctypes.c_uint32()
    if api(0, None, ctypes.byref(size)) or ctypes.get_last_error() != 122:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(size.value)
    if not api(0, buffer, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    cores = parse_core_records(buffer.raw[:size.value])
    if any(p["group"] != 0 for core in cores for p in core["processors"]):
        raise ValueError("This single-machine screen supports one Windows processor group")
    allowed = sorted(psutil.Process().cpu_affinity())
    logical = sorted(p["logical_cpu"] for core in cores for p in core["processors"])
    if not set(allowed) <= set(logical) or len(logical) != psutil.cpu_count(logical=True):
        raise ValueError("Windows core records disagree with psutil processor visibility")
    return {"schema_version": 1, "source": "GetLogicalProcessorInformationEx",
            "physical_core_count": len(cores), "logical_cpu_count": len(logical),
            "allowed_logical_cpus": allowed, "cores": cores}


def validate_layout(topology, slots):
    """Require each allowed logical CPU exactly once; report actual core membership."""
    flat = [cpu for slot in slots for cpu in slot]
    if (not slots or any(not slot for slot in slots)
            or any(type(cpu) is not int for cpu in flat)
            or len(flat) != len(set(flat))
            or set(flat) != set(topology["allowed_logical_cpus"])):
        raise ValueError("Slots must cover the allowed logical CPUs exactly once")
    core_of = {p["logical_cpu"]: core["core_index"]
               for core in topology["cores"] for p in core["processors"] if p["group"] == 0}
    return [{"slot": index, "logical_cpus": list(slot),
             "physical_cores": [core_of[cpu] for cpu in slot]}
            for index, slot in enumerate(slots)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    topology = collect_topology()
    if topology["allowed_logical_cpus"] != list(range(16)):
        raise ValueError("The frozen baseline requires logical CPU IDs 0 through 15")
    topology["layouts"] = {
        "8x2_current": validate_layout(topology, [[i, i + 8] for i in range(8)]),
        "16x1_candidate": validate_layout(topology, [[i] for i in range(16)]),
    }
    report = json.dumps(topology, indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(report)
    print(report, end="")


if __name__ == "__main__":
    main()
