"""Prospective immutable resources and metadata-only CV allocation bounds."""
from __future__ import annotations
import math
import time


def validate_resources(values):
    integer_keys = ("cpu", "maximum_named_array_bytes", "maximum_job_rss_bytes", "headroom_reserve_bytes")
    if any(type(values.get(key)) is not int or values[key] < (0 if key == "cpu" else 1)
           for key in integer_keys):
        raise ValueError("An explicit nonnegative CPU and positive integer byte limits/reserve are required")
    for key in ("maximum_job_seconds", "headroom_wait_seconds", "headroom_poll_seconds"):
        if type(values.get(key)) not in (int, float) or not math.isfinite(values[key]) or values[key] <= 0:
            raise ValueError("Explicit finite positive wall/wait/poll limits are required")
    if (values["maximum_job_seconds"] > 1200 or values["headroom_poll_seconds"] > values["headroom_wait_seconds"]
            or values["maximum_named_array_bytes"] >= values["maximum_job_rss_bytes"]):
        raise ValueError("Wall time must be <=1200s; named arrays must leave room below RSS; poll must fit wait")
    if values.get("workers", 1) != 1 or values.get("threads_per_fit", 1) != 1:
        raise ValueError("Exactly one worker and one thread per fit are required")
    return {**values, "workers": 1, "threads_per_fit": 1}


def wait_for_headroom(resources, *, available, clock=time.monotonic, sleep=time.sleep):
    """Bounded pre-dispatch wait. Insufficiency is terminal, never a new budget."""
    needed = resources["maximum_job_rss_bytes"] + resources["headroom_reserve_bytes"]
    started, samples = clock(), []
    while True:
        current, now = int(available()), clock()
        samples.append({"elapsed_seconds": now - started, "available_bytes": current})
        if current >= needed:
            return {"sufficient": True, "required_available_bytes": needed, "samples": samples,
                    "process_rss_allowance_bytes": resources["maximum_job_rss_bytes"],
                    "headroom_reserve_bytes": resources["headroom_reserve_bytes"]}
        remaining = resources["headroom_wait_seconds"] - (now - started)
        if remaining <= 0:
            return {"sufficient": False, "required_available_bytes": needed, "samples": samples,
                    "process_rss_allowance_bytes": resources["maximum_job_rss_bytes"],
                    "headroom_reserve_bytes": resources["headroom_reserve_bytes"],
                    "failure": "insufficient_system_headroom"}
        sleep(min(resources["headroom_poll_seconds"], remaining))


def check_named_bounds(report, cases, jobs, limit):
    """Check existing metadata estimates against every current policy shape.

    The report budgets all scalar corrections conservatively; current identity
    profiles need no design. New17b controls may change lambda/shrinkage, but not
    the report's rows, class layout, tree capacity or requested anchor rounds.
    All three fold AND full-union profiles must fit before launch.
    """
    lookup = {case["dataset_name"]: case for case in cases}
    result = []
    for job in jobs:
        case, params = lookup[job["dataset"]], job["parameters"]
        n = (job["expanded_source"]["receipt"]["rows"] if job.get("expanded_source") else
             case["roles"]["train"]["rows"] + case["roles"]["stop"]["rows"])
        problem, library = case["problem_type"], job["library"]
        classes = case["num_classes"] if problem == "multiclass" else 1
        vector = problem == "multiclass" and params.get("multi_strategy") == "multi_output_tree"
        factor = classes if problem == "multiclass" and not vector else 1
        if params.get("num_parallel_tree", 1) != 1:
            raise ValueError("Unbudgeted parallel trees")
        if library == "lightgbm":
            leaves = int(params["num_leaves"])
            if int(params.get("max_depth", -1)) > 0:
                leaves = min(leaves, 2 ** int(params["max_depth"]))
        else:
            if int(params["max_depth"]) <= 0:
                raise ValueError("Unbounded native tree depth")
            leaves = 2 ** int(params["max_depth"])
            if int(params.get("max_leaves", 0)) > 0:
                leaves = min(leaves, int(params["max_leaves"]))
        field = {"regression": "leaf_refit", "binary": "binary_refit", "multiclass": "multiclass_refit"}[problem]
        active = job.get(field) is not None
        if active and problem != "regression" and job[field]["solver"].get("maxcor", 20) != 20:
            raise ValueError("Unbudgeted L-BFGS history")
        rows = [row for row in report["estimates"] if (row["dataset"], row["library"]) == (job["dataset"], library)]
        expected = {(multiplier, stage) for multiplier in ("0.5", "1", "2") for stage in ("fivefold_train_max", "full_union")}
        if len(rows) != 6 or {(row["multiplier"], row["stage"]) for row in rows} != expected:
            raise ValueError("Each policy needs all six frozen prospective bounds")
        for row in rows:
            requested = {"0.5": (job["rounds"] + 1) // 2, "1": job["rounds"], "2": 2 * job["rounds"]}[row["multiplier"]]
            fitting = n if row["stage"] == "full_union" else n - n // 5
            if (row["current_requested_rounds"] != job["rounds"] or row["requested_rounds"] != requested
                    or row["problem"] != problem or row["classes"] != classes or row["vector_leaves"] != vector
                    or row["fitting_rows"] != fitting or row["full_fitting_rows"] != n
                    or row["physical_trees_per_requested_round"] != factor
                    or row["maximum_native_leaves_per_tree"] != min(fitting, leaves)):
                raise ValueError("Current policy differs from prospective metadata bound")
            if active and row["named_total_bytes"] > limit:
                raise MemoryError("A prospective fold/full-union profile exceeds the fixed named-array cap")
        result.append({"dataset": job["dataset"], "library": library, "correction_active": active,
                       "maximum_named_bound_bytes": max(row["named_total_bytes"] for row in rows) if active else 0,
                       "profiles_checked": 6})
    return result
