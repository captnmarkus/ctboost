"""Audit closed selection and authorized saved binary STOP archives only.

No feature frames, model objects, regression arrays or DEV/test data are decoded.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / "benchmarks/results/improvements_20261016_followup"
OUT = REPORT / "accuracy_wave17b"
HASHES = {}


def digest(path):
    path = Path(path)
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    result = value.hexdigest()
    HASHES[str(path)] = result
    return result


def read(path):
    path = Path(path)
    digest(path)
    return json.loads(path.read_text(encoding="utf-8"))


def key(score, controls):
    assert math.isfinite(score)
    return (score, 0, 0., 0.) if controls is None else (score, 1, -controls["lambda"], controls["shrinkage"])


def candidate_files(directory, candidate):
    index = candidate["index"]
    assert read(directory / f"candidate_{index}.json") == candidate
    assert digest(directory / f"candidate_{index}.npz") == candidate["predictions_sha256"]
    assert candidate["status"] == "ok"
    evidence = candidate["evidence"]
    assert evidence["independent_sparse_raw_margin_oracle"] is True
    assert all(check["leaf_routing_exact"] is True for check in evidence["checks"].values())


def certify(solve, penalty, rows):
    assert solve["variant"] == "unscaled_lsqr_10000" and solve["penalty"] == penalty
    assert solve["iterations_limit"] == 10000 and solve["initial_coefficients"] == "zero"
    assert solve["atol"] == solve["btol"] == 1e-12 and solve["conlim"] == 1e8
    assert solve["success"] is True
    native, certificate = solve["native"], solve["certificate"]
    assert native["istop"] in (0, 1, 2, 4, 5) and 0 <= native["iterations"] <= 10000
    assert certificate["finite"] is True and certificate["certified"] is True
    assert all(math.isfinite(value) for value in certificate.values())
    gap = certificate["gradient_norm"] ** 2 / (2 * penalty)
    assert math.isclose(gap, certificate["original_gradient_dual_gap"], rel_tol=1e-12, abs_tol=1e-30)
    denominator = max(certificate["primal"], sys.float_info.epsilon * certificate["primal_zero"])
    relative = gap / denominator if denominator else 0.
    assert math.isclose(relative, certificate["relative_gap"], rel_tol=1e-12, abs_tol=1e-30)
    assert relative <= 1e-10 and certificate["normalized_original_gradient"] <= 1e-8
    assert math.isclose(math.sqrt(2 * gap / rows), certificate["fitting_prediction_rms_error_bound"], rel_tol=1e-12, abs_tol=1e-30)
    return {"status": native["istop"], "iterations": native["iterations"],
            "relative_gap": certificate["relative_gap"],
            "normalized_original_gradient": certificate["normalized_original_gradient"]}


def main():
    plan = read(OUT / "plan.json")
    assert Path(sys.executable).resolve() == Path(plan["runtime"]["python"]).resolve()
    assert platform.python_version() == plan["runtime"]["python_version"]
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "psutil")}
    assert all(version == plan["runtime"]["versions"][name] for name, version in versions.items())
    import psutil
    psutil.Process().cpu_affinity([0])
    import numpy as np
    metric_source = ROOT / "benchmarks/fair_accuracy_scout.py"
    assert digest(metric_source) == plan["dependencies"][str(metric_source)]
    metric_node = [node for node in ast.parse(metric_source.read_text(encoding="utf-8")).body
                   if isinstance(node, ast.FunctionDef) and node.name == "error"]
    assert len(metric_node) == 1
    metric_globals = {}
    exec(compile(ast.Module(body=metric_node, type_ignores=[]), str(metric_source), "exec"), metric_globals)
    frozen_error = metric_globals["error"]
    decoded_archives = []

    def binary_archive(path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            assert set(archive.files) == {"predictions", "labels", "row_ids"}
            predictions, labels, row_ids = (archive[field] for field in ("predictions", "labels", "row_ids"))
        assert predictions.ndim == labels.ndim == row_ids.ndim == 1
        assert len(predictions) == len(labels) == len(row_ids) > 0
        assert set(np.unique(labels)) == {0, 1} and len(np.unique(row_ids)) == len(row_ids)
        decoded_archives.append(str(path))
        return frozen_error(predictions, labels, "binary"), labels, row_ids

    document = read(OUT / "scalar_controls.json")
    assert plan["wave"] == "17b" and document["all_fits_successful"] is True
    assert document["plan_sha256"] == digest(OUT / "plan.json")
    assert digest(ROOT / "benchmarks/fair_accuracy_scalar_numerical_retry.py") == plan["script_sha256"]
    source_count = 0
    for path, expected in plan["dependencies"].items():
        if Path(path).suffix == ".py" or Path(path).name == "robustness_protocol_scalar_numerical.json":
            assert digest(path) == expected
            source_count += 1
    cases = {case["dataset_name"]: case for case in plan["cases"]}
    choices = {(c["dataset"], c["library"]): c for c in document["choices"]}
    jobs = {(j["dataset"], j["library"]): j for j in plan["jobs"] if cases[j["dataset"]]["problem_type"] != "multiclass"}
    assert len(choices) == len(document["choices"]) == len(jobs) == 24 and choices.keys() == jobs.keys()
    anchors = {(a["dataset"], a["library"]): a for a in plan["source_anchors"]}
    assert anchors.keys() == jobs.keys()
    for identity, anchor in anchors.items():
        assert digest(anchor["anchor_path"]) == anchor["anchor_sha256"]
        assert read(anchor["anchor_path"]) == anchor["anchor"]
        assert digest(anchor["identity_path"]) == anchor["identity_sha256"]
        model = anchor["anchor"]["model"]
        assert digest(model["model_path"]) == model["model_sha256"]
        assert anchor["anchor"]["old_scalar_corrections_disabled_before_fit"] is True
        assert anchor["anchor"]["fit"]["requested_rounds"] == jobs[identity]["rounds"]
    previous = Path(plan["previous"])
    prior = read(previous / "plan.json")
    binary_checks, shared_binary_roles = [], {}
    for reused in plan["reused_binary"]:
        identity = reused["dataset"], reused["library"]
        assert cases[identity[0]]["problem_type"] == "binary"
        record_path = Path(reused["job_path"])
        assert digest(record_path) == reused["job_sha256"]
        for path, expected in reused["artifact_sha256"].items():
            assert digest(path) == expected
        record = read(record_path)
        chosen = choices[identity]
        assert chosen == reused["choice"] == record["choice"]
        assert record["status"] == "ok" and record["anchor"] == anchors[identity]["anchor"]["model"]
        candidates = record["candidates"]
        assert len(candidates) == 8 and {c["index"] for c in candidates} == set(range(8))
        identity_error, labels, row_ids = binary_archive(anchors[identity]["identity_path"])
        if identity[0] in shared_binary_roles:
            expected_labels, expected_ids = shared_binary_roles[identity[0]]
            assert np.array_equal(labels, expected_labels) and np.array_equal(row_ids, expected_ids)
        else:
            shared_binary_roles[identity[0]] = (labels, row_ids)
        expected_choice = {"dataset": identity[0], "library": identity[1], "correction": None,
            "stop_error": identity_error, "anchor": record["anchor"], "fit": record["fit"],
            "fitting_data": record["fitting_data"]}
        best_key = key(identity_error, None)
        for candidate in candidates:
            assert candidate["controls"] == prior["grid"][candidate["index"]]
            candidate_files(record_path.parent, candidate)
            recomputed, candidate_labels, candidate_ids = binary_archive(record_path.parent / f"candidate_{candidate['index']}.npz")
            assert np.array_equal(candidate_labels, labels) and np.array_equal(candidate_ids, row_ids)
            assert recomputed == candidate["stop_error"]
            candidate_key = key(recomputed, candidate["controls"])
            if candidate_key < best_key:
                best_key = candidate_key
                expected_choice.update(stop_error=recomputed, correction={**candidate["controls"],
                    "solver": prior["solvers"]["binary"], "max_working_bytes": prior["resources"]["maximum_named_array_bytes"]})
        assert expected_choice == chosen
        binary_checks.append({"dataset": identity[0], "library": identity[1], "controls": chosen["correction"],
                              "identity_error_recomputed": identity_error,
                              "identity_and_eight_candidate_selection_reconstructed": True})
    assert len(binary_checks) == 12 and len(decoded_archives) == len(set(decoded_archives)) == 108
    regression_checks, certificates, resources = [], [], []
    expected_penalty_paths = set()
    for identity in jobs:
        if cases[identity[0]]["problem_type"] != "regression":
            continue
        records = []
        for penalty_index in range(4):
            directory = OUT / "penalty_fits" / identity[0] / identity[1] / str(penalty_index)
            expected_penalty_paths.add(directory / "job.json")
            record = read(directory / "job.json")
            assert (record["dataset"], record["library"], record["penalty_index"]) == (*identity, penalty_index)
            assert record["status"] == "ok" and record["new_native_tree_fits"] == 0
            assert record["anchor"] == anchors[identity]["anchor"]["model"]
            assert record["fitting_data"] == anchors[identity]["anchor"]["fitting_data"]
            assert record["fitting_data"]["stop_excluded"] is True
            assert record["solve"] == read(directory / "solve.json")
            penalty = plan["grid"][2 * penalty_index]["lambda"]
            certificates.append(certify(record["solve"], penalty, record["fitting_data"]["rows"]))
            memory = record["memory"]
            assert memory == read(directory / "memory.json")
            assert memory["refit_array_budget_bytes"] + memory["extra_certificate_vector_bytes"] <= plan["resources"]["maximum_named_array_bytes"]
            assert len(record["candidates"]) == 2
            assert {c["index"] for c in record["candidates"]} == {2*penalty_index, 2*penalty_index+1}
            for candidate in record["candidates"]:
                assert candidate["controls"] == plan["grid"][candidate["index"]]
                candidate_files(directory, candidate)
            resource = read(OUT / "penalty_logs" / f"{identity[0]}-{identity[1]}-{penalty_index}-resource.json")
            assert resource["exit_code"] == 0 and resource["resource_failure"] is None and resource["cleanup_complete"] is True
            assert resource["peak_rss_bytes"] <= plan["resources"]["maximum_job_rss_bytes"]
            resources.append(resource)
            records.append(record)
        assert len({r["identity_error"] for r in records}) == 1
        first = records[0]
        expected_choice = {"dataset": identity[0], "library": identity[1], "correction": None,
            "stop_error": first["identity_error"], "anchor": first["anchor"], "fit": first["fit"], "fitting_data": first["fitting_data"]}
        best_key = key(first["identity_error"], None)
        for candidate in [c for r in records for c in r["candidates"]]:
            candidate_key = key(candidate["stop_error"], candidate["controls"])
            if candidate_key < best_key:
                best_key = candidate_key
                expected_choice.update(stop_error=candidate["stop_error"], correction={**candidate["controls"],
                    "solver": plan["solvers"]["regression"], "max_working_bytes": plan["resources"]["maximum_named_array_bytes"]})
        assert expected_choice == choices[identity]
        regression_checks.append({"dataset": identity[0], "library": identity[1], "controls": expected_choice["correction"],
                                  "identity_and_eight_candidate_selection_reconstructed": True})
    assert set((OUT / "penalty_fits").glob("*/*/*/job.json")) == expected_penalty_paths and len(expected_penalty_paths) == 48
    assert len(regression_checks) == 12 and len(certificates) == len(resources) == 48
    assert document["new_anchor_fits_completed"] == 0 and document["regression_identity_prediction_checks"] == 48
    evidence = {"date": "2026-10-16", "status": "passed_selection_audit",
        "scope": "Closed selection only. Source/JSON metadata and byte hashes plus108 previously used binary STOP prediction/label/row-ID archives. No feature frames, model objects, regression arrays, DEV, confirmation or official data decoded; no fitting or model prediction.",
        "metric_runtime": {"python": sys.executable, "python_version": platform.python_version(), "versions": versions,
                           "metric_source_sha256": digest(metric_source), "cpu_affinity": psutil.Process().cpu_affinity()},
        "decoded_binary_stop_archives": decoded_archives,
        "plan_sha256": digest(OUT / "plan.json"), "controls_sha256": digest(OUT / "scalar_controls.json"),
        "source_files_verified": source_count, "anchor_models_and_identity_archives_byte_verified": 24,
        "reused_binary": binary_checks, "new_regression": regression_checks,
        "successful_new_penalty_jobs": 48, "successful_new_certified_solves": 48,
        "successful_new_nonidentity_evaluations": 96, "reused_nonidentity_evaluations": 96,
        "maximum_iterations": max(c["iterations"] for c in certificates),
        "maximum_relative_gap": max(c["relative_gap"] for c in certificates),
        "maximum_normalized_original_gradient": max(c["normalized_original_gradient"] for c in certificates),
        "maximum_worker_seconds": max(r["wall_seconds"] for r in resources),
        "maximum_peak_rss_bytes": max(r["peak_rss_bytes"] for r in resources),
        "resource_failures": 0, "new_model_training_runs": 0,
        "limitations": "Regression certificates and identity/candidate STOP errors are checked from recorded evidence, not recomputed from Z/targets/predictions. Binary metrics are independently recomputed from saved STOP archives with the exact frozen metric/runtime. No union/DEV outcome follows from selection success.",
        "audit_source_sha256": digest(__file__), "verified_file_sha256": HASHES}
    target = Path(__file__).with_name("wave17b_selection_independent_review.json")
    assert not target.exists()
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({key: evidence[key] for key in ("status", "source_files_verified", "successful_new_penalty_jobs", "maximum_iterations", "controls_sha256")}))


if __name__ == "__main__":
    main()
