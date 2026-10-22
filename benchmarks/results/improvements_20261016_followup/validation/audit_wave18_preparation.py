"""Read only frozen wave-18 preparation metadata, sources and hashes."""
import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
ROOT = HERE.parents[3]
OUT = REPORT / "accuracy_wave18"
HASHES = {}


def digest(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    HASHES[str(path)] = h.hexdigest()
    return h.hexdigest()


def read(path):
    digest(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def index_hash(indices):
    return hashlib.sha256(struct.pack("<" + "q" * len(indices), *indices)).hexdigest()


def keyed(rows):
    result = {(row["dataset"], row["library"]): row for row in rows}
    assert len(result) == len(rows)
    return result


def main():
    plan = read(OUT / "plan.json")
    manifest = read(OUT / "fold_manifest.json")
    prior_dir = Path(plan["previous"])
    prior, frozen, development = (read(prior_dir / filename) for filename in ("plan.json", "frozen_selections.json", "development.json"))
    terminal = read(HERE / "wave17b_terminal_independent_review.json")
    assert plan["wave"] == 18 and plan["protocol"] == "full_current_fivefold_round_cv_20261016"
    assert plan["seed"] == 47 and plan["conditional_seed"] == 677 and plan["split_seed"] == 18018
    assert plan["admission"] == "primary47_failed" and plan["failed_seed"] is None
    assert terminal["status"] == "passed_closed_terminal_metadata_audit"
    assert terminal["development_sha256"] == digest(prior_dir / "development.json")
    assert terminal["frozen_selections_sha256"] == digest(prior_dir / "frozen_selections.json")
    assert terminal["plan_sha256"] == digest(prior_dir / "plan.json")
    assert frozen["all_fits_successful"] is True and development["development_target_met"] is False
    assert plan["runtime"] == prior["runtime"] and plan["scipy_version"] == prior["scipy_version"]
    assert plan["jobs"] == frozen["jobs"] and plan["cases"] == prior["cases"]
    assert plan["expanded_data"] == prior["expanded_data"] and len(plan["expanded_data"]) == 4
    assert all(doc["official_test_scored"] is False for doc in (plan, prior, frozen, development))
    assert plan["fold_manifest_sha256"] == digest(OUT / "fold_manifest.json")
    assert plan["script_sha256"] == digest(ROOT / "benchmarks/fair_accuracy_current_round_cv.py")
    source_count = 0
    for path, expected in plan["dependencies"].items():
        # Metadata/source evidence only; do not load role/model payloads or current fit receipts.
        if Path(path).suffix in (".py", ".json"):
            assert digest(path) == expected
            source_count += 1
    resources = plan["resources"]
    assert resources == {"cpu": 4, "headroom_poll_seconds": 5., "headroom_reserve_bytes": 4*2**30,
        "headroom_wait_seconds": 300., "maximum_job_rss_bytes": 10*2**30, "maximum_job_seconds": 1200.,
        "maximum_named_array_bytes": 6*2**30, "threads_per_fit": 1, "workers": 1}
    required_memory = resources["maximum_job_rss_bytes"] + resources["headroom_reserve_bytes"]
    assert plan["quiet_baseline_memory"]["available_bytes"] >= required_memory
    memory = read(REPORT / "full_current_cv_memory/estimates.json")
    jobs, bounds = keyed(plan["jobs"]), keyed(plan["memory_bounds"])
    cases = {c["dataset_name"]: c for c in plan["cases"]}
    manifests = {m["dataset"]: m for m in manifest}
    assert len(jobs) == len(bounds) == 36 and bounds.keys() == jobs.keys()
    assert len(manifests) == len(manifest) == len(cases) == 12 and manifests.keys() == cases.keys()
    all_estimates = memory["estimates"]
    assert len(all_estimates) == 216
    audit_rows, active_maximum, profile_count = [], 0, 0
    for identity, job in jobs.items():
        case, bound = cases[identity[0]], bounds[identity]
        field = {"binary": "binary_refit", "regression": "leaf_refit", "multiclass": "multiclass_refit"}[case["problem_type"]]
        active = job.get(field) is not None
        rows = [row for row in all_estimates if (row["dataset"], row["library"]) == identity]
        assert len(rows) == 6 and {(r["multiplier"], r["stage"]) for r in rows} == {(m,s) for m in ("0.5","1","2") for s in ("full_union","fivefold_train_max")}
        n = job["expanded_source"]["receipt"]["rows"] if job.get("expanded_source") else sum(case["roles"][r]["rows"] for r in ("train", "stop"))
        for row in rows:
            counts = {"0.5": (job["rounds"]+1)//2, "1": job["rounds"], "2": job["rounds"]*2}
            assert row["current_requested_rounds"] == job["rounds"] and row["requested_rounds"] == counts[row["multiplier"]]
            assert row["full_fitting_rows"] == n and row["fitting_rows"] == (n if row["stage"] == "full_union" else n-n//5)
            assert row["named_total_bytes"] == sum(row["named_array_budgets_bytes"].values())
            assert row["physical_trees_upper_bound"] == row["physical_trees_per_requested_round"] * row["requested_rounds"]
            if active:
                assert row["named_total_bytes"] <= resources["maximum_named_array_bytes"]
                active_maximum = max(active_maximum, row["named_total_bytes"])
        assert bound == {"dataset": identity[0], "library": identity[1], "correction_active": active,
            "maximum_named_bound_bytes": max(r["named_total_bytes"] for r in rows) if active else 0, "profiles_checked": 6}
        profile_count += len(rows)
    for dataset, m in manifests.items():
        case, data = cases[dataset], m["data"]
        assert data["normalization_performed"] is False
        assert data["source_roles"] == {r: case["roles"][r] for r in ("train", "stop")}
        assert data["rows"] == sum(data[k] for k in ("original_train_rows", "original_stop_rows", "outside_cap_rows"))
        assert data["ordered_ids_sha256"] == m["ordered_union_ids_sha256"] and m["split_seed"] == 18018
        assignments = m["assignments"]
        assert len(assignments) == data["rows"] and set(assignments) == set(range(5))
        assert len(m["folds"]) == 5 and {f["fold"] for f in m["folds"]} == set(range(5))
        totals, validation_totals = [], None
        for fold in m["folds"]:
            train = [i for i,a in enumerate(assignments) if a != fold["fold"]]
            validation = [i for i,a in enumerate(assignments) if a == fold["fold"]]
            assert len(train) == fold["train_rows"] and len(validation) == fold["validation_rows"]
            assert index_hash(train) == fold["train_index_sha256"] and index_hash(validation) == fold["validation_index_sha256"]
            assert type(fold["train_category_schema"]) is dict
            assert set(fold["train_category_schema"]).issubset(set(m["feature_names"]))
            if case["problem_type"] != "regression":
                a,b = fold["train_class_counts"], fold["validation_class_counts"]
                assert len(a) == len(b) == case["num_classes"] and min(a) > 0 and min(b) > 0
                assert sum(a) == len(train) and sum(b) == len(validation)
                totals.append([x+y for x,y in zip(a,b)])
                validation_totals = b[:] if validation_totals is None else [x+y for x,y in zip(validation_totals,b)]
            else:
                assert fold["train_class_counts"] is fold["validation_class_counts"] is None
        if totals:
            assert all(total == totals[0] for total in totals) and validation_totals == totals[0]
        audit_rows.append({"dataset": dataset, "authorized_fitting_rows": data["rows"], "outside_cap_rows": data["outside_cap_rows"],
            "class_counts": totals[0] if totals else None, "all_five_assignment_hashes_reconstructed": True})
    assert profile_count == 216 and active_maximum == 4515168888
    evidence = {"status": "passed_frozen_preparation_metadata_audit", "date": "2026-10-16",
        "scope": "Frozen preparation JSON/source/hash evidence only. No feature, target, model or prediction payload decoded; no fit/scoring/current-worker receipt inspected.",
        "plan_sha256": digest(OUT / "plan.json"), "manifest_sha256": digest(OUT / "fold_manifest.json"),
        "admission": plan["admission"], "verified_source_and_metadata_files": source_count,
        "shared_fold_datasets": audit_rows, "prospective_profiles_verified": profile_count,
        "maximum_active_named_bound_bytes": active_maximum, "resources": resources,
        "quiet_available_bytes": plan["quiet_baseline_memory"]["available_bytes"], "required_available_bytes": required_memory,
        "planned_fold_fits": 540, "planned_mandatory_union_fits": 36,
        "limitations": "Class counts, row-ID digests and vocabularies are source-bound preparation receipts, not recomputed from raw targets/features. Assignment ordinal hashes and class-count consistency are independently recomputed. Named-array bounds exclude native/input/allocator overhead; process RSS and bounded pre-dispatch headroom remain authoritative. No outcome is inferred from preparation success.",
        "audit_source_sha256": digest(__file__), "verified_file_sha256": HASHES}
    target = HERE / "wave18_preparation_independent_review.json"
    assert not target.exists()
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({k:evidence[k] for k in ("status","plan_sha256","prospective_profiles_verified","maximum_active_named_bound_bytes")}))


if __name__ == "__main__":
    main()
