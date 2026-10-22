"""Metadata-only prospective resource table; no model/data imports or fitting."""
from __future__ import annotations

import ast
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REPORTS = HERE.parent
GIB = 1024**3


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pure_function(path, name, namespace=None):
    # Execute only the reviewed arithmetic function, never module imports,
    # native exports or dataset/model loaders.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    scope = {} if namespace is None else dict(namespace)
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


def atomic(path, text):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    paths = {
        "complete_policy": REPORTS / "accuracy_wave16/frozen_selections.json",
        "inherited_controls_and_row_manifests": REPORTS / "accuracy_wave17/plan.json",
        "terminal_wave17_resources": REPORTS / "accuracy_wave17/summary.json",
        "scalar_array_formula": ROOT / "benchmarks/fixed_clean_refit_check.py",
        "multiclass_array_formula": ROOT / "benchmarks/fixed_multiclass_leaf_refit.py",
        "new_regression_vector_reserve": ROOT / "benchmarks/ridge_numerical_diagnostic.py",
        "generator": Path(__file__),
    }
    plan, complete = read(paths["inherited_controls_and_row_manifests"]), read(paths["complete_policy"])
    summary = read(paths["terminal_wave17_resources"])
    cases = {c["dataset_name"]: c for c in plan["cases"]}
    jobs = {(j["dataset"], j["library"]): j for j in plan["jobs"]}
    saved_jobs = {(j["dataset"], j["library"]): j for j in complete["jobs"]}
    selections = {(s["dataset"], s["library"]): s for s in complete["selections"]}
    assert len(cases) == 12 and len(jobs) == len(saved_jobs) == len(selections) == 36
    # The failed refresh has no authority to supply new selected controls.
    for key, job in jobs.items():
        old = saved_jobs[key]
        for field in ("rounds", "parameters", "leaf_refit", "binary_refit", "multiclass_refit", "temperature", "expanded_source"):
            assert job.get(field) == old.get(field), (key, field)
    scalar = pure_function(paths["scalar_array_formula"], "memory_preflight",
        {"_snapshot": lambda payload: {"native_leaf_counts": payload["counts"]}})
    multiclass = pure_function(paths["multiclass_array_formula"], "sparse_memory_estimate")
    rows, compact = [], []
    for key, job in jobs.items():
        dataset, library = key
        case, parameters = cases[dataset], job["parameters"]
        problem, classes = case["problem_type"], case["num_classes"]
        n = (job["expanded_source"]["receipt"]["rows"] if job.get("expanded_source")
             else case["roles"]["train"]["rows"] + case["roles"]["stop"]["rows"])
        fold_min, fold_max = n - math.ceil(n / 5), n - n // 5
        rounds = job["rounds"]
        assert selections[key]["refit_details"]["requested_rounds"] == rounds
        retained = selections[key]["refit_details"]["retained_rounds"]
        assert int(parameters.get("num_parallel_tree", 1)) == 1
        if library == "lightgbm":
            capacity = int(parameters["num_leaves"])
            if int(parameters.get("max_depth", -1)) > 0:
                capacity = min(capacity, 2 ** int(parameters["max_depth"]))
        else:
            depth = int(parameters["max_depth"])
            assert depth > 0
            capacity = 2**depth
            if int(parameters.get("max_leaves", 0)) > 0:
                capacity = min(capacity, int(parameters["max_leaves"]))
        vector = problem == "multiclass" and parameters.get("multi_strategy") == "multi_output_tree"
        factor = classes if problem == "multiclass" and not vector else 1
        profile_rounds = [max(1, math.ceil(rounds / 2)), rounds, 2 * rounds]
        active = bool(job.get({"binary": "binary_refit", "regression": "leaf_refit", "multiclass": "multiclass_refit"}[problem]))
        entries = []
        for profile, requested in zip(("0.5", "1", "2"), profile_rounds):
            trees = requested * factor
            for stage, count in (("fivefold_train_max", fold_max), ("full_union", n)):
                leaf_counts = [capacity] * trees
                if problem == "multiclass":
                    supports = tuple([-1] * trees if vector else [i % classes for i in range(trees)])
                    estimate = multiclass(count, classes, supports, leaf_counts)
                else:
                    estimate = scalar({"problem": problem, "counts": leaf_counts}, count,
                                      max_working_bytes=2**100, solver_kind=problem)
                arrays = dict(estimate["array_budgets_bytes"])
                if problem == "regression":
                    arrays["new_augmented_solver_and_certificate_extra_reserve"] = 256 * (count + estimate["parameters_upper_bound"])
                total = sum(arrays.values())
                entry = dict(dataset=dataset, library=library, problem=problem,
                    current_requested_rounds=rounds, current_retained_rounds_metadata=retained,
                    multiplier=profile, requested_rounds=requested, stage=stage,
                    fitting_rows=count, full_fitting_rows=n, fivefold_training_rows_min=fold_min,
                    fivefold_training_rows_max=fold_max, physical_trees_upper_bound=trees,
                    physical_trees_per_requested_round=factor, maximum_native_leaves_per_tree=capacity,
                    classes=classes if problem == "multiclass" else 1, vector_leaves=vector,
                    unique_csr_matrices=(1 if vector else classes) if problem == "multiclass" else 1,
                    current_fixed_correction_nonzero=active,
                    prospective_scalar_refresh_assumes_nonzero=problem != "multiclass",
                    correction_allocation_required_for_planning=active or problem != "multiclass",
                    parameters_upper_bound=estimate["parameters_upper_bound"],
                    csr_nonzeros=estimate["csr_nonzeros"], named_array_budgets_bytes=arrays,
                    named_total_bytes=total, named_total_gib=total/GIB,
                    largest_named_array_term=max(arrays, key=arrays.get),
                    largest_named_array_term_bytes=max(arrays.values()),
                    hypothetical_if_current_multiclass_identity=problem == "multiclass" and not active)
                rows.append(entry)
                entries.append(entry)
        compact.append(dict(dataset=dataset, library=library, problem=problem, classes=classes,
            rows=n, fold_min=fold_min, fold_max=fold_max, rounds=profile_rounds, factor=factor,
            vector=vector, active=active, retained=retained,
            full=[e["named_total_gib"] for e in entries if e["stage"] == "full_union"],
            fold=[e["named_total_gib"] for e in entries if e["stage"] == "fivefold_train_max"]))
    assert len(rows) == 216
    applicable = [e for e in rows if e["correction_allocation_required_for_planning"]]
    result = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose="prospective metadata-only capacity estimate, not an experiment plan or launch authorization",
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths.values()},
        source_consistency="wave17 inherited jobs equal complete wave16 controls; no partial wave17 choices used",
        selected_metadata_sources_only=True, model_or_data_payloads_loaded=0, fits=0, score_computations=0,
        existing_wave17_maximum_process_tree_rss_bytes=summary["maximum_process_tree_rss_bytes"],
        existing_wave17_maximum_named_array_estimate_bytes=summary["maximum_named_array_estimate_bytes"],
        existing_wave17_maximum_job_seconds=summary["maximum_job_seconds"],
        max_applicable_full=max((e for e in applicable if e["stage"] == "full_union"), key=lambda e:e["named_total_bytes"]),
        max_applicable_fold=max((e for e in applicable if e["stage"] == "fivefold_train_max"), key=lambda e:e["named_total_bytes"]),
        estimates=rows)
    atomic(HERE / "estimates.json", json.dumps(result, indent=2)+"\n")
    lines = ["# Prospective full-current-data round CV: memory", "",
        "16 October 2026. Metadata-only resource review; **no policy freeze, fit, score or data/model payload read**. "
        "Counts and controls come from the latest complete wave-16 policy, cross-checked against the inherited wave-17 job manifest. "
        "Wave 17 failed numerically and supplies no replacement selected controls.", "",
        "## Decision", "",
        "A single worker is the appropriate starting scope, but 4 GiB of named refit arrays is insufficient for every potential final union refit. "
        "The worst case is diamonds/LightGBM at 2,390 requested rounds. "
        f"Its bound is **{result['max_applicable_full']['named_total_gib']:.3f} GiB** on all 33,460 fitting rows, versus "
        f"**{result['max_applicable_fold']['named_total_gib']:.3f} GiB** on a fivefold training partition. "
        "A 6 GiB named-array budget would cover the calculated cases; this is a feasibility observation, not an adopted limit. "
        "An 8 GiB whole-process ceiling cannot be guaranteed from named arrays alone. Do not copy the old limits blindly or relax them after failure.", "",
        "## All 36 policies", "",
        "Every triple is ordered **0.5 / 1 / 2** times the current requested count; half uses ceiling and minimum one. "
        "GiB means 2^30 bytes. Values sum conservative named-allocation budgets, not simultaneously measured RSS. "
        "`S` is scalar; `V(K)` shares one CSR across K class blocks; `S×K` uses K class-specific CSRs and K physical trees per round. "
        "`I` marks an inherited multiclass identity: its displayed bound is hypothetical and no correction design is needed under fixed identity. "
        "Scalar bounds assume a prospective refresh chooses a nonzero correction, even where the current policy is identity.", "",
        "| Task | Library / layout | Full N; fold train N | Requested rounds (½/1/2) | Full named GiB (½/1/2) | Fold named GiB (½/1/2) |",
        "|---|---|---:|---:|---:|---:|"]
    short={"blood-transfusion-service-center":"blood", "in_vehicle_coupon_recommendation":"coupons", "Another-Dataset-on-used-Fiat-500":"Fiat500", "QSAR_fish_toxicity":"QSAR", "GiveMeSomeCredit":"Credit"}
    for c in compact:
        layout = (f"V({c['classes']})" if c['vector'] else f"S×{c['classes']}") if c['problem']=="multiclass" else "S"
        if c['problem']=="multiclass" and not c['active']:
            layout += " I"
        fold = str(c['fold_min']) if c['fold_min']==c['fold_max'] else f"{c['fold_min']}–{c['fold_max']}"
        triple=lambda values: " / ".join("<0.001" if 0 < v < .0005 else f"{v:.3f}" for v in values)
        lines.append(f"| {short.get(c['dataset'],c['dataset'])} | {c['library']} {layout} | {c['rows']}; {fold} | {' / '.join(map(str,c['rounds']))} | {triple(c['full'])} | {triple(c['fold'])} |")
    lines += ["", "All current retained counts equal their requested counts in the saved refit receipts. "
        "Future folds may structurally stop earlier, especially LightGBM, so the table budgets the requested count. "
        "Old nested correction receipts can contain historical tree counts and were deliberately not used as current model-size evidence.", "",
        "## Accounting and uncertainty", "",
        "Let n be fitting rows, T physical trees, q=nT, L the sum of observed leaf columns, and P the coefficient count. "
        "For prospective bounds, each tree has at most min(n, 2^depth) leaves for CTBoost/XGBoost, or min(n, num_leaves) for LightGBM; "
        "any explicit additional depth/leaf cap is intersected. No minimum-leaf heuristic is treated as a hard bound. "
        "The manifest has no parallel-tree multiplicity above one. Actual topology and observed leaves remain unknown until fitting.", "",
        "The common leading budgets are 16q bytes for native leaf IDs/copy, 16q for resident CSR data/indices, and 24q for design construction: "
        "**56nT bytes combined**. Maps add 8L. CSR row pointers add 8(n+1) per distinct matrix; construction adds 32n. "
        "Native margins/copy add 16n for scalar or 16nK for multiclass. "
        "For scalar models P=L; multiclass scalar class blocks partition the trees and also have P=L; vector leaves share one CSR but have P=KL.", "",
        "Regression includes the existing 64(n+P) LSQR vector budget **plus 256(n+P)** for the prospective augmented operator/certificate. "
        "Binary L-BFGS-B includes 96n objective workspace, 8(45P+4,560)+12P optimizer work arrays (history 20), and 100P vector reserve. "
        "Multiclass uses the same optimizer terms and 8n(8K+4) objective workspace. "
        "The generator executes only the pure arithmetic function ASTs from the frozen helpers and adds the explicitly named new regression reserve; "
        "it does not import native modules or instantiate payloads. [The JSON table](estimates.json) contains all 216 stage/profile rows, term-by-term bytes and source hashes.", "",
        "For diamonds/LightGBM at double rounds/full union, q=79,969,400: the three leading terms alone consume 4.171 GiB. "
        "Right column preconditioning is matrix-free and must not create another full CSR or a dense Gram matrix. "
        "Bounds are deliberately loose when trees are shallow/root-only or leaves are unobserved. "
        "They do not cover native fitting histograms, model objects, input frames, raw-fold vocabulary/CTR workspaces, test predictions, "
        "Python containers, BLAS caches, serialization copies or allocator retention. Named totals therefore are neither an RSS bound nor a native-training memory estimate.", "",
        "## Host and empirical context", "",
        f"Wave 17's largest completed worker used **{summary['maximum_process_tree_rss_bytes']/GIB:.3f} GiB RSS**; "
        f"its largest named estimate was **{summary['maximum_named_array_estimate_bytes']/GIB:.3f} GiB**, and its longest job took "
        f"{summary['maximum_job_seconds']:.1f} seconds. These are separate maxima, not a paired overhead factor, and only TRAIN anchors ran. "
        "The numerical failure occurred despite staying within resource limits; additional memory cannot establish convergence.", "",
        "A contemporaneous OS metadata snapshot reported 31.925 GiB visible and 14.100 GiB free physical memory. "
        "Other diagnostic/build activity can change availability; this is not a quiet baseline or a reserved capacity guarantee. "
        "Before any new freeze, obtain a no-build/no-timing baseline and predeclare a one-worker RSS limit plus independent system-headroom preflight. "
        "A prospective 6 GiB named / 8–12 GiB owned-process envelope is arithmetically plausible on this host, but the higher RSS end requires "
        "sufficient available RAM and a fixed additional headroom reserve before dispatch. A bounded wait or terminal insufficient-memory outcome must be decided before running. "
        "Keep the owned-process-tree watchdog; do not claim that this table validates an 8 GiB ceiling.", "",
        "Fivefold × three independent counts × 36 policies means **540 fold fits plus 36 mandatory union fits**, before any conditional seed replication. "
        "Do not multiply a one-job memory estimate by the fit count: workers must execute serially and release models/designs between jobs. "
        "No early stopping or slicing is assumed. Every fold needs raw fold-local normalization and TRAIN-fold-only correction solving at its actual count; "
        "fixed multiclass temperature is applied once. Preserve the four expanded-source exclusions and full class vocabulary checks. "
        "Numerical solver adoption and resource limits require separate prospective decisions; this report makes neither.", ""]
    atomic(HERE / "README.md", "\n".join(lines))
    print(json.dumps({"rows":len(rows),"max_full_gib":result['max_applicable_full']['named_total_gib'],"max_fold_gib":result['max_applicable_fold']['named_total_gib'],"artifacts":[str(HERE/'README.md'),str(HERE/'estimates.json')]}))


if __name__ == "__main__":
    main()
