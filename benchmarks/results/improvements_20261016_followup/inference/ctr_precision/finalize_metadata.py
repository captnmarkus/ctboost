"""Bind final evidence, including repaired failures and explicit skips."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(r"C:/apps/ctboost")
WORK = ROOT / ".tmp/improvements-20261016/ctr-precision"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((WORK / name).read_text())


full_log = (WORK / "full-consistent-cp310.log").read_text()
assert "41 failed, 2484 passed, 125 skipped" in full_log
assert "55 passed" in (WORK / "controllers-two-slots.log").read_text()
assert "56 passed" in (WORK / "normal-pytest-affinity-isolation.log").read_text()
skips = next(json.loads(line)["validation_only_optional_dependency_skips"]
             for line in full_log.splitlines() if line.startswith('{"validation_only_optional_dependency_skips"'))
assert len(skips) == 19
(WORK / "scientific-stack-skiplist.json").write_text(json.dumps({
    "count":len(skips),"nodes":skips,
    "reason":"CP310 lacks the complete supported comparison stack. Existing isolated LGB4.7 imports but toy fitting raises access violations; XGB3.2 differs from frozen3.4 vector layout.",
    "alternative_validation":"All34 cases across the same four modules passed under CP312 NumPy2/XGB3.4.1/LGB4.7."},indent=2)+"\n")
pipeline_reference = read("legacy-reference.json")
real_reference = read("real-model-reference.json")
real_verified = read("real-model-verified.json")
assert len(pipeline_reference["records"]) == 16
assert real_reference["records"] == real_verified["records"] and len(real_verified["records"]) == 12
assert real_verified["all_exact"]
native_snapshot = read("native-source-hashes.json")
assert all(digest(ROOT / relative) == value for relative,value in native_snapshot.items())
frozen_accuracy_native = ROOT / ".tmp/improvements-20261016/followup/runtime/ctboost/_core.cp312-win_amd64.pyd"
assert digest(frozen_accuracy_native) == "c5ea7e026942c3f1190ca5511fbf74f353bfeb08e7771f76bc5f6c1ca3e4d197"
builds = {}
for abi in ("cp310","cp312"):
    receipt = read(f"runtime-{abi}/build_receipt.json")
    assert digest(ROOT / f"ctboost/_core.{abi}-win_amd64.pyd") == receipt["native_sha256"]
    assert all(digest(ROOT / "ctboost" / relative) == value for relative,value in receipt["python_sha256"].items())
    builds[abi] = {"package":receipt["package"],"native_sha256":receipt["native_sha256"],
                   "build_receipt_sha256":digest(WORK / f"runtime-{abi}/build_receipt.json")}
packaging = read("sdist-membership.json")
assert "tests/conftest.py" in packaging["checked_members"]
assert packaging["verification_script_sha256"] == digest(WORK / "verify_sdist.py")
receipt = {
    "status":"numerical_fix_validated_with_explicit_test_repairs_and_skips",
    "production_scope":"Mean CTR precision, format5 and legacy compatibility only; tree training principle/objectives/metrics unchanged",
    "builds":builds,"native_source_manifest_sha256":digest(WORK / "native-source-hashes.json"),
    "native_source_entries":len(native_snapshot),"current_source_equals_built_source":True,
    "frozen_accuracy_native_sha256":digest(frozen_accuracy_native),
    "initial_full_setup_diagnostic":{"passed":2442,"failed":59,"skipped":106,
        "failures":{"missing_comparison_dependency":19,"insufficient_or_leaked_cpu_affinity":40},
        "subprocess_native_provenance":"previous checkout native; this is not full candidate coverage"},
    "consistent_full_run":{"passed":2484,"failed":41,"skipped":125,
        "failure_causes":{"distributed_format_expectation":1,"leaked_affinity_controller_preflight":40},
        "collection_limit":"Latest ridge-diagnostic test source changed during this run; its latest revision has separate validation."},
    "repairs":{"distributed_format5_cp310":{"passed":1},"distributed_format5_cp312":{"passed":1},
        "controllers_two_worker_slots":{"passed":55,"cpus":[0,2,6,8]},
        "normal_pytest_affinity_isolation":{"passed":56,"external_validation_plugin":False,
            "fixture":"tests/conftest.py","fixture_sha256":digest(ROOT / "tests/conftest.py"),
            "sequence":"known CPU4-setting identity transfer followed by both controller modules; no initial affinity override"}},
    "combined_full_collection_passing_coverage":2525,"full_collection_explicit_skips":125,
    "single_zero_failure_full_run_claim":False,"scientific_stack_validation":{"cp312_passed":34,"cp310_validation_only_skipped":19},
    "cp312_targeted":{"focused":{"passed":103,"skipped":3},"wider":{"passed":125},
        "final_precision":{"passed":38},"subprocess_exports_before_expectation_repair":{"passed":73,"failed":1,"skipped":3},
        "counts_overlap":True},
    "native_checks":{"cp310":{"checks":230,"failures":0},"cp312":{"checks":230,"failures":0}},
    "legacy_oracle":{"pipeline_cases":16,"old_model_and_warmstart":True,"all_bytes_and_states_exact":True,
        "reference_sha256":digest(WORK / "legacy-reference.json")},
    "saved_real_models":{"models":12,"source":"frozen wave16 CT models","input_scope":"originalTRAIN features; full frame and1/64/1000 batches",
        "fits":0,"scores":0,"confirmation_or_official_test_opened":False,
        "reference_sha256":digest(WORK / "real-model-reference.json"),"verified_sha256":digest(WORK / "real-model-verified.json")},
    "sdist":{"artifact_sha256":packaging["artifact_sha256"],"members":packaging["archive_member_count"],
        "checked_members":len(packaging["checked_members"]),"receipt_sha256":digest(WORK / "sdist-membership.json")},
    "limitations":["No accuracy or latency improvement has been established for this precision change.",
        "Double accumulation is not exact arbitrary summation.","Mean CTR prior/total storage grows from float32 to float64.",
        "Older runtimes reject newly fitted format5; old loaded pipelines intentionally retain legacy arithmetic.",
        "CUDA builds and Python ABIs other than3.10/3.12 were not built here."],
}
(WORK / "validation.json").write_text(json.dumps(receipt,indent=2)+"\n")
(WORK / "sdist-initial-superseded.json").write_text(json.dumps({
    "status":"superseded_not_acceptance_evidence",
    "reason":"Affinity fixture and verifier source were updated while initial packaging was running. Final sdist was rebuilt after both stabilized.",
    "retained_artifact":"sdist/ctboost-0.1.61-pre-affinity.tar.gz"},indent=2)+"\n")
print(json.dumps({"builds":builds,"native_source_manifest_sha256":receipt["native_source_manifest_sha256"],
                  "combined_passing_coverage":2525,"explicit_skips":125,"sdist_members":packaging["archive_member_count"]}),flush=True)
