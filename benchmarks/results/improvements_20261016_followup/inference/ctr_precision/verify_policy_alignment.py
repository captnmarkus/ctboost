"""Read-only provenance cross-check against the prospective serving policy."""
import hashlib
import json
from pathlib import Path

ROOT = Path(r"C:/apps/ctboost")
WORK = ROOT / ".tmp/improvements-20261016/ctr-precision"
ARCHIVE = ROOT / "benchmarks/results/improvements_20261016_followup/inference/ctr_precision"
policy_path = ARCHIVE.parent / "final_runner_preparation/selection_policy_v2.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert digest(policy_path) == "da146217cf0663ed4c71cff1fba80c712a1614e969d095c14f0e73e916fc5ab2"
policy = json.loads(policy_path.read_text())
assert digest(policy_path.parent / policy["previous_policy"]) == policy["previous_policy_sha256"]
assert digest(ROOT / policy["runner"]) == policy["runner_sha256"]
package = Path(policy["runtime_path"])
assert digest(package / "ctboost/_core.cp312-win_amd64.pyd") == policy["runtime_native_sha256"]
for path, expected in policy["runtime_provenance"].items():
    assert digest(path) == expected
    archived = ARCHIVE / ("build-receipt-cp312.json" if Path(path).name == "build_receipt.json" else Path(path).name)
    assert digest(archived) == expected
build = json.loads((package / "build_receipt.json").read_text())
assert build["native_sha256"] == policy["runtime_native_sha256"]
assert all(digest(package / "ctboost" / relative) == value for relative,value in build["python_sha256"].items())
validation = json.loads((ARCHIVE / "validation.json").read_text())
assert validation["builds"]["cp312"]["native_sha256"] == policy["runtime_native_sha256"]
assert validation["native_source_manifest_sha256"] == digest(WORK / "native-source-hashes.json")
assert policy["prediction_seed"] == 47 and policy["batch_rows"] == [1,64,1000]
assert policy["library_orders"] == [["ctboost","xgboost","lightgbm"],["lightgbm","ctboost","xgboost"],["xgboost","lightgbm","ctboost"]]
receipt = {"status":"all_policy_runtime_and_archived_provenance_checks_passed",
    "policy_path":str(policy_path),"policy_sha256":digest(policy_path),
    "runner_sha256":policy["runner_sha256"],"runtime_path":str(package),
    "native_sha256":policy["runtime_native_sha256"],"runtime_python_files_checked":len(build["python_sha256"]),
    "provenance_entries_checked_against_original_and_archive":len(policy["runtime_provenance"]),
    "seed_and_batch_and_balanced_orders_unchanged":True,"timings_started":False,
    "script_sha256":digest(__file__)}
(WORK / "timing-policy-alignment.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt),flush=True)
