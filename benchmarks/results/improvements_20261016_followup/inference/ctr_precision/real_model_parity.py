"""Compare existing models on TRAIN features only; no refitting or scoring."""
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "CTBOOST_HIST_THREADS", "CTBOOST_NODE_HIST_THREADS"):
    os.environ[key] = "1"
import psutil
psutil.Process().cpu_affinity([0])
ROOT = Path(r"C:/apps/ctboost")
OUT = ROOT / ".tmp/improvements-20261016/ctr-precision"
package = Path(sys.argv[2]).resolve()
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != "_editable_skbc_ctboost"]
sys.path[:0] = [str(package), str(ROOT / ".tmp/improvements-20261016/followup/deps"), str(ROOT)]
import numpy as np
import ctboost
from benchmarks import fair_accuracy_scout as accuracy
assert Path(ctboost.__file__).resolve().is_relative_to(package)
assert Path(ctboost._core.__file__).resolve().is_relative_to(package)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=True).encode()).hexdigest()


def array_receipt(value):
    array = np.ascontiguousarray(value)
    return {"shape":list(array.shape), "dtype":str(array.dtype),
            "sha256":hashlib.sha256(array.view(np.uint8)).hexdigest()}


selection_path = ROOT / "benchmarks/results/improvements_20261016_followup/accuracy_wave16/frozen_selections.json"
plan_path = selection_path.with_name("plan.json")
plan = json.loads(plan_path.read_text())
frozen = json.loads(selection_path.read_text())
assert digest(plan_path) == frozen["plan_sha256"]
cases = {case["dataset_name"]:case for case in plan["cases"]}
records = []
for selected in frozen["selections"]:
    if selected["library"] != "ctboost":
        continue
    name = selected["dataset"]
    role = cases[name]["roles"]["train"]
    assert digest(role["path"]) == role["sha256"]
    assert digest(selected["model_path"]) == selected["model_sha256"]
    with open(selected["model_path"], "rb") as stream:
        payload = pickle.load(stream)
    with open(role["path"], "rb") as stream:
        raw = pickle.load(stream)["X"]
    normalized, _ = accuracy.normalize(raw, payload["schema"])
    model = payload["model"]
    pipeline = model._feature_pipeline
    record = {"dataset":name, "model_sha256":selected["model_sha256"],
              "train_sha256":role["sha256"], "rows":len(raw),
              "schema_sha256":value_digest(payload["schema"]),
              "pipeline_sha256":None if pipeline is None else value_digest(pipeline.to_state()),
              "raw_predictions":array_receipt(model.predict(normalized)), "batches":[]}
    for rows in (1,64,1000):
        batch = normalized.iloc[np.arange(rows) % len(normalized)].copy()
        result = {"rows":rows,"raw_predictions":array_receipt(model.predict(batch))}
        if pipeline is not None:
            result["transformed"] = array_receipt(pipeline.transform_array(batch)[0])
        record["batches"].append(result)
    records.append(record)
    print(json.dumps({"checked":name,"rows":len(raw)}), flush=True)
    del model, payload, pipeline, raw, normalized, batch
    gc.collect()

receipt = {"native":str(ctboost._core.__file__),"native_sha256":digest(ctboost._core.__file__),
           "selection_sha256":digest(selection_path),"plan_sha256":digest(plan_path),
           "script_sha256":digest(__file__),"normalization_helper_sha256":digest(accuracy.__file__),
           "records":records,"fits":0,"scores":0,"confirmation_or_official_test_opened":False,
           "input_scope":"existing TRAIN frame features only; labels not used"}
if sys.argv[1] == "capture":
    (OUT / "real-model-reference.json").write_text(json.dumps(receipt,indent=2)+"\n")
else:
    reference_path = OUT / "real-model-reference.json"
    reference = json.loads(reference_path.read_text())
    assert all(receipt[key] == reference[key] for key in (
        "records","selection_sha256","plan_sha256","script_sha256","normalization_helper_sha256"))
    receipt["reference_sha256"] = digest(reference_path)
    receipt["all_exact"] = True
    (OUT / "real-model-verified.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({"models":len(records),"all_exact":sys.argv[1] != "capture", "fits":0,"scores":0}), flush=True)
