"""Capture/verify old-runtime pipelines, ordered replay and continued models."""
import hashlib
import json
import os
from pathlib import Path
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
assert Path(ctboost.__file__).resolve().is_relative_to(package)
assert Path(ctboost._core.__file__).resolve().is_relative_to(package)


def bits(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32).tolist()


def equal_state(actual, expected):
    # JSON preserves the native serialized model contract while canonicalizing
    # pybind tuple/list representations and NaN defaults in non-leaf state.
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


if sys.argv[1] == "capture":
    records = []
    X = np.asarray([["a", "x"], ["b", "x"], ["a", "y"], [None, "y"]] * 9, dtype=object)
    query = np.asarray([["a", "x"], ["b", "y"], ["unseen", "x"], [None, None]], dtype=object)
    for version in [1, 2, 3, 4]:
        for objective in ["regression", "binary", "multiclass", "frequency"]:
            y = (np.arange(len(X)) % 3).astype(np.float32) if objective == "multiclass" else (
                (np.arange(len(X)) % 2).astype(np.float32) if objective == "binary" else
                np.tile([100000000., 1., -100000000., 5.5], 9).astype(np.float32))
            model = ctboost.FeaturePipeline(cat_features=[0, 1], ordered_ctr=True,
                simple_ctr=["Frequency"] if objective == "frequency" else ["Mean"],
                categorical_combinations=[[0, 1]], combinations_ctr=["Mean", "Frequency"],
                random_seed=47, ctr_prior_strength=0.2)
            initial = model.to_state()
            if version < 3:
                initial["feature_pipeline_format_version"] = version
                initial["categorical_key_encoding_version"] = 1
                model = ctboost.FeaturePipeline.from_state(initial)
            model.fit(X, y)
            initial = model.to_state()
            initial["feature_pipeline_format_version"] = version
            model = ctboost.FeaturePipeline.from_state(initial)
            record = {"version":version, "objective":objective, "input":X.tolist(), "labels":y.tolist(),
                      "query":query.tolist(), "state":initial, "canonical_state":model.to_state(),
                      "query_bits":bits(model.transform_array(query)[0])}
            record["training_bits"] = bits(model.fit_transform_array(X, y)[0])
            record["replayed_state"] = model.to_state()
            records.append(record)
    small_X = np.asarray([["a"], ["b"], ["a"], ["c"]] * 8, dtype=object)
    small_y = np.asarray([2., 20., 3., 70.] * 8, dtype=np.float32)
    params = {"objective":"RMSE", "cat_features":[0], "ordered_ctr":True,
              "ctr_prior_strength":0.2, "alpha":1.0, "max_depth":2, "random_seed":47}
    model = ctboost.train(small_X, params, label=small_y, num_boost_round=3)
    model_path = OUT / "legacy-model.ctb"
    model.save_model(model_path)
    small_query = np.asarray([["a"], ["b"], ["c"], ["unseen"], [None]], dtype=object)
    continuation = ctboost.train(small_X, params, label=small_y, init_model=model, num_boost_round=2)
    warmstart = {"input":small_X.tolist(), "labels":small_y.tolist(), "query":small_query.tolist(),
                 "params":params, "model_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
                 "before_bits":bits(model.predict(small_query)),
                 "after_bits":bits(continuation.predict(small_query)),
                 "continued_state":continuation._handle.export_state(),
                 "pipeline":continuation._feature_pipeline.to_state()}
    receipt = {"native":ctboost._core.__file__, "native_sha256":hashlib.sha256(Path(ctboost._core.__file__).read_bytes()).hexdigest(),
               "records":records, "warmstart":warmstart, "synthetic_only":True}
    (OUT / "legacy-reference.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"captured":len(records), "native_sha256":receipt["native_sha256"]}), flush=True)
else:
    receipt = json.loads((OUT / "legacy-reference.json").read_text())
    for r in receipt["records"]:
        model = ctboost.FeaturePipeline.from_state(r["state"])
        if not equal_state(model.to_state(), r["canonical_state"]):
            actual = model.to_state()
            print([(key, actual[key], r["canonical_state"].get(key)) for key in actual
                   if not equal_state(actual[key], r["canonical_state"].get(key))], flush=True)
            raise AssertionError((r["version"],r["objective"],"load"))
        assert bits(model.transform_array(np.asarray(r["query"], dtype=object))[0]) == r["query_bits"]
        assert bits(model.fit_transform_array(np.asarray(r["input"], dtype=object), r["labels"])[0]) == r["training_bits"]
        assert equal_state(model.to_state(), r["replayed_state"]), (r["version"],r["objective"],"replay")
    r = receipt["warmstart"]
    model_path = OUT / "legacy-model.ctb"
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == r["model_sha256"]
    model = ctboost.Booster.load_model(model_path)
    query = np.asarray(r["query"], dtype=object)
    assert bits(model.predict(query)) == r["before_bits"]
    continuation = ctboost.train(np.asarray(r["input"], dtype=object), r["params"],
                                 label=r["labels"], init_model=model, num_boost_round=2)
    assert bits(continuation.predict(query)) == r["after_bits"]
    assert json.dumps(continuation._handle.export_state(), sort_keys=True) == json.dumps(r["continued_state"], sort_keys=True)
    assert equal_state(continuation._feature_pipeline.to_state(), r["pipeline"])
    print(json.dumps({"all_checks_exact":True, "pipelines":len(receipt["records"]),
                      "model_and_warmstart":True, "native_sha256":hashlib.sha256(Path(ctboost._core.__file__).read_bytes()).hexdigest()}))
