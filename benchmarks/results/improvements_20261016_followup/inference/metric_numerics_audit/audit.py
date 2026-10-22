"""Synthetic raw-margin metric audit; no model, dataset, fitting or timing access."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import psutil
    psutil.Process().cpu_affinity([0])
    package = args.package.resolve()
    sys.meta_path = [f for f in sys.meta_path if type(f).__module__ != "_editable_skbc_ctboost"]
    sys.path.insert(0, str(package))
    import ctboost
    import numpy as np
    native_path = Path(ctboost._core.__file__).resolve()
    assert Path(ctboost.__file__).resolve().is_relative_to(package)
    assert native_path.is_relative_to(package)
    native_hash = hashlib.sha256(native_path.read_bytes()).hexdigest()
    assert native_hash == "c5ea7e026942c3f1190ca5511fbf74f353bfeb08e7771f76bc5f6c1ca3e4d197"
    binary = []
    for magnitude in (17.0, 40.0, 80.0):
        for sign in (-1.0, 1.0):
            margin = sign * magnitude
            for target in (0, 1):
                actual = ctboost._core._evaluate_metric(
                    np.array([margin], dtype=np.float32), np.array([target], dtype=np.float32),
                    np.ones(1, dtype=np.float32), "Logloss", 1)
                # Direct exact-real identity is safe over these bounded logits.
                reference = math.log1p(math.exp(margin if target == 0 else -margin))
                binary.append(dict(margin=margin, target=target, native=float(actual),
                                   float64_reference=reference, difference=float(actual) - reference))
    multiclass = []
    for gap in (17.0, 40.0, 80.0):
        logits = [0.0, -gap, -2.0 * gap]
        for target in (0, 1, 2):
            actual = ctboost._core._evaluate_metric(
                np.array([logits], dtype=np.float32), np.array([target], dtype=np.float32),
                np.ones(1, dtype=np.float32), "MultiClass", 3)
            reference = math.log1p(math.exp(-gap) + math.exp(-2.0 * gap)) - logits[target]
            multiclass.append(dict(logits=logits, target=target, native=float(actual),
                                       float64_reference=reference,
                                       difference=float(actual) - reference))
    assert binary[6]["native"] > 27.0  # +40, wrong target, clipped below its true loss.
    assert max(abs(x["difference"]) for x in binary) > 50.0
    assert max(abs(x["difference"]) for x in multiclass) > 100.0
    result = dict(
        purpose=__doc__, python=platform.python_version(), numpy=np.__version__,
        package=str(package), native=str(native_path), native_sha256=native_hash,
        build_info=ctboost.build_info(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpu_affinity=psutil.Process().cpu_affinity(), binary=binary, multiclass=multiclass,
        real_models_data_predictions_scores_read=False, fitting=False, timing_claims=False,
        implementation_changed=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(args.output)
    print(json.dumps({"binary_cases": len(binary), "multiclass_cases": len(multiclass),
                      "native_sha256": native_hash, "output": str(args.output)}))


if __name__ == "__main__":
    main()
