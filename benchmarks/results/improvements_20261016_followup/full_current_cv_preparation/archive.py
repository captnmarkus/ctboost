"""Archive inactive preparation evidence without freezing an experiment."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "benchmarks/results/improvements_20261016_followup/full_current_cv_preparation"
OUT.mkdir(parents=True, exist_ok=False)
files = [HERE / name for name in ("validate.py", "validation.json", "validation.log", "validate_native.py",
                                  "native-validation.json", "native-validation.log", "archive.py")]
files += [ROOT / "benchmarks" / (name + ".py") for name in ("fair_accuracy_current_round_cv_support",
    "test_fair_accuracy_current_round_cv_support", "fair_accuracy_current_round_cv_backend",
    "test_fair_accuracy_current_round_cv_native")]
for source in files:
    shutil.copyfile(source, OUT / source.name)
review = {"status": "inactive preparation only", "independent_reviewer": "/root/multiplication_research",
    "support": "Only material review finding was missing expanded-source fallback; corrected before46-pass validation. No remaining blocker for inactive support handoff.",
    "backend": "Clean union.fit_fixed dispatch, copied memory-only override, fixed temperature, metadata guards and toy native coverage reviewed without blocker.",
    "counts": {"synthetic_orchestration_tests": 46, "native_integration_tests": 30,
               "synthetic_four_round_native_fits": 22, "real_data_reads": 0, "real_quality_scores": 0},
    "limitations": "No policy activation, parent17b completion/admission, real fold manifest or resource freeze. The17b whole-source hash was26dae7b5…61f6 during native validation and may subsequently change while the parent runner is prepared. Preserve the recorded historical hash and rebind final callbacks before any experiment."}
(OUT / "review.json").write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
(OUT / "README.md").write_text("""# Inactive full-current-data round CV preparation

This archive contains reviewed support/backend prototypes and synthetic validation. It is not a frozen or activated experiment.

- **46 synthetic orchestration checks passed**: authorized raw row order and exact original TRAIN/STOP matches, mandatory expanded sources, observed fold-only vocabulary, no validation labels passed to fitting/correction, fixed temperature, complete540 fold receipt identities and deterministic mean-error/tie rules.
- **30 native integration checks passed**, including **22 synthetic four-round fits**, on immutable accuracy CPython3.12/native c5ea7e026942c3f1190ca5511fbf74f353bfeb08e7771f76bc5f6c1ca3e4d197, NumPy2.0.2, SciPy1.16.3, XGBoost3.4.1 and LightGBM4.7.0. Regression/binary and scalar/vector multiclass cover identity/nonzero corrections, exact routes, ordered CTR replay, native/sparse oracles, fixed temperature once, pickle bytes and original metrics.
- **Zero real benchmark rows/models opened, real fits or real quality scores.** Tests ran onCPU0 with one thread. The64MiB native-test named-array budget is not an experiment resource decision.

The receipts bind exact source/runtime hashes and preserve logs. Frozen wave13/16/17 sources matched their original plans. Native validation used the then-current17b source, unchanged during that run;17b was still being prepared, so its eventual final hash must be rebound and affected callbacks reviewed before real activation.

The metadata-only resource analysis is [full_current_cv_memory](../full_current_cv_memory/README.md). Future preparation requires a completed, quality-failed17b policy, shared fold/source manifests, all36 fixed policies, the certified regression solver, explicit immutable resource limits and quiet-memory headroom. Planned work is540 individually owned fold-fit jobs followed by36 mandatory full-current-union fits. Seed677 is prospective and fixed before outcomes; no seed companion or official-test gate has run.
""", encoding="utf-8")
manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(OUT.iterdir()) if path.is_file()}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"archive": str(OUT), "artifacts": len(manifest)}, indent=2))
