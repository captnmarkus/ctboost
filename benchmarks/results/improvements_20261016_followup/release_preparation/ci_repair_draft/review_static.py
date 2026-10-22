"""Hash/AST-only draft audit. Does not import or run any test/learner/workflow."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import copy
import hashlib
import json
import re
import tomllib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
assert (ROOT / "pyproject.toml").is_file()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
checks = []
for item in manifest["files"]:
    original, draft = ROOT / item["path"], HERE / "files" / item["path"]
    assert sha(draft) == item["draft_sha256"]
    if item["original_sha256"] is None:
        assert not original.exists()
    else:
        assert sha(original) == item["original_sha256"]
    if draft.suffix == ".py":
        old = ast.parse((HERE / "originals" / item["path"]).read_text(encoding="utf-8"))
        new = ast.parse(draft.read_text(encoding="utf-8"))
        tests = lambda tree: {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
        assert tests(old) == tests(new)
        old_asserts = sum(isinstance(n, ast.Assert) for n in ast.walk(old))
        new_asserts = sum(isinstance(n, ast.Assert) for n in ast.walk(new))
        assert new_asserts >= old_asserts
        checks.append(dict(path=item["path"], syntax_parsed_without_import=True,
            test_functions_preserved=len(tests(old)), original_assertions=old_asserts, draft_assertions=new_asserts))

workflow_path = HERE / "files/.github/workflows/scientific-tests.yml"
workflow = workflow_path.read_text(encoding="utf-8")
paths = re.findall(r'"((?:tests|benchmarks)/test_[^"]+\.py)"', workflow)
assert len(paths) == len(set(paths)) == 23
skips, test_sources = [], []
for name in paths + ["tests/conftest.py"]:
    source = HERE / "files" / name
    if not source.exists():
        source = ROOT / name
    contents = source.read_text(encoding="utf-8")
    tree = ast.parse(contents)
    test_sources.append(dict(path=name, reviewed_source_sha256=sha(source)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"skip", "skipif", "xfail", "importorskip"}:
            call = ast.unparse(node)
            assert call.startswith("pytest."), call
            if node.func.attr == "importorskip":
                package = ast.literal_eval(node.args[0])
                assert package in {"numpy", "pandas", "scipy", "scipy.sparse.linalg", "xgboost", "lightgbm", "psutil"}, package
                resolution = "Required package is installed and imported by the pinned job preflight"
            else:
                assert node.func.attr == "skip" and "3.4.1" in call, call
                resolution = "Only unsupported-XGBoost-version gate; pinned job asserts version3.4.1 before tests"
            skips.append(dict(path=name, line=node.lineno, call=call, pinned_stack_resolution=resolution))

for name in ("include", "spec", "scripts/prepare_cuda_runtime_license.py", "scripts/validate_release_artifacts.py"):
    assert (ROOT / name).exists()
for token in ("NUMEXPR_NUM_THREADS", "CTBOOST_HIST_THREADS", "d23441a48e516b6c34aea4fa41551a30e30af803", "ece7cb06caefa5fff74198d8649806c4678c61a1"):
    assert token in workflow
original_project = tomllib.loads((HERE / "originals/pyproject.toml").read_text(encoding="utf-8"))
draft_project = tomllib.loads((HERE / "files/pyproject.toml").read_text(encoding="utf-8"))
expected_project = copy.deepcopy(original_project)
dev = expected_project["project"]["optional-dependencies"]["dev"]
dev.insert(dev.index("pytest>=8.0") + 1, "psutil>=7")
source_resources = ["/benchmarks/fair_accuracy_current_round_cv_policy.json",
    "/benchmarks/results/improvements_20261016_followup/robustness_protocol_multiclass.json",
    "/benchmarks/results/improvements_20261016_followup/robustness_protocol_scalar_refresh.json",
    "/benchmarks/results/improvements_20261016_followup/robustness_protocol_scalar_numerical.json"]
includes = expected_project["tool"]["scikit-build"]["sdist"]["include"]
start = includes.index("/benchmarks/*.cpp") + 1
includes[start:start] = source_resources
assert expected_project == draft_project
resource_receipts = [dict(path=name.lstrip("/"), sha256=sha(ROOT / name.lstrip("/")),
    bytes=(ROOT / name.lstrip("/")).stat().st_size) for name in source_resources]
receipt = dict(recorded_at=datetime.now(timezone.utc).isoformat(),
    status="source_only_static_review_not_execution", manifest_sha256=sha(HERE / "manifest.json"),
    changes_sha256=sha(HERE / "changes.diff"), python_source_checks=checks,
    scientific_selection_paths=paths, reviewed_scientific_sources=test_sources,
    skip_path_inventory=skips, skipif_or_xfail_paths=0, gpu_or_optional_backend_skip_paths=0,
    source_layout_review="Selected tests resolve benchmark siblings/policy JSON copied with benchmarks; synthetic cache reads are stubbed or temp_path. No selected test requires include/spec/release scripts, but the workflow now copies them to match ordinary installed-package CI. No .tmp datasets/models are copied.",
    packaging_review=dict(semantic_delta_only_dev_psutil_and_four_exact_sdist_json_paths=True,
        wheel_exclusions_unchanged=True, runtime_dependencies_unchanged=True, added_resources=resource_receipts,
        unbundled_replay_scope="Real historical study replay also needs the explicitly bound archive and user-supplied authorized caches/parent models; these remain outside runtime wheels. No complete replay claim."),
    negative_version_test="Supported3.4.1 model is constructed, then the reported version is changed to3.4.2 and rejection is asserted. It is not skipped in the pinned stack.",
    active_sources_unchanged=True, tests_or_native_imports_run=False, workflow_executed=False,
    limits="AST/path/hash review makes zero skips logically attainable, not experimentally proven. Hosted installation, fixture behavior and workflow execution still require a real CI run.")
(HERE / "static_review.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
print(json.dumps(dict(manifest_sha256=receipt["manifest_sha256"], static_review_sha256=sha(HERE / "static_review.json"),
    selected_files=len(paths), skip_paths=len(skips), tests_executed=False)))
