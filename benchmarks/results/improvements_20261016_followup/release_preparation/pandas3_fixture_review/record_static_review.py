"""Hash and syntax-check source text only; never import or run the test modules."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
release = Path(r"C:\Users\captn\.codex\worktrees\release-0-1-62\ctboost")
original_checkout = Path(r"C:\apps\ctboost")
names = ["test_native_categorical_codes.py", "test_feature_pipeline_categorical_row.py",
         "test_categorical_codes_prototype.py"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


records = []
patch = []
for name in names:
    rel = Path("tests") / name
    before = root / "originals" / rel
    after = root / "proposed" / rel
    before_text = before.read_text(encoding="utf-8")
    after_text = after.read_text(encoding="utf-8")
    ast.parse(after_text, filename=str(rel))
    if digest(release / rel) != digest(before):
        raise RuntimeError("release source changed during draft: " + str(rel))
    records.append({"path": rel.as_posix(), "original_sha256": digest(before),
                    "proposed_sha256": digest(after),
                    "release_still_matches_original": True,
                    "original_scientific_checkout_sha256": digest(original_checkout / rel),
                    "original_scientific_checkout_matches_snapshot": digest(original_checkout / rel) == digest(before),
                    "syntax_ast_parse": "passed_without_import_or_execution"})
    patch.extend(difflib.unified_diff(before_text.splitlines(keepends=True),
                                    after_text.splitlines(keepends=True),
                                    fromfile="a/" + rel.as_posix(), tofile="b/" + rel.as_posix()))
(root / "fixture_changes.patch").write_text("".join(patch), encoding="utf-8", newline="\n")
receipt = {"status": "draft_only_not_applied", "tests_run": False,
           "production_changes": False, "performance_claims": False,
           "release_root": str(release), "files": records,
           "patch_sha256": digest(root / "fixture_changes.patch"),
           "review_sha256": digest(root / "README.md"),
           "recorder_sha256": digest(Path(__file__)),
           "sources": ["https://pandas.pydata.org/docs/whatsnew/v3.0.6.html",
                       "https://pandas.pydata.org/docs/user_guide/migration-3-strings.html",
                       "https://pandas.pydata.org/docs/whatsnew/v3.0.0.html"]}
(root / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": receipt["status"], "files": len(records),
                  "syntax_only": True, "tests_run": False,
                  "manifest_sha256": digest(root / "manifest.json"),
                  "patch_sha256": receipt["patch_sha256"]}))
