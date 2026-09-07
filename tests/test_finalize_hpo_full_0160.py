"""Synthetic, offline finalization fixtures; never official training evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from benchmarks.tabarena import finalize_hpo_full_0160 as final


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@pytest.fixture
def optional_api_doubles(monkeypatch):
    """Core CI has no benchmark extras; real API smoke runs in the pinned env."""

    def get_split_idx(fold=0, repeat=0, sample=0, n_folds=1, n_repeats=1, n_samples=1):
        assert fold < n_folds and repeat < n_repeats and sample < n_samples
        return n_folds * n_samples * repeat + n_samples * fold + sample

    for name in ("tabarena", "tabarena.benchmark", "tabarena.benchmark.task"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    utils = ModuleType("tabarena.benchmark.task.utils")
    utils.get_split_idx = get_split_idx
    monkeypatch.setitem(sys.modules, utils.__name__, utils)

    # JSON is a YAML subset. Only metadata flag rewriting is under test here.
    yaml = ModuleType("yaml")
    yaml.safe_load = json.loads
    yaml.safe_dump = json.dumps
    monkeypatch.setitem(sys.modules, "yaml", yaml)

    class RepositoryNotFoundError(Exception):
        pass

    def unexpected_hub_call(*args, **kwargs):
        pytest.fail("Unexpected Hugging Face access")

    hub = ModuleType("huggingface_hub")
    hub.HfApi = unexpected_hub_call
    hub.snapshot_download = unexpected_hub_call
    errors = ModuleType("huggingface_hub.errors")
    errors.RepositoryNotFoundError = RepositoryNotFoundError
    hub.errors = errors
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)


@pytest.fixture
def study(tmp_path, monkeypatch, optional_api_doubles):
    """Two synthetic configs, two outer repeats, three folds; no actual models."""
    splits = [
        {"dataset": "fixture", "repeat": repeat, "fold": fold}
        for repeat in range(2)
        for fold in range(3)
    ]
    parents = [
        {
            **split,
            "task_id": 123,
            "config_index": index,
            "config_name": f"C{index}",
            "parent_id": f"fixture-r{split['repeat']}-f{split['fold']}-c{index}",
            "owner": "local" if index == 0 else "kaggle",
        }
        for index in range(2)
        for split in splits
    ]
    native_hash = "a" * 64
    wheel = {"filename": "synthetic.whl", "sha256": "b" * 64}
    plan = {
        "fixture_only": True,
        "parents": parents,
        "expected_parent_count": 12,
        "expected_child_count": 96,
        "outer_splits": splits,
        "outer_splits_sha256": digest(splits),
        "package_pins": {"ctboost": "0.1.60"},
        "source_sha256": {"fixture.py": "c" * 64},
        "tabarena_commit": "d" * 40,
        "public_wheels": [wheel],
        "resources": {"num_cpus": 2, "memory_limit_gb": 8},
    }
    receipt = {
        "repository": "captnmarkus/ctboost",
        "commit": "e" * 40,
        "plan_path": "fixture/plan.json",
        "plan_sha256": digest(plan),
    }
    worker = SimpleNamespace(
        plan_hash=digest,
        source_hash=final.file_hash,
        validate_plan=lambda p: (
            p if p["fixture_only"] else pytest.fail("Expected synthetic fixture")
        ),
    )
    identity = final.registration_identity(plan, receipt, worker)
    worker.verify_registration = lambda *args: identity
    calls = []

    def validate(raw, parent, plan_sha256, *, outer_split):
        calls.append(parent["parent_id"])
        payload = final.read_json(raw)
        assert payload["parent_id"] == parent["parent_id"]
        assert payload["outer_split"] == outer_split
        assert plan_sha256 == digest(plan)
        return {
            "raw_sha256": final.file_hash(raw),
            "size_bytes": raw.stat().st_size,
            "runtime_sha256": payload["runtime_sha256"],
            "bag_children": 8,
            "outer_split": payload["outer_split"],
            "repeat": parent["repeat"],
            "fold": parent["fold"],
        }

    worker.validate_parent_result = validate
    monkeypatch.setattr(final, "protocol", lambda: worker)
    runtime = {
        "packages": plan["package_pins"],
        "source_sha256": plan["source_sha256"],
        "tabarena_commit": plan["tabarena_commit"],
        "outer_splits_sha256": plan["outer_splits_sha256"],
        "public_wheel": wheel,
        "python": "3.12.11 fixture",
        "ctboost_build_info": {"version": "0.1.60"},
        "environment": {},
        "environment_sha256": digest({}),
        "package_files_sha256": {"native.pyd": native_hash},
        "native_extension_sha256": native_hash,
    }
    inputs = [tmp_path / "local", tmp_path / "remote"]
    for directory in inputs:
        directory.mkdir()
    paths = []
    for parent in parents:
        root = inputs[parent["config_index"]]
        key = (
            Path(parent["config_name"])
            / str(parent["task_id"])
            / f"{parent['repeat']}_{parent['fold']}"
        )
        raw = root / "data" / key / "results.pkl"
        outer_split = {
            "repeat": parent["repeat"],
            "fold": parent["fold"],
            "fixture_rows": 16,
        }
        final.write_json(
            raw,
            {
                "parent_id": parent["parent_id"],
                "outer_split": outer_split,
                "runtime_sha256": digest(runtime),
            },
        )
        validation = validate(raw, parent, digest(plan), outer_split=outer_split)
        path = root / "artifacts" / key / "manifest.json"
        manifest = {
            "parent": parent,
            "raw_path": raw.relative_to(root).as_posix(),
            "plan_sha256": digest(plan),
            "host": parent["owner"],
            "resources": plan["resources"],
            "registration": identity,
            "runtime": runtime,
            "status": "complete",
            "validation": validation,
            "elapsed_seconds": 1.0,
            "outer_split": outer_split,
        }
        final.write_json(path, manifest)
        paths.append((path, raw))
    calls.clear()
    receipt_path = tmp_path / "registration.json"
    final.write_json(receipt_path, receipt)
    return SimpleNamespace(
        plan=plan,
        receipt=receipt,
        worker=worker,
        inputs=inputs,
        paths=paths,
        calls=calls,
        receipt_path=receipt_path,
        output=tmp_path / "finalized",
        source=tmp_path / "source",
    )


def alter(path, function):
    value = final.read_json(path)
    function(value)
    final.write_json(path, value)


def test_complete_parent_membership_and_all_bag_children(study):
    report, paths = final.collect(study.plan, study.inputs, study.receipt)
    assert report["status"] == "complete"
    assert report["complete_parents"] == 12 and report["validated_children"] == 96
    assert len(paths) == len(study.calls) == 12


def test_missing_parent_is_pending_without_deserializing(study):
    manifest, raw = study.paths[-1]
    manifest.unlink()
    raw.unlink()
    report, _ = final.collect(study.plan, study.inputs, study.receipt)
    assert report["status"] == "pending" and len(report["missing_parents"]) == 1
    assert study.calls == []


@pytest.mark.parametrize("status", ["preparing", "running", "failed"])
def test_noncomplete_parent_never_scores_or_publishes(study, monkeypatch, status):
    alter(study.paths[0][0], lambda m: m.update(status=status))
    monkeypatch.setattr(
        final, "evaluate", lambda *args: pytest.fail("Premature evaluation")
    )
    monkeypatch.setattr(
        study.worker,
        "verify_registration",
        lambda *args: pytest.fail("Premature network"),
    )
    with pytest.raises(ValueError, match="incomplete"):
        final.stage(
            study.plan, study.inputs, study.receipt_path, study.output, study.source
        )
    assert not study.output.exists()
    assert study.calls == []


def test_extra_parent_rejected(study):
    alter(study.paths[0][0], lambda m: m["parent"].update(parent_id="foreign"))
    with pytest.raises(ValueError, match="Extra or foreign"):
        final.collect(study.plan, study.inputs, study.receipt)


def test_duplicate_parent_rejected(study, tmp_path):
    duplicate = tmp_path / "duplicate"
    shutil.copytree(study.inputs[0], duplicate)
    with pytest.raises(ValueError, match="Duplicate parent"):
        final.collect(study.plan, [*study.inputs, duplicate], study.receipt)


@pytest.mark.parametrize(
    "field", ["plan_sha256", "host", "resources", "registration", "runtime"]
)
def test_parent_provenance_mismatch_rejected(study, field):
    alter(
        study.paths[0][0],
        lambda m: m.update(
            {
                field: {}
                if field in {"resources", "registration", "runtime"}
                else "foreign"
            }
        ),
    )
    with pytest.raises(ValueError):
        final.collect(study.plan, study.inputs, study.receipt)


def test_round_mapping_cannot_reuse_r0_manifest(study):
    source = study.paths[0][0]
    target = study.paths[3][0]
    shutil.copyfile(source, target)
    with pytest.raises(ValueError, match="Duplicate parent|repeat/fold"):
        final.collect(study.plan, study.inputs, study.receipt)


def test_exact_outer_split_receipt_is_required(study):
    alter(study.paths[0][0], lambda m: m.pop("outer_split"))
    with pytest.raises(ValueError, match="outer-split receipt"):
        final.collect(study.plan, study.inputs, study.receipt)


def test_changed_outer_split_receipt_is_not_self_attested(study):
    alter(study.paths[0][0], lambda m: m["outer_split"].update(fixture_rows=17))
    with pytest.raises(AssertionError):
        final.collect(study.plan, study.inputs, study.receipt)


def test_raw_checksum_and_undeclared_raw_rejected(study):
    raw = study.paths[0][1]
    original = raw.read_bytes()
    raw.write_bytes(original.replace(b'"fixture_rows": 16', b'"fixture_rows": 17'))
    with pytest.raises(ValueError, match="checksum"):
        final.collect(study.plan, study.inputs, study.receipt)
    raw.write_bytes(original)
    extra = study.inputs[0] / "foreign/results.pkl"
    extra.parent.mkdir()
    extra.write_bytes(b"foreign")
    with pytest.raises(ValueError, match="Undeclared"):
        final.collect(study.plan, study.inputs, study.receipt)


def tables(plan):
    mapping = final.split_keys(plan)
    common = {
        "metric_error": 0.1,
        "metric_error_val": 0.2,
        "time_train_s": 1.0,
        "time_infer_s": 0.01,
    }
    models = pd.DataFrame(
        [
            {
                **common,
                "method": p["config_name"],
                "dataset": p["dataset"],
                "fold": mapping[p["dataset"], p["repeat"], p["fold"]],
            }
            for p in plan["parents"]
        ]
    )
    hpo = pd.DataFrame(
        [
            {
                **common,
                "dataset": key[0],
                "fold": index,
                "method_subtype": kind,
                "imputed": False,
            }
            for key, index in mapping.items()
            for kind in ("default", "tuned", "tuned_ensemble")
        ]
    )
    return models, hpo


def test_flattened_repeat_fold_mapping_and_exact_hpo_coverage(study):
    models, hpo = tables(study.plan)
    assert final.split_keys(study.plan)["fixture", 1, 0] == 3
    assert set(models["fold"]) == set(range(6))
    final.validate_tables(study.plan, models, hpo)
    models["fold"] %= 3
    with pytest.raises(ValueError, match="collapse"):
        final.validate_tables(study.plan, models, hpo)


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "imputed", "nan", "predictions"]
)
def test_invalid_result_tables_rejected(study, change):
    models, hpo = tables(study.plan)
    if change == "missing":
        hpo = hpo.iloc[:-1]
    elif change == "duplicate":
        hpo.iloc[-1] = hpo.iloc[0]
    elif change == "imputed":
        hpo.loc[0, "imputed"] = True
    elif change == "nan":
        models.loc[0, "metric_error"] = float("nan")
    else:
        models["predictions"] = "must remain private"
    with pytest.raises(ValueError):
        final.validate_tables(study.plan, models, hpo)


@pytest.fixture
def staged(study, monkeypatch):
    def evaluate(plan, paths, output, source):
        assert len(paths) == 12
        canonical = output / "canonical-results"
        (canonical / "results").mkdir(parents=True)
        (canonical / "metadata.yaml").write_text(
            json.dumps({"can_hpo": True, "has_raw": True, "has_processed": True})
        )
        for name in final.TABLES:
            # Staging copies bytes; Parquet encoding is covered by the real API smoke.
            (canonical / "results" / name).write_bytes(
                b"synthetic result-table fixture"
            )
        return canonical

    monkeypatch.setattr(final, "evaluate", evaluate)
    return study, final.stage(
        study.plan, study.inputs, study.receipt_path, study.output, study.source
    )


def test_staging_allowlist_and_idempotent_resume(staged, monkeypatch):
    study, folder = staged
    assert set(final.verify_stage(folder)) == final.PUBLIC_FILES
    assert not list(folder.rglob("*.pkl")) and not list(folder.rglob("*.npz"))
    assert all(raw.is_file() for _, raw in study.paths)
    assert (
        final.read_json(folder / "canonical-results/metadata.yaml")["has_raw"] is False
    )
    original = (folder / "SHA256SUMS").read_bytes()
    monkeypatch.setattr(
        final, "evaluate", lambda *args: pytest.fail("Repeated evaluation")
    )
    assert (
        final.stage(
            study.plan, study.inputs, study.receipt_path, study.output, study.source
        )
        == folder
    )
    assert (folder / "SHA256SUMS").read_bytes() == original


def test_public_allowlist_rejects_even_rechecksummed_raw(staged):
    _, folder = staged
    (folder / "raw.pkl").write_bytes(b"not for publication")
    (folder / "SHA256SUMS").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in final.checksums(folder).items())
    )
    with pytest.raises(ValueError, match="inventory"):
        final.verify_stage(folder)


@pytest.fixture
def hub(staged, monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub import errors

    RepositoryNotFoundError = errors.RepositoryNotFoundError

    study, folder = staged
    remote = tmp_path / "fake-hub"
    operations = []

    class Api:
        def __init__(self, token):
            self.token = token

        def whoami(self):
            return {"name": "Maiernator"}

        def repo_info(self, *args, **kwargs):
            if not remote.exists():
                raise RepositoryNotFoundError("Synthetic absent repository")
            return SimpleNamespace(sha="f" * 40, private=False)

        def create_repo(self, **kwargs):
            operations.append("create")
            assert self.token == "synthetic-secret"
            remote.mkdir()

        def upload_folder(self, **kwargs):
            operations.append("upload")
            assert kwargs["parent_commit"] == "f" * 40
            shutil.copytree(kwargs["folder_path"], remote, dirs_exist_ok=True)
            return SimpleNamespace(oid="f" * 40)

        def list_repo_files(self, *args, **kwargs):
            return [
                p.relative_to(remote).as_posix()
                for p in remote.rglob("*")
                if p.is_file()
            ]

    def snapshot_download(**kwargs):
        operations.append("anonymous" if kwargs["token"] is False else "recovery")
        shutil.copytree(remote, kwargs["local_dir"], dirs_exist_ok=True)
        return kwargs["local_dir"]

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(final, "credential", lambda: "synthetic-secret")
    return SimpleNamespace(
        study=study, folder=folder, remote=remote, operations=operations
    )


def test_explicit_publication_and_restart_verify_anonymous_bytes(hub):
    receipt = final.publish(hub.folder, hub.study.output)
    assert receipt["anonymous_download_verified"] is True
    assert hub.operations == ["create", "upload", "anonymous"]
    final.publish(hub.folder, hub.study.output)
    assert hub.operations == ["create", "upload", "anonymous", "anonymous"]


def test_existing_foreign_dataset_is_never_overwritten(hub):
    hub.remote.mkdir()
    (hub.remote / "README.md").write_text("Someone else's existing data")
    with pytest.raises(ValueError, match="foreign"):
        final.publish(hub.folder, hub.study.output)
    assert hub.operations == []
    assert (hub.remote / "README.md").read_text() == "Someone else's existing data"


def test_upload_before_receipt_crash_recovers_exact_commit(hub):
    shutil.copytree(hub.folder, hub.remote)
    receipt = final.publish(hub.folder, hub.study.output)
    assert receipt["commit"] == "f" * 40
    assert hub.operations == ["recovery", "anonymous"]


def test_recovery_rejects_foreign_bytes_with_matching_marker(hub):
    shutil.copytree(hub.folder, hub.remote)
    (hub.remote / "README.md").write_text("Changed after another publication")
    with pytest.raises(ValueError, match="foreign or changed"):
        final.publish(hub.folder, hub.study.output)
    assert "upload" not in hub.operations


def test_changed_empty_repository_is_not_overwritten_on_resume(hub):
    hub.remote.mkdir()
    marker = final.read_json(hub.folder / final.MARKER)
    final.write_json(
        hub.study.output / "publication_receipt.json",
        {
            "repo_id": final.REPO_ID,
            "stage_sha256": final.file_hash(hub.folder / "SHA256SUMS"),
            "binding": marker["binding"],
            "created": True,
            "created_head": "9" * 40,
        },
    )
    with pytest.raises(ValueError, match="foreign"):
        final.publish(hub.folder, hub.study.output)
    assert hub.operations == []


def test_pending_cli_check_does_not_publish(study, monkeypatch, capsys, tmp_path):
    alter(study.paths[0][0], lambda m: m.update(status="running"))
    path = tmp_path / "plan.json"
    final.write_json(path, study.plan)
    monkeypatch.setattr(
        final, "publish", lambda *args: pytest.fail("Premature publication")
    )
    monkeypatch.setattr(
        final,
        "sys",
        SimpleNamespace(flags=SimpleNamespace(isolated=True), path=list(sys.path)),
    )
    installed = ModuleType("ctboost")
    installed.__file__ = str(tmp_path / "site-packages/ctboost/__init__.py")
    monkeypatch.setitem(sys.modules, "ctboost", installed)
    result = final.main(
        [
            "--plan",
            str(path),
            "--input",
            str(study.inputs[0]),
            "--input",
            str(study.inputs[1]),
            "--registration",
            str(study.receipt_path),
            "--output",
            str(study.output),
            "--check",
        ]
    )
    assert result == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pending"


def test_changed_publication_completeness_never_reaches_hub(staged, monkeypatch):
    study, folder = staged
    alter(folder / "validation/completeness.json", lambda m: m.update(status="pending"))
    (folder / "SHA256SUMS").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in final.checksums(folder).items())
    )
    monkeypatch.setattr(
        final, "credential", lambda: pytest.fail("Premature credential/network access")
    )
    with pytest.raises(ValueError, match="completeness"):
        final.publish(folder, study.output)
