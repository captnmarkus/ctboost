"""Offline full-split transport checks with real shared raw-result validation."""

import gzip
import hashlib
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.tabarena import hpo_full_0160 as shared
from benchmarks.tabarena import kaggle_hpo_full_0160 as controller
from benchmarks.tabarena import kaggle_hpo_worker_full_0160 as worker


def plan_fixture(count=27):
    parents = []
    for ordinal in range(count):
        config, split = divmod(ordinal, 816)
        task, split = divmod(split, 16)
        repeat, fold = divmod(split, 8)
        parents.append(
            {
                "parent_id": f"openml-{task}-r{repeat}-f{fold}-c{config:03d}",
                "ordinal": ordinal,
                "owner": "local" if ordinal % 9 < 4 else "kaggle",
                "dataset": f"dataset-{task}",
                "task_id": task,
                "problem_type": "binary",
                "config_index": config,
                "config_name": f"CTBoost_c{config:03d}",
                "repeat": repeat,
                "fold": fold,
                "child_seeds": list(range(config * 8, config * 8 + 8)),
            }
        )
    return {
        "parents": parents,
        "source_sha256": {},
        "resources": dict(shared.RESOURCES),
    }


def registration(plan):
    return {
        "repository": "captnmarkus/ctboost",
        "commit": "a" * 40,
        "plan_path": "benchmarks/tabarena/full-plan.json",
        "plan_sha256": worker.json_hash(plan),
        "registered_at_utc": "2026-09-07T00:00:00Z",
    }


def raw_result(parent, digest, runtime):
    flags = {"is_fit": True, "is_valid": True, "can_infer": True}
    names = [f"S1F{i + 1}" for i in range(8)]
    seeds = parent["child_seeds"]
    outer = shared._outer_split_receipt(
        SimpleNamespace(get_split_indices=lambda **_: (np.arange(8), np.arange(8, 10))),
        parent,
    )
    return {
        "framework": parent["config_name"],
        "problem_type": "binary",
        "metric_error": 0.3,
        "metric_error_val": 0.4,
        "time_train_s": 1.0,
        "time_infer_s": 0.01,
        "task_metadata": {
            "name": parent["dataset"],
            "tid": parent["task_id"],
            "repeat": parent["repeat"],
            "fold": parent["fold"],
        },
        "method_metadata": {
            "info": {
                **flags,
                "children_info": {
                    name: {**flags, "hyperparameters": {"random_seed": seed}}
                    for name, seed in zip(names, seeds)
                },
                "bagged_info": {"num_child_models": 8, "child_model_names": names},
            }
        },
        "simulation_artifacts": {
            "pred_proba_dict_val": {parent["config_name"]: np.full(8, 0.5)},
            "pred_proba_dict_test": {parent["config_name"]: np.array([0.4, 0.7])},
            "y_val": np.arange(8) % 2,
            "y_test": np.array([0, 1]),
            "bag_info": {
                "val_idx_per_child": [np.array([i]) for i in range(8)],
                "pred_proba_test_per_child": [np.array([0.4, 0.7]) for _ in names],
            },
        },
        "hpo_full_0160": {
            "ctboost_version": "0.1.60",
            "portfolio_id": shared.PORTFOLIO_ID,
            "plan_sha256": digest,
            "runtime_sha256": worker.json_hash(runtime),
            "parent_id": parent["parent_id"],
            "owner": parent["owner"],
            "repeat": parent["repeat"],
            "fold": parent["fold"],
            "sample": 0,
            "outer_split": outer,
            "children": {
                name: {
                    "random_seed": seed,
                    "task_type": "CPU",
                    "feature_test": "quadratic",
                    "native_controls": dict(shared._base().BASELINE_CONTROLS),
                }
                for name, seed in zip(names, seeds)
            },
        },
    }


def completed_bundle(tmp_path, *, mutate_raw=None, mutate_child=None):
    plan = plan_fixture()
    # Both indices are nonzero; changing them must affect paths and raw identity.
    parent = plan["parents"][4]
    parent.update(repeat=2, fold=3, parent_id="openml-0-r2-f3-c000")
    digest = worker.json_hash(plan)
    receipt = registration(plan)
    workspace = tmp_path / "remote-workspace"
    output = workspace / "output"
    destination = tmp_path / "download"
    artifacts = destination / "artifacts"
    artifacts.mkdir(parents=True)
    child_path, raw = worker.parent_paths(output, parent)
    runtime = {"synthetic": True}
    result = raw_result(parent, digest, runtime)
    child = {
        "status": "complete",
        "parent": parent,
        "plan_sha256": digest,
        "runtime": runtime,
        "host": "kaggle",
        "resources": plan["resources"],
        "outer_split": dict(result["hpo_full_0160"]["outer_split"]),
        "registration": {
            "registration_sha256": worker.json_hash(receipt),
            "plan_sha256": digest,
            "receipt": receipt,
        },
    }
    if mutate_raw:
        mutate_raw(result)
    if mutate_child:
        mutate_child(child)
    raw.parent.mkdir(parents=True)
    with gzip.open(raw, "wb") as stream:
        pickle.dump(result, stream)
    child["validation"] = {"raw_sha256": worker.file_hash(raw)}
    worker.write_json(child_path, child)
    record = {
        "parent_id": parent["parent_id"],
        "status": "complete",
        "exit_code": 0,
        "worker_status": "complete",
        "manifest_path": child_path.relative_to(output).as_posix(),
        "manifest_sha256": worker.file_hash(child_path),
        "raw_path": raw.relative_to(output).as_posix(),
        "raw_sha256": worker.file_hash(raw),
    }
    worker.write_json(output / "scheduler" / f"{parent['parent_id']}.json", record)
    specs = worker.shard_specs(plan)
    manifest = {
        **specs[0],
        "status": "complete",
        "parents": [record],
        "worker_sha256": "b" * 64,
        "ctboost_version": worker.CTBOOST_VERSION,
        "benchmark_name": worker.BENCHMARK_NAME,
        "tabarena_commit": worker.TABARENA_COMMIT,
        "portfolio_25_sha256": worker.PORTFOLIO_25_SHA256,
        "plan_sha256": digest,
        "package_sha256": "c" * 64,
    }
    worker.checkpoint(workspace, artifacts, manifest)
    execution = {
        "shards": specs,
        "plan_sha256": digest,
        "payload_sha256": "c" * 64,
        "registration_sha256": worker.json_hash(receipt),
    }
    return destination, plan, execution, manifest


def validate(bundle):
    destination, plan, execution, _ = bundle
    return controller.validate_download(
        destination, 0, plan=plan, execution=execution, worker_hash="b" * 64
    )


def test_full_split_partition_and_paths_have_no_collisions(tmp_path):
    plan = plan_fixture(26 * 816)
    specs = worker.shard_specs(plan)
    ids = [identity for spec in specs for identity in spec["parent_ids"]]
    assert sum(row["owner"] == "local" for row in plan["parents"]) == 9431
    assert len(ids) == len(set(ids)) == 11785
    assert ids == [
        row["parent_id"] for row in plan["parents"] if row["owner"] == "kaggle"
    ]
    assert len(specs) == 983
    assert len(specs[0]["parent_ids"]) == 1
    assert all(len(spec["parent_ids"]) == 12 for spec in specs[1:])
    assert sum(spec["expected_child_fits_in_shard"] for spec in specs) == 94280
    paths = [worker.parent_paths(tmp_path, parent) for parent in plan["parents"]]
    assert len({str(path) for path, _ in paths}) == len(paths)
    assert len({str(path) for _, path in paths}) == len(paths)
    parent = plan["parents"][11]
    manifest, raw = worker.parent_paths(tmp_path, parent)
    assert (
        manifest.relative_to(tmp_path).as_posix()
        == "artifacts/CTBoost_c000/0/1_3/manifest.json"
    )
    assert raw.relative_to(tmp_path).as_posix() == "data/CTBoost_c000/0/1_3/results.pkl"
    assert parent["child_seeds"] == plan["parents"][0]["child_seeds"] == list(range(8))


def test_nonzero_split_archive_validates_real_raw_and_reuses_verified_evidence(
    tmp_path,
):
    bundle = completed_bundle(tmp_path)
    result = validate(bundle)
    assert result["completed_parent_ids"] == ["openml-0-r2-f3-c000"]
    assert result["failed_parents"] == []
    assert result["terminal_parent_count"] == 1
    assert validate(bundle) == result
    assert (
        bundle[0] / "verified/workspace/output/data/CTBoost_c000/0/2_3/results.pkl"
    ).is_file()


@pytest.mark.parametrize(
    "field,value",
    [
        ("portfolio_25_sha256", "0" * 64),
        ("benchmark_name", "ctboost_0160_hpo200"),
        ("worker_sha256", "d" * 64),
    ],
)
def test_collection_rejects_other_run_identity(tmp_path, field, value):
    bundle = completed_bundle(tmp_path)
    bundle[3][field] = value
    worker.write_json(bundle[0] / "artifacts/manifest.json", bundle[3])
    with pytest.raises(ValueError, match="Full-run remote manifest identity differs"):
        validate(bundle)


def test_rehashed_raw_with_wrong_outer_fold_is_rejected(tmp_path):
    bundle = completed_bundle(
        tmp_path, mutate_raw=lambda raw: raw["task_metadata"].update(fold=0)
    )
    with pytest.raises(ValueError, match="outer repeat/fold changed"):
        validate(bundle)


@pytest.mark.parametrize("missing", [False, True])
def test_prefit_split_receipt_is_required_and_bound_to_raw(tmp_path, missing):
    def mutate(child):
        if missing:
            del child["outer_split"]
        else:
            child["outer_split"]["train_indices_sha256"] = "0" * 64
            child["outer_split"]["sha256"] = worker.json_hash(
                {
                    key: value
                    for key, value in child["outer_split"].items()
                    if key != "sha256"
                }
            )

    bundle = completed_bundle(tmp_path, mutate_child=mutate)
    with pytest.raises(
        ValueError, match="[Oo]uter.split|pre-fit|lacks its complete raw result"
    ):
        validate(bundle)


def test_registered_payload_sources_and_rendered_workers_are_immutable(
    tmp_path, monkeypatch
):
    # Only protocol validation is stubbed: this tiny inventory is not the official plan.
    monkeypatch.setattr(shared, "validate_plan", lambda *args, **kwargs: None)
    plan = plan_fixture()
    source = tmp_path / "benchmarks/example.py"
    source.parent.mkdir()
    source.write_bytes(b"example = 1\r\n")
    plan["source_sha256"] = {
        "benchmarks/example.py": hashlib.sha256(b"example = 1\n").hexdigest()
    }
    receipt = registration(plan)
    payload, bundle = controller.make_payload(plan, receipt, source_root=tmp_path)
    assert controller.make_payload(plan, receipt, source_root=tmp_path) == (
        payload,
        bundle,
    )
    assert bundle["allocation"]["cpu_per_parent"] == 2
    assert bundle["allocation"]["memory_limit_gb"] == 8
    monkeypatch.setattr(worker, "PAYLOAD_BASE64", payload)
    package = tmp_path / "unpacked"
    assert worker.unpack_payload(package) == (bundle, plan)
    assert (package / "benchmarks/example.py").read_bytes() == b"example = 1\n"
    assert worker.unpack_payload(package, existing=True) == (bundle, plan)
    (package / "benchmarks/example.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Previously unpacked source changed"):
        worker.unpack_payload(package, existing=True)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Registered source changed"):
        controller.make_payload(plan, receipt, source_root=tmp_path)

    plan["source_sha256"] = {}
    plan_path, receipt_path = tmp_path / "plan.json", tmp_path / "registration.json"
    worker.write_json(plan_path, plan)
    worker.write_json(receipt_path, registration(plan))
    root = tmp_path / "queue"
    prepared = controller.prepare_run(root, plan_path, receipt_path, owner="maiernator")
    _, execution, state = prepared
    assert (
        controller.prepare_run(root, plan_path, receipt_path, owner="maiernator")
        == prepared
    )
    path, kernel, digest = controller.prepare_package(
        root, state, execution, state["slots"][0], 1
    )
    assert kernel.startswith("maiernator/ctboost-0160-full25-")
    assert digest == execution["rendered_worker_sha256"]["1"]
    assert "SHARD_INDEX = 1\n" in (path / "worker.py").read_text(encoding="utf-8")
    metadata = json.loads((path / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["is_private"] == "true" and metadata["enable_gpu"] == "false"
    (root / "worker_template.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen remote worker template changed"):
        controller.prepare_run(root, plan_path, receipt_path, owner="maiernator")


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r\r\n"])
def test_inventory_accepts_trailing_curly_apostrophe_cli_notice(
    monkeypatch, line_ending
):
    notice = "Warning: Looks like you\u2019re using an outdated `kaggle` version (installed: 2.2.0), please consider upgrading to the latest version (2.2.2)"
    output = line_ending.join(
        [
            "ref,title,author,lastRunTime,totalVotes",
            "maiernator/full,Full,maiernator,2026-09-07,0",
            notice,
            "",
        ]
    )

    def remote(_executable, args, **kwargs):
        return output if args[0] == "list" else "KernelWorkerStatus.RUNNING"

    monkeypatch.setattr(controller.transport, "kaggle_command", remote)
    state = {"slots": [{"phase": "submitted", "kernel": "maiernator/full"}]}
    assert controller.occupied_kernels("unused", "maiernator", state) == {
        "maiernator/full"
    }
