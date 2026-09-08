"""Validate all full-run parents, stage result-only artifacts, optionally publish.

Use only trusted outputs from this run: validating a pickle executes its contents.
--check reports pending coverage without scoring or contacting Hugging Face. Once
coverage is complete it validates every raw result. Publication requires --publish;
neither failed nor missing parents are imputed. Raw targets/predictions stay local.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "Maiernator/ctboost-tabarena-full-hpo25-0.1.60"
SUITE = "ctboost_0160_full_hpo25"
TABLES = ("model_results.parquet", "hpo_results.parquet")
MARKER = "publication.json"
PUBLIC_FILES = {
    "canonical-results/metadata.yaml",
    "plan.json",
    "validation/completeness.json",
    "validation/registration.json",
    MARKER,
    "README.md",
    *(f"canonical-results/results/{name}" for name in TABLES),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol():
    from benchmarks.tabarena import hpo_full_0160

    return hpo_full_0160


def registration_identity(plan, receipt, worker):
    require(
        receipt.get("repository") == "captnmarkus/ctboost",
        "Foreign registration repository",
    )
    require(
        re.fullmatch(r"[a-f0-9]{40}", receipt.get("commit", "")),
        "Invalid registration commit",
    )
    name = receipt.get("plan_path", "")
    require(
        name
        and not name.startswith("/")
        and ".." not in Path(name).parts
        and "\\" not in name
        and ":" not in name,
        "Unsafe registration plan path",
    )
    digest = worker.plan_hash(plan)
    require(
        receipt.get("plan_sha256") == digest, "Registration belongs to another plan"
    )
    return {
        "registration_sha256": worker.plan_hash(receipt),
        "plan_sha256": digest,
        "receipt": receipt,
        "public_url": f"https://raw.githubusercontent.com/{receipt['repository']}/{receipt['commit']}/{name}",
    }


def validate_runtime(runtime, plan, worker):
    for key, expected in (
        ("packages", plan["package_pins"]),
        ("source_sha256", plan["source_sha256"]),
        ("tabarena_commit", plan["tabarena_commit"]),
        ("outer_splits_sha256", plan["outer_splits_sha256"]),
    ):
        require(
            runtime.get(key) == expected, f"Runtime {key} differs from the full plan"
        )
    require(
        runtime.get("public_wheel") in plan["public_wheels"],
        "Unregistered public wheel",
    )
    require(runtime.get("python", "").startswith("3.12."), "Unexpected Python version")
    require(
        runtime.get("ctboost_build_info", {}).get("version") == "0.1.60",
        "Unexpected native version",
    )
    require(
        runtime.get("environment_sha256")
        == worker.plan_hash(runtime.get("environment")),
        "Runtime environment digest differs",
    )
    files = runtime.get("package_files_sha256", {})
    require(
        files and all(re.fullmatch(r"[a-f0-9]{64}", value) for value in files.values()),
        "Missing public package byte hashes",
    )
    require(
        runtime.get("native_extension_sha256") in files.values(),
        "Missing native package hash",
    )


def utc_time(value):
    require(isinstance(value, str), "Missing recovery timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(
        parsed.utcoffset() == timedelta(0), "Recovery timestamps must be explicit UTC"
    )
    return parsed


def projected_runtime(manifest):
    return {
        key: value
        for key, value in manifest["runtime"].items()
        if key != "python_executable"
    }


def validate_recovery_audit(plan, inputs, receipt, audit, worker):
    """Validate the public, pre-authorized exception; archives were checked at recovery."""
    require(audit.get("schema_version") == 1, "Unknown recovery audit schema")
    declaration = audit.get("declaration", {})
    identity = registration_identity(plan, receipt, worker)
    for key, value in {
        "schema_version": 1,
        "plan_sha256": worker.plan_hash(plan),
        "registration_commit": receipt["commit"],
        "registration_sha256": worker.plan_hash(receipt),
        "reason": "user_requested_sleep_shutdown",
        "policy": "one explicit fresh attempt per listed parent; no automatic retries",
    }.items():
        require(declaration.get(key) == value, f"Recovery declaration {key} differs")
    authorized = utc_time(declaration.get("authorized_at_utc"))
    for key in ("shutdown_receipt_sha256", "controller_state_sha256"):
        require(
            re.fullmatch(r"[a-f0-9]{64}", declaration.get(key, "")),
            "Missing recovery evidence hash",
        )
    require(
        re.fullmatch(r"[a-f0-9]{40}", audit.get("public_declaration_commit", "")),
        "Recovery needs an immutable public commit",
    )
    name = audit.get("public_declaration_path", "")
    require(
        name.startswith("benchmarks/tabarena/")
        and name.endswith(".json")
        and ".." not in Path(name).parts
        and "\\" not in name
        and ":" not in name,
        "Unsafe public recovery declaration path",
    )
    require(
        audit.get("public_declaration_sha256") == worker.plan_hash(declaration),
        "Recovery public declaration digest differs",
    )
    if audit.get("archive_root"):
        archive = Path(audit["archive_root"]).resolve()
        require(
            all(
                archive != Path(root).resolve()
                and Path(root).resolve() not in archive.parents
                and archive not in Path(root).resolve().parents
                for root in inputs
            ),
            "Interrupted archives must remain outside collection roots",
        )
    entries = declaration.get("parents", [])
    require(
        len(entries) == 8 and len({entry["parent_id"] for entry in entries}) == 8,
        "Recovery must list exactly eight distinct interrupted parents",
    )
    expected = {parent["parent_id"]: parent for parent in plan["parents"]}
    archived = set()
    for entry in entries:
        parent = expected.get(entry["parent_id"])
        require(
            parent is not None and parent["owner"] == "local",
            "Recovery includes a foreign or remote parent",
        )
        key = f"{parent['config_name']}/{parent['task_id']}/{parent['repeat']}_{parent['fold']}"
        require(entry.get("key") == key, "Recovery parent repeat/fold path differs")
        previous = entry.get("previous_manifest", {})
        require(
            previous.get("parent") == parent
            and previous.get("plan_sha256") == identity["plan_sha256"]
            and previous.get("resources") == plan["resources"]
            and previous.get("host") == "local"
            and previous.get("status") == "running"
            and isinstance(previous.get("outer_split"), dict),
            "Recovery previous parent identity, status, resources, or split differs",
        )
        require(
            all(
                previous.get("registration", {}).get(k) == v
                for k, v in identity.items()
            ),
            "Recovery previous registration differs",
        )
        require(
            "python_executable" not in previous.get("runtime", {}),
            "Recovery declaration exposes local executable path",
        )
        validate_runtime(previous.get("runtime", {}), plan, worker)
        require(
            utc_time(previous.get("started_at"))
            <= utc_time(previous.get("fit_started_at"))
            < authorized,
            "Prior interrupted attempt did not precede recovery authorization",
        )
        files = entry.get("archived_files", [])
        for record in files:
            path = record.get("path", "")
            require(
                isinstance(path, str)
                and ".." not in Path(path).parts
                and "\\" not in path
                and ":" not in path
                and any(
                    path.startswith(f"{kind}/{key}/")
                    for kind in ("artifacts", "controller", "data")
                ),
                "Recovery archive path escapes its parent",
            )
            require(
                path not in archived
                and isinstance(record.get("size_bytes"), int)
                and record["size_bytes"] >= 0
                and re.fullmatch(r"[a-f0-9]{64}", record.get("sha256", "")),
                "Duplicate or invalid recovery archive inventory",
            )
            archived.add(path)
        require(
            f"artifacts/{key}/manifest.json" in archived
            and f"controller/{key}/claim.json" in archived,
            "Recovery lacks archived manifest or execution claim",
        )
    return {
        key: audit[key]
        for key in (
            "schema_version",
            "declaration",
            "public_declaration_path",
            "public_declaration_commit",
            "public_declaration_sha256",
        )
    }


def validate_replacement(manifest, entry, declaration):
    previous = entry["previous_manifest"]
    require(
        projected_runtime(manifest) == previous["runtime"],
        "Recovered parent runtime changed",
    )
    require(
        manifest.get("outer_split") == previous["outer_split"],
        "Recovered parent outer split changed",
    )
    require(
        utc_time(declaration["authorized_at_utc"])
        <= utc_time(manifest.get("started_at"))
        <= utc_time(manifest.get("fit_started_at"))
        <= utc_time(manifest.get("finished_at")),
        "Replacement did not start after explicit recovery authorization",
    )


def verify_public_recovery(audit, worker):
    url = (
        "https://raw.githubusercontent.com/captnmarkus/ctboost/"
        f"{audit['public_declaration_commit']}/{audit['public_declaration_path']}"
    )
    published = worker._base()._fetch_json(url)
    require(
        published == audit["declaration"]
        and worker.plan_hash(published) == audit["public_declaration_sha256"],
        "Immutable public recovery declaration differs",
    )


def collect(plan, inputs, receipt, *, worker=None, recovery_audit=None):
    """Discover exact parent coverage; deserialize only after all parents complete."""
    worker = worker or protocol()
    worker.validate_plan(plan)
    identity = registration_identity(plan, receipt, worker)
    recovery = (
        None
        if recovery_audit is None
        else validate_recovery_audit(plan, inputs, receipt, recovery_audit, worker)
    )
    recovered = (
        {}
        if recovery is None
        else {entry["parent_id"]: entry for entry in recovery["declaration"]["parents"]}
    )
    expected = {parent["parent_id"]: parent for parent in plan["parents"]}
    require(len(expected) == plan["expected_parent_count"], "Duplicate expected parent")
    found, raw_paths, roots = {}, set(), [Path(path).resolve() for path in inputs]
    require(
        roots and all(path.is_dir() for path in roots),
        "Every input must be an existing run directory",
    )
    require(
        not any(
            a == b or a in b.parents or b in a.parents
            for index, a in enumerate(roots)
            for b in roots[index + 1 :]
        ),
        "Overlapping collection roots",
    )
    for root in roots:
        for path in sorted(root.rglob("manifest.json")):
            manifest = read_json(path)
            if "parent" not in manifest:
                continue  # Transport bundle manifests are validated by the downloader.
            require(
                root in path.resolve().parents, "Manifest escapes its input directory"
            )
            parent = manifest["parent"]
            parent_id = parent.get("parent_id")
            require(
                parent_id in expected and parent == expected[parent_id],
                "Extra or foreign parent",
            )
            require(parent_id not in found, f"Duplicate parent: {parent_id}")
            key = (
                Path(parent["config_name"])
                / str(parent["task_id"])
                / f"{parent['repeat']}_{parent['fold']}"
            )
            require(
                len(path.parents) >= 5
                and path.relative_to(path.parents[4])
                == Path("artifacts") / key / "manifest.json",
                "Manifest path has a different repeat/fold identity",
            )
            raw = path.parents[4] / "data" / key / "results.pkl"
            require(
                root in raw.resolve().parents, "Raw result escapes its input directory"
            )
            require(
                manifest.get("raw_path")
                == (Path("data") / key / "results.pkl").as_posix(),
                "Manifest raw path differs",
            )
            require(
                manifest.get("plan_sha256") == identity["plan_sha256"]
                and manifest.get("host") == parent["owner"]
                and manifest.get("resources") == plan["resources"],
                "Parent plan, owner, or resource mismatch",
            )
            require(
                all(
                    manifest.get("registration", {}).get(k) == v
                    for k, v in identity.items()
                ),
                "Parent public registration differs",
            )
            runtime = manifest.get("runtime", {})
            validate_runtime(runtime, plan, worker)
            status = manifest.get("status")
            require(
                status in {"preparing", "running", "complete", "failed"},
                "Unknown parent status",
            )
            if status == "complete":
                if parent_id in recovered:
                    validate_replacement(
                        manifest, recovered[parent_id], recovery["declaration"]
                    )
                require(
                    not (path.parent / "resource_failure.json").exists()
                    and not any(
                        manifest.get(k) for k in ("error", "error_type", "traceback")
                    ),
                    "Complete parent retains failure evidence",
                )
                validation = manifest.get("validation", {})
                require(
                    raw.is_file()
                    and raw.stat().st_size == validation.get("size_bytes"),
                    "Missing or resized raw result",
                )
                require(
                    validation.get("runtime_sha256") == worker.plan_hash(runtime),
                    "Manifest runtime digest differs",
                )
                require(
                    math.isfinite(manifest.get("elapsed_seconds", float("nan")))
                    and manifest["elapsed_seconds"] >= 0,
                    "Invalid parent elapsed time",
                )
                raw_paths.add(raw.resolve())
            found[parent_id] = (path, raw, manifest)
    actual_raw = {
        path.resolve() for root in roots for path in root.rglob("results.pkl")
    }
    # Failed/interrupted workers may retain a partial raw result; it cannot enter evaluation.
    require(
        actual_raw <= {raw.resolve() for _, raw, _ in found.values()},
        "Undeclared raw result",
    )
    missing = sorted(set(expected) - set(found))
    failed = sorted(key for key, (_, _, m) in found.items() if m["status"] == "failed")
    active = sorted(
        key
        for key, (_, _, m) in found.items()
        if m["status"] in {"preparing", "running"}
    )
    report = {
        "schema_version": 1,
        "plan_sha256": identity["plan_sha256"],
        "expected_parents": len(expected),
        "complete_parents": len(raw_paths),
        "missing_parents": missing,
        "failed_parents": failed,
        "active_parents": active,
        "status": "failed"
        if failed
        else "pending"
        if missing or active
        else "complete",
    }
    if recovery is not None:
        report["recovery"] = recovery
    records = []
    if report["status"] == "complete":
        for parent in plan["parents"]:
            path, raw, manifest = found[parent["parent_id"]]
            require(
                file_hash(raw) == manifest["validation"].get("raw_sha256"),
                "Raw result checksum differs",
            )
            require(
                isinstance(manifest.get("outer_split"), dict),
                "Missing actual outer-split receipt",
            )
            validation = worker.validate_parent_result(
                raw,
                parent,
                identity["plan_sha256"],
                outer_split=manifest["outer_split"],
            )
            require(
                validation == manifest["validation"],
                "Revalidated parent differs from its manifest",
            )
            require(
                validation.get("outer_split") == manifest["outer_split"],
                "Raw and manifest outer-split receipts differ",
            )
            records.append(
                {
                    "parent_id": parent["parent_id"],
                    "host": parent["owner"],
                    "manifest_sha256": file_hash(path),
                    **validation,
                }
            )
        report.update(
            validated_children=sum(row["bag_children"] for row in records),
            parents=records,
        )
        require(
            report["validated_children"] == plan["expected_child_count"],
            "Incomplete child coverage",
        )
    return report, sorted(raw_paths)


def split_keys(plan):
    """TabArena's result column 'fold' is flattened repeat/fold, with sample=0."""
    from tabarena.benchmark.task.utils import get_split_idx

    dimensions = {}
    for split in plan["outer_splits"]:
        dims = dimensions.setdefault(split["dataset"], [0, 0])
        dims[0] = max(dims[0], split["fold"] + 1)
        dims[1] = max(dims[1], split["repeat"] + 1)
    return {
        (s["dataset"], s["repeat"], s["fold"]): get_split_idx(
            fold=s["fold"],
            repeat=s["repeat"],
            n_folds=dimensions[s["dataset"]][0],
            n_repeats=dimensions[s["dataset"]][1],
            n_samples=1,
        )
        for s in plan["outer_splits"]
    }


def validate_tables(plan, models, hpo):
    import numpy as np

    mapping = split_keys(plan)
    expected = {
        (p["config_name"], p["dataset"], mapping[p["dataset"], p["repeat"], p["fold"]])
        for p in plan["parents"]
    }
    require(
        len(models) == len(expected)
        and set(
            models[["method", "dataset", "fold"]].itertuples(index=False, name=None)
        )
        == expected,
        "Canonical model results omit, duplicate, or collapse an outer split",
    )
    expected_hpo = {
        (kind, key[0], index)
        for key, index in mapping.items()
        for kind in ("default", "tuned", "tuned_ensemble")
    }
    require(
        len(hpo) == len(expected_hpo)
        and set(
            hpo[["method_subtype", "dataset", "fold"]].itertuples(
                index=False, name=None
            )
        )
        == expected_hpo,
        "Canonical HPO results do not cover every full split and subtype",
    )
    for table in (models, hpo):
        require(
            "imputed" not in table or not table["imputed"].any(),
            "Imputed results are forbidden",
        )
        for column in (
            "metric_error",
            "metric_error_val",
            "time_train_s",
            "time_infer_s",
        ):
            require(
                np.isfinite(table[column].to_numpy(dtype=float)).all(),
                f"Nonfinite canonical {column}",
            )
            if column.startswith("time_"):
                require((table[column] >= 0).all(), "Negative canonical timing")
        require(
            not any(
                re.search(r"(^y_|pred|target)", name, re.IGNORECASE)
                for name in table.columns
            ),
            "Prediction/target column in public results",
        )


def evaluate(plan, raw_paths, output, source):
    """Official pinned API; no context download, test-set selection, or imputation."""
    import tabarena
    from tabarena.benchmark.task.metadata import TaskMetadataCollection
    from tabarena.end_to_end import EndToEnd, EndToEndResults
    from tabarena.models._method_metadata import MethodMetadata

    worker = protocol()
    require(
        source in Path(tabarena.__file__).resolve().parents,
        "Imported TabArena is not the specified pinned source",
    )
    require(
        worker.load_population(source / worker.METADATA_PATH) == plan["outer_splits"],
        "Evaluator metadata differs",
    )
    require(
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        == plan["tabarena_commit"],
        "Evaluator source commit differs",
    )
    require(
        subprocess.run(
            ["git", "-C", str(source), "diff", "--quiet", "HEAD", "--", "packages"],
            check=False,
        ).returncode
        == 0,
        "Evaluator source has changed",
    )
    require(
        {name: importlib.metadata.version(name) for name in plan["package_pins"]}
        == plan["package_pins"],
        "Evaluator package pins differ",
    )
    canonical = output / "canonical-results"
    done = output / "evaluation_complete.json"
    if not done.exists():
        EndToEnd.from_path_raw(
            path_raw=sorted({path.parents[3] for path in raw_paths}),
            file_paths=raw_paths,
            task_metadata=TaskMetadataCollection.from_source(
                source / worker.METADATA_PATH
            ),
            method="CTBoost",
            suite=SUITE,
            artifact_dir=canonical,
            backend="native",
            num_cpus=1,
            cache=True,
            cache_raw=False,
            cache_processed=False,
            cache_hpo_trajectories=False,
        )
    else:
        require(
            read_json(done)
            == {name: file_hash(canonical / "results" / name) for name in TABLES},
            "Cached evaluation changed",
        )
    metadata = MethodMetadata.from_yaml(path=canonical / "metadata.yaml")
    require(metadata.can_hpo, "Canonical metadata does not enable HPO")
    results = EndToEndResults.from_cache(methods=[metadata])
    validate_tables(
        plan, results.get_results(use_model_results=True), results.get_results()
    )
    write_json(done, {name: file_hash(canonical / "results" / name) for name in TABLES})
    return canonical


def checksums(folder):
    return {
        path.relative_to(folder).as_posix(): file_hash(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path != folder / "SHA256SUMS"
    }


def verify_stage(folder):
    entries = {}
    for line in (folder / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(
            name not in entries and re.fullmatch(r"[a-f0-9]{64}", digest),
            "Invalid public SHA manifest",
        )
        entries[name] = digest
    require(
        set(entries) == PUBLIC_FILES and entries == checksums(folder),
        "Public artifact inventory changed",
    )
    return entries


def stage(plan, inputs, receipt_path, output, source, *, recovery_audit=None):
    import yaml

    worker = protocol()
    receipt = read_json(receipt_path)
    report, raw_paths = collect(plan, inputs, receipt, recovery_audit=recovery_audit)
    require(
        report["status"] == "complete",
        "Full run is incomplete; no evaluation or publication permitted",
    )
    if recovery_audit is not None:
        verify_public_recovery(report["recovery"], worker)
    output.mkdir(parents=True, exist_ok=True)
    registration = worker.verify_registration(plan, receipt_path, output)
    binding = {
        "plan_sha256": worker.plan_hash(plan),
        "validation_sha256": worker.plan_hash(report),
        "registration_sha256": worker.plan_hash(receipt),
        "finalizer_sha256": worker.source_hash(__file__),
    }
    bound = output / "finalization_binding.json"
    if bound.exists():
        require(
            read_json(bound) == binding, "Finalization inputs or implementation changed"
        )
    else:
        require(
            not (output / "canonical-results").exists()
            and not (output / "public").exists(),
            "Unbound prior evaluation exists",
        )
        write_json(bound, binding)
    public = output / "public"
    if public.exists():
        verify_stage(public)
        require(
            read_json(public / MARKER)["binding"] == binding,
            "Staged publication belongs to other inputs",
        )
        return public
    canonical = evaluate(plan, raw_paths, output, source)
    temporary = output / "public-staging"
    temporary.mkdir(exist_ok=True)
    allowed = PUBLIC_FILES | {"SHA256SUMS"}
    require(
        {
            p.relative_to(temporary).as_posix()
            for p in temporary.rglob("*")
            if p.is_file()
        }
        <= allowed,
        "Foreign file in interrupted public staging",
    )
    (temporary / "canonical-results/results").mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        shutil.copyfile(
            canonical / "results" / name, temporary / "canonical-results/results" / name
        )
    metadata = yaml.safe_load((canonical / "metadata.yaml").read_text(encoding="utf-8"))
    metadata.update(has_raw=False, has_processed=False, has_results=True)
    (temporary / "canonical-results/metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8"
    )
    write_json(temporary / "plan.json", plan)
    write_json(temporary / "validation/completeness.json", report)
    write_json(temporary / "validation/registration.json", registration)
    write_json(
        temporary / MARKER,
        {
            "schema_version": 1,
            "repo_id": REPO_ID,
            "binding": binding,
            "all_parents_valid": True,
            "raw_and_target_arrays_published": False,
        },
    )
    card = f"""---
license: mit
language:
- en
tags:
- tabarena
- tabular-machine-learning
---
# CTBoost 0.1.60 — full TabArena, default + 25 HPO configurations

All 51 TabArena-v0.1 datasets and 816 official outer splits are included: 21,216
validated parent results and 169,728 bag children. Each parent uses eight sequential
bag folds, a 3,600-second training budget, two CPU threads and an 8 GiB memory limit.
This author-run study uses heterogeneous local/remote hardware; its timings are
**not canonical TabArena 8-CPU/32-GB timings**. Leaderboard admission remains subject
to maintainer review. No Elo or improvement claim is implied by this publication.

The default and first 25 frozen configurations were evaluated without test-guided
changes. The pinned official TabArena API selects tuned configurations and ensemble
weights from OOF validation predictions. No missing/failed results were imputed.
The [public protocol]({registration["public_url"]}) and validation inventory bind
the population, seeds, public wheels, sources, resource limits and parent hashes.

`canonical-results/metadata.yaml` and the two result Parquets are the official result
tier. Raw pickles, training targets and prediction arrays remain local and are not
included. `SHA256SUMS` covers every public artifact. The result files can be loaded
with `MethodMetadata.from_yaml` and `EndToEndResults.from_cache` from TabArena commit
`{plan["tabarena_commit"]}` after placing metadata beside its `results/` directory.
"""
    disclosure = ""
    if recovery_audit is not None:
        disclosure = (
            "Eight prior attempts were interrupted by a user-requested local shutdown, "
            "archived, and explicitly authorized for one fresh attempt each. The unchanged "
            "worker still disables automatic retries. The counts above describe retained "
            "results; interrupted work is excluded from successful-fit timings. The immutable "
            "declaration and prior-attempt hash inventory are included in "
            "`validation/completeness.json` under `recovery`.\n"
        )
        card += "\n" + disclosure
    (temporary / "README.md").write_text(card, encoding="utf-8")
    inventory = checksums(temporary)
    (temporary / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in inventory.items()),
        encoding="utf-8",
    )
    verify_stage(temporary)
    temporary.rename(public)
    (output / "draft_lennart_reply.md").write_text(
        f"The CTBoost 0.1.60 full TabArena run is complete: default + 25 HPO configurations "
        f"on all 51 datasets / 816 outer splits, with eight bag folds and no imputed failures. "
        f"Validated result tables and provenance: https://huggingface.co/datasets/{REPO_ID}. "
        "These are author-run 2-CPU/8-GiB results on mixed hardware, so timings are not "
        "canonical. Raw prediction artifacts remain local for an agreed review channel.\n"
        + disclosure,
        encoding="utf-8",
    )
    return public


def credential():
    from huggingface_hub import get_token

    token = get_token()
    if token:
        return token
    path = ROOT / ".env"
    require(path.is_file(), "No configured Hugging Face credential")
    matches = [
        re.fullmatch(
            r"""(?:[A-Za-z_][A-Za-z0-9_]*\s*=\s*["']?)?(hf_[A-Za-z0-9]+)["']?""",
            line.strip(),
        )
        for line in path.read_text(encoding="utf-8-sig").splitlines()
    ]
    values = [match.group(1) for match in matches if match]
    require(len(values) == 1, "Expected one configured Hugging Face credential")
    return values[0]


def publish(folder, output):
    """Explicit, restartable single-dataset publication; never update foreign data."""
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    inventory = verify_stage(folder)
    marker = read_json(folder / MARKER)
    require(
        marker.get("repo_id") == REPO_ID and marker.get("all_parents_valid") is True,
        "Publication has no complete full-run validation",
    )
    evidence = read_json(folder / "validation/completeness.json")
    if evidence.get("recovery") is not None:
        require(
            "archive_root" not in evidence["recovery"],
            "Public recovery audit contains a local archive path",
        )
        validate_recovery_audit(
            read_json(folder / "plan.json"),
            [],
            read_json(folder / "validation/registration.json")["receipt"],
            evidence["recovery"],
            protocol(),
        )
    require(
        evidence.get("status") == "complete"
        and evidence.get("complete_parents") == evidence.get("expected_parents")
        and not any(
            evidence.get(k)
            for k in ("missing_parents", "failed_parents", "active_parents")
        )
        and protocol().plan_hash(evidence) == marker["binding"]["validation_sha256"],
        "Publication completeness evidence differs from its binding",
    )
    binding = {
        "repo_id": REPO_ID,
        "stage_sha256": file_hash(folder / "SHA256SUMS"),
        "binding": marker["binding"],
    }
    receipt_path = output / "publication_receipt.json"
    receipt = read_json(receipt_path) if receipt_path.exists() else binding
    require(
        all(receipt.get(k) == v for k, v in binding.items()),
        "Prior publication receipt belongs to other artifacts",
    )
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    api = HfApi(token=credential())
    require(
        api.whoami()["name"].casefold() == "maiernator",
        "Unexpected Hugging Face account",
    )
    try:
        existing = api.repo_info(REPO_ID, repo_type="dataset")
    except RepositoryNotFoundError:
        existing = None
    expected = set(inventory) | {"SHA256SUMS"}
    if existing is None:
        require(
            not receipt.get("commit"), "Previously published repository disappeared"
        )
        api.create_repo(
            repo_id=REPO_ID, repo_type="dataset", private=False, exist_ok=False
        )
        receipt.update(
            created=True, created_head=api.repo_info(REPO_ID, repo_type="dataset").sha
        )
        write_json(receipt_path, receipt)
    elif receipt.get("commit"):
        require(
            existing.sha == receipt["commit"],
            "Published repository head changed after the recorded upload",
        )
    elif not receipt.get("commit"):
        remote_names = set(api.list_repo_files(REPO_ID, repo_type="dataset"))
        if MARKER in remote_names:
            # An upload can finish before the local receipt write. Recover only exact bytes.
            cached = Path(
                snapshot_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    revision=existing.sha,
                    local_dir=output / "upload-recovery",
                    token=credential(),
                )
            )
            require(
                remote_names - {".gitattributes"} == expected
                and (cached / "SHA256SUMS").read_bytes()
                == (folder / "SHA256SUMS").read_bytes()
                and all(
                    file_hash(cached / name) == digest
                    for name, digest in inventory.items()
                ),
                "Existing repository contains foreign or changed artifacts",
            )
            receipt["commit"] = existing.sha
        else:
            require(
                receipt.get("created") is True
                and receipt.get("created_head") == existing.sha
                and remote_names <= {".gitattributes"},
                "Refusing to overwrite an existing foreign dataset",
            )
    if not receipt.get("commit"):
        commit = api.upload_folder(
            repo_id=REPO_ID,
            repo_type="dataset",
            folder_path=folder,
            commit_message="Publish validated full CTBoost 0.1.60 HPO25 results",
            parent_commit=receipt["created_head"],
        )
        receipt["commit"] = commit.oid
    write_json(receipt_path, receipt)
    public = HfApi(token=False)
    info = public.repo_info(REPO_ID, repo_type="dataset", revision=receipt["commit"])
    require(not info.private, "Published dataset is not anonymously accessible")
    require(
        set(
            public.list_repo_files(
                REPO_ID, repo_type="dataset", revision=receipt["commit"]
            )
        )
        - {".gitattributes"}
        == expected,
        "Published file inventory differs",
    )
    downloaded = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=receipt["commit"],
            local_dir=output / "anonymous-verification",
            token=False,
        )
    )
    require(
        (downloaded / "SHA256SUMS").read_bytes() == (folder / "SHA256SUMS").read_bytes()
        and all(
            file_hash(downloaded / name) == digest for name, digest in inventory.items()
        ),
        "Anonymous download checksum mismatch",
    )
    receipt.update(
        anonymous_download_verified=True,
        url=f"https://huggingface.co/datasets/{REPO_ID}",
    )
    write_json(receipt_path, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--recovery-audit", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    import ctboost

    require(
        sys.flags.isolated
        and (ROOT / "ctboost") not in Path(ctboost.__file__).resolve().parents,
        "Use python -I with the installed public wheel",
    )
    sys.path.insert(0, str(ROOT))
    plan = read_json(args.plan)
    recovery = None if args.recovery_audit is None else read_json(args.recovery_audit)
    if args.check:
        report, _ = collect(
            plan, args.input, read_json(args.registration), recovery_audit=recovery
        )
        print(
            json.dumps(
                {k: v for k, v in report.items() if k != "parents"}, sort_keys=True
            ),
            flush=True,
        )
        return 0
    require(args.source is not None, "--source must name the pinned TabArena checkout")
    folder = stage(
        plan,
        args.input,
        args.registration,
        args.output.resolve(),
        args.source.resolve(),
        recovery_audit=recovery,
    )
    result = (
        publish(folder, args.output.resolve())
        if args.publish
        else {"status": "staged", "path": str(folder)}
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
