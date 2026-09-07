"""Dispatch only the frozen local assignments of the public 0.1.60 HPO200 plan.

Uses the existing owned-process, durable launch, RAM admission, pause and
no-retry controller. Started failures are terminal; other assignments continue.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_BASE_PATH = ROOT / "benchmarks/tabarena/local_hpo_controller.py"
_spec = importlib.util.spec_from_file_location("_ctboost_hpo_controller_base", _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


class Controller(base.Controller):
    def __init__(self, *, output, plan_path, wheel, registration, worker,
                 python=None, max_workers=8, poll_seconds=5):
        self.output = Path(output).resolve()
        self.plan_path = Path(plan_path).resolve()
        self.wheel = Path(wheel).resolve()
        self.registration = Path(registration).resolve()
        self.worker = worker
        self.worker_script = Path(worker.__file__).resolve()
        self.python = str(Path(python or sys.executable).resolve())
        if self.python != str(Path(sys.executable).resolve()):
            raise ValueError("Run the controller with the same Python used for its parents")
        self.plan = worker.validate_plan(worker.read_json(self.plan_path))
        self.plan_hash = worker.plan_hash(self.plan)
        self.policy = base.allocation_policy(base.psutil.Process().cpu_affinity(), max_workers)
        if self.plan["resources"] != worker.RESOURCES:
            raise ValueError("The local worker resource contract differs from the plan")
        if not math.isfinite(poll_seconds) or not 0.1 <= poll_seconds <= 30:
            raise ValueError("poll_seconds must be between 0.1 and 30")
        self.poll_seconds = poll_seconds
        self.registration_verified = worker.verify_registration(
            self.plan, self.registration, self.output
        )
        self.runtime = worker.runtime_provenance(self.plan, self.wheel)
        self.runtime_hash = worker.plan_hash(self.runtime)
        local = [parent for parent in self.plan["parents"] if parent["owner"] == "local"]
        self.parents = {base.parent_key(parent): parent for parent in local}
        if not local or len(self.parents) != len(local):
            raise ValueError("Missing or duplicate local parent assignments")
        self.processes = {}
        self.state_path = self.output / "controller_state.json"
        self._freeze_execution()
        if self.state_path.exists():
            self.state = base.read_json(self.state_path)
            if (self.state.get("plan_sha256") != self.plan_hash
                    or set(self.state.get("parents", {})) != set(self.parents)
                    or any(r.get("status") not in base.ACTIVE | base.TERMINAL | {"queued"}
                           for r in self.state["parents"].values())):
                raise ValueError("Controller state differs from the frozen local assignments")
        else:
            self.state = {"schema_version": 1, "plan_sha256": self.plan_hash,
                          "parents": {key: {"status": "queued"} for key in self.parents},
                          "observed_dataset_peak_bytes": {}}
        self.save()

    def _freeze_execution(self):
        binding = {
            "schema_version": 1, "controller": "ctboost_local_hpo_0160",
            "plan_sha256": self.plan_hash, "plan_path": str(self.plan_path),
            "controller_source_sha256": self.worker.source_hash(__file__),
            "base_controller_source_sha256": self.worker.source_hash(_BASE_PATH),
            "worker_source_sha256": self.worker.source_hash(self.worker_script),
            "python": self.python, "python_sha256": base.file_hash(self.python),
            "base_python": str(Path(sys._base_executable).resolve()),
            "base_python_sha256": base.file_hash(sys._base_executable),
            "psutil_version": base.psutil.__version__,
            "wheel_path": str(self.wheel), "wheel_sha256": base.file_hash(self.wheel),
            "runtime_sha256": self.runtime_hash,
            "registration_path": str(self.registration),
            "registration": self.registration_verified,
            "allocation_policy": self.policy, "poll_seconds": self.poll_seconds,
        }
        path = self.output / "controller_execution.json"
        if path.exists():
            existing = base.read_json(path)
            if {key: value for key, value in existing.items() if key != "created_at"} != binding:
                raise ValueError("Frozen local controller execution or provenance changed")
        else:
            base.write_json(path, {**binding, "created_at": base.now()})

    def _validate_completed(self, parent, manifest, raw):
        if (manifest.get("status") != "complete" or manifest.get("parent") != parent
                or manifest.get("plan_sha256") != self.plan_hash
                or manifest.get("host") != "local"
                or manifest.get("resources") != self.plan["resources"]
                or manifest.get("registration") != self.registration_verified
                or self.worker.plan_hash(manifest.get("runtime", {})) != self.runtime_hash
                or manifest.get("validation", {}).get("raw_sha256") != base.file_hash(raw)):
            raise ValueError("Completed parent identity, runtime, resources or artifact changed")
        validated = self.worker.validate_parent_result(raw, parent, self.plan_hash)
        if validated.get("runtime_sha256") != self.runtime_hash:
            raise ValueError("Raw parent runtime differs from the attested local worker")
        return validated

    def _worker_command(self, parent, slot, claim, permit, token):
        return [
            self.python, "-I", str(Path(__file__).resolve()), "_worker",
            "--claim", str(claim), "--permit", str(permit), "--token", token, "--",
            str(self.worker_script), "run-parent", "--plan", str(self.plan_path),
            "--output", str(self.output), "--parent-id", parent["parent_id"],
            "--host", "local", "--wheel", str(self.wheel),
            "--registration", str(self.registration), "--affinity",
            ",".join(str(cpu) for cpu in self.policy["cpu_slots"][slot]),
        ]

    def launch(self, key, slot, reservation):
        try:
            super().launch(key, slot, reservation)
        except Exception as exc:
            # Base launch has durably recorded a terminal failure and stopped
            # its verified process tree. Never retry it; continue other parents.
            record = self.state["parents"][key]
            if record["status"] != "launch_failed":
                raise
            record["error"] = f"{type(exc).__name__}: {exc}"
            self.save()

    def _finish_from_artifacts(self, key, *, fallback, exit_code=None):
        super()._finish_from_artifacts(key, fallback=fallback, exit_code=exit_code)
        manifest_path, _, _ = base.parent_paths(self.output, self.parents[key])
        record = self.state["parents"][key]
        if manifest_path.exists():
            manifest = base.read_json(manifest_path)
            record["manifest_sha256"] = base.file_hash(manifest_path)
            if manifest.get("error"):
                record["worker_error"] = manifest["error"]
        failure = manifest_path.with_name("resource_failure.json")
        if failure.exists():
            record["resource_failure"] = base.read_json(failure)
            if record["resource_failure"].get("status") == "timeout":
                record["status"] = "timed_out"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_worker":
        return base._worker_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "plan", "wheel", "registration"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args(argv)
    # Native package must be imported from the wheel before benchmark source is exposed.
    import ctboost
    if (ROOT / "ctboost") in Path(ctboost.__file__).resolve().parents:
        raise RuntimeError("Use python -I with the installed public 0.1.60 wheel")
    sys.path.insert(0, str(ROOT))
    from benchmarks.tabarena import hpo_0160 as worker
    with base.controller_lock(args.output):
        controller = Controller(output=args.output, plan_path=args.plan, wheel=args.wheel,
                                registration=args.registration, worker=worker,
                                max_workers=args.max_workers, poll_seconds=args.poll_seconds)
        return controller.run()


if __name__ == "__main__":
    raise SystemExit(main())
