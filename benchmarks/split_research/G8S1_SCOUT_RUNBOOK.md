# Grouped-8 scout runbook

This runbook is part of the sealed grouped-statistic scout identity. The scout
must fail closed if this file, its manifest entry, the runtime sources, the
candidate installation, or the output namespace differs from the reviewed
state.

## Pristine candidate installation

Use a fresh Python 3.12 environment for the candidate wheel. Do not reuse an
environment that previously imported or installed CTBoost. Install the wheel
without generating bytecode and keep bytecode disabled for every later scout
process. In PowerShell:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m pip install --no-compile --no-deps <candidate-wheel>
```

The installed `ctboost` directory must contain exactly the release-source
Python files and one top-level `_core` extension. It must contain no `.pyc`, no
second `.pyd` or `.so`, no linked importable file, and no stale file from an
older installation. Every importable file must have an exact hashed wheel
`RECORD` entry. The harness verifies those conditions and binds the installed
Python source map to the requested clean, merged CTBoost commit before it
imports CTBoost.

## Canonical invocation

Set `PYTHONDONTWRITEBYTECODE=1` and invoke every `preflight`, `run`, and
`summarize` command through the external bootstrap with both `-I` and `-B`:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -I -B benchmarks/split_research/g8s1_scout_bootstrap.py <command> `
  --tabarena-root <pinned-tabarena-checkout> `
  --expected-ctboost-commit <full-40-character-commit> `
  --expected-native-sha256 <installed-core-sha256>
```

Do not run `python -m g8s1_scout` or import the harness by adding its directory
to `PYTHONPATH`. `-I` keeps caller-controlled paths out of initial import
resolution, `-B` prevents the bootstrap process from writing bytecode, and
`PYTHONDONTWRITEBYTECODE=1` carries that protection into child processes.

## Interrupted publication and recovery

Ordinary Python interruptions, including `KeyboardInterrupt` and `SystemExit`,
run the success/failure staging cleanup before they propagate. An operating
system hard termination (`SIGKILL`, `TerminateProcess`), machine reset, or
power loss cannot run Python cleanup. It can therefore leave an abandoned
`.g8s1-success-*` or `.g8s1-failure-*` staging directory under the exact scout
report directory. Namespace validation treats any files in such a directory as
unexpected and refuses to continue. An empty abandoned staging directory is
also not evidence of a completed publication.

To recover, first stop every process using the namespace. Resolve and verify
the exact candidate namespace and report directory, then move the entire
abandoned staging directory to a private quarantine outside the benchmark
results tree. If `sanitized` exists but is not one of the exact complete
success or failure layouts, quarantine the entire `sanitized` directory too.
Never merge, rename, or copy individual partial files into `sanitized`, and do
not edit the sealed raw artifacts or provenance. Rerun `preflight`; only after
the namespace validates should `summarize` be rerun from the sealed raw
artifacts.
