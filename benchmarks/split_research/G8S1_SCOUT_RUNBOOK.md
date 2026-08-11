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

## Pristine pinned TabArena checkout

The pinned `packages/tabarena/src` import root must contain exactly the tracked
`tabarena` source tree: no `__pycache__`, `.pyc`, `.pyd`, `.so`, symlink,
untracked namespace, or ignored/generated sibling. Python's `-B` option
prevents new bytecode writes but can still read an existing valid cache,
including a Git-ignored cache that is invisible to ordinary status output.
Start from a fresh checkout or remove/quarantine only verified generated
entries beneath that exact import root before preflight. This includes editable
install residue such as `tabarena.egg-info`; the pinned environment's installed
`.dist-info` remains outside the source root. The harness compares every source
entry with the pinned commit before adding the root to `sys.path` or importing
any TabArena module.

## Frozen portfolio portability

The exact P200 schedule source is the hash-sealed `g8s1_scout/p200.json`
document. The live adapter generator is checked against that document for
identical configuration/key order and exact non-floating values; every float
must be equal or at most one IEEE-754 binary64 ULP away. It is never used as
the schedule source.

Before PR #17, the adapter's logarithmic sampler exposed host `libm` drift:
Linux and Windows differed by one ULP at P200 cells 87 (`ctr_prior_strength`)
and 197 (`alpha`), despite identical source, seed, and NumPy 2.5.2. PR #17,
now included in current `master`, supplies exact tuple canonicalization for
those two frozen samples. The hash-sealed P200 document remains the schedule
source, and the live adapter generator remains audited against it; do not
replace the document with platform-generated output or allowlist
platform-specific portfolio hashes.

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
report directory. Namespace validation treats every such directory as
unexpected and refuses to continue, even when the abandoned directory is
empty. An empty `sanitized` directory is likewise invalid.

To recover, first stop every process using the namespace. Resolve and verify
the exact candidate namespace and report directory, then move the entire
abandoned staging directory to a private quarantine outside the benchmark
results tree. If `sanitized` exists but is not one of the exact complete
success or failure layouts, quarantine the entire `sanitized` directory too.
Never merge, rename, or copy individual partial files into `sanitized`, and do
not edit the sealed raw artifacts or provenance. Rerun `preflight`; only after
the namespace validates should `summarize` be rerun from the sealed raw
artifacts.
