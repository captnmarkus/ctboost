# Final CI evidence for CTBoost 0.1.61

The recorded CPU and release dry-run gates passed on source
`5780b41c7321c70e508dccf34e2f76f2503d1175`, branch
`improve/inference-accuracy-20260908`. This run did not create or update a tag,
published package, or GitHub release asset: **both publication jobs were
skipped**. Monitoring used read-only GitHub APIs.

- [CPU CI 34270193368](cpu-34270193368.md): all 13 jobs passed. Twelve jobs ran
  the full Python suite (1816 passed, 49 skipped) plus native CTest; the separate
  MSVC-fast guard ran three focused tests with 74 deselected.
- [Release dry run 34270927962](publish-34270927962.md): 17 successful jobs and
  two skipped publication jobs. Validation covered 26 wheels (10 CUDA-enabled,
  16 CPU-only) and one source distribution.
- The source distribution's JVM conformance suite passed nine tests, R reported
  `Status: OK`, and rebuilding the source archive passed four installed smoke
  tests. Exact excerpts and required step outcomes are in
  [final-gates.json](final-gates.json).

The validated [release-dist artifact](https://github.com/captnmarkus/ctboost/actions/runs/34270927962/artifacts/10074783361)
is ID `10074783361`, **75,764,206 bytes**, digest
`sha256:f1325aa5289960055aeb657606948f303edc7f8b80a7f10b8cc80dd583f454dd`.
[Artifact metadata](release-dist-artifact.json) records the API identity; no
large binaries were downloaded for this receipt.

Final run metadata, per-job step outcomes, test counts and selected exact log
lines are retained here. Full successful logs remain in local scratch and at
stable GitHub job URLs; original log hashes appear in each summary. Historical
failures remain in the earlier phase archives. These checks establish the
recorded CI gates, not new TabArena accuracy or Elo results.
