# CTBoost 0.1.60 local worker comparison

**Decision: retain eight workers with two CPU threads per parent (layout A).**
Neither candidate met the registered requirement of at least 10% higher steady
throughput in both paired comparisons. The official local HPO queue resumed on
2026-09-07 at 13:15:31 UTC with its original resource policy. All 19 parents that
were active or completed before the pause finished successfully; none was
interrupted for this test. Remote execution continued independently.

## Results

The machine was a Ryzen 7 5800X3D with eight physical cores, sixteen logical
CPUs and 32 GiB RAM. The test used the installed public CTBoost 0.1.60 Windows
wheel, Python 3.12.11, four fixed synthetic profiles and the order A, B, C, C,
B, A. Each batch ran the same 32 jobs for exactly 128 iterations. There were
192 successful timed fits and 64 separate four-iteration warmups.

| Layout | Workers × threads | Steady throughput gain, first / second pair | Throughput gain including startup, first / second pair | Sampled peak pool RSS | Qualifies |
| --- | --- | --- | --- | --- | --- |
| A: existing cross-core pairs | 8 × 2 | reference | reference | 1.81 GiB | retained |
| B: one logical CPU per worker | 16 × 1 | +6.1% / +18.0% | +17.7% / +4.1% | 3.34–3.46 GiB | no |
| C: actual SMT sibling pairs | 8 × 2 | +5.8% / +12.0% | +34.7% / +8.6% | 1.75–1.77 GiB | no |

The first B and C batches are compared with the first A batch; their second
batches are compared with the final A batch. Gains describe throughput, not
the percentage reduction in elapsed time.

| Batch | Layout | Seconds after readiness | Seconds including startup and exit |
| --- | --- | ---: | ---: |
| 0 | A | 8.409 | 14.225 |
| 1 | B | 7.926 | 12.088 |
| 2 | C | 7.951 | 10.564 |
| 3 | C | 8.051 | 11.054 |
| 4 | B | 7.642 | 11.528 |
| 5 | A | 9.016 | 12.002 |

Predictions were bitwise identical for corresponding jobs across all six
batches; the maximum absolute difference was zero. Every timed fit retained
128 iterations and produced finite predictions. Sampled free system memory
stayed above 16.0 GiB, exceeding the 4 GiB guard. There were no failed timing
workers.

These short synthetic batches are an initial screen. Their roughly eight-second
steady intervals and visible startup variation do not establish a reliable
speedup across the 51 official datasets. They also do not estimate HPO scores
or completion time. The larger numeric profile crosses the parallel histogram
threshold, but this does not make its runtime representative of every HPO fit.
Sixteen workers used about 1.9 times the sampled pool RSS of A in this screen.

## Evidence and verification

The [protocol](../../tabarena/WORKER_THROUGHPUT_0160.md) and [specification](spec.json)
were published before timing at commit
`5151d0cad969c54c728ed73caf48b322f513153b`. The earlier v1 specification was
[retired without any fits](../worker_throughput_20260907/retired-without-fits.json).

- [Machine-readable summary](summary.json), including unrounded timings and ratios.
- [Public registration verification](registration.json) and [dispatch resumption receipt](operator.json).
- [Raw timing evidence](timing-evidence.zip): all jobs, predictions, worker readiness,
  affinities, claims, logs, batch records and the frozen specification.
- [Archive and per-file SHA-256 manifest](evidence-manifest.json).

The archive was read back and every entry checked against its SHA-256 digest.
The harness and topology tests passed (33 tests), and all nine cross-platform
CMake CI jobs passed for the registered test commit. No library code, official
HPO resource policy or TabArena PR was changed by the decision.
