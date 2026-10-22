# Independent follow-up candidates

These measurements compare each candidate with the already accepted builtin missing-value and zero-active-bin-allocation changes. They do not use the original unoptimized checkout as baseline. The six-pass order was **accepted–exact-string–nodes24–nodes24–exact-string–accepted**, with the same five saved real fixtures and a separate high-cardinality categorical-2000 synthetic regression check. The frozen protocol, source snapshots, build receipts, every raw pass, and summaries are retained.

## Retained: exact Python-string conversion

`NormalizeKey` bypasses the redundant `PyObject_Str` conversion only for exact builtin strings. Custom subclasses retain `__str__`, and all strings still pass through the existing versioned key escaping. All 133 key-codec/numeric/CTR tests passed, and every timed pass passed the full transformed-feature and prediction parity checks.

The targeted 1,000-row real cases improved consistently in both mirrored comparisons:

| Case | Baseline pass medians ms | Candidate pass medians ms | Geometric speedup |
| --- | --- | --- | ---: |
| hiva_agnostic | 232.610, 233.721 | 215.197, 212.029 | 1.092x |
| splice | 14.509, 14.345 | 13.767, 13.655 | 1.052x |

The original 15-case real-input raw aggregate was **1.0001x**, so this is not a general inference-speed claim. Its first coil-64 candidate pass contained a transient raw timing sequence of 1.149, 1.097, 0.879, 0.980, 0.753, 0.721, and 0.716 ms; the second candidate pass was 0.693 ms, beside baseline medians 0.696 and 0.699 ms. A separately frozen coil-64-only replication investigated this apparent regression and was neutral: **0.9997x**, with paired ratios 0.9952x and 1.0042x. The original outlier remains in the original aggregate and all raw receipts; the replication does not replace it.

The categorical-2000 synthetic raw geometric ratio was **1.028x** across batches 1, 64, and 4,096; prepared-Pool ratio was 1.003x and the unchanged CatBoost control ratio was 0.986x. No equal-quality comparison is implied by this control.

## Rejected: 24-byte cached nodes

A union reduced each cached node from 32 to 24 bytes by sharing the mutually exclusive leaf update and internal split metadata. The original training/serialized node representation stayed unchanged. It passed 83 prediction-reference tests, 230 native smoke checks, and every timed parity check, but materially slowed inference:

| Suite | Raw geometric ratio | Prepared ratio | Native ratio / unchanged CatBoost control |
| --- | ---: | ---: | ---: |
| Five real cases × three batches | 0.946x | 0.889x | 0.853x native |
| Categorical-2000 × three batches | 0.955x | 0.847x | 0.986x control |

The candidate was removed from production. Its source and complete negative results remain here. Smaller node storage did not produce faster traversal in these measurements; the study did not isolate the microarchitectural cause.

## First Python one-row candidate: mixed, requires refinement

`row-timing/` records an independent accepted–row-fast–row-fast–accepted sequence for rows 1 and 64. Native code was identical. The candidate avoided pandas' common matrix conversion only for a strictly guarded one-row categorical string frame. It passed 23 focused extraction tests and all timed transform/output checks.

| Case, one row | Baseline ms | Candidate ms | Speedup |
| --- | ---: | ---: | ---: |
| hiva_agnostic | 37.237 | 28.677 | 1.298x |
| splice | 1.632 | 1.388 | 1.176x |
| coil2000_insurance_policies | 0.383 | 0.444 | 0.862x |
| GiveMeSomeCredit | 0.194 | 0.247 | 0.786x |
| SDSS17 | 0.308 | 0.368 | 0.839x |

The 64-row controls stayed near parity. The repeated one-row regressions on frames that should fall back show that the guard itself needs improvement. This measured version is not accepted as a broad serving optimization; any refined candidate must receive a separate check and receipt.

## Reproduction and limitations

`../runner.py` and `measure_synthetic.py` are frozen runners. Runtime packages and saved model/input fixtures remain local as described in the parent README. Every receipt records binary/source/model/input hashes. Both candidates used one CPU/thread and identical dependencies. Precise timing windows were coordinated with other agents; ordinary operating-system activity remains uncontrolled. Raw prediction includes preprocessing, and prepared/native paths are diagnostic. All fixtures are reused development inputs; no new training, label-driven model selection, independent holdout assessment, or leaderboard evaluation occurred here.

`../summarize_candidates.py <suite-directory> <candidate-name>` regenerates the summaries from raw receipts; use `exact-string`, `nodes24`, or `row-fast` as appropriate. The raw real and synthetic suites remain separate, as does the coil replication.
