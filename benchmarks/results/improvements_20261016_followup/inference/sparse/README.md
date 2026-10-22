# Compact prediction bins and categorical dictionary experiment

These experiments use the previously validated composed runtime as the baseline.
They preserve the fitted conditional-inference trees, split decisions, leaf
values, accumulation order, and public histogram layout. No model fitting or
accuracy selection occurs in these latency measurements.

The private native candidate stores bins only for features used anywhere in the
cached ensemble and remaps cached traversal addresses. Schema lookups and prefix
activity remain indexed by the original feature ids. The identity variant leaves
fully used schemas in their original layout. Root-only prefixes, forbidden NaNs,
two bin widths, sparse inputs, multiclass vector/scalar leaves, and original leaf
ids are covered by 25 additional contract tests. Those tests also pass against
the unchanged baseline; all 108 prediction tests and 230 native checks pass on
both candidate revisions. Full transformed matrices and predictions match the
saved baseline bit for bit on all five real workloads at 1, 64, and 1000 rows.

`initial/` preserves the original ABBA experiment, including all unfavorable
results. Its five-case geometric raw speedup was 1.0000; prepared 1.0889 and
native 1.2062. Hiva uses 22 of 6468 transformed columns: at 1000 rows the native call
fell from 1.1903 to 0.1664 ms (7.155×), while raw preprocessing remained dominant.
Other native/prepared workloads showed small repeatable regressions, including
high-cardinality 4096-row prepared 0.965× and SDSS 64-row native 0.957×. An isolated
splice 1-row raw outlier is retained, not removed from the aggregate.

`identity/` contains a balanced six-pass comparison of the composed baseline,
the independent Python categorical-dictionary candidate, and the native
identity-layout refinement: A–B–C–C–B–A. The Python candidate keeps the native
binary unchanged. Its actual Python hashes and prototype receipt are embedded
in the frozen plan and timing receipts; a copied native-build receipt alone
would not identify that candidate accurately. The two candidates are measured
independently; their gains must not be multiplied to imply a composed result.

The dictionary candidate's current `len(data) > 1` dispatch is rejected. Its
raw geometric speedup is 0.9563, despite repeatable 1000-row gains on hiva
(212.12→143.57 ms, 1.477×) and splice (13.642→10.862 ms, 1.256×). At 64 rows it
regresses hiva 48.55→100.57 ms (0.483×) and splice 2.637→4.550 ms (0.580×).
Prepared/native controls are 1.0025×/1.0019×; the high-cardinality control raw
speedup is 1.0033×. This is a batch-size crossover, not a broad improvement.

The identity native revision repeats the targeted memory-layout benefit:
real raw 1.0026×, prepared 1.0817×, native 1.2052×. Hiva 1000-row native latency
falls 1.1757→0.1672 ms (7.030×), and prepared improves 3.520×. Non-target costs
remain: splice 1000-row native 0.975×, and high-cardinality 4096-row prepared
0.963×. SDSS 64-row native is 0.988× with the identity bypass. A transient
first-pass splice 64-row prepared outlier remains in the 0.902× summary.
Neither native revision is in the shared production checkout at archive time;
the tradeoff must be explicitly decided rather than hidden by the aggregate.

The runners use one fixed CPU, one thread, warmup/calibration, seven timing
blocks, and fresh processes. Other team builds, tests, fitting, and solves were
paused during each timed window. Real raw timings include the existing shared
frame normalization; prepared timings include wrapper/Pool construction on
already transformed arrays; native timings reuse a prepared Pool. The synthetic
case additionally measures an unchanged CatBoost control. Source patches,
runner copies, source/native hashes, protocols, raw blocks, and summaries are
retained. Model/input fixtures remain in the local cache and are identified by
their hashes in every raw receipt; this archive does not contain those large
fixtures and is not independently runnable without them.

These saved development workloads diagnose inference behavior. They do not
establish an accuracy ranking or an end-to-end advantage over other libraries.
