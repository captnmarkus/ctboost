# Local CPU worker throughput comparison

This is a performance screen for the published CTBoost 0.1.60 Windows wheel on
the author's Ryzen 7 5800X3D (8 physical cores, 16 logical processors, 32 GiB RAM).
It does not produce HPO scores or alter CTBoost's learning rules. The official
HPO200 run stops admitting local parents while its active parents finish;
remote execution continues. No active official fit is interrupted for this test.

## Fixed comparison

The Windows topology API reports the actual physical-core relationships. Every
layout covers the same 16 process-visible logical CPUs exactly once:

- **A:** the existing eight workers, two CPU threads each, with slots
  `(0,8), (1,9), ... (7,15)`.
- **B:** sixteen workers, one CPU thread each.
- **C:** eight workers, two CPU threads each, placing both SMT siblings of one
  physical core in each worker's slot. On this machine those pairs are
  `(0,1), (2,3), ... (14,15)`.

Run the layouts in the fixed order **A, B, C, C, B, A**. Every batch receives the
same 32 jobs, interleaved equally across numeric binary classification, numeric
regression, numeric multiclass classification, and categorical classification.
The immutable workload specification fixes all synthetic data, seeds, learner
parameters, and 128 training iterations. There is no validation set, early
stopping, test-score selection, or per-job training deadline. A separate
four-iteration warmup precedes each worker's readiness barrier.

The numeric binary profile uses 32,768 rows and 96 numeric columns, crossing
CTBoost's default parallel-node-histogram threshold of 1,048,576 values even
after its fixed 80% row subsample. The other profiles remain smaller, exercising
serial work as well. This v2 specification supersedes the unused v1 specification
after static source review; no timing fits were run under v1.

The harness records the workload and source digests, public-wheel runtime,
actual CPU topology and affinities, per-job results, batch wall time, startup
overhead, total process-tree memory, and minimum free system memory. It reports
both throughput after workers are ready and throughput including startup.
Abort the timing batch if free system memory falls below 4 GiB, a worker fails,
startup exceeds two minutes, or a batch exceeds twenty minutes. Only owned
timing processes may be stopped; preserve all diagnostics.

## Decision rule

All 192 timed fits must finish their 128 iterations with finite predictions.
For each job, check data/specification identity and prediction agreement across
all six batches. Report bitwise equality separately; numerical acceptance uses
absolute and relative tolerances of `1e-12`. Warmups are separate evidence and
are not counted as timed fits.

A candidate must improve throughput after readiness by at least 10% in both
comparisons against A: its first batch against the first A batch, and its second
batch against the final A batch. Throughput including startup must also avoid a
regression in both comparisons. The memory floor must hold throughout.

If both candidates qualify, prefer C when its geometric-mean throughput is
within 5% of B, because C preserves the existing per-parent CPU-thread count.
Otherwise select the qualifying candidate with higher measured throughput.
If neither qualifies, retain the current layout. Any failed or incomplete screen
leaves the existing official execution policy in place.

Four synthetic profiles and two repetitions are an initial throughput screen,
not proof of a speedup or safe fixed concurrency across all 51 datasets. Memory
admission remains necessary. A change to the official run must be documented
before applying it, preserving the original plan, started-parent evidence and
source/runtime receipts. Changing CPU resources can change how much training
fits within the official time limit, so results from differing resource policies
must remain distinguishable.
