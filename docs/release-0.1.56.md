# CTBoost 0.1.56

CTBoost 0.1.56 is an alpha release focused on training throughput, CUDA and
distributed foundations, and portable inference. It does not change the
conditional-inference learner: the protected split-statistics sources are
byte-identical to 0.1.55. The quadratic feature test with no multiplicity
adjustment remains the default, grouped testing remains opt-in, and Bonferroni
remains a stopping-threshold correction rather than a score optimizer.

## Highlights

- Sparse CSC quantization now scans each column linearly, and CPU node
  histograms can run deterministically across features.
- `Pool.from_cuda_quantized(...)` accepts validated, single-device CUDA Array
  Interface buffers without a host-data round trip.
- GPU training uses the shared conditional-inference candidate selector,
  including grouped statistics and per-node Bonferroni adjustment, while split
  optimization continues to use the original raw bins.
- The Spark adapter adds fail-closed barrier execution, stable global query IDs,
  and explicit collection rather than silently collecting in `mode="auto"`.
- Version-2 JSON predictors can carry a fitted Python preprocessing pipeline;
  loaders now reject oversized, duplicate-key, non-finite, cyclic, out-of-bounds,
  and structurally inconsistent artifacts.
- Prepared-numeric, inference-only source runtimes are included for Java 17 and
  R, backed by a shared conformance corpus.

## Compatibility and boundaries

The Python runtime reads valid version-1 and version-2 JSON predictors. Older
version-1-only readers reject new version-2 artifacts. Raw fitted preprocessing
runs only in Python; the R and JVM modules require prepared numeric features and
are source modules in this release, not CRAN or Maven publications.

Spark training now defaults to fail-closed `mode="auto"`; pass
`mode="collect"` explicitly to permit driver collection. The initial
distributed GPU transport remains TCP/barrier based and is not an NCCL,
GPU-direct, elastic, or fault-tolerant multi-node stack.

The native `Pool` C++ class layout changed. Ordinary wheel users receive a
matching rebuilt extension, but applications that compile or link CTBoost's
repository-native headers or static core must rebuild. Generated model-specific
C/C++ scorer interfaces are unchanged.

Model fingerprints detect accidental inconsistency; they are not cryptographic
signatures. Exported models remain trusted artifacts. Valid legacy model and
pipeline states remain supported, while malformed states that older versions
tolerated may now fail closed.

## Benchmark interpretation

The throughput work is intended to preserve deterministic model behavior, not
to claim a predictive-score improvement. The latest measured TabArena evidence
remains the 0.1.55 three-dataset default smoke at 1054.0 provisional Elo. It is
not a full or official leaderboard result and is not evidence of a 0.1.56 Elo
gain.
