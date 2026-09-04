# Compatibility and limits

## Runtime matrix

- Python: 3.8 through 3.14 in package metadata;
- CPU wheels: Linux, Windows, and macOS according to release artifacts;
- CUDA: Linux x86-64 and Windows AMD64 only, according to release artifacts;
- source builds: CMake 3.24+, a C++17 compiler, and pybind11 through the build backend.

## Explicit limitations

- Multi-output and multilabel wrappers fit independent boosters rather than vector-leaf trees.
- Compact vector leaves are opt-in for CPU multiclass classification with more
  than two classes; GPU and distributed vector training are unsupported.
- AFT uses a Python log-normal objective, numeric/prepared input, and no distributed fit.
- Continuing a callable-objective model requires supplying the callable again.
- Exact SHAP is empirical interventional and currently CPU-oriented.
- Object importance is approximate shared-leaf influence.
- Spark barrier training is an initial native-shard path; `mode="collect"` remains the
  explicit driver-memory fallback. Distributed evaluation sets are not supported.
- Ordinary Arrow/Polars/cuDF/CuPy/DLPack inputs materialize into CTBoost-owned host
  arrays. The explicit quantized-CUDA pool API has a narrower device-resident contract.
- Streaming Pools are numeric-only.
- Generated Python/C++ and ONNX exports require prepared numeric features for models with
  fitted categorical, text, or embedding pipelines.
- The repository's R and JVM packages are inference-only and prepared-feature-only;
  they are not training bindings and are not yet published to CRAN/Maven Central.
  They accept scalar JSON predictor versions 1 and 2 and reject vector version 3.
- No CoreML or PMML export is currently shipped.
- Multi-node GPU still uses the trusted-network TCP reference coordinator rather than
  mature NCCL/GPU-direct collectives, elasticity, or fault recovery.

## Security boundary for distributed training

Authenticated TCP roots contain a high-entropy per-run bearer token. Requests are
bounded and authenticated before payload dispatch, and tokens are excluded from saved
models and snapshots. The transport is not TLS: use a trusted/private network and do
not expose the coordinator directly to the public internet.
