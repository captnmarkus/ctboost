# Compatibility and limits

## Runtime matrix

- Python: 3.8 through 3.14 in package metadata;
- CPU wheels: Linux, Windows, and macOS according to release artifacts;
- CUDA: Linux x86-64 and Windows AMD64 only, according to release artifacts;
- source builds: CMake 3.24+, a C++17 compiler, and pybind11 through the build backend.

## Explicit limitations

- Multi-output and multilabel wrappers fit independent boosters rather than vector-leaf trees.
- AFT uses a Python log-normal objective, numeric/prepared input, and no distributed fit.
- Continuing a callable-objective model requires supplying the callable again.
- Exact SHAP is empirical interventional and currently CPU-oriented.
- Object importance is approximate shared-leaf influence.
- Spark training collects to the driver.
- Arrow/Polars/cuDF/CuPy/DLPack inputs currently materialize into CTBoost-owned host arrays.
- Streaming Pools are numeric-only.
- Generated Python/C++ and ONNX exports require prepared numeric features for models with
  fitted categorical, text, or embedding pipelines.
- No CoreML, PMML, Java, or R binding is currently shipped.

## Security boundary for distributed training

Authenticated TCP roots contain a high-entropy per-run bearer token. Requests are
bounded and authenticated before payload dispatch, and tokens are excluded from saved
models and snapshots. The transport is not TLS: use a trusted/private network and do
not expose the coordinator directly to the public internet.
