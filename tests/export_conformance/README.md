# JSON predictor conformance fixtures

The prepared regression, binary, and multiclass artifacts are shared golden
models for every non-Python JSON scorer. Their adjacent `*.cases.json` files
contain prepared rows and expected raw/probability/class outputs.

The format-v2 binary and multiclass fixtures are release-0.1.56 producer
artifacts. The format-v1 regression fixture intentionally remains a 0.1.55
artifact so every runtime continuously checks backward compatibility.

`raw_pipeline_v2.json` is intentionally incomplete beyond the raw-input flag;
it proves that prepared-only runtimes reject raw preprocessing before touching
model or pipeline fields. `duplicate_prepared_flag_v2.json` proves that readers
reject ambiguous duplicate keys.

From `bindings/jvm`, run `mvn test`. From `bindings/R/ctboost`, run
`testthat::test_local()`. Set `CTBOOST_CONFORMANCE_DIR` to this directory when
the package is tested outside the repository tree.
