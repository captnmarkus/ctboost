# CTBoost R predictor

This package loads CTBoost JSON predictor format versions 1 and 2 and scores
already-prepared numeric features.

```r
model <- ctboost_load_predictor("model.json")
predict(model, matrix(c(1, 2), nrow = 1), type = "raw")
predict(model, matrix(c(1, 2), nrow = 1), type = "probability")
```

It is an inference package, not an R training binding. Artifacts with
`expects_prepared_features=false` are rejected; the package never executes an
embedded fitted feature pipeline.

Loaded predictors use locked bindings. The loader bounds artifact size, rejects
duplicate keys, and validates all model indices and tree topology before
scoring. JSON artifacts are not cryptographically authenticated, so load them
only from a trusted source.

From this package directory, run `testthat::test_local()` to exercise the same
fixtures used by the JVM runtime in `tests/export_conformance`.
