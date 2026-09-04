# Portable R and JVM inference

CTBoost ships source packages for scoring prepared numeric JSON-predictor artifacts in
R and Java 17. They are inference runtimes, not training bindings, and they do not alter
the model's conditional-inference feature selection or split points.

## Supported profile

Both runtimes implement the repository's
[`json-predictor-prepared`](https://github.com/captnmarkus/ctboost/blob/master/spec/json-predictor-prepared.md)
profile for scalar format versions 1 and 2. Version-3 vector predictors are
unsupported and rejected; use the Python runtime for compact vector inference.
They validate the envelope, quantization arrays,
finite model values, tree dimensions, child and feature indices, reachability, and
acyclic traversal before scoring. Duplicate JSON keys are rejected.

The input must already have exactly `num_features` numeric columns in the order used by
the exported model. Artifacts with `expects_prepared_features=false` are rejected before
the scorer reads fitted-pipeline fields. Use the Python JSON runtime when inference must
reproduce CTBoost's fitted categorical, text, or embedding preprocessing.

## JVM

Build and test the Java 17 module:

```bash
cd bindings/jvm
mvn test
```

Load `JsonPredictor` from the package `io.github.ctboost.inference`, then call its raw,
probability, or class-prediction methods. The module is source-distributed and has not
yet been published to Maven Central.

## R

Build and check the pure-R package:

```bash
R CMD build bindings/R/ctboost
R CMD check --no-manual --no-build-vignettes ctboost_*.tar.gz
```

Use `ctboost_load_predictor(path)` followed by `ctboost_predict(model, data, type=...)`.
The package is source-distributed and has not yet been published to CRAN.

## Security and conformance

The JSON model must be trusted. Structural validation and traversal limits make malformed
artifacts fail closed, but the model fingerprint is a consistency checksum, not a digital
signature. The shared fixtures under `tests/export_conformance` bind Python, R, and JVM
raw/probability/class outputs for regression, binary, and multiclass models.

Training APIs, JNI/native bindings, raw fitted preprocessing, and model-registry
authentication are outside this first portable-inference slice.
