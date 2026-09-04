# CTBoost JVM predictor

This Maven module loads CTBoost JSON predictor format versions 1 and 2 and
scores already-prepared numeric features on Java 17 or newer.

```java
JsonPredictor model = JsonPredictor.load(Path.of("model.json"));
double[] raw = model.predictRaw(new float[] {1.0f, 2.0f});
double[] probabilities = model.predictProba(new float[] {1.0f, 2.0f});
```

This is an inference runtime, not a training binding. It deliberately rejects
artifacts with `expects_prepared_features=false`; fitted categorical, text, and
embedding preprocessing must be applied before calling it.

The loader bounds artifact size, rejects duplicate keys, and validates all
model indices and tree topology before scoring. The JSON artifact is not
cryptographically authenticated, so load it only from a trusted source.

Run the shared conformance tests from this directory with `mvn test`. The tests
read `../../tests/export_conformance` so JVM and R exercise identical artifacts.
