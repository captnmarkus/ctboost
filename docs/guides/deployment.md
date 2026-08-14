# Deployment

Every supported export carries or can be paired with an inference manifest describing
input schema, preprocessing expectations, objective/link semantics, outputs, build
identity, and a deterministic model fingerprint.

```python
model.export_inference_manifest("model.manifest.json")
model.save_model("model.json", model_format="json")
```

## Save, resume, and inspect

JSON is the safe default for native CTBoost model persistence. Pickle supports
Python convenience wrappers and resumable wrapper state, but it must be treated
as trusted code/data. Training snapshots validate their saved configuration and
data schema before resuming:

```python
booster = ctboost.train(
    pool,
    {"objective": "RMSE", "learning_rate": 0.1},
    num_boost_round=200,
    snapshot_path="run_snapshot.ctb",
    resume_from_snapshot=True,
)

manifest = booster.get_inference_manifest()
booster.export_inference_manifest("model.manifest.json")
```

Snapshot resume is a validated warm-start convenience, not a blanket
bit-for-bit equivalence promise for every training path. Use `init_model` when
you intentionally change a later-stage configuration.

## Export choices

Install table-oriented CLI support with `python -m pip install "ctboost[cli]"`.
ONNX export additionally requires `python -m pip install "ctboost[onnx]"`.

| Format | Best for | Important limit |
|---|---|---|
| pickle | Python round-trip | Treat files as trusted code/data |
| JSON predictor | Python inference, including fitted preprocessing | CTBoost Python runtime; trusted artifact |
| generated Python | dependency-free numeric inference | prepared numeric features for fitted pipelines |
| generated C++ / C ABI | native embedding | model-specific generated interface |
| ONNX | interoperable numeric inference | prepared numeric features for fitted pipelines |
| R/JVM JSON scorers | portable inference | prepared numeric features only; inference-only |

The version-2 JSON predictor can embed a fitted categorical, text, or embedding pipeline
for the Python `load_exported_predictor` runtime. Raw-feature exports require a matching
inference manifest and the current pipeline/key-codec versions; fingerprints and the
full native pipeline state are validated before construction. Treat the JSON as trusted
input: its SHA-256 fingerprint detects accidental or uncoordinated changes but is not a
signature or authenticity boundary.

Generated Python/C++/ONNX predictors and the R/JVM scorers do not silently reproduce a
fitted pipeline. Their manifest/profile requires prepared features. See
[portable inference](portable-inference.md) for the cross-language boundary.

Classification exports preserve the fitted class-label order. Standalone
Python and JSON predictors expose raw prediction; classification helpers also
provide probabilities/classes where supported. A deterministic model
fingerprint in the manifest binds the model state used for deployment.

## CLI

```bash
ctboost train --input train.csv --target label --model model.ctboost
ctboost predict --model model.ctboost --input test.csv --output predictions.csv
ctboost inspect --model model.ctboost
ctboost export --model model.ctboost --format cpp --output generated_model.cpp
```

NumPy inputs are always loaded with pickling disabled. Output files are not
overwritten unless requested. The unsafe-pickle opt-in applies to trusted model
serialization, not NumPy input arrays.

The CLI accepts NPY, NPZ, CSV, TSV, Parquet, and Feather input. Prediction
output additionally supports JSON. Explicit flags override values supplied by
`--params`; expected input/model failures return a nonzero status with an
actionable message. Loading or creating a pickle model requires the explicit
unsafe-pickle opt-in.
