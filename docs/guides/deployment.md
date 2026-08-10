# Deployment

Every supported export carries or can be paired with an inference manifest describing
input schema, preprocessing expectations, objective/link semantics, outputs, build
identity, and a deterministic model fingerprint.

```python
model.export_inference_manifest("model.manifest.json")
model.save_model("model.json", format="json")
```

## Export choices

| Format | Best for | Important limit |
|---|---|---|
| pickle | Python round-trip | Treat files as trusted code/data |
| JSON | inspection and CTBoost reload | CTBoost runtime required for training |
| generated Python | dependency-free numeric inference | prepared numeric features for fitted pipelines |
| generated C++ / C ABI | native embedding | model-specific generated interface |
| ONNX | interoperable numeric inference | prepared numeric features for fitted pipelines |

Generated predictors and ONNX do not silently reproduce a fitted categorical, text, or
embedding pipeline. The manifest marks `prepared_features=True` where preprocessing must
remain outside the exported scorer.

## CLI

```bash
ctboost train --input train.csv --target label --model-out model.ctboost
ctboost predict --model model.ctboost --input test.csv --output predictions.csv
ctboost inspect --model model.ctboost
ctboost export --model model.ctboost --format cpp --output generated_model.cpp
```

NumPy pickle-backed input is disabled unless explicitly trusted. Output files are not
overwritten unless requested.
