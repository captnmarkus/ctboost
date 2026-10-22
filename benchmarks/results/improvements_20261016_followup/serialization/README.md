# Native model persistence follow-up

Three already assessed phase-one models were reused without fitting or model
selection. Removing JSON whitespace preserved the parsed model document and
reduced file size by 82.65–85.88%. In 27 fresh-process measurements, all loaded
models produced exactly the reference float32 prediction bits.

| Model | Indented JSON | Compact JSON | Median load, indented → compact | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Breast cancer, scalar | 4.65 MB | 0.81 MB | 32.75 → 22.56 ms | 1.45x |
| Digits, scalar | 449.32 MB | 63.43 MB | 3.202 → 2.129 s | 1.50x |
| Digits, vector | 48.68 MB | 8.35 MB | 333.48 → 230.00 ms | 1.45x |

MB means decimal bytes. The existing public pickle API loaded these models
2.94–3.26x faster than indented JSON, with identical prediction bits. Its files
were 0.75 MB, 62.51 MB and 7.26 MB respectively. This is a trusted local pickle
comparison, not a new persistence format.

The large scalar model's median whole-process peak working set fell from
1.413 GB to 0.403 GB with compact JSON. This peak includes imports, file hashing,
loading and first prediction; it is neither an isolated native-model size nor
an allocation profiler's load-only peak. Retained RSS is recorded separately in
the raw measurements.

## Timing scope

- One CPU (logical CPU 6), one configured OpenMP/BLAS thread, CPython 3.10.14.
- Each load ran in a fresh process. Imports, artifact hashing and input loading
  occurred before the timer. Hashing warms the OS file cache: these are **warm
  cache load times**, not cold-disk or process-startup latencies.
- Three repeats used balanced Latin ordering across the three formats. Raw
  measurements are retained; this small scout does not support precise confidence
  intervals or a claim about every machine/model.
- Other benchmark workers, tests and builds were paused during timing. Model
  creation, output formatting and filesystem writes were not timed.
- First prediction was timed separately. It was broadly unchanged and no warm
  traversal speedup is claimed from JSON formatting.

## Retained change

The two native JSON writers in `ctboost/_serialization.py` now use
`separators=(",", ":")` and retain `sort_keys=True`. Fields, float encoding,
schema versions, tree order, readers, pickle handling and exported predictor
formats are unchanged. The existing booster, classifier, feature-pipeline and
vector-leaf persistence tests passed (44 tests).

`prepare` constructs compact JSON from the original document, checks parsed
equality, and writes pickle through `Booster.save_model(model_format="pickle")`.
Before any timed load, review corrected an initial preparation that had pickled
the parsed document directly; that would not represent the normal library API.
It also added balanced ordering and Python-source fingerprints. See
`preparation_note.json`; no measurements were discarded for these changes.

## Reproduction

The runner is `benchmarks/serialization_load_scout.py`. The immutable runtime is
`.tmp/improvements-20261016/candidate-bins`; native and Python source hashes,
dependency versions, artifact hashes and exact fixture paths are in `plan.json`.
Large local models and arrays are intentionally outside the Git evidence archive.
The input rows had already been assessed in phase one and are used here solely
for output parity, not a new accuracy comparison.

From the repository root, using the original CPython 3.10 environment:

```powershell
python benchmarks/serialization_load_scout.py prepare --output <fresh-report-dir> --artifacts <fresh-artifact-dir> --accuracy benchmarks/results/improvements_20261016/accuracy --package .tmp/improvements-20261016/candidate-bins
python benchmarks/serialization_load_scout.py measure --output <fresh-report-dir>
```

`summary.json` contains the computed medians. `measurements.json` contains all
27 observations and parity checks. These results compare serialization choices
for fixed CTBoost models; they do not establish parity with XGBoost or LightGBM.
