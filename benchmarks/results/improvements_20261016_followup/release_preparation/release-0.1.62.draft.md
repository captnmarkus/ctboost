# CTBoost 0.1.62

Release preparation draft. Publication, final accuracy results, final inference
measurements and release validation are pending. This draft must be finalized
before inclusion in the public release notes.

This release reduces CPU preprocessing and prediction overhead, makes JSON
models smaller, corrects precision in categorical target statistics, and
updates the JVM JSON dependency. The conditional-inference feature test still
selects a feature before cut-point optimization. Default tree-growth and
hyperparameter settings are unchanged.

## CPU inference and persistence

- Cached CPU prediction and leaf-index paths avoid copying the full quantization
  schema into each prediction histogram. Root-only cached traversal skips feature-bin
  allocation while retaining the input-column-count check.
- Common categorical keys use exact native normalization. Small single-row
  categorical inputs reuse fitted metadata, while eligible wide categorical
  batches use native integer-code conversion. The batch path requires an ordinary
  pandas DataFrame with at least 1,000 rows and 32 input columns, all independently
  categorical, ordinary string column names, exact-string vocabularies of at most
  32 categories, and supported native-endian signed integer codes. Only single-source
  categorical statistics are supported; one-hot, categorical combinations, text
  and embedding pipelines use the general path. Older native extensions without
  the new method also retain the existing path.
- Native-model JSON writers omit formatting whitespace. This formatting change
  preserves parsed model fields and prediction semantics; it does not change
  pickle handling or exported-predictor formats.

The categorical conversion acceptance study checked transformed-feature bits
and saved-model prediction bits across 102 synthetic cases and 35 real-model
case/batch pairs in four balanced measurement passes. Across the 35 pairs from
five frozen models, the geometric mean raw-prediction speedup was 1.136x for
this change. At 1,000 rows, hiva improved from 212.05 to 51.32 ms and splice from
13.69 to 7.84 ms. Eligibility and model size matter; this is not a universal
speedup or a comparison at matched accuracy. These measurements precede the
CTR-precision runtime used for the final release checks.

Compact JSON reduced three saved native-model artifacts by 82.65–85.88% and
improved load time by 1.45–1.50x against indented JSON. The study made 27
fresh-process measurements across indented JSON, compact JSON and the existing
pickle format. File hashing warmed the OS cache before each load; these are not
cold-disk or process-startup latencies. All loaded predictions matched the
reference bits. These are local measurements, not guarantees for every model
or storage system.

Final selected-model inference evidence: PENDING.

## Categorical statistics and compatibility

Fresh feature pipelines use double precision for Mean categorical target
statistic priors, category totals and ordered-prefix totals. This fixes
demonstrated accumulation drift and cancellation, including finite means whose
totals overflow float32. Final features remain checked float32 values. Double
accumulation improves precision; it is not exact summation for arbitrary dynamic
ranges. The persisted pipeline format advances to version 5 to distinguish the
new arithmetic.

Existing pipeline formats 1–4 retain their legacy accumulation arithmetic when
loaded. Format 4 preserves predictions, ordered replay and warm-start behavior.
Formats 1/2 re-export as format 3, as before; explicitly refitting a loaded format
1–3 pipeline still upgrades its smoothing to format 4 while retaining legacy
accumulation. Upgrading the package does not silently reinterpret an old pipeline
as version 5. A fresh pipeline is required to obtain the new Mean arithmetic and
can produce different feature values and trained models. This correction is not
evidence of a general accuracy improvement. Frequency and embedding arithmetic
are unchanged.

The Python export runtime accepts the new pipeline format. Older runtimes
reject it, so deploy compatible runtimes with new exports. Existing limitations
of prepared-feature R/JVM predictors and raw/vector exports remain documented
in the deployment guide. Applications linking native headers or the static
library must rebuild.

## JVM security dependency

The JVM binding upgrades `jackson-databind` from 2.21.5 to 2.21.7. Upstream
advisories identify this version as fixing the five reported issues:

- [Path deserialization scheme handling](https://github.com/FasterXML/jackson-databind/security/advisories/GHSA-wjgm-6hv5-3cvf).
- [Duration/XMLGregorianCalendar number parsing](https://github.com/FasterXML/jackson-databind/security/advisories/GHSA-q4xh-88c3-wmh7).
- [Comparable unsafe-base-type validation](https://github.com/FasterXML/jackson-databind/security/advisories/GHSA-gx83-3vf8-gh7j).
- [Quadratic forward-reference completion](https://github.com/FasterXML/jackson-databind/security/advisories/GHSA-cxp5-3px4-pw24).
- [Unknown raw type-ID retention](https://github.com/FasterXML/jackson-databind/security/advisories/GHSA-wv8q-qhhj-9h54).

The local Java 17/Maven check resolved databind and core to 2.21.7 and annotations
to 2.21, matching the upstream dependency metadata. All nine JVM conformance
tests passed, with no failures, errors or skips. Dependency remediation does
not imply that every vulnerable upstream code path was exposed by CTBoost's
predictor. GitHub alert closure will be checked after the default-branch update.

## Accuracy evidence

The October development study compares CTBoost, XGBoost and LightGBM on 12
metadata-selected tasks. It preserves CTBoost's tree-selection design. Controls
include separate training, stopping and development roles, training-only
categorical fitting, fixed per-stage configurations, and recorded failures.
Repeated development evaluation is adaptive; its results are not an unbiased
leaderboard estimate or an official TabArena ranking.

Final current-round results and any remaining gap: PENDING. No new learning
default or general superiority claim is justified before that evidence is
complete. The separate historical TabArena bundle remains attributed to 0.1.58.

## Validation

Final local tests, scientific integration checks, platform/compiler CI,
documentation build, complete wheel/source artifact dry run, R/JVM checks,
publication catalog verification and Dependabot closure: PENDING.

Earlier failed experiments and test attempts remain in the audit archive;
later successful checks must not overwrite them. Inactive research prototypes
are archived as unrun and are not validated release functionality.
