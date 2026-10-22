# Raw-margin metric numerical audit

16 October 2026. Synthetic evaluation only; no fits, real data, saved models,
development predictions or official-test values were accessed. Production
metric code is unchanged. [The receipt](receipt.json) binds the executable
[audit script](audit.py), Python/NumPy versions and immutable accuracy native
binary `c5ea7e026942c3f1190ca5511fbf74f353bfeb08e7771f76bc5f6c1ca3e4d197`.

| Binary margin / target | Native loss, rounded | Float64 logit loss, rounded |
| --- | ---: | ---: |
| -17 / 1 | 17.000000 | 17.000000 |
| +17 / 0 | 27.631043 | 17.000000 |
| -40 / 1 | 27.631021 | 40.000000 |
| +40 / 0 | 27.631043 | 40.000000 |
| -80 / 1 | 27.631021 | 80.000000 |
| +80 / 0 | 27.631043 | 80.000000 |

The script also checks correct-label cases and nine multiclass cases. For
logits `[0, -40, -80]` and target class 2, native loss is 27.631021 rather
than 80. This follows from float sigmoid rounding in binary evaluation and
probability clipping at `1e-12` in both native classification metrics.

The contract is raw margins: `src/core/booster_fit_metrics.cpp` passes native
prediction buffers directly; `src/bindings/module_registration.cpp` and
`ctboost/training/_eval_runtime.py` likewise pass float32 margins to metrics.
Stable double softplus and shifted log-sum-exp compute the intended loss.
They affect histories and possibly early stopping, not objective gradients,
tree splitting or predictions from an existing model. No actual checkpoint or
portfolio effect has been measured. The frozen benchmark's external multiclass
scorer already uses stable log-sum-exp.

**Deferred compatibility issue:** native ordinary warm start and the Python
multi-evaluation/callback loop reuse old history and best scores. Replacing
the formula alone would compare old clipped scores with new unclipped scores.
The narrow formula correction therefore remains experimental, separate from
the CTR precision patch. The robust future route is a persisted metric
arithmetic version: absent means legacy, new models use stable arithmetic,
and loaded legacy models keep legacy metrics throughout both execution paths.
Version propagation must cover native factories/state/loaders, Python builtin
metric evaluation, multi-evaluation/distributed paths and snapshots. A model
format guard should prevent old runtimes silently ignoring stable arithmetic.

Rescoring every historical checkpoint is not a general small alternative:
DART's historical ensembles are not recoverable after tree rescaling, and
original multi-evaluation/custom-metric data or callbacks may be unavailable.
Documentation alone does not resolve mixed arithmetic histories.

Reproduce from the repository root:

```powershell
& .tmp/inference-accuracy-20260908/baseline-env/Scripts/python.exe -I `
  benchmarks/results/improvements_20261016_followup/inference/metric_numerics_audit/audit.py `
  --package .tmp/improvements-20261016/followup/runtime `
  --output .tmp/metric-numerics-repeat.json
```

The initial attempt with the unrelated `.venv` interpreter stopped before
import/evaluation because `psutil` was unavailable. The recorded successful
run used the frozen accuracy interpreter; no dependency was installed.
