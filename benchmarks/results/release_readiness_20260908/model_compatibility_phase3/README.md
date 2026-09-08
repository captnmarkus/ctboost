# Phase-three saved-model compatibility

All 14 saved public CTBoost 0.1.60 models pass the same prediction checks on the rebuilt 0.1.61 Windows wheel from source commit `5780b41c7321c70e508dccf34e2f76f2503d1175`. The 6,610 previously unsealed development-fold-6 rows produce predictions matching the archived public arrays bit for bit, with identical dtype and shape, through both raw normalized input and prepared-Pool APIs. Existing 1,000-row cycled batches match both public reference repetitions. Hiva's saved 37-by-6,468 prepared matrix, feature names and categorical indices also match exactly. Serialized booster and pipeline state remain unchanged by prediction.

No models were fitted, no scores or timings were calculated, and no confirmation-fold-7 or outer-test rows were opened. This is reused-data compatibility evidence on Windows, not an independent quality study or proof that the ARM/GNU-FMA training tests passed. Earlier phase-one and phase-two evidence remains unchanged in `../model_compatibility/`.

| Artifact | SHA-256 |
| --- | --- |
| Phase-three Windows wheel | `b9fb7b598395a64380ee2f5a2395269ef567169edd151916790087dd71177f94` |
| Installed native extension | `5b91bb3139dd54ab05f5c8dea2b1bd1caabf0b9dcb2a2e76cafb01b30281e1b5` |
| Installed feature pipeline Python module | `9e5a75e0b4588436371e0752059e8aa676c1407583a9d7dcf83ab32126fbe5bd` |

The verifier checks the externally supplied wheel/native hashes and verifies the native bytes inside the wheel. The separately preserved wheel identity receipt also verifies that the installed native and Python pipeline files equal their wheel contents. NumPy 2.0.2, pandas 2.3.3 and scikit-learn 1.6.1 match the original fixture environment. The verification receipt records installed Python and current native source hashes, the build/verification Git commit and the working-tree status; it does not claim pending documentation/archive files were part of the wheel.

The first preflight rejected an initially supplied wheel hash that had been read before archive completion, before loading any models or data. `initial_preflight.json` preserves that event and its resolution. After the completed build and installed-file checks, the verifier ran against the final hash above and passed. Earlier wheel files and verification receipts were not replaced.

Reproduction requires the existing local fixtures named and hashed in the receipt, and the matching installed-wheel interpreter. Run:

```text
python -I verify_public_models.py 5b91bb3139dd54ab05f5c8dea2b1bd1caabf0b9dcb2a2e76cafb01b30281e1b5 b9fb7b598395a64380ee2f5a2395269ef567169edd151916790087dd71177f94
```

The script preserves the prior data/model/reference assertions and adds phase-three provenance/output binding. It targets the separately stored `dist-portable` wheel. The archive contains only verification code, logs, hashes and receipts; feature data, labels, predictions and models are not included. `manifest.json` hashes all payload files and records their original paths. `SHA256SUMS` additionally covers the manifest; local Git attributes preserve exact file bytes.
