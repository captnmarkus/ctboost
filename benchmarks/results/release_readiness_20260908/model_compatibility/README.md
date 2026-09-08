# Saved-model prediction compatibility for 0.1.61

Both verification phases passed all 14 saved public CTBoost 0.1.60 models on the previously unsealed development fold 6 (6,610 rows total). Raw normalized inputs and prepared-Pool predictions match the original saved prediction arrays bit for bit, including dtype and shape. The existing 1,000-row cycled batches also match both archived public baseline repeats. Hiva's saved 37 ? 6,468 prepared matrix, feature names and categorical indices match exactly. Prediction leaves serialized booster and pipeline state unchanged.

No models were fitted, no scores or timings were calculated, and neither confirmation fold 7 nor outer-test rows were opened. These are compatibility checks on reused development data, not independent accuracy evidence. The archive contains hashes and verification results, not source feature tables, labels, models or prediction arrays.

| Phase | Native SHA-256 | Wheel SHA-256 |
| --- | --- | --- |
| Initial candidate | `836a16402c52337673450cf9be24c7ab50cd7ad9c6fc54bc74c4ead0f66859dd` | `eabd3197d9bed5eb0267f436200268b847cebc0d6e6edad2eeff43be1ee79bb5` |
| Final sNaN-policy candidate | `3fe462617f2ba45ad1b159b8a6a35808095c441d7e1d2c8d7f1f0aaae139ba65` | `9413d08ac1380f0222de639fc73967b233aa9ee67879755d8fb401af43a6dd43` |

Both candidates report version 0.1.61. The reference model runtime is public 0.1.60; its native identity is recorded in each receipt. NumPy 2.0.2, pandas 2.3.3 and scikit-learn 1.6.1 match the reference environment. The final receipt records Git commit `aaf50d7a6d40449efc9171a6b3f634156398fad0`, installed Python file hashes and native source hashes. It also discloses pending working-tree edits; the verified artifact is the wheel identified above, not a claim that every working-tree file equals that commit.

`phase1/` preserves the first receipt, log and verifier byte for byte. `final/` contains the subsequent check against the separately stored `dist-final` wheel. The final verifier differs only in output/provenance binding: it pins the expected final native hash and records the source commit and working-tree status. Both use identical model/data/reference checks. Reproduction requires the pre-existing local fixtures referenced by the verifiers; their paths and expected hashes are recorded in the receipts. Execute the relevant script with the corresponding isolated installed-wheel interpreter (`python -I`). No old environment needs modification.

`manifest.json` hashes every evidence file. `SHA256SUMS` additionally covers that manifest. Local Git attributes preserve exact bytes across line-ending conventions. The first attempt at the final preflight pointed to the deliberately retained initial wheel in `dist/`; native-byte verification rejected it before any model or data was opened. The corrected run binds `dist-final/` and passes all checks.
