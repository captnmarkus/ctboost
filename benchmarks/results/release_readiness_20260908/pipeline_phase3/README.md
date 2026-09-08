# Pipeline portability checks: phase 3

The rebuilt Windows x86-64 wheel passed the **same 323 pipeline tests in both NumPy environments** after float16 was moved back to the original object-conversion path. This includes direct native inputs and DataFrames containing float16. No new tests, fits, or timing workloads were added.

| Python | NumPy | pandas | Arrow / Polars | Result |
|---|---|---|---|---|
| 3.12.11 | 1.26.4 | 2.1.4 | 15.0.2 / 0.20.31 | 323 passed |
| 3.12.11 | 2.5.3 | 3.0.5 | 25.0.1 / 1.44.1 | 323 passed |

These are actual native-wheel tests, using isolated Python, the installed package, and pytest importlib mode. [matrix/](matrix/) preserves the original logs and runtime receipts; [dependencies/](dependencies/) records the installed versions. The ten executed test files and runner are included. Their contents match the phase2 test scope.

## Why another build was necessary

The previous ARM suite, at source `aaf50d7a6d40449efc9171a6b3f634156398fad0`, exposed **12 float16 signaling-NaN warning/callback mismatches**: three policies across NumPy, direct native, pandas, and mixed pandas inputs. [preceding-arm-failure/](preceding-arm-failure/) preserves the complete original log and identifies those cases. The log's other 30 failures concerned separate native FMA training-equivalence behavior.

The earlier [phase1/2 pipeline evidence](../pipeline/README.md), including the float32 side-effect probe and raw timing blocks, remains unchanged. Phase3 adds no ARM or newly compiled Python3.8 result locally. The broader [13-job CPU CI run](https://github.com/captnmarkus/ctboost/actions/runs/34270193368) is tracked separately; this archive does not assert its outcome or final release approval.

## Completed wheel identity

Source: `5780b41c7321c70e508dccf34e2f76f2503d1175`.

Completed wheel SHA256: `b9fb7b598395a64380ee2f5a2395269ef567169edd151916790087dd71177f94`.

An earlier `b3c83491...` hash was read while the wheel archive was still being written and is superseded. After completion, **all 181 wheel members other than the installer-rewritten RECORD file matched the tested installations byte-for-byte in both environments**. There were no extra CTBoost or benchmark package files. Thus the existing successful test runs bind to the completed wheel without repeating them. [runtime/installed-wheel-identity.json](runtime/installed-wheel-identity.json) records each member hash and the correction; [provenance.json](provenance.json) records native and wrapper identities.

The archive contains no wheels, models, or datasets. Original logs retain their original encoding and bytes; `.gitattributes` prevents newline conversion. Runners retain their original workspace paths. Run `python verify_archive.py` from this directory to verify the manifest and unchanged-original inventory.
