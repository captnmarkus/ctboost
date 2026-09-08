# Training arithmetic: compiler proof and deferred baseline limitation

This archive contains two separate deterministic investigations. It contains no benchmark rows, fitted models or benchmark scores. Original probe and receipt bytes are preserved; `manifest.json` records their original local paths and hashes. `SHA256SUMS` covers every archive file except itself.

## GNU fused multiply/add proof

`fma/probe.cpp` extracts the unchanged prediction-update loop bodies from `booster_prediction.cpp` and `booster_tree_updates.cpp` into minimal stand-in tree structures. It compares 3,000 finite float32 updates across scalar and vector paths with identical inputs. This is an arithmetic probe using GCC 14.2.0 targeting 32-bit Windows, not validation of a complete ARM or Linux package.

With `-O3 -mavx2 -mfma -mfpmath=sse`, scalar leaf-range training updates differ from histogram prediction in 343 of 3,000 values. Vector leaf ranges, both leaf-index paths and explicit `std::fma` agree with histogram prediction. Scalar leaf ranges precompute and round the learning-rate product outside the row loop; the compiler contracts multiply/add in the other tested loops.

With `-ffp-contract=off` added, all existing paths agree, while explicit `std::fma` differs in 343 values. Hardware feature macros therefore cannot justify unconditionally introducing `std::fma` into training. The correction in commit `5780b41c7321c70e508dccf34e2f76f2503d1175` routes affected GNU CPU training updates through the unchanged histogram prediction helper for each new tree. It preserves that compiler's inference arithmetic, including contraction-disabled builds. Ordinary Windows, Clang and default x86 builds retain the leaf-range update. GPU training is excluded because its histogram bin storage may be released.

Reproduce from this directory using the recorded GCC toolchain (put its `bin` directory on `PATH`):

```text
g++ -std=c++17 -O3 -mavx2 -mfma -mfpmath=sse -static fma/probe.cpp -o fast.exe
fast.exe
g++ -std=c++17 -O3 -mavx2 -mfma -mfpmath=sse -ffp-contract=off -static fma/probe.cpp -o off.exe
off.exe
```

`fma/result.json` preserves the actual compiler, flags, outputs and source hashes. Real ARM, GNU-FMA and contraction-disabled package tests are separate CI evidence; this receipt does not declare their outcome.

## Deferred nonzero Pool.baseline warm start

`baseline/reproduce.py` and `baseline/result.json` document a separate existing limitation on the earlier 0.1.61 Windows candidate, native SHA-256 `3fe462617f2ba45ad1b159b8a6a35808095c441d7e1d2c8d7f1f0aaae139ba65`. No baseline behavior or saved-model inference was changed by the GNU fix.

Fresh training starts with the per-row baseline and adds tree updates. Resumed training reconstructs the tree ensemble before adding that baseline. Float addition order differs. Warm start with a nonzero `Pool.baseline` therefore does not guarantee exact equivalence to uninterrupted training, and small rounding changes can affect later split decisions.

The synthetic reproduction uses 96 rows, fixed seed 471, fractional weights, learning rate 0.123456789 and seven rounds, comparing one uninterrupted fit with a three-round fit followed by four more rounds. Five of eight objective/storage/root-only cases differ. Four have maximum raw-prediction differences between 1.49e-8 and 5.96e-8. One regression case changes later tree topology and reaches an absolute raw-prediction difference of 0.48253047 on four of 96 rows; its largest loss-history difference is 0.00177482. Three root-only classification cases agree exactly. These are reproduction results, not bounds on the limitation in other data.

Run the preserved script in a compatible installed-wheel environment, providing a new output path:

```text
python -I baseline/reproduce.py new-baseline-result.json
```

The script fits synthetic unit models only and records its own hash, installed runtime identity, per-case comparisons and full loss histories. The archived receipt is the original run on the earlier candidate; it must not be described as a phase-three rerun or as a public 0.1.60 runtime test.
