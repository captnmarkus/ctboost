# Floating-point contraction compatibility probe

This standalone C++ probe compares CTBoost's legacy indirect-leaf update pattern
(`const float update = float(rate) * weight; prediction += update`) with cached,
preweighted float products. It uses 1,000 deterministic finite inputs; it does
not train a model or use benchmark data.

On the existing GCC 14.2.0 i686-w64-mingw32 compiler, the FMA-enabled build differs
on 86/1,000 rows. The first mismatch agrees with `std::fma`; disassembly confirms
`vfmadd213ss` in `legacy_update`. The otherwise identical no-FMA build has zero
mismatches. These are explicit FMA-target builds, not default x86_64-wheel results.

Reproduce with GCC (run the commands from this directory):

```sh
g++ -std=c++17 -O3 -mavx2 -mfma -mfpmath=sse -static probe.cpp -o fma-probe
./fma-probe
g++ -std=c++17 -O3 -mavx2 -mno-fma -mfpmath=sse -static probe.cpp -o no-fma-probe
./no-fma-probe
```

Windows executables can use `.exe` suffixes. `result.json` records the outputs,
compiler, source hash, and flags; `legacy-fma-assembly.txt` preserves the relevant
disassembly. No global floating-point flag or training calculation was changed.

The inference implementation retains the legacy helper on GNU FMA-capable and
`__FAST_MATH__` builds. Exact compatibility tests compare against that same
compiler's legacy helper through a private diagnostic; exported routing and leaf
indices retain independent reference checks.

Compiler behavior: [GCC optimization options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
and [Clang floating-point controls](https://clang.llvm.org/docs/UsersManual.html).
