# Direct C++ tree histogram bounds regression

This probe exercises `Tree::PredictBinnedLeafIndex`, `PredictBinnedRow`, and
`AccumulateBinnedContributions` directly, outside the Python model API. It covers
8-bit and 16-bit histogram storage, normal left/right routing, too few histogram
columns, valid narrower histograms using only available features, and trees
without attached schemas.

Before the fix, six expected `std::out_of_range` exceptions are missing (three
methods times two storage widths). After the schema-coverage guard, all 22
checks pass. The defect predates the compact prediction cache: the affected
`tree_predict.cpp` was already present at commit `c5b6149`.

The too-few-columns cases keep padding allocated in the buffer. This safely
proves that the old fast path bypassed the declared histogram column count,
without intentionally reading unallocated memory.

The probe was compiled with existing MSVC 19.50.35717 x64, `/std:c++17 /EHsc /MD
/O2`, Python/pybind11 include directories, and the existing isolated static core
library. The before executable links that core directly. The after executable
compiles the fixed `src/core/tree_predict.cpp` alongside the same probe before
linking the same core, so only the relevant implementation changes. No model was
trained. `result.json` records both implementation hashes and the exact outputs.

Example from a configured MSVC shell (replace the include/library placeholders):

```bat
cl /std:c++17 /EHsc /MD /O2 /I <repo>/include /I <python-include> /I <pybind11-include> probe.cpp <core-library> /Fe:before.exe /link /LIBPATH:<python-libs>
cl /std:c++17 /EHsc /MD /O2 /I <repo>/include /I <python-include> /I <pybind11-include> probe.cpp <repo>/src/core/tree_predict.cpp <core-library> /Fe:after.exe /link /LIBPATH:<python-libs>
```

Python's shared-library directory must be on the child process's search path.
The checked fallback retains the previous exception and continues to accept a
narrower histogram when the tree visits only features that are actually present.
