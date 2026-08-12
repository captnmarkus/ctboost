# Development

## Install from source

From a checkout, build and install CTBoost with:

```bash
python -m pip install .
```

For an editable checkout with the test dependencies:

```bash
python -m pip install -e ".[dev]"
```

Force a CPU-only source build on POSIX shells with:

```bash
CMAKE_ARGS="-DCTBOOST_ENABLE_CUDA=OFF" python -m pip install .
```

On PowerShell:

```powershell
$env:CMAKE_ARGS="-DCTBOOST_ENABLE_CUDA=OFF"
python -m pip install .
```

See [GPU installation](gpu.md) before attempting a CUDA source build.

## Direct CMake build

Install the native build and test dependencies, then point CMake at the
pybind11 package installed for the active Python interpreter:

```bash
python -m pip install "pybind11>=2.12,<3" numpy pandas scikit-learn scipy pytest
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCTBOOST_ENABLE_CUDA=OFF \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build --config Release --parallel
```

CTBoost requires CMake 3.24 or newer and a C++17 compiler. Use the same Python
interpreter for dependency installation, configuration, and tests.

## Tests and distributions

Run the Python suite from the repository root:

```bash
pytest tests
```

Build a source distribution with the `build` frontend:

```bash
python -m pip install build
python -m build --sdist
```

## Project layout

```text
ctboost/       Python API surface
include/       public C++ headers
src/core/      core training, data, objectives, trees, and statistics
src/bindings/  pybind11 extension bindings
cuda/          optional CUDA backend
tests/         Python test suite
demo/          local example workflows
docs/          documentation sources
```

## Documentation

Install the pinned documentation dependencies and run the same strict build
used in CI:

```bash
python -m pip install --requirement requirements-docs.txt
python -m mkdocs build --strict
```

For a local preview with automatic rebuilding:

```bash
python -m mkdocs serve
```
