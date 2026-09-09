# GPU installation

CTBoost reports CUDA support at runtime:

```python
import ctboost
print(ctboost.build_info())
```

Train on CUDA by setting `task_type="GPU"`:

```python
from ctboost import CTBoostClassifier

model = CTBoostClassifier(task_type="GPU", iterations=500)
model.fit(X_train, y_train)
```

CTBoost 0.1.61 provides unified Linux x86-64 and Windows AMD64 wheels for
CPython 3.10 through 3.14. The ordinary installation supports CPU training and
CUDA training when a compatible NVIDIA device and driver are available.

```bash
python -m pip install -U ctboost
```

!!! note "Released artifacts are authoritative"
    Check the [0.1.61 release](https://github.com/captnmarkus/ctboost/releases/tag/v0.1.61)
    for the available Python and platform builds. macOS and ARM wheels are CPU-only.
    A CUDA-enabled build still works for CPU training on a machine without an NVIDIA GPU.
    The bundled CUDA runtime remains subject to the NVIDIA CUDA Toolkit license
    included in each CUDA-enabled wheel.

The unified wheels bundle the CUDA 12.8 runtime; a local CUDA toolkit is not
required. GPU training needs a compatible NVIDIA driver and compute capability
6.0 or newer. Linux aarch64, macOS, and CPython 3.8/3.9 artifacts are CPU-only.
