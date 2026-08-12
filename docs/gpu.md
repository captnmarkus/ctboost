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

Starting with CTBoost 0.1.54, releases use unified Linux x86-64 and Windows
AMD64 wheels for CPython 3.10 through 3.14: the ordinary command below installs
one wheel that can train on CPU and use CUDA when an NVIDIA device is available.

```bash
python -m pip install -U ctboost
```

!!! note "Released artifacts are authoritative"
    Check the [GitHub release notes](https://github.com/captnmarkus/ctboost/releases)
    for the authoritative Python, operating-system, CUDA-runtime, architecture,
    and minimum-driver matrix. macOS and ARM wheels are CPU-only.
    A CUDA-enabled build still works for CPU training on a machine without an NVIDIA GPU.
    The bundled CUDA runtime remains subject to the NVIDIA CUDA Toolkit license
    included in each CUDA-enabled wheel.

The 0.1.54/0.1.55 unified wheels bundle the CUDA 12.8 runtime. GPU use requires
an NVIDIA driver compatible with CUDA 12.x (at least 525.60.13 on Linux or
528.33 on Windows) but does not require a local CUDA toolkit. Released CUDA
wheels target compute capability 6.0 or newer, with native code through current
architectures and forward-compatible PTX. Linux aarch64, macOS, and CPython
3.8/3.9 artifacts are CPU-only.

Older releases that published CUDA wheels only as GitHub assets retain the legacy
`ctboost-install-gpu` command. It verifies the GitHub-provided SHA-256 digest before
asking pip to replace the installed wheel.
