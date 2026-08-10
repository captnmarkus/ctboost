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

The release workflow is moving to unified Linux x86-64 and Windows AMD64 wheels:
on supported CPython versions, the ordinary command below installs one wheel that can
train on CPU and use CUDA when an NVIDIA device is available.

```bash
python -m pip install -U ctboost
```

!!! note "Released artifacts are authoritative"
    Check the release notes for the exact Python, operating-system, CUDA-runtime,
    architecture, and minimum-driver matrix. macOS and ARM wheels are CPU-only.
    A CUDA-enabled build still works for CPU training on a machine without an NVIDIA GPU.
    The bundled CUDA runtime remains subject to the NVIDIA CUDA Toolkit license
    included in each CUDA-enabled wheel.

Older releases that published CUDA wheels only as GitHub assets retain the legacy
`ctboost-install-gpu` command. It verifies the GitHub-provided SHA-256 digest before
asking pip to replace the installed wheel.
