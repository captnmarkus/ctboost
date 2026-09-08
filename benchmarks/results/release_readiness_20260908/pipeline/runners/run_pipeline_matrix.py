import os
for name in ('CTBOOST_HIST_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
import argparse
import ctypes
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(), (1 << 5) | (1 << 13))
import ctboost
import numpy
import pandas
import polars
import pyarrow
import pytest

parser = argparse.ArgumentParser()
parser.add_argument('--python-wrapper-overlay', action='store_true')
parser.add_argument('--report', required=True, type=Path)
args = parser.parse_args()
root = Path(r'C:\apps\ctboost')
if args.python_wrapper_overlay:
    old = ctboost.FeaturePipeline
    spec = importlib.util.spec_from_file_location('ctboost.feature_pipeline', root / 'ctboost/feature_pipeline.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for imported in list(sys.modules.values()):
        if imported is not None and getattr(imported, 'FeaturePipeline', None) is old:
            imported.FeaturePipeline = module.FeaturePipeline
    ctboost.feature_pipeline = module
sys.path.insert(0, str(root))
files = [
    'test_feature_pipeline_numeric_inference.py', 'test_feature_pipeline_ctr_inference.py',
    'test_ctr_prior_smoothing.py', 'test_ctr_export_compatibility.py',
    'test_sklearn_feature_pipeline.py', 'test_feature_pipeline_state_validation.py',
    'test_feature_pipeline_persistence.py', 'test_text_embedding_pipeline.py', 'test_columnar_input.py',
]
extra = root / 'tests/test_feature_pipeline_optional_input_inference.py'
if extra.exists():
    files.append(extra.name)
result = pytest.main([str(root / 'tests' / name) for name in files] + ['--import-mode=importlib', '-q'])
report = {'python': sys.version, 'numpy': numpy.__version__, 'pandas': pandas.__version__,
          'pyarrow': pyarrow.__version__, 'polars': polars.__version__,
          'native_sha256': hashlib.sha256(Path(ctboost._core.__file__).read_bytes()).hexdigest(),
          'pipeline_sha256': hashlib.sha256(Path(ctboost.feature_pipeline.__file__).read_bytes()).hexdigest(),
          'python_wrapper_only_overlay': args.python_wrapper_overlay, 'pytest_exit_code': int(result)}
args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report))
raise SystemExit(result)
