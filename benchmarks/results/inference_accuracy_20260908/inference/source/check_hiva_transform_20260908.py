import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

for key in ('CTBOOST_HIST_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import psutil
psutil.Process().cpu_affinity([6])
import numpy as np
import ctboost

root = Path(r'C:\apps\ctboost')
sys.path.insert(0, str(root))
from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame

parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
source = root / '.tmp/inference-accuracy-20260908/accuracy-grouped-scout-v1'
output = root / '.tmp/inference-accuracy-20260908/hiva-profile'
with (source / 'fits/hiva_agnostic/ctboost_default/model.pkl').open('rb') as handle:
    payload = pickle.load(handle)
with (source / 'data/hiva_agnostic/development.pkl').open('rb') as handle:
    development = pickle.load(handle)
frame = normalize_tabarena_frame(development['X'].iloc[:37], categorical_columns=payload['categorical_columns'])[0]
matrix, categories, names = payload['model']._feature_pipeline.transform_array(frame)
report = {
    'native_sha256': hashlib.sha256(Path(ctboost._core.__file__).read_bytes()).hexdigest(),
    'shape': list(matrix.shape),
    'matrix_sha256': hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest(),
    'names': names,
    'cat_features': categories,
}
path = output / 'public-transformed-37.npz'
if args.check:
    with np.load(path) as archive:
        np.testing.assert_array_equal(matrix.view(np.uint32), archive['matrix'].view(np.uint32))
    old = json.loads(path.with_suffix('.json').read_text())
    assert report['names'] == old['names'] and report['cat_features'] == old['cat_features']
    check = {key: value for key, value in report.items() if key not in ('names', 'cat_features')}
    check['bitwise_equal'] = True
    check['reference_npz_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'candidate-transform-check.json').write_text(json.dumps(check, indent=2) + '\n')
else:
    if path.exists():
        raise RuntimeError('reference already exists')
    np.savez_compressed(path, matrix=matrix)
    path.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key not in ('names', 'cat_features')}))
