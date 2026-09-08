import os
for name in ('CTBOOST_HIST_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import ctypes
import hashlib
import json
import statistics
import time
from pathlib import Path
import ctboost
import numpy as np
import pandas as pd

ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(), 1 << 5)
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
records = []
for rows in (16, 1000):
    for kind in ('array_f32', 'array_f64', 'frame_f32', 'frame_f64', 'frame_f32_int', 'frame_f64_int'):
        values = np.arange(rows * 32).reshape(rows, 32).astype(np.float32 if 'f32' in kind else np.float64)
        frame = pd.DataFrame(values) if kind.startswith('frame') else values
        if kind.endswith('_int'):
            frame[0] = np.arange(rows, dtype=np.int64)
        pipeline = ctboost.FeaturePipeline().fit(np.zeros((3, 32)), [0, 1, 2])
        expected = pipeline.transform_array(np.asarray(frame, dtype=object))[0]
        np.testing.assert_array_equal(pipeline.transform_array(frame)[0].view(np.uint32), expected.view(np.uint32))
        for _ in range(10):
            pipeline.transform_array(frame)
        begin = time.perf_counter()
        for _ in range(20):
            pipeline.transform_array(frame)
        elapsed = time.perf_counter() - begin
        count = max(10, min(10000, round(0.06 / (elapsed / 20))))
        blocks = []
        for _ in range(7):
            begin = time.perf_counter()
            for _ in range(count):
                pipeline.transform_array(frame)
            blocks.append((time.perf_counter() - begin) / count)
        records.append({'rows': rows, 'columns': 32, 'kind': kind, 'calls_per_block': count,
                        'seconds_per_call': blocks, 'median_seconds': statistics.median(blocks)})
report = {'native_sha256': hashlib.sha256(Path(ctboost._core.__file__).read_bytes()).hexdigest(),
          'pipeline_sha256': hashlib.sha256(Path(ctboost.feature_pipeline.__file__).read_bytes()).hexdigest(),
          'numpy': np.__version__, 'pandas': pd.__version__, 'cpu_affinity': [5],
          'threads': 1, 'purpose': 'bounded normal-path regression check; concurrent background workloads',
          'records': records}
args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(args.output)
