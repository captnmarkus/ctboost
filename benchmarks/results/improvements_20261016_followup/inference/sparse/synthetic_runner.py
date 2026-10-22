"""Saved-model inference experiment; no fitting and no quality claim."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import statistics
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LegacyNumpyUnpickler(pickle.Unpickler):
    """The frozen NumPy 2 fixture uses the renamed NumPy 1 core modules."""
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def measure(call):
    for _ in range(3):
        call()
    start = time.perf_counter()
    for _ in range(10):
        call()
    estimate = (time.perf_counter() - start) / 10
    count = min(1000, max(2, int(0.05 / max(estimate, 1e-6))))
    blocks = []
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(count):
            call()
        blocks.append(1000 * (time.perf_counter() - start) / count)
    return dict(median_ms=statistics.median(blocks), block_ms=blocks, calls_per_block=count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--cpu', type=int, default=6)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--cases', nargs='+', default=['categorical_2000'])
    args = parser.parse_args()
    destination = args.output / f'{args.label}.json'
    if destination.exists():
        raise ValueError('Refusing to overwrite a timing receipt')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'CTBOOST_HIST_THREADS'):
        os.environ[name] = '1'
    import psutil
    psutil.Process().cpu_affinity([args.cpu])
    sys.meta_path = [finder for finder in sys.meta_path
                     if type(finder).__module__ != '_editable_skbc_ctboost']
    sys.path.insert(0, str(args.package.resolve()))
    import ctboost
    assert Path(ctboost.__file__).resolve().is_relative_to(args.package.resolve())
    import numpy as np
    from catboost import CatBoostClassifier
    native = Path(ctboost._core.__file__)
    references = args.output / 'reference'
    references.mkdir(parents=True, exist_ok=True)
    cases = args.cases
    records = []
    for name in cases:
        directory = args.fixtures / name
        with (directory / 'input.pkl').open('rb') as stream:
            frame = LegacyNumpyUnpickler(stream).load()
        model = ctboost.CTBoostClassifier.load_model(directory / 'ctboost.json')
        control = CatBoostClassifier()
        control.load_model(str(directory / 'catboost.cbm'))
        model_state = model.get_booster()._handle.export_state()
        for rows in (1, 64, 4096):
            batch = frame.iloc[np.arange(rows) % len(frame)]
            prepared = model._transform_prediction_pool(batch)
            start = time.perf_counter()
            actual = model.predict_proba(batch)
            cold_ms = 1000 * (time.perf_counter() - start)
            expected_path = references / f'{name}-{rows}.npy'
            if args.reference:
                if expected_path.exists():
                    raise ValueError(f'Refusing to replace {expected_path}')
                np.save(expected_path, actual)
            expected = np.load(expected_path)
            for values in (actual, model.predict_proba(prepared)):
                assert values.shape == expected.shape and values.dtype == expected.dtype
                np.testing.assert_array_equal(values.view(np.uint8), expected.view(np.uint8))
            raw_native = np.asarray(model.get_booster()._handle.predict(prepared._handle))
            legacy_native = np.asarray(model.get_booster()._handle._predict_uncached(prepared._handle))
            np.testing.assert_array_equal(raw_native.view(np.uint8), legacy_native.view(np.uint8))
            record = dict(case=name, rows=rows, trees=len(model_state['trees']),
                          nodes=sum(len(t['nodes']) for t in model_state['trees']),
                          model_sha256=sha(directory / 'ctboost.json'),
                          catboost_model_sha256=sha(directory / 'catboost.cbm'),
                          input_sha256=sha(directory / 'input.pkl'),
                          reference_sha256=sha(expected_path), exact_parity=True,
                          first_call_this_batch_ms=cold_ms)
            if not args.check_only:
                record['raw'] = measure(lambda: model.predict_proba(batch))
                record['prepared'] = measure(lambda: model.predict_proba(prepared))
                record['catboost_control'] = measure(lambda: control.predict_proba(batch, thread_count=1))
            records.append(record)
            print(json.dumps(record), flush=True)
    result = dict(label=args.label, package=str(Path(ctboost.__file__).resolve()),
                  native_sha256=sha(native), runner_sha256=sha(__file__),
                  build_info=ctboost.build_info(), python=sys.version,
                  versions={name: importlib.metadata.version(name)
                            for name in ('numpy', 'pandas', 'catboost', 'psutil')},
                  build_receipt=json.loads((args.package / 'build_receipt.json').read_text()),
                  affinity=psutil.Process().cpu_affinity(),
                  protocol='One CPU/thread; 3 warmups; 7 blocks targeting 50ms each; no fitting.',
                  cases=records)
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

