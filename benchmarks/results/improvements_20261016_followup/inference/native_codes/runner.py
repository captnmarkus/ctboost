"""Fixed saved-model CPU inference study, including preprocessing parity.

No fitting, label inspection, or quality selection. Exact fixtures stay local.
"""
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


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(call):
    for _ in range(3):
        call()
    started = time.perf_counter()
    for _ in range(5):
        call()
    count = min(1000, max(2, int(.05 / max((time.perf_counter() - started) / 5, 1e-6))))
    samples = []
    for _ in range(7):
        started = time.perf_counter()
        for _ in range(count):
            call()
        samples.append(1000 * (time.perf_counter() - started) / count)
    return dict(median_ms=statistics.median(samples), block_ms=samples, calls_per_block=count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--scout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--cases', nargs='+', default=[
        'hiva_agnostic', 'coil2000_insurance_policies', 'splice', 'GiveMeSomeCredit', 'SDSS17'])
    parser.add_argument('--rows', nargs='+', type=int, default=[1, 64, 1000])
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    result_path = args.output / f'{args.label}.json'
    if result_path.exists():
        raise ValueError('Refusing to overwrite an experiment receipt')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'CTBOOST_HIST_THREADS'):
        os.environ[name] = '1'
    import psutil
    psutil.Process().cpu_affinity([6])
    sys.meta_path = [finder for finder in sys.meta_path
                     if type(finder).__module__ != '_editable_skbc_ctboost']
    sys.path.insert(0, str(args.package.resolve()))
    import ctboost
    import numpy as np
    assert Path(ctboost.__file__).resolve().is_relative_to(args.package.resolve())
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame
    reference_dir = args.output / 'reference'
    reference_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for case in args.cases:
        model_path = args.scout / 'fits' / case / 'ctboost_default' / 'model.pkl'
        data_path = args.scout / 'data' / case / 'development.pkl'
        with model_path.open('rb') as stream:
            payload = CompatibleUnpickler(stream).load()
        with data_path.open('rb') as stream:
            development = CompatibleUnpickler(stream).load()
        model = payload['model']
        categorical_columns = payload['categorical_columns']
        source_frame = development['X']
        raw = source_frame.iloc[np.arange(max(args.rows)) % len(source_frame)].copy().reset_index(drop=True)
        normalize = lambda batch: normalize_tabarena_frame(batch, categorical_columns=categorical_columns)[0]
        native = model.get_booster()._handle
        trees = native.export_state()['trees']
        state_info = dict(trees=len(trees), nodes=sum(len(tree['nodes']) for tree in trees),
                          active_features=len({node['split_feature_id'] for tree in trees
                                               for node in tree['nodes'] if not node['is_leaf']}))
        transformed, _, _ = model._feature_pipeline.transform_array(normalize(raw))
        transformed_reference = reference_dir / f'{case}-transformed.npy'
        if args.reference:
            if transformed_reference.exists():
                raise ValueError(f'Refusing to overwrite {transformed_reference}')
            np.save(transformed_reference, transformed)
        expected_transform = np.load(transformed_reference)
        np.testing.assert_array_equal(np.ascontiguousarray(transformed).view(np.uint32),
                                      np.ascontiguousarray(expected_transform).view(np.uint32))
        for rows in args.rows:
            batch = raw.iloc[:rows]
            normalized = normalize(batch)
            prepared = model._transform_prediction_pool(normalized)
            predict = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
            expected_path = reference_dir / f'{case}-{rows}.npy'
            actual = predict(normalized)
            if args.reference:
                if expected_path.exists():
                    raise ValueError(f'Refusing to overwrite {expected_path}')
                np.save(expected_path, actual)
            expected = np.load(expected_path)
            for values in (actual, predict(prepared)):
                assert values.shape == expected.shape and values.dtype == expected.dtype
                np.testing.assert_array_equal(np.ascontiguousarray(values).view(np.uint8),
                                              np.ascontiguousarray(expected).view(np.uint8))
            scores = np.asarray(native.predict(prepared._handle))
            legacy = np.asarray(native._predict_uncached(prepared._handle))
            np.testing.assert_array_equal(scores.view(np.uint32), legacy.view(np.uint32))
            record = dict(case=case, rows=rows, input_columns=raw.shape[1],
                          categorical_columns=len(categorical_columns),
                          transformed_columns=transformed.shape[1], **state_info,
                          model_sha256=digest(model_path), input_sha256=digest(data_path),
                          expected_sha256=digest(expected_path),
                          transformed_reference_sha256=digest(transformed_reference),
                          exact_prediction_parity=True, exact_transform_parity=True)
            if not args.check_only:
                record['raw'] = measure(lambda: predict(normalize(batch)))
                record['prepared'] = measure(lambda: predict(prepared))
                record['native'] = measure(lambda: native.predict(prepared._handle))
            records.append(record)
            print(json.dumps(record), flush=True)
    receipt = dict(label=args.label, runtime=ctboost.build_info(), python=sys.version,
                   package=str(Path(ctboost.__file__).resolve()),
                   native_sha256=digest(ctboost._core.__file__), runner_sha256=digest(__file__),
                   build_receipt=json.loads((args.package / 'build_receipt.json').read_text()),
                   versions={name: importlib.metadata.version(name) for name in
                             ('numpy', 'pandas', 'scikit-learn', 'psutil')},
                   cpu_affinity=psutil.Process().cpu_affinity(),
                   protocol='One CPU/thread; 3 warmups; 5 calibration calls; 7 blocks targeting50ms.',
                   limitations='Reused development inputs, cycled/truncated; no new quality or leaderboard evidence.',
                   records=records)
    result_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
