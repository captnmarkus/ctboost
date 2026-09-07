"""Paired development check of CTR smoothing; never reads outer-test rows.

Use the existing isolated benchmark Python for prepare/run/report. Each fit uses
the baseline or fixed wheel in its own process. This is not an HPO admission gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import queue
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.tmp/tabarena-local-0159/pilot'
PYTHONS = {
    'baseline': ROOT / '.tmp/tabarena-hpo0159-venv/Scripts/python.exe',
    'fixed': ROOT / '.tmp/score-audit-0159-venv/Scripts/python.exe',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def bootstrap():
    import ctboost
    assert not Path(ctboost.__file__).resolve().is_relative_to(ROOT / 'ctboost')
    sys.path.insert(0, str(ROOT))
    return ctboost


def prepare(output):
    bootstrap()
    from autogluon.common.utils.cv_splitter import CVSplitter
    from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame

    if output.exists():
        raise ValueError('Use a new output directory; preserve previous attempts')
    protocol = json.loads((ROOT / 'benchmarks/tabarena/pilot_0159_v1.json').read_text())
    cases = []
    for dataset in protocol['datasets']:
        source = SOURCE / 'data' / dataset['dataset_name'] / 'outer_train.pkl'
        with source.open('rb') as stream:
            data = pickle.load(stream)
        _, categoricals = normalize_tabarena_frame(data['X'])
        if not categoricals:
            continue
        splitter = CVSplitter(n_splits=8, n_repeats=1, random_state=0,
                              stratify=dataset['problem_type'] != 'regression')
        splits = list(splitter.split(data['X'], data['y']))
        payload = {'X': data['X'], 'y': data['y'],
                   'splits': {fold: splits[fold] for fold in (2, 3)}}
        destination = output / 'data' / (dataset['dataset_name'] + '.pkl')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('wb') as stream:
            pickle.dump(payload, stream, protocol=5)
        cases.append({**dataset, 'source_sha256': digest(source), 'data_sha256': digest(destination)})
    native_hashes = {}
    for version, python in PYTHONS.items():
        native = list((python.parents[1] / 'Lib/site-packages/ctboost').glob('_core*.pyd'))
        assert len(native) == 1
        native_hashes[version] = digest(native[0])
    plan = {'purpose': 'Exploratory bugfix comparison; no official test rows or HPO admission.',
            'selection': 'All 10 datasets with categorical columns from the previous metadata-selected pilot panel.',
            'reuse_disclosure': 'Inner folds 2/3 were not scored in the previous pilot; training rows overlap. This is development evidence, not independent validation.',
            'datasets': cases, 'folds': [2, 3], 'prior_strengths': [0.2, 1.0],
            'params': protocol['base_params'], 'workers': 8, 'threads': 2,
            'fit_seconds_limit': 300, 'expected_fits': len(cases) * 8,
            'source_sha256': digest(__file__), 'native_sha256': native_hashes,
            'negative_control': 'Strength1 predictions must agree exactly between wheels.',
            'analysis': 'Report all paired errors, equal-task median relative change and all regressions; no parameter selection.'}
    assert len(cases) == 10
    write(output / 'plan.json', plan)
    print(json.dumps({'datasets': len(cases), 'fits': plan['expected_fits'], 'plan_sha256': digest(output / 'plan.json')}))


def fit(output, dataset, fold, strength, version, slot):
    import psutil
    os.environ['CTBOOST_HIST_THREADS'] = '2'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    psutil.Process().cpu_affinity([slot * 2, slot * 2 + 1])
    ctboost = bootstrap()
    import numpy as np
    from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
    from sklearn.preprocessing import LabelEncoder
    from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame
    from benchmarks.tabarena.local_pilot import MemorySampler

    plan = json.loads((output / 'plan.json').read_text())
    assert plan['source_sha256'] == digest(__file__)
    assert plan['native_sha256'][version] == digest(ctboost._core.__file__)
    metadata = next(row for row in plan['datasets'] if row['dataset_name'] == dataset)
    data_path = output / 'data' / (dataset + '.pkl')
    assert digest(data_path) == metadata['data_sha256']
    with data_path.open('rb') as stream:
        data = pickle.load(stream)
    train, validation = data['splits'][fold]
    X_train, categoricals = normalize_tabarena_frame(data['X'].iloc[train])
    X_val, _ = normalize_tabarena_frame(data['X'].iloc[validation], categorical_columns=categoricals)
    y_train, y_val = data['y'].iloc[train], data['y'].iloc[validation]
    family = metadata['problem_type']
    if family != 'regression':
        encoder = LabelEncoder().fit(y_train)
        assert len(encoder.classes_) == metadata['num_classes']
        y_train, y_val = encoder.transform(y_train), encoder.transform(y_val)
    params = dict(plan['params'])
    patience = params.pop('early_stopping_rounds')
    params.update(ctr_prior_strength=strength, random_seed=fold, cat_features=categoricals,
                  eval_metric={'binary': 'AUC', 'multiclass': 'MultiClass', 'regression': 'RMSE'}[family])
    estimator = ctboost.CTBoostRegressor if family == 'regression' else ctboost.CTBoostClassifier
    model = estimator(**params)
    started = time.monotonic()
    record = {'dataset': dataset, 'fold': fold, 'strength': strength, 'version': version,
              'family': family, 'plan_sha256': digest(output / 'plan.json'), 'deadline_stopped': False}

    def deadline(env):
        elapsed = time.monotonic() - started
        stop = elapsed + 2 * elapsed / (env.iteration + 1) >= plan['fit_seconds_limit']
        record['deadline_stopped'] |= stop
        return stop

    with MemorySampler() as memory:
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=patience, callbacks=[deadline])
    elapsed = time.monotonic() - started
    predictions = model.predict(X_val) if family == 'regression' else model.predict_proba(X_val)
    if family == 'binary':
        predictions = predictions[:, 1]
        error = 1 - roc_auc_score(y_val, predictions)
    elif family == 'multiclass':
        error = log_loss(y_val, predictions, labels=np.arange(metadata['num_classes']))
    else:
        error = np.sqrt(mean_squared_error(y_val, predictions))
    assert np.isfinite(error) and np.isfinite(predictions).all()
    target = output / 'fits' / f'{dataset}-f{fold}-s{strength}-{version}.json'
    assert not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target.with_suffix('.npz'), predictions=predictions, labels=y_val, indices=validation)
    record.update(error=float(error), fit_seconds=elapsed, peak_rss_bytes=memory.peak,
                  rounds=int(model.get_booster().num_iterations_trained), status='ok',
                  prediction_sha256=digest(target.with_suffix('.npz')))
    write(target, record)


def run(output):
    bootstrap()
    from benchmarks.tabarena.local_pilot import kill_worker, process_tree_rss
    from benchmarks.tabarena.local_resources import worker_environment
    plan = json.loads((output / 'plan.json').read_text())
    assert plan['source_sha256'] == digest(__file__)
    jobs = [(d['dataset_name'], fold, strength) for d in plan['datasets']
            for fold in plan['folds'] for strength in plan['prior_strengths']]
    random.Random(160).shuffle(jobs)
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    (output / 'logs').mkdir(exist_ok=True)

    def slot_worker(slot):
        while True:
            try:
                dataset, fold, strength = pending.get_nowait()
            except queue.Empty:
                return
            versions = ['baseline', 'fixed']
            random.Random(f'{dataset}:{fold}:{strength}').shuffle(versions)
            for version in versions:
                stem = f'{dataset}-f{fold}-s{strength}-{version}'
                target = output / 'fits' / (stem + '.json')
                if target.exists():
                    continue
                command = [str(PYTHONS[version]), '-I', str(Path(__file__).resolve()), 'fit',
                           '--output', str(output), '--dataset', dataset, '--fold', str(fold),
                           '--strength', str(strength), '--version', version, '--slot', str(slot)]
                with (output / 'logs' / (stem + '.log')).open('w') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                        env=worker_environment(2), creationflags=subprocess.CREATE_NO_WINDOW)
                    launched = time.monotonic()
                    try:
                        while process.poll() is None:
                            if time.monotonic() - launched > 390 or process_tree_rss(process) > 3 * 1024**3:
                                kill_worker(process)
                                break
                            time.sleep(.25)
                    finally:
                        if process.poll() is None:
                            kill_worker(process)
                if not target.exists():
                    write(target, {'dataset': dataset, 'fold': fold, 'strength': strength,
                                   'version': version, 'status': 'failed', 'exit_code': process.returncode})
                print(json.dumps({'finished': stem, 'returncode': process.returncode}), flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(slot_worker, range(8)))


def report(output):
    import numpy as np
    plan = json.loads((output / 'plan.json').read_text())
    records = [json.loads(path.read_text()) for path in (output / 'fits').glob('*.json')]
    assert len(records) == plan['expected_fits'] and all(r['status'] == 'ok' for r in records)
    for path in (output / 'fits').glob('*.json'):
        record = json.loads(path.read_text())
        assert record['plan_sha256'] == digest(output / 'plan.json')
        assert record['prediction_sha256'] == digest(path.with_suffix('.npz'))
    rows = []
    for dataset in plan['datasets']:
        for strength in plan['prior_strengths']:
            means = {}
            for version in PYTHONS:
                paired = [r for r in records if r['dataset'] == dataset['dataset_name']
                          and r['strength'] == strength and r['version'] == version]
                assert {r['fold'] for r in paired} == {2, 3} and len(paired) == 2
                means[version] = float(np.mean([r['error'] for r in paired]))
            if strength == 1.0:
                for fold in (2, 3):
                    prefix = output / 'fits' / f"{dataset['dataset_name']}-f{fold}-s1.0"
                    with np.load(str(prefix) + '-baseline.npz') as old, np.load(str(prefix) + '-fixed.npz') as new:
                        assert np.array_equal(old['predictions'], new['predictions']), 'Control predictions changed'
            rows.append({'dataset': dataset['dataset_name'], 'strength': strength, **means,
                         'relative_error_improvement': (means['baseline'] - means['fixed']) / max(means['baseline'], 1e-12)})
    affected = [row for row in rows if row['strength'] == .2]
    summary = {'fits': len(records), 'control_predictions_exact': True, 'rows': rows,
               'median_relative_error_improvement': float(np.median([r['relative_error_improvement'] for r in affected])),
               'wins': sum(r['relative_error_improvement'] > 1e-12 for r in affected),
               'losses': sum(r['relative_error_improvement'] < -1e-12 for r in affected),
               'deadline_stops': sum(r['deadline_stopped'] for r in records)}
    write(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'fit', 'run', 'report'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--fold', type=int)
    parser.add_argument('--strength', type=float)
    parser.add_argument('--version', choices=list(PYTHONS))
    parser.add_argument('--slot', type=int)
    args = parser.parse_args()
    if args.stage == 'fit':
        fit(args.output.resolve(), args.dataset, args.fold, args.strength, args.version, args.slot)
    else:
        globals()[args.stage](args.output.resolve())
