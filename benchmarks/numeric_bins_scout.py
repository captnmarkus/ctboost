"""Frozen 64-vs-256-bin development comparison; outer-training rows only.

Reuses 20 fixed-wheel, prior-1 baselines. Timings are descriptive, not a paired
runtime experiment. Neither the production defaults nor HPO admission change.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / '.tmp/ctr-prior-audit-20260907-v2'


def prepare(output):
    import numpy as np
    from autogluon.common.utils.cv_splitter import CVSplitter

    from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame
    from benchmarks.tabarena.local_pilot import indices_hash

    assert not output.exists(), 'Use a new output directory'
    protocol_path = ROOT / 'benchmarks/tabarena/pilot_0159_v1.json'
    protocol = json.loads(protocol_path.read_text())
    prior = json.loads((PRIOR / 'plan.json').read_text())
    assert prior['params'] == protocol['base_params'] and prior['folds'] == [2, 3]
    assert prior['params']['max_bins'] == 256 and prior['threads'] == 2
    assert prior['source_sha256'] == ctr.digest(ctr.__file__)
    native = list((ctr.PYTHONS['fixed'].parents[1] / 'Lib/site-packages/ctboost').glob('_core*.pyd'))
    assert len(native) == 1 and ctr.digest(native[0]) == prior['native_sha256']['fixed']
    cases, jobs, reuse = [], [], []
    previous = {row['dataset_name']: row for row in prior['datasets']}
    for dataset in protocol['datasets']:
        name = dataset['dataset_name']
        source = ctr.SOURCE / 'data' / name / 'outer_train.pkl'
        prepared = json.loads(source.with_name('prepared.json').read_text())
        assert prepared['contains_outer_test_rows'] is False
        assert prepared['protocol_sha256'] == ctr.digest(protocol_path)
        assert prepared['data_sha256'] == ctr.digest(source)
        with source.open('rb') as stream:
            data = pickle.load(stream)
        _, categoricals = normalize_tabarena_frame(data['X'])
        assert bool(categoricals) == (name in previous)
        splitter = CVSplitter(n_splits=8, n_repeats=1, random_state=0,
                              stratify=dataset['problem_type'] != 'regression')
        splits = list(splitter.split(data['X'], data['y']))
        payload = {'X': data['X'], 'y': data['y'], 'splits': {f: splits[f] for f in (2, 3)}}
        destination = output / 'bins64/data' / (name + '.pkl')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('wb') as stream:
            pickle.dump(payload, stream, protocol=5)
        case = {**dataset, 'source_sha256': ctr.digest(source), 'data_sha256': ctr.digest(destination),
                'split_sha256': {str(f): indices_hash(*splits[f]) for f in (2, 3)}}
        cases.append(case)
        if name in previous:
            old_data = PRIOR / 'data' / (name + '.pkl')
            assert previous[name]['source_sha256'] == case['source_sha256']
            assert ctr.digest(old_data) == previous[name]['data_sha256']
            with old_data.open('rb') as stream:
                old = pickle.load(stream)
            for fold in (2, 3):
                for actual, expected in zip(old['splits'][fold], splits[fold]):
                    np.testing.assert_array_equal(actual, expected)
                reuse.append(str(PRIOR / 'fits' / f'{name}-f{fold}-s1.0-fixed.json'))
        else:
            target = output / 'bins256/data' / destination.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(destination, target)
            jobs.extend((256, name, f) for f in (2, 3))
        jobs.extend((64, name, f) for f in (2, 3))
    assert len(cases) == 14 and len(reuse) == 20 and len(jobs) == 36
    dependencies = [Path(__file__), Path(ctr.__file__), protocol_path,
                    ROOT / 'benchmarks/tabarena/ctboost_model.py', ROOT / 'benchmarks/tabarena/learning_options.py',
                    ROOT / 'benchmarks/tabarena/local_pilot.py', ROOT / 'benchmarks/tabarena/local_resources.py']
    for bins in (64, 256):
        ctr.write(output / f'bins{bins}/plan.json', {
            'source_sha256': prior['source_sha256'], 'native_sha256': prior['native_sha256'],
            'datasets': cases, 'params': {**prior['params'], 'max_bins': bins}, 'fit_seconds_limit': 300})
    ctr.write(output / 'plan.json', {
        'purpose': __doc__, 'hypothesis': 'Fewer numeric bins may improve quadratic CIT power and fit cost.',
        'selection': 'All 14 original metadata-selected pilot tasks; no score selection.',
        'split_policy': 'Official outer train only; CVSplitter8 seed0, folds2/3, classification stratified.',
        'analysis': 'Report every task/fold, equal-task median relative error change, wins/losses and deadlines; no tuning or default promotion.',
        'reuse_disclosure': '20 prior1 fixed-wheel baselines reused on the same folds; training/validation rows overlap earlier development studies. No independent confirmation or rigorous runtime claim.',
        'dependencies': {str(p): ctr.digest(p) for p in dependencies},
        'prior_plan_sha256': ctr.digest(PRIOR / 'plan.json'), 'datasets': cases,
        'native_sha256': prior['native_sha256']['fixed'], 'params': prior['params'],
        'arms': [64, 256], 'prior_strength': 1.0, 'model_seed': 'inner fold index',
        'callback_policy': 'Unchanged ctr_prior_scout.fit callback: early stop50; stop when elapsed+2*elapsed/(iteration+1)>=300s;400tree cap.',
        'resolved_fit_params': 'base params, max_bins arm, ctr_prior_strength1, random_seed fold2/3, cat_features from unchanged normalizer, eval_metric AUC/MultiClass/RMSE by family.',
        'workers': 8, 'threads': 2, 'fit_seconds_limit': 300, 'jobs': jobs, 'reuse': reuse,
        'arm_plan_sha256': {str(b): ctr.digest(output / f'bins{b}/plan.json') for b in (64, 256)},
        'expected_new_fits': 36, 'expected_reused_fits': 20, 'expected_total_records': 56})
    print(json.dumps({'plan_sha256': ctr.digest(output / 'plan.json'), 'new_fits': 36, 'reused': 20}))


def verify(output):
    plan = json.loads((output / 'plan.json').read_text())
    assert all(ctr.digest(p) == h for p, h in plan['dependencies'].items())
    assert ctr.digest(PRIOR / 'plan.json') == plan['prior_plan_sha256']
    assert all(ctr.digest(output / f'bins{b}/plan.json') == h for b, h in plan['arm_plan_sha256'].items())
    return plan


def records(output, plan, paths):
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    from benchmarks.tabarena.local_pilot import indices_hash
    result = []
    for path in map(Path, paths):
        row = json.loads(path.read_text())
        assert row['status'] == 'ok' and row['strength'] == 1.0 and row['version'] == 'fixed'
        assert path.name == f"{row['dataset']}-f{row['fold']}-s1.0-fixed.json"
        assert row['plan_sha256'] == ctr.digest(path.parents[1] / 'plan.json')
        assert row['prediction_sha256'] == ctr.digest(path.with_suffix('.npz'))
        case = next(d for d in plan['datasets'] if d['dataset_name'] == row['dataset'])
        data_path = output / 'bins64/data' / (row['dataset'] + '.pkl')
        assert ctr.digest(data_path) == case['data_sha256']
        with data_path.open('rb') as stream:
            data = pickle.load(stream)
        train, validation = data['splits'][row['fold']]
        assert indices_hash(train, validation) == case['split_sha256'][str(row['fold'])]
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['indices'], validation)
            labels = data['y'].iloc[validation]
            if case['problem_type'] != 'regression':
                labels = LabelEncoder().fit(data['y'].iloc[train]).transform(labels)
            np.testing.assert_array_equal(archive['labels'], labels)
            assert len(archive['predictions']) == len(validation)
            assert np.isfinite(archive['predictions']).all() and np.isfinite(row['error'])
        result.append({**row, 'record_path': str(path), 'record_sha256': ctr.digest(path)})
    return result


def run(output):
    from benchmarks.tabarena.local_pilot import kill_worker, process_tree_rss
    from benchmarks.tabarena.local_resources import worker_environment
    plan = verify(output)
    completed = list((PRIOR / 'fits').glob('*.json'))
    assert len(completed) == 80 and all(json.loads(p.read_text())['status'] == 'ok' for p in completed), 'Wait for the CTR study to finish'
    reused = records(output, plan, plan['reuse'])
    manifest = output / 'reused.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == reused
    else:
        ctr.write(manifest, reused)
    jobs = list(plan['jobs'])
    random.Random(640256).shuffle(jobs)
    (output / 'logs').mkdir(exist_ok=True)

    def slot_worker(slot):
        for bins, name, fold in jobs[slot::8]:
            target = output / f'bins{bins}/fits/{name}-f{fold}-s1.0-fixed.json'
            if target.exists():
                records(output, plan, [target])
                continue
            command = [str(ctr.PYTHONS['fixed']), '-I', str(Path(__file__).resolve()), 'fit',
                       '--output', str(output), '--dataset', name, '--fold', str(fold), '--bins', str(bins), '--slot', str(slot)]
            with (output / 'logs' / f'{name}-f{fold}-b{bins}.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    env=worker_environment(2), creationflags=subprocess.CREATE_NO_WINDOW)
                started = time.monotonic()
                try:
                    while process.poll() is None:
                        if time.monotonic() - started > 390 or process_tree_rss(process) > 3 * 1024**3:
                            kill_worker(process)
                            break
                        time.sleep(.25)
                finally:
                    if process.poll() is None:
                        kill_worker(process)
            assert process.returncode == 0 and target.exists(), f'Worker failed: {name} fold{fold} bins{bins}'
            records(output, plan, [target])
            print(json.dumps({'dataset': name, 'fold': fold, 'bins': bins, 'status': 'ok'}), flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(slot_worker, range(8)))


def report(output):
    import numpy as np
    plan = verify(output)
    reused = records(output, plan, plan['reuse'])
    assert json.loads((output / 'reused.json').read_text()) == reused
    all_rows = [{**r, 'max_bins': 256, 'reused': True} for r in reused]
    for bins, name, fold in plan['jobs']:
        path = output / f'bins{bins}/fits/{name}-f{fold}-s1.0-fixed.json'
        all_rows.append({**records(output, plan, [path])[0], 'max_bins': bins, 'reused': False})
    assert len(all_rows) == 56 and len({(r['dataset'], r['fold'], r['max_bins']) for r in all_rows}) == 56
    comparisons = []
    for dataset in plan['datasets']:
        name = dataset['dataset_name']
        means = {b: float(np.mean([r['error'] for r in all_rows if r['dataset'] == name and r['max_bins'] == b])) for b in (64, 256)}
        comparisons.append({'dataset': name, 'error_bins64': means[64], 'error_bins256': means[256],
                            'relative_improvement': (means[256] - means[64]) / max(means[256], 1e-12)})
    changes = [r['relative_improvement'] for r in comparisons]
    ctr.write(output / 'summary.json', {'plan_sha256': ctr.digest(output / 'plan.json'), 'comparisons': comparisons,
              'median_relative_improvement': float(np.median(changes)), 'wins': sum(c > 1e-12 for c in changes),
              'losses': sum(c < -1e-12 for c in changes), 'deadline_stops': sum(r['deadline_stopped'] for r in all_rows),
              'reuse_disclosure': plan['reuse_disclosure'], 'records': all_rows})
    print(json.dumps({'records': len(all_rows), 'median_relative_improvement': float(np.median(changes))}))


if __name__ == '__main__':
    import ctboost
    assert not Path(ctboost.__file__).resolve().is_relative_to(ROOT / 'ctboost')
    sys.path.insert(0, str(ROOT))
    from benchmarks import ctr_prior_scout as ctr
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'fit', 'run', 'report'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--fold', type=int, choices=[2, 3])
    parser.add_argument('--bins', type=int, choices=[64, 256])
    parser.add_argument('--slot', type=int, choices=range(8))
    args = parser.parse_args()
    if args.stage == 'fit':
        plan = verify(args.output)
        assert [args.bins, args.dataset, args.fold] in plan['jobs']
        ctr.fit(args.output / f'bins{args.bins}', args.dataset, args.fold, 1.0, 'fixed', args.slot)
    else:
        globals()[args.stage](args.output.resolve())
