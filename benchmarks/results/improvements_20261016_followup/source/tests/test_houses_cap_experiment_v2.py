"""Synthetic role, parser integration and fixed-policy tests; no network/data."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

spec = importlib.util.spec_from_file_location("tested_houses_cap",
    Path(__file__).resolve().parents[1] / "benchmarks/fair_accuracy_houses_cap_v2.py")
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)


def role_fixture():
    train, test = np.roll(np.arange(13760, dtype=np.int64), 73), np.arange(13760, 20640, dtype=np.int64)
    selected, _ = train_test_split(np.arange(len(train)), train_size=10000, random_state=47)
    selected.sort()
    cap = train[selected]
    parts = {"train": cap[:6250], "stop": cap[6250:7500], "development": cap[7500:]}
    frame = pd.DataFrame({name: np.arange(20640, dtype=np.float64) + column
                          for column, name in enumerate(main.selector.ATTRIBUTES[:-1])})
    labels = np.arange(20640, dtype=np.float64) / 10
    roles = {name: {"X": frame.iloc[ids], "y": labels[ids], "outer_train_positions": ids}
             for name, ids in parts.items()}
    case = {"dataset_name": "houses", "problem_type": "regression", "num_features": 8, "num_classes": -1,
            "original_rows": len(train), "outer_train_index_sha256": main.cap.data.index_hash(train),
            "outer_test_index_sha256": main.cap.data.index_hash(test),
            "roles": {name: {"rows": len(ids), "index_sha256": main.cap.data.index_hash(ids)} for name, ids in parts.items()}}
    return case, train, test, roles, cap, frame, labels


def test_global_id_mapping_preserves_official_order_and_exclusions():
    case, train, test, roles, cap, _, _ = role_fixture()
    manifest = main.authorized_manifest(case, train, test, roles)
    allowed = np.asarray(manifest["authorized_global_rows"])
    np.testing.assert_array_equal(allowed[:7500], np.concatenate([roles['train']['outer_train_positions'], roles['stop']['outer_train_positions']]))
    np.testing.assert_array_equal(allowed[7500:], np.setdiff1d(train, cap))
    assert len(allowed) == 11260 and manifest['added_rows'] == 3760
    assert not np.intersect1d(allowed, np.concatenate([test, roles['development']['outer_train_positions']])).size


def test_overlap_is_rejected_even_with_a_recomputed_role_hash():
    case, train, test, roles, _, _, _ = role_fixture()
    roles['development']['outer_train_positions'][0] = roles['train']['outer_train_positions'][0]
    case['roles']['development']['index_sha256'] = main.cap.data.index_hash(roles['development']['outer_train_positions'])
    with pytest.raises(ValueError, match='exhaust'):
        main.authorized_manifest(case, train, test, roles)


def test_materialization_never_parses_excluded_fields_and_checks_exact_prefix(tmp_path, monkeypatch):
    case, train, test, roles, _, frame, labels = role_fixture()
    manifest = main.authorized_manifest(case, train, test, roles)
    allowed = set(manifest['authorized_global_rows'])
    header = (b'@RELATION houses\n' + b''.join(f'@ATTRIBUTE {name} NUMERIC\n'.encode() for name in main.selector.ATTRIBUTES) + b'@DATA\n')
    lines = []
    for row in range(20640):
        lines.append((','.join(repr(float(v)) for v in [*frame.iloc[row], labels[row]])).encode() if row in allowed else b'\xff,UNPARSEABLE')
    raw = header + b'\n'.join(lines) + b'\n'
    args = SimpleNamespace(output=tmp_path/'output', artifacts=tmp_path/'artifacts')
    args.artifacts.mkdir()
    main.base.write(args.output/'authorized_rows.json', manifest)
    plan = {'source_md5': hashlib.md5(raw).hexdigest(), 'cases': [case], 'all_anchor_jobs': [],
            'inherited_expanded_data': [], 'authorized_rows_sha256': main.base.digest(args.output/'authorized_rows.json')}
    main.base.write(args.output/'plan.json', plan)
    source_path = args.artifacts/'raw.bin'
    source_path.write_bytes(raw)
    main.base.write(args.output/'acquired_source.json', {'plan_sha256': main.base.digest(args.output/'plan.json'),
        'md5': hashlib.md5(raw).hexdigest(), 'sha256': hashlib.sha256(raw).hexdigest(), 'path': str(source_path), 'data_rows_parsed': 0})
    monkeypatch.setattr(main, 'checked', lambda _: plan)
    monkeypatch.setattr(main.base, 'load_role', lambda _, role: roles[role])
    main.materialize(args)
    prepared = main.base.read(args.output/'prepared_data.json')
    source = prepared['expanded_data'][0]
    assert source['receipt']['excluded_rows_decoded'] == source['receipt']['excluded_rows_tokenized'] == 0
    assert source['receipt']['original_train_stop_exact_parity'] is True
    with Path(source['path']).open('rb') as stream:
        actual = pickle.load(stream)
    np.testing.assert_array_equal(actual['y'], labels[manifest['authorized_global_rows']])


@pytest.mark.parametrize('library', ['ctboost','xgboost','lightgbm'])
def test_native_houses_policy_roundtrip_and_seed349(tmp_path, library):
    rng = np.random.default_rng(62)
    frame = pd.DataFrame(rng.normal(size=(80, 8)), columns=main.selector.ATTRIBUTES[:-1])
    labels = 2 * frame.iloc[:, 0].to_numpy() + frame.iloc[:, 1].to_numpy()
    expanded = {'X': frame, 'y': labels, 'verification_frame': frame.iloc[:8], 'receipt': {'synthetic': True}}
    path = tmp_path/'expanded.pkl'
    path.write_bytes(pickle.dumps(expanded, protocol=5))
    source = {'dataset':'houses','path':str(path),'sha256':main.base.digest(path),'receipt':expanded['receipt']}
    if library == 'ctboost':
        params = {'learning_rate':.1,'max_depth':3,'min_data_in_leaf':3,'alpha':.5,'verbose':False}
    elif library == 'xgboost':
        params = {'objective':'reg:squarederror','tree_method':'hist','eta':.1,'max_depth':3,'nthread':1}
    else:
        params = {'objective':'regression','learning_rate':.1,'num_leaves':7,'min_data_in_leaf':3,'num_threads':1,'verbosity':-1}
    job = {'dataset':'houses','library':library,'parameters':params,'rounds':4,'expanded_source':source,
           'binary_refit':None,'temperature':None,'leaf_refit':{'lambda':10.,'shrinkage':.25,'solver':{'atol':1e-8,'btol':1e-8,'iter_lim':2000}}}
    snapshot = copy.deepcopy(job)
    case = {'dataset_name':'houses','problem_type':'regression','num_classes':-1}
    for seed in (47,349):
        payload, details, verification = main.fit_policy(job, case, seed)
        assert details['requested_rounds'] == details['retained_rounds'] == 4
        assert details['early_stopping'] is False
        predictions = main.predict(payload, verification)
        np.testing.assert_array_equal(predictions, main.predict(pickle.loads(pickle.dumps(payload, protocol=5)), verification))
    assert job == snapshot


def test_other_three_expanded_tasks_and_eight_unions_delegate_unchanged(monkeypatch):
    calls = []
    def delegated(job, case, seed):
        calls.append((copy.deepcopy(job),case['dataset_name'],seed))
        return 'payload','details','frame'
    monkeypatch.setattr(main.cap, 'fit_policy', delegated)
    for dataset in (*main.cap.data.EXPAND_DATASETS,'churn'):
        job = {'dataset':dataset,'rounds':123,'expanded_source':{'unchanged':True} if dataset != 'churn' else None}
        assert main.fit_policy(job, {'dataset_name':dataset},349) == ('payload','details','frame')
        assert calls[-1] == (job,dataset,349)
