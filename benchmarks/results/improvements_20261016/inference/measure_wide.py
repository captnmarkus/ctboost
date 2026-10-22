"""Fixed schema-copy stress case, separate from fitted-model evidence."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--package', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--label', required=True)
parser.add_argument('--prepare', action='store_true')
parser.add_argument('--check-only', action='store_true')
args = parser.parse_args()
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
spec = importlib.util.spec_from_file_location('measure_common', Path(__file__).with_name('measure_inference.py'))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
args.output.mkdir(exist_ok=True, parents=True)
if args.prepare:
    schema = dict(num_bins_per_feature=[256] * 512,
                  cut_offsets=list(range(0, 512 * 255 + 1, 255)),
                  cut_values=list(np.linspace(-3, 3, 255)) * 512,
                  categorical_mask=[0] * 512, missing_value_mask=[0] * 512,
                  nan_mode=0, nan_modes=[0] * 512)
    def node(feature=-1, left=-1, right=-1, threshold=0, weight=0):
        return dict(is_leaf=feature < 0, is_categorical_split=False,
                    split_feature_id=feature, left_child=left, right_child=right,
                    split_bin_index=threshold, leaf_weight=weight, left_categories=[0] * 256)
    state = ctboost._core.GradientBooster(objective='RMSE').export_state()
    state.update(quantization_schema=schema, base_score=[0.25],
                 tree_learning_rates=[0.05] * 64, loss_history=[1.0] * 64,
                 best_iteration=63, best_score=1.0)
    state['trees'] = [dict(nodes=[node(t % 4, 1, 2, 128), node((t + 1) % 4, 3, 4, 100),
                                  node((t + 2) % 4, 5, 6, 150),
                                  *[node(weight=(k - 1.25) / (t + 1)) for k in range(4)]],
                           feature_importances=[1.0] * 4 + [0.0] * 508) for t in range(64)]
    (args.output / 'state.json').write_text(json.dumps(state))
    np.save(args.output / 'input.npy', np.random.default_rng(64116).normal(size=(64, 512)).astype('float32'))
state = json.loads((args.output / 'state.json').read_text())
model = ctboost.Booster(ctboost._core.GradientBooster.from_state(state))
data = np.load(args.output / 'input.npy')
records = []
for rows in (1, 64):
    batch = data[:rows]
    prepared = ctboost.Pool(batch)
    expected_path = args.output / f'reference-{rows}.npy'
    if args.prepare:
        np.save(expected_path, model.predict(batch))
    expected = np.load(expected_path)
    for actual in (model.predict(batch), model.predict(prepared),
                   np.asarray(model._handle._predict_uncached(prepared._handle))):
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
    row = dict(rows=rows, columns=512, active_columns=4, trees=64, depth=2,
               exact_parity=True, reference_sha256=common.sha(expected_path))
    if not args.check_only:
        row['raw'] = common.measure(lambda: model.predict(batch))
        row['prepared'] = common.measure(lambda: model.predict(prepared))
    records.append(row)
result = dict(label=args.label, purpose='Deliberately wide schema-copy stress case; no quality/general speed claim',
              runner_sha256=common.sha(__file__), common_runner_sha256=common.sha(common.__file__),
              state_sha256=common.sha(args.output / 'state.json'), input_sha256=common.sha(args.output / 'input.npy'),
              native_sha256=common.sha(ctboost._core.__file__), build_info=ctboost.build_info(),
              build_receipt=json.loads((args.package / 'build_receipt.json').read_text()),
              versions={name: common.importlib.metadata.version(name) for name in ('numpy', 'psutil')},
              python=sys.version, affinity=psutil.Process().cpu_affinity(), cases=records)
destination = args.output / f'{args.label}.json'
if destination.exists():
    raise ValueError('Refusing to overwrite receipt')
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
