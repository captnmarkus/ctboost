from pathlib import Path
import hashlib
import json

root = Path.cwd()
work = root / '.tmp/improvements-20261016/followup-native'
output = work / 'fair/wave11-ensemble'
reference = output / 'wave11-final-check-v2.json'
receipt = json.loads(reference.read_text(encoding='utf-8'))
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert receipt['check_only'] and len(receipt['records']) == 108
frozen_files = [root / 'benchmarks/fair_inference_ensemble_scout.py',
                root / 'benchmarks/summarize_fair_inference.py',
                root / 'benchmarks/fixed_seed_ensemble.py',
                root / 'benchmarks/fixed_temperature.py',
                root / 'benchmarks/fair_accuracy_scout.py',
                work / 'run-ensemble-latency.ps1', reference]
plan = {'protocol': 'final_native_wave11_ensemble_latency_v1',
        'runtime': receipt['runtime'], 'frozen_files': {str(path): digest(path) for path in frozen_files},
        'accuracy_plan_sha256': receipt['plan_sha256'], 'selection_sha256': receipt['selection_sha256'],
        'scope': 'Diagnostic serving costs of the failed-quality wave11 two-seed policy on final production CP312; no fitting or new quality scoring.',
        'cases': 12, 'libraries': 3, 'batches': [1, 64, 1000], 'cpu': 6, 'threads': 1,
        'passes': [{'label': f'wave11-ensemble-order-{index}', 'order': order}
                   for index, order in enumerate((['ctboost', 'xgboost', 'lightgbm'],
                                                   ['xgboost', 'lightgbm', 'ctboost'],
                                                   ['lightgbm', 'ctboost', 'xgboost']), 1)],
        'measurement': 'Each path: 3 warmups, 5 calibration calls and 7 blocks targeting 50 ms; raw includes one normalization, two complete model calls, component temperatures and mixing.',
        'acceptance': 'Every pass must match the reference source/runtime/model/input metadata. Full-source and all batch outputs and XGB component/mixture inplace outputs must be bit exact. Preserve all passes; do not discard a slow or adverse pass.',
        'timing_authorization': 'Prepared only; launch requires explicit coordinated CPU release from parent/accuracy agent.'}
destination = output / 'latency-plan.json'
if destination.exists():
    raise ValueError('Preserve existing timing plans')
destination.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
print(destination)
