"""Summarize three balanced, immutable library-order latency receipts."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(directory):
    runs = [json.loads((directory / f'wave6-composed-order-{index}.json').read_text())
            for index in (1, 2, 3)]
    for field in ('selection_sha256', 'plan_sha256', 'development_quality_sha256',
                  'runner_sha256', 'preprocessing_runner_sha256', 'runtime'):
        assert all(run[field] == runs[0][field] for run in runs), field
    indexed = [{(r['dataset'], r['rows'], r['library']): r for r in run['records']}
               for run in runs]
    rows = []
    for dataset, count in dict.fromkeys((r['dataset'], r['rows']) for r in runs[0]['records']):
        result = {'dataset': dataset, 'rows': count, 'libraries': {}}
        for library in ('ctboost', 'xgboost', 'lightgbm'):
            records = [run[dataset, count, library] for run in indexed]
            for field in ('model_sha256', 'input_sha256', 'development_error', 'retained_rounds'):
                assert all(r[field] == records[0][field] for r in records), field
            result['libraries'][library] = {
                **{key: records[0][key] for key in ('development_error', 'retained_rounds', 'model_bytes',
                                                   'model_sha256', 'problem_type')},
                **{path: {'median_ms': statistics.geometric_mean(r[path]['median_ms'] for r in records),
                          'order_medians_ms': [r[path]['median_ms'] for r in records]}
                   for path in ('raw', 'normalized_frame')},
            }
        result['ctboost_speedup'] = {
            other: {path: result['libraries'][other][path]['median_ms'] /
                          result['libraries']['ctboost'][path]['median_ms']
                    for path in ('raw', 'normalized_frame')}
            for other in ('xgboost', 'lightgbm')}
        rows.append(result)
    aggregate = {}
    for batch in (None, 1, 64, 1000):
        subset = [r for r in rows if batch is None or r['rows'] == batch]
        aggregate['all' if batch is None else str(batch)] = {
            other: {path: {'geometric_speedup': statistics.geometric_mean(
                              r['ctboost_speedup'][other][path] for r in subset),
                           'faster_cases': sum(r['ctboost_speedup'][other][path] > 1 for r in subset),
                           'cases': len(subset)}
                    for path in ('raw', 'normalized_frame')}
            for other in ('xgboost', 'lightgbm')}
    return {'scope': 'Wave6 diagnostic; differing selected quality/capacity, current composed runtime. '
                     'Speedup is competitor latency / CTBoost latency; >1 means CTBoost faster.',
            'aggregate': aggregate, 'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.directory), indent=2) + '\n')
