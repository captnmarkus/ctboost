import json
from pathlib import Path
import statistics

directory = Path(__file__).parent / 'timing'
passes = {label: json.loads((directory / f'{label}.json').read_text())
          for label in ('baseline-1', 'specialized-1', 'specialized-2', 'baseline-2')}
rows = []
for index, first in enumerate(passes['baseline-1']['cases']):
    row = {key: first[key] for key in ('case', 'rows', 'trees', 'nodes')}
    for path in ('raw', 'prepared', 'catboost_control'):
        values = {label: receipt['cases'][index][path]['median_ms']
                  for label, receipt in passes.items()}
        before = statistics.geometric_mean([values['baseline-1'], values['baseline-2']])
        after = statistics.geometric_mean([values['specialized-1'], values['specialized-2']])
        row[path] = dict(baseline_ms=before, specialized_ms=after, speedup=before / after,
                         ordering_speedups=[values['baseline-1'] / values['specialized-1'],
                                            values['baseline-2'] / values['specialized-2']])
    row['raw_control_adjusted_speedup'] = row['raw']['speedup'] / row['catboost_control']['speedup']
    rows.append(row)
summary = dict(rows=rows, geometric_speedup={
    path: statistics.geometric_mean(row[path]['speedup'] for row in rows)
    for path in ('raw', 'prepared', 'catboost_control')},
    raw_control_adjusted_geometric_speedup=statistics.geometric_mean(
        row['raw_control_adjusted_speedup'] for row in rows))
(directory / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
