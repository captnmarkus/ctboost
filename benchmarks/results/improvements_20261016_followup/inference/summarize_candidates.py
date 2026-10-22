"""Compare independent candidates against the shared balanced baseline passes."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(directory, candidate, baseline="accepted"):
    files = {arm: [json.loads((directory / f"{arm}-{index}.json").read_text())
                   for index in (1, 2)] for arm in (baseline, candidate)}
    record_field = "records" if "records" in files[baseline][0] else "cases"
    paths = ("raw", "prepared", "native") if record_field == "records" else ("raw", "prepared", "catboost_control")
    runs = {arm: [{(r["case"], r["rows"]): r for r in run[record_field]} for run in values]
            for arm, values in files.items()}
    rows = []
    for key in runs[baseline][0]:
        row = dict(case=key[0], rows=key[1])
        for path in paths:
            base = [run[key][path]["median_ms"] for run in runs[baseline]]
            values = [run[key][path]["median_ms"] for run in runs[candidate]]
            baseline_ms, measured = statistics.geometric_mean(base), statistics.geometric_mean(values)
            row[path] = dict(baseline_ms=baseline_ms, candidate_ms=measured, speedup=baseline_ms / measured,
                             ordering_speedups=[a / b for a, b in zip(base, values)])
        rows.append(row)
    return dict(candidate=candidate, rows=rows, geometric_speedup={
        path: statistics.geometric_mean(row[path]["speedup"] for row in rows) for path in paths})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("candidate")
    parser.add_argument("--baseline", default="accepted")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(summarize(args.directory, args.candidate, args.baseline), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
