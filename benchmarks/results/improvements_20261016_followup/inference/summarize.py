"""Summarize paired fixed-order inference passes without discarding raw blocks."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(directory, baseline="baseline", candidate="combined"):
    arms = {
        name: [json.loads((directory / f"{name}-{index}.json").read_text())
               for index in (1, 2)]
        for name in (baseline, candidate)
    }
    indexed = {
        name: [{(row["case"], row["rows"]): row for row in run["records"]}
               for run in runs]
        for name, runs in arms.items()
    }
    rows = []
    for key in indexed[baseline][0]:
        row = dict(case=key[0], rows=key[1])
        for path in ("raw", "prepared", "native"):
            base = [run[key][path]["median_ms"] for run in indexed[baseline]]
            cand = [run[key][path]["median_ms"] for run in indexed[candidate]]
            base_ms, cand_ms = statistics.geometric_mean(base), statistics.geometric_mean(cand)
            row[path] = dict(baseline_ms=base_ms, candidate_ms=cand_ms,
                             speedup=base_ms / cand_ms,
                             ordering_speedups=[a / b for a, b in zip(base, cand)])
        rows.append(row)
    return dict(rows=rows, geometric_speedup={
        path: statistics.geometric_mean(row[path]["speedup"] for row in rows)
        for path in ("raw", "prepared", "native")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--candidate", default="combined")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(summarize(args.directory, args.baseline, args.candidate), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
