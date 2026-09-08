"""Independently reproduce the geometric aggregate from the four raw summaries."""
from pathlib import Path
import json
import math
import statistics


def summarize(raw):
    passes = [json.loads((raw / f"pass-{index}-{arm}/summary.json").read_text())
              for index, arm in enumerate(("baseline", "final", "final", "baseline"), start=1)]
    rows = []
    for index in range(14):
        current = [item["rows"][index] for item in passes]
        assert all(row["dataset"] == current[0]["dataset"] for row in current)
        ct = [row["models"]["ctboost_default"]["median_ms"] for row in current]
        cb = [row["models"]["catboost_ag_default"]["median_ms"] for row in current]
        baseline = statistics.geometric_mean((ct[0], ct[3]))
        final = statistics.geometric_mean((ct[1], ct[2]))
        baseline_cb = statistics.geometric_mean((cb[0], cb[3]))
        final_cb = statistics.geometric_mean((cb[1], cb[2]))
        rows.append({"dataset": current[0]["dataset"], "speedup": baseline / final,
                     "final_cb_ratio": final / final_cb,
                     "cb_mid_endpoint_ratio": final_cb / baseline_cb,
                     "pairs": [ct[0] / ct[1], ct[3] / ct[2]]})
    return {"datasets": 14,
            "public_speedup": statistics.geometric_mean(row["speedup"] for row in rows),
            "faster_than_public": sum(row["speedup"] > 1 for row in rows),
            "final_cb_ratio": statistics.geometric_mean(row["final_cb_ratio"] for row in rows),
            "faster_than_cb": sum(row["final_cb_ratio"] < 1 for row in rows),
            "cb_mid_endpoint_ratio": statistics.geometric_mean(row["cb_mid_endpoint_ratio"] for row in rows),
            "cb_ratio_range": [min(row["cb_mid_endpoint_ratio"] for row in rows),
                               max(row["cb_mid_endpoint_ratio"] for row in rows)],
            "pair_speedups": [statistics.geometric_mean(row["pairs"][index] for row in rows)
                              for index in range(2)], "rows": rows}


def compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            compare(a, b)
    elif isinstance(expected, float):
        # Equivalent geometric-mean formulas can round differently in float64.
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    else:
        assert actual == expected


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    result = summarize(directory / "raw")
    compare(result, json.loads((directory / "root-independent-summary.json").read_text()))
    print(json.dumps(result, indent=2))
