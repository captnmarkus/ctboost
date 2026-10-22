"""Render every real case from the frozen categorical batching replication."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    source = args.archive / "profile/native_guard_replication/comparison.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    rows = [row for row in report["rows"] if row["kind"] == "real"]
    cases = list(dict.fromkeys(row["case"] for row in rows))
    batches = sorted({row["rows"] for row in rows})
    assert report["passed"] and len(cases) == 5 and len(batches) == 7 and len(rows) == 35
    assert len({(row["case"], row["rows"]) for row in rows}) == 35
    cells = [[next(row for row in rows if row["case"] == case and row["rows"] == batch)
              for batch in batches] for case in cases]
    values = np.array([[cell["speedup"] for cell in line] for line in cells])
    colors = LinearSegmentedColormap.from_list("slower_neutral_faster", ["#bc5b38", "#f5f6f6", "#317d9b"])
    fig, axis = plt.subplots(figsize=(11.8, 5.5))
    axis.imshow(np.log2(values), cmap=colors, norm=TwoSlopeNorm(vmin=-3., vcenter=0., vmax=3.), aspect="auto")
    for i, line in enumerate(cells):
        for j, cell in enumerate(line):
            text = f"{cell['speedup']:.2f}×" + ("*" if cell["eligible"] else "")
            axis.text(j, i, text, ha="center", va="center", fontsize=12,
                      color="white" if cell["speedup"] > 3 else "#173848")
    axis.set_xticks(range(len(batches)), [f"{n:,}" for n in batches])
    axis.set_yticks(range(len(cases)), cases)
    axis.set_xlabel("Prediction batch (rows)", labelpad=12)
    axis.set_xticks(np.arange(-.5, len(batches), 1), minor=True)
    axis.set_yticks(np.arange(-.5, len(cases), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=3)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.tick_params(which="major", length=0, pad=9)
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.suptitle("CTBoost categorical batching: all 35 real prediction cases", fontsize=16,
                 x=.04, ha="left", y=.96)
    fig.text(.04, .88, "Speedup over the prior composed implementation · 1× is unchanged · higher is faster",
             fontsize=11, color="#444444")
    fig.text(.04, .045,
        f"Fixed models and exact outputs; balanced ABBA timing. Geometric mean: {report['real_raw_geomean_speedup']:.3f}×.\n"
        "* Eligible native batching: at least 1,000 rows and 32 supported categorical columns. Other cases use the existing path.\n"
        "CPU: Ryzen 7 5800X3D, one pinned thread. These are CTBoost implementation comparisons, not cross-library rankings.",
        fontsize=9.5, color="#444444")
    fig.subplots_adjust(left=.235, right=.97, top=.80, bottom=.25)
    destination = args.archive / "figures"
    destination.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"categorical_batching.{suffix}", dpi=160, facecolor="white")
    plt.close(fig)
    receipt = {"input": str(source.relative_to(args.archive)),
               "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "cases": len(rows), "scope": "All real cases from the accepted guarded replication"}
    (destination / "categorical_batching_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
