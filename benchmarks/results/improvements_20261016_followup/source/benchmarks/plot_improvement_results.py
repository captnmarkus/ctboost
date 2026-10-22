"""Render completed development evidence; no fitting or model selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = []
    for path in args.archive.glob("accuracy_wave*/development.json"):
        wave = int(re.search(r"accuracy_wave(\d+)", path.parent.name).group(1))
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append((wave, path, report))
    reports.sort(key=lambda item: (item[0], item[1].parent.name))
    if not reports:
        raise ValueError("No completed assessments")
    positions = list(range(1, len(reports) + 1))
    policy_labels = [re.match(r"accuracy_wave(\d+[a-z]*)", path.parent.name).group(1)
                     for _, path, _ in reports]
    labels = [label if policy_labels.count(label) == 1 else
              f"{label}{chr(ord('A') + policy_labels[:index].count(label))}"
              for index, label in enumerate(policy_labels)]

    def overall(report, library):
        rows = [row for row in report["summary"]
                if row["family"] == "all" and row["comparator"] == library
                and row.get("ensemble_pair", "mean") == "mean"
                and row.get("seed", "mean") == "mean"]
        if len(rows) != 1:
            raise ValueError("Expected exactly one overall decision row per comparator")
        return rows[0]
    colors = {"xgboost": "#2869a6", "lightgbm": "#cb671f"}
    names = {"xgboost": "XGBoost", "lightgbm": "LightGBM"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for library in colors:
        values = [100 * overall(report, library)["median_relative_error_gain"]
                  for _, _, report in reports]
        axes[0].plot(positions, values, "o-", color=colors[library],
                     label=f"Versus {names[library]}", linewidth=2)
        axes[0].annotate(f"{values[-1]:+.2f}%", (positions[-1], values[-1]),
                         xytext=(7, -5 if library == "lightgbm" else 6), textcoords="offset points",
                         color=colors[library], fontsize=10)
        wins = [overall(report, library)["wins"] + overall(report, library)["ties"]
                for _, _, report in reports]
        axes[1].plot(positions, wins, "o-", color=colors[library],
                     label=f"Versus {names[library]}", linewidth=2)
        axes[1].annotate(f"{wins[-1]}/12", (positions[-1], wins[-1]),
                         xytext=(7, -5 if library == "lightgbm" else 6), textcoords="offset points",
                         color=colors[library], fontsize=10)
    axes[0].axhline(0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set(title="Median error improvement across 12 tasks",
                xlabel="Completed development wave", ylabel="CTBoost improvement (%) · higher is better",
                xticks=positions, xticklabels=labels)
    axes[0].set_xlim(.7, positions[-1] + .9)
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(axis="y", alpha=.2)

    axes[1].axhline(6, color="#333333", linestyle="--", linewidth=1)
    axes[1].set(title="Tasks where CTBoost wins or ties",
                xlabel="Completed development wave", ylabel="Tasks out of 12 · higher is better",
                xticks=positions, xticklabels=labels, yticks=range(0, 13, 2), ylim=(0, 12))
    axes[1].set_xlim(.7, positions[-1] + .9)
    axes[1].grid(axis="y", alpha=.2)
    axes[1].legend(frameon=False, loc="lower right")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("CTBoost accuracy research — development results", fontsize=16, x=.04, ha="left")
    caption = (
        "Same 12 metadata-selected tasks; model choices use training/stopping rows. Shared development rows evaluate frozen policies.\n"
        "Historically reused, capped development data. These are not official test results or a leaderboard ranking.")
    if len(set(policy_labels)) != len(reports):
        caption += "\nUppercase suffixes distinguish multiple completed assessments of the same policy, including seed-sensitivity checks."
    fig.text(.04, .035, caption,
             fontsize=10, color="#444444")
    fig.subplots_adjust(left=.07, right=.97, top=.85, bottom=.19, wspace=.25)
    destination = args.archive / "figures"
    destination.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"accuracy_development.{suffix}", dpi=160, facecolor="white")
    receipt = {"input_reports": [{"path": str(path.relative_to(args.archive)), "label": label,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for label, (_, path, _) in zip(labels, reports)],
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "latest_wave": reports[-1][0], "latest_policy_label": labels[-1],
               "scope": "Completed development evidence only"}
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"latest_wave": reports[-1][0], "output": str(destination)}))


if __name__ == "__main__":
    main()
