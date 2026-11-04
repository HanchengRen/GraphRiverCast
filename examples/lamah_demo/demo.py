"""Validate the archived LamaH gauge subsets and reproduce score summaries."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_splits(config):
    rows = read_csv(ROOT / "station_subsets.csv")
    nodes = [int(row["node_index"]) for row in rows]
    if len(nodes) != config["total_gauges"] or len(set(nodes)) != len(nodes):
        raise ValueError("Gauge count or node-index uniqueness check failed")
    subsets = {}
    previous = set()
    for percent in config["supervision_percentages"]:
        column = f"supervised_{percent}pct"
        if any(row[column] not in ("0", "1") for row in rows):
            raise ValueError(f"Invalid membership value in {column}")
        subset = {int(row["node_index"]) for row in rows if row[column] == "1"}
        if len(subset) != config["supervised_counts"][str(percent)]:
            raise ValueError(f"Wrong supervised-gauge count for {percent}%")
        if not previous.issubset(subset):
            raise ValueError(f"Gauge subsets are not nested at {percent}%")
        subsets[percent] = subset
        previous = subset
    return set(nodes), subsets


def summarize(config, nodes, subsets):
    per_seed = []
    summary = []
    for condition in config["conditions"]:
        for percent in config["supervision_percentages"]:
            name = f"{condition}_r{percent / 100:g}_per_station.csv"
            rows = read_csv(ROOT / "metrics" / name)
            row_nodes = [int(row["node_index"]) for row in rows]
            if len(row_nodes) != len(nodes) or set(row_nodes) != nodes:
                raise ValueError(f"Gauge population mismatch in {name}")
            if any(row["supervised"].lower() not in ("true", "false") for row in rows):
                raise ValueError(f"Invalid supervised flag in {name}")
            supervised = {
                int(row["node_index"]) for row in rows
                if row["supervised"].lower() == "true"
            }
            if supervised != subsets[percent]:
                raise ValueError(f"Gauge subset mismatch in {name}")
            for metric in ("NSE", "KGE"):
                seed_medians = []
                for seed in config["training_seeds"]:
                    column = f"{metric}_s{seed}"
                    values = [float(row[column]) for row in rows]
                    if any(math.isinf(value) for value in values):
                        raise ValueError(f"Infinite score in {name}: {column}")
                    valid = [value for value in values if not math.isnan(value)]
                    if not valid:
                        raise ValueError(f"No valid scores in {name}: {column}")
                    value = median(valid)
                    seed_medians.append(value)
                    per_seed.append({
                        "condition": condition, "supervision_percent": percent,
                        "metric": metric, "training_seed": seed,
                        "n_scored_gauges": len(valid), "station_median": value,
                    })
                lo, hi = min(seed_medians), max(seed_medians)
                summary.append({
                    "condition": condition, "supervision_percent": percent,
                    "n_supervised": len(supervised), "metric": metric,
                    "median_across_seeds": median(seed_medians),
                    "min_across_seeds": lo, "max_across_seeds": hi,
                    "half_range": (hi - lo) / 2,
                })
    return per_seed, summary


def make_plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = [
        ("finetune", "GRC, pre-trained", "#279174", "o", "-"),
        ("scratch", "GRC, from scratch", "#555555", "o", "--"),
        ("lstm_finetune", "LSTM, pre-trained", "#D67A22", "s", "-."),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric in zip(axes, ("NSE", "KGE")):
        for condition, label, color, marker, style in curves:
            rows = sorted(
                (row for row in summary
                 if row["condition"] == condition and row["metric"] == metric),
                key=lambda row: row["supervision_percent"],
            )
            x = [row["supervision_percent"] for row in rows]
            ax.fill_between(
                x, [row["min_across_seeds"] for row in rows],
                [row["max_across_seeds"] for row in rows], color=color, alpha=0.16,
            )
            ax.plot(x, [row["median_across_seeds"] for row in rows],
                    color=color, marker=marker, linestyle=style, label=label)
        ax.set(xlabel="Gauges used for supervision (%)", ylabel=metric,
               title=f"LamaH-CE: {metric}", xticks=[10, 30, 50, 70, 90])
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Median across three runs; shading shows their min-max range", fontsize=11)
    fig.savefig(output / "lamah_supervision_demo.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--plot", action="store_true", help="Also plot curves using Matplotlib")
    args = parser.parse_args()
    config = json.loads((ROOT / "experiment.json").read_text(encoding="utf-8"))
    nodes, subsets = load_splits(config)
    per_seed, summary = summarize(config, nodes, subsets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_seed_station_medians.csv", per_seed)
    write_csv(args.output_dir / "supervision_summary.csv", summary)
    for percent, subset in subsets.items():
        print(f"{percent:2d}% supervision: {len(subset):3d}/{len(nodes)} gauges")
    print("Validated identical, nested station subsets across all three models.")
    print(f"Reproduced {len(summary)} metric summaries from {len(per_seed)} per-run medians.")
    if args.plot:
        make_plot(summary, args.output_dir)
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
