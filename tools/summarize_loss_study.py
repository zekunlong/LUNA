#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASETS = [
    "caltech101",
    "dtd",
    "eurosat",
    "fgvc_aircraft",
    "food101",
    "oxford_flowers",
    "oxford_pets",
    "stanford_cars",
    "sun397",
    "ucf101",
]
LOSSES = ["loss_kd", "loss_cc", "loss_bc"]
CFG_NAME = "vit_b16_c2_ep20_batch8_4+4ctx"
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and summarize the seed-1 LUNA loss study."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="LUNA package root (defaults to the parent of this tools directory).",
    )
    return parser.parse_args()


def read_dataset(root: Path, dataset: str):
    csv_path = (
        root
        / "output"
        / "base2new"
        / "train_base"
        / dataset
        / "shots_0"
        / "LUNA"
        / CFG_NAME
        / "seed_1"
        / "loss_trends.csv"
    )
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing loss CSV: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5 or [int(row["epoch"]) for row in rows] != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected epochs 1..5 in {csv_path}, got {len(rows)} rows")

    parsed = []
    for row in rows:
        item = {"dataset": dataset, "epoch": int(row["epoch"])}
        for loss in LOSSES:
            item[loss] = float(row[loss])
        parsed.append(item)
    return parsed


def main():
    args = parse_args()
    root = args.root.resolve()
    summary_dir = root / "output" / "luna_loss_study_seed1"
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    by_dataset = {}
    for dataset in DATASETS:
        rows = read_dataset(root, dataset)
        all_rows.extend(rows)
        by_dataset[dataset] = rows

    summary_csv = summary_dir / "loss_study_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "epoch", *LOSSES])
        writer.writeheader()
        writer.writerows(all_rows)

    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    titles = {
        "loss_kd": "Instance-level distillation loss (loss_kd)",
        "loss_cc": "Class-correlation distillation loss (loss_cc)",
        "loss_bc": "Batch-correlation distillation loss (loss_bc)",
    }
    for axis, loss in zip(axes, LOSSES):
        for dataset in DATASETS:
            rows = by_dataset[dataset]
            axis.plot(
                [row["epoch"] for row in rows],
                [row[loss] for row in rows],
                marker="o",
                linewidth=1.5,
                label=dataset,
            )
        axis.set_title(titles[loss])
        axis.set_ylabel("Epoch mean (coefficient-free)")
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("LUNA base-to-new distillation-loss trends (seed 1)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 0.84, 0.97))
    overview = summary_dir / "loss_study_overview.png"
    fig.savefig(overview, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Validated {len(DATASETS)} datasets and {len(all_rows)} epoch rows")
    print(f"Saved summary CSV to {summary_csv}")
    print(f"Saved overview plot to {overview}")


if __name__ == "__main__":
    main()
