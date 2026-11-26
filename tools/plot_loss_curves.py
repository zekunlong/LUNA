import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_SPECS = (
    (
        "loss_kd",
        r"$\mathcal{L}_{\mathrm{ins}}$",
        "#0072B2",
        "o",
        "-",
    ),
    (
        "loss_bc",
        r"$\mathcal{L}_{\mathrm{batch}}$",
        "#D55E00",
        "s",
        "--",
    ),
    (
        "loss_cc",
        r"$\mathcal{L}_{\mathrm{class}}$",
        "#009E73",
        "^",
        "-.",
    ),
)

DISPLAY_NAMES = {
    "caltech101": "Caltech101",
    "dtd": "DTD",
    "eurosat": "EuroSAT",
    "fgvc_aircraft": "FGVC Aircraft",
    "food101": "Food101",
    "oxford_flowers": "Oxford Flowers",
    "oxford_pets": "Oxford Pets",
    "stanford_cars": "Stanford Cars",
    "sun397": "SUN397",
    "ucf101": "UCF101",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the three epoch-averaged LUNA distillation losses."
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", required=True)
    return parser.parse_args()


def publication_title(raw_title):
    match = re.fullmatch(r"(.+?)\s*\(seed\s+(\d+)\)", raw_title, flags=re.IGNORECASE)
    if not match:
        return raw_title
    dataset, _seed = match.groups()
    return DISPLAY_NAMES.get(dataset.lower(), dataset)


def main():
    args = parse_args()
    with args.csv.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(f"No loss rows found in {args.csv}")

    required = {"epoch", *(name for name, *_ in LOSS_SPECS)}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    epochs = [int(row["epoch"]) for row in rows]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.8,
        }
    )

    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.16, top=0.87)

    for name, label, color, marker, linestyle in LOSS_SPECS:
        values = [float(row[name]) for row in rows]
        if any(value <= 0 for value in values):
            raise ValueError(f"Log-scale plot requires positive {name} values")
        axis.plot(
            epochs,
            values,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )

    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss (log scale)")
    axis.set_xticks(epochs)
    axis.set_title(publication_title(args.title), pad=10, fontweight="semibold")
    axis.grid(axis="y", which="major", color="#CBD5E1", linewidth=0.7, alpha=0.8)
    axis.grid(axis="x", which="major", color="#E2E8F0", linewidth=0.6, alpha=0.65)
    axis.tick_params(direction="out", length=4, width=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#CBD5E1",
        handlelength=2.7,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output.with_suffix(".png")
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved loss curves to {png_path}")


if __name__ == "__main__":
    main()
