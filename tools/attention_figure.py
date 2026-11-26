#!/usr/bin/env python3
"""Publication-sized attention figures from measured arrays, without model imports.

The NPZ contract uses images[N,H,W,3], maps[N,M,h,w], methods[M],
classnames[N] and a scalar JSON metadata string. No pickled/object arrays.
Missing methods are never imputed. Complete exports require all five methods.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.cm import ScalarMappable
import numpy as np
from PIL import Image

METHODS = ("CLIP", "CoOp", "KgCoOp", "DeKg", "LUNA")


def load_bundle(path, allow_partial=False):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    required = {"images", "maps", "methods", "classnames", "metadata"}
    if not required.issubset(result):
        raise ValueError("Missing bundle fields: " + str(sorted(required - result.keys())))
    metadata = json.loads(str(result["metadata"].item()))
    images = np.asarray(result["images"], dtype=np.float32)
    maps = np.asarray(result["maps"], dtype=np.float64)
    methods = result["methods"].tolist()
    if len(methods) != len(set(methods)) or any(m not in METHODS for m in methods):
        raise ValueError("Methods must be unique canonical names: " + str(METHODS))
    if images.ndim != 4 or images.shape[-1] != 3 or not len(images):
        raise ValueError("images must have shape [N,H,W,3], N > 0")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("images must be finite RGB values in [0,1]")
    if maps.ndim != 4 or maps.shape[:2] != (len(images), len(methods)):
        raise ValueError("maps must have shape [N,M,h,w]")
    if min(maps.shape[-2:]) < 2:
        raise ValueError("Native attention grid must have at least 2x2 cells")
    if result["classnames"].shape != (len(images),):
        raise ValueError("One class label is required for each image")
    available = []
    for i, method in enumerate(methods):
        values = maps[:, i]
        if np.isnan(values).all():
            continue
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(method + ": maps must be nonnegative/finite or entirely NaN")
        available.append(method)
    if not available:
        raise ValueError("No measured attention maps are available")
    missing = [m for m in METHODS if m not in available]
    if missing and not allow_partial:
        raise ValueError("Incomplete comparison; missing " + ", ".join(missing)
                         + ". Use --allow-partial only for a labelled draft.")
    for key in ("predictions", "target_probabilities", "correct"):
        if key in result and result[key].shape != (len(images), len(methods)):
            raise ValueError(key + " must have shape [N,M]")
    if "target_probabilities" in result:
        p = result["target_probabilities"][:, [methods.index(m) for m in available]]
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("Available target probabilities must be finite and in [0,1]")
    result.update(images=images, maps=maps, methods=methods, metadata=metadata,
                  available=available, missing=missing)
    return result


def probability_map(values):
    values = np.asarray(values, dtype=np.float64)
    total = values.sum()
    return values / total if total > 0 else np.zeros_like(values)


def shared_row_maps(maps, shape):
    """Unit-sum native maps, bilinear resize, then ONE common range per row.

    This is a relative spatial mass display, not absolute attribution magnitude.
    No quantile clipping, power transform or method-dependent transparency.
    """
    upsampled = []
    for values in maps:
        p = probability_map(values).astype(np.float32)
        upsampled.append(np.asarray(Image.fromarray(p).resize(
            (shape[1], shape[0]), Image.Resampling.BILINEAR), dtype=np.float64))
    stack = np.stack(upsampled)
    low, high = float(stack.min()), float(stack.max())
    if high - low <= max(abs(high), 1e-30) * 1e-7:
        return np.zeros_like(stack), (low, high)
    return (stack - low) / (high - low), (low, high)


def overlay(image, heatmap, cmap="jet", alpha=0.48, display_gamma=1.0):
    display = heatmap if display_gamma == 1 else PowerNorm(display_gamma, vmin=0, vmax=1)(heatmap)
    colors = plt.get_cmap(cmap)(display)
    return np.clip((1-alpha)*image + alpha*colors[..., :3], 0, 1)


def spatial_metrics(values):
    p = probability_map(values).reshape(-1)
    if not p.any():
        return {"normalized_entropy": None, "top20_mass": None, "zero_map": True}
    positive = p[p > 0]
    k = max(1, math.ceil(len(p) * 0.2))
    return {"normalized_entropy": float(-(positive*np.log(positive)).sum()/np.log(len(p))),
            "top20_mass": float(np.partition(p, -k)[-k:].sum()), "zero_map": False}


def save_figure(bundle, target, columns, cmap, alpha, dpi, show_scores=True, display_gamma=1.0):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none"})
    n = len(bundle["images"])
    width = 7.0 if len(columns) == 5 else 4.65
    left, right, gap = .055, .055, .045
    cell = (width-left-right-gap*len(columns))/(len(columns)+1)
    header, label_band, score_band, row_gap, footer = .54, .20, .17 if show_scores else .04, .10, .43
    row_height = label_band + cell + score_band + row_gap
    height = header + n*row_height + footer
    fig = plt.figure(figsize=(width, height), facecolor="white")
    title = bundle["metadata"].get("display_name", bundle["metadata"].get("dataset", "Attention comparison"))
    fig.text(left/width, 1-.12/height, title, fontsize=10, fontweight="bold", va="top")
    if bundle["missing"]:
        fig.text(1-right/width, 1-.13/height, "DRAFT · incomplete method comparison",
                 ha="right", va="top", color="#8b2e27", fontsize=6.8)
    for c, method in enumerate(("Input",)+tuple(columns)):
        x = left + c*(cell+gap)
        fig.text((x+cell/2)/width, 1-.43/height, method, ha="center", va="center",
                 fontsize=9, fontweight="bold" if method == "LUNA" else "normal")
    ranges = []
    for row, (im, classname) in enumerate(zip(bundle["images"], bundle["classnames"])):
        top = header + row*row_height
        fig.text(left/width, 1-(top+.03)/height, str(classname).replace("_", " "),
                 fontsize=7.6, va="top")
        active_indices = [bundle["methods"].index(m) for m in bundle["available"]]
        display_maps, scale = shared_row_maps(bundle["maps"][row, active_indices], im.shape[:2])
        ranges.append({"row": row, "minimum": scale[0], "maximum": scale[1]})
        display_by_method = dict(zip(bundle["available"], display_maps))
        y = 1-(top+label_band+cell)/height
        for c, method in enumerate(("Input",)+tuple(columns)):
            x = left+c*(cell+gap)
            ax = fig.add_axes([x/width, y, cell/width, cell/height])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(.85 if method == "LUNA" else .35)
                spine.set_edgecolor("#a6302d" if method == "LUNA" else "#535353")
            if method == "Input":
                ax.imshow(im, interpolation="nearest")
                continue
            if method not in display_by_method:
                ax.set_facecolor("#f5f5f5")
                ax.text(.5, .5, "Weights / maps\nunavailable", transform=ax.transAxes,
                        ha="center", va="center", color="#727272", fontsize=7)
                continue
            ax.imshow(overlay(im, display_by_method[method], cmap, alpha, display_gamma), interpolation="nearest")
            idx = bundle["methods"].index(method)
            raw = bundle["maps"][row, idx]
            if not raw.any() or np.ptp(raw) <= max(float(raw.max()), 1e-30)*1e-7:
                ax.text(.5, .06, "No spatial preference", transform=ax.transAxes, ha="center", fontsize=6,
                        bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 1})
            if show_scores and "target_probabilities" in bundle:
                score = float(bundle["target_probabilities"][row, idx])
                label = "$p(y)$="+f"{score:.3f}"
                if "correct" in bundle:
                    label += "  " + ("correct" if bundle["correct"][row, idx] else "wrong")
                fig.text((x+cell/2)/width, y-.045/height, label, ha="center", va="top", fontsize=6.1)
    color_ax = fig.add_axes([.60, .19/height, .27, .065/height])
    color_norm = Normalize(0,1) if display_gamma == 1 else PowerNorm(display_gamma, vmin=0, vmax=1)
    ticks = [0,1] if display_gamma == 1 else [0,.1,.25,.5,1]
    bar = fig.colorbar(ScalarMappable(norm=color_norm, cmap=cmap), cax=color_ax,
                       orientation="horizontal", ticks=ticks)
    bar.ax.set_xticklabels(["Low", "High"] if display_gamma == 1 else ["0", "0.1", "0.25", "0.5", "1"])
    bar.ax.tick_params(labelsize=6, length=0, pad=1)
    bar.outline.set_linewidth(.3)
    fig.text(.60, .30/height, "Relative spatial relevance", fontsize=6.2)
    note = bundle["metadata"].get("figure_note", "Same target, crop, scale and opacity across methods")
    if display_gamma != 1:
        note = (f"Shared display: power scale (gamma={display_gamma:g})\n"
                "Measured maps unchanged; same settings for all methods")
    fig.text(left/width, .32/height, note, fontsize=5.8, va="top")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(target.with_suffix("."+extension), dpi=dpi, facecolor="white")
    plt.close(fig)
    return ranges


def render_bundle(bundle_path, output_dir, allow_partial=False, cmap="jet", alpha=.48,
                  dpi=600, show_scores=True, display_gamma=1.0):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if cmap not in ("jet", "turbo", "viridis"):
        raise ValueError("Choose jet, turbo or viridis")
    if not np.isfinite(display_gamma) or display_gamma <= 0:
        raise ValueError("display_gamma must be finite and positive")
    bundle = load_bundle(bundle_path, allow_partial)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    stem = "comparison_DRAFT" if bundle["missing"] else "attention_comparison"
    ranges = save_figure(bundle, out/stem, METHODS, cmap, alpha, dpi, show_scores, display_gamma)
    if bundle["missing"]:
        save_figure(bundle, out/"available_methods", bundle["available"], cmap, alpha, dpi, show_scores, display_gamma)
    rows = []
    for i, classname in enumerate(bundle["classnames"]):
        for method in bundle["available"]:
            j = bundle["methods"].index(method)
            record = {"row": i+1, "classname": str(classname), "method": method,
                      **spatial_metrics(bundle["maps"][i,j])}
            for key in ("predictions", "target_probabilities", "correct"):
                if key in bundle:
                    record[key] = bundle[key][i,j].item()
            rows.append(record)
    with (out/"sample_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest = {"status": "incomplete_draft" if bundle["missing"] else "complete",
                "source_bundle": str(Path(bundle_path).resolve()),
                "source_sha256": hashlib.sha256(Path(bundle_path).read_bytes()).hexdigest(),
                "column_order": ["Input", *METHODS], "missing_methods": bundle["missing"],
                "colormap": cmap, "alpha": alpha, "dpi": dpi,
                "display_gamma": display_gamma,
                "color_mapping": "linear" if display_gamma == 1 else "shared monotonic power scale; colorbar uses the same transform",
                "normalization": "native unit sum; bilinear; pooled min-max across methods per image",
                "quantile_clipping": False, "method_specific_style": False,
                "row_scale_ranges": ranges, "inference_metadata": bundle["metadata"]}
    (out/"render_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--cmap", choices=("jet", "turbo", "viridis"), default="jet")
    parser.add_argument("--alpha", type=float, default=.48)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--no-scores", action="store_true")
    parser.add_argument("--display-gamma", type=float, default=1.0,
                        help="Shared positive color-scale exponent for every method; arrays are unchanged")
    args = parser.parse_args()
    result = render_bundle(args.bundle, args.output_dir, args.allow_partial, args.cmap,
                           args.alpha, args.dpi, not args.no_scores, args.display_gamma)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
