#!/usr/bin/env python3
"""Prediction-independent localization audit with paired class-bootstrap intervals.

Preparation reads original novel test splits and fixes images before inference.
The audit uses native nonnegative relevance and official annotation rectangles;
display normalization and selected qualitative success cases are not involved.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys


METHODS = ("CLIP", "CoOp", "KgCoOp", "DeKg", "LUNA")
METRICS = ("bbox_mass", "background_mass", "bbox_enrichment", "pointing_inside_bbox")
SAMPLES_PER_CLASS = {"fgvc_aircraft": 2, "stanford_cars": 1}
NAMES = {"fgvc_aircraft": "FGVC Aircraft", "stanford_cars": "Stanford Cars"}
TOOLS = Path(__file__).resolve().parent
COMPARISON_ROOT = Path("E:/LUNA/attention_comparison")
SEED = 1
BOOTSTRAP_RESAMPLES = 2000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def annotation_files(dataset, data_root):
    if dataset == "fgvc_aircraft":
        return [data_root/dataset/"images_box.txt"]
    return [data_root/dataset/"devkit/cars_train_annos.mat", data_root/dataset/"cars_test_annos_withlabels.mat"]


def prepare(dataset, comparison_root, output_root, data_root):
    """Freeze class-balanced samples without reading predictions/checkpoints."""
    from visualize_attention_comparison import read_config

    dataset_path = comparison_root/"datasets"/(dataset+".json")
    models_path = comparison_root/"configs"/(dataset+"_models.json")
    manifest = read_json(dataset_path)
    models = read_config(models_path)
    classes = manifest["classnames"]
    if models.get("classnames") != classes or manifest.get("dataset") != dataset:
        raise ValueError("Dataset and model configs must have identical complete original class order")
    if tuple(spec["name"] for spec in models["methods"]) != METHODS:
        raise ValueError("Audit requires all five canonical method specs in comparison order")
    cutoff = (len(classes)+1)//2
    by_class = defaultdict(list)
    for sample in manifest["test"]:
        label = sample["label"]
        if classes[label] != sample["classname"]:
            raise ValueError("Class label/name mismatch in original test manifest")
        if label >= cutoff:
            by_class[label].append(sample)
    if sorted(by_class) != list(range(cutoff, len(classes))):
        raise ValueError("Every novel class must be represented in the source test split")
    rng = random.Random(SEED)
    samples, image_hashes = [], {}
    for label in range(cutoff, len(classes)):
        k = SAMPLES_PER_CLASS[dataset]
        if len(by_class[label]) < k:
            raise ValueError("Insufficient test images for original class label {}".format(label))
        for sample in rng.sample(by_class[label], k):
            row = dict(sample, source_split="test", class_group="novel", selection="predeclared class-balanced random localization audit")
            path = Path(row["impath"])
            if not path.is_file():
                raise FileNotFoundError(path)
            image_hashes[str(path.resolve())] = sha256(path)
            samples.append(row)
    selection = {
        "purpose": "prediction-independent class-balanced localization audit",
        "prediction_independent": True,
        "source_split": "test",
        "class_group": "novel",
        "seed": SEED,
        "n_classes": len(by_class),
        "n_images": len(samples),
        "images_per_class": SAMPLES_PER_CLASS[dataset],
        "rule": "Visit novel classes in original integer-label order; random.Random(1).sample the prescribed number of images from each original test list; no prediction/correctness/relevance filtering",
        "selected_before_inference": True,
    }
    config = {
        "dataset": dataset,
        "title": models.get("title", NAMES[dataset]),
        "classnames": classes[cutoff:],
        "clip_path": models["clip_path"],
        "methods": models["methods"],
        "samples": samples,
        "sample_selection": selection,
        "figure_note": "Preselected novel-test localization audit · seed 1",
        "variant_note": models.get("variant_note", ""),
    }
    root = output_root/dataset
    root.mkdir(parents=True, exist_ok=True)
    config_path = root/"config.json"
    source_paths = annotation_files(dataset, data_root)
    receipt = {
        "status": "PREPARED_PREDICTION_INDEPENDENT_SELECTION",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "selection": selection,
        "dataset_manifest": {"path": str(dataset_path.resolve()), "sha256": sha256(dataset_path)},
        "models_config_at_preparation": {"path": str(models_path.resolve()), "sha256": sha256(models_path)},
        "annotation_sources": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in source_paths],
        "sample_image_sha256": image_hashes,
        "preparation_source_sha256": sha256(__file__),
        "protocol_note": "Only manifests, source annotation files and selected images were read to prepare this audit. Checkpoints and predictions are not read by --prepare-only.",
    }
    if config_path.is_file():
        existing = read_json(config_path)
        if existing != config:
            raise FileExistsError("Predeclared audit config differs from proposed config; use a separate output root: " + str(config_path))
        previous = read_json(root/"selection_receipt.json")
        if previous["sample_image_sha256"] != image_hashes:
            raise ValueError("Predeclared audit images changed")
        if previous["dataset_manifest"]["sha256"] != receipt["dataset_manifest"]["sha256"] or previous["annotation_sources"] != receipt["annotation_sources"]:
            raise ValueError("Predeclared dataset/annotation sources changed")
    else:
        write_json(config_path, config)
        receipt["config_sha256"] = sha256(config_path)
        write_json(root/"selection_receipt.json", receipt)
    print(json.dumps({"dataset": dataset, "status": "PREPARED", "config": str(config_path), **selection}), flush=True)
    return config_path


def paired_cluster_bootstrap(luna, rival, classes, resamples=BOOTSTRAP_RESAMPLES, seed=SEED):
    """Equal-class paired mean; resample eligible classes as intact clusters."""
    import numpy as np

    left, right = np.asarray(luna, dtype=np.float64), np.asarray(rival, dtype=np.float64)
    labels = np.asarray(classes)
    if left.shape != right.shape or left.shape != labels.shape or left.ndim != 1:
        raise ValueError("Paired values and class labels must have the same one-dimensional shape")
    valid = np.isfinite(left) & np.isfinite(right)
    labels, left, right = labels[valid], left[valid], right[valid]
    unique = np.unique(labels)
    if not len(unique):
        return {"estimate": None, "ci95": None, "valid_images": 0, "valid_classes": 0, "valid_images_by_class": {}, "paired_luna_mean": None, "paired_rival_mean": None, "interval_status": "NO_VALID_PAIRS"}
    luna_class_means = np.asarray([left[labels == key].mean() for key in unique])
    rival_class_means = np.asarray([right[labels == key].mean() for key in unique])
    differences = luna_class_means-rival_class_means
    report = {
        "estimate": float(differences.mean()), "valid_images": int(valid.sum()), "valid_classes": len(unique),
        "valid_images_by_class": {str(key): int((labels == key).sum()) for key in unique},
        "paired_luna_mean": float(luna_class_means.mean()), "paired_rival_mean": float(rival_class_means.mean()),
    }
    if len(unique) < 2:
        return dict(report, ci95=None, interval_status="INSUFFICIENT_CLASS_CLUSTERS")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(resamples, len(unique)))
    distribution = differences[draws].mean(axis=1)
    report.update(ci95=np.quantile(distribution, [.025, .975]).tolist(), interval_status="DEGENERATE_IDENTICAL_CLASS_DIFFERENCES" if np.ptp(differences) == 0 else "ESTIMATED")
    return report


def checked_box_metrics(raw, box, resolution):
    """Keep valid mass metrics; do not assign arbitrary pointing scores to ties."""
    import numpy as np
    from attention_localization import box_metrics

    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2:
        return dict({metric: None for metric in METRICS}, bbox_area_fraction=None,
                    invalid_reason="invalid_map_shape", pointing_status="undefined_invalid_map",
                    validity="INVALID_RELEVANCE", pointing_tied=False)
    values = box_metrics(raw, box, resolution)
    invalid = values.get("invalid_reason")
    if invalid:
        state = {"zero_map": "ZERO_RELEVANCE", "empty_cropped_bbox": "BBOX_OUTSIDE_CROP"}.get(invalid, "INVALID_RELEVANCE")
        return dict(values, validity=state, pointing_tied=False)
    ties = np.count_nonzero(raw == raw.max()) > 1
    if ties:
        values["pointing_inside_bbox"] = None
        values["pointing_status"] = "undefined_tied_maxima"
    return dict(values, validity="VALID_MASS_METRICS", pointing_tied=bool(ties))


def summarize(dataset, config_path, bundle_path, data_root, output_dir):
    import numpy as np
    from PIL import Image
    from attention_localization import analyze, annotations, crop_box

    config = read_json(config_path)
    receipt = read_json(config_path.parent/"selection_receipt.json")
    if sha256(config_path) != receipt["config_sha256"]:
        raise ValueError("Prepared config changed after its selection receipt")
    with np.load(bundle_path, allow_pickle=False) as archive:
        methods = archive["methods"].tolist()
        maps = archive["maps"].copy()
        metadata = json.loads(archive["metadata"].item())
        resolution = int(archive["images"].shape[1])
        classnames = archive["classnames"].tolist()
    if methods != list(METHODS) or maps.shape[:2] != (len(config["samples"]), len(METHODS)):
        raise ValueError("Audit bundle must contain the same selected samples and all five methods")
    if metadata.get("config_sha256") != receipt["config_sha256"]:
        raise ValueError("Attribution bundle was computed from a different selection/config")
    settings = metadata["inference_settings"]
    if settings.get("attribution") != "gradient_weighted_attention_rollout" or settings.get("score") != "target-margin" or settings.get("negative_k") != 5 or settings.get("rollout_last_layers") != 6 or settings.get("rollout_power") != 1.0 or settings.get("quantile_threshold") is not None:
        raise ValueError("Attribution settings differ from the predeclared native rollout audit protocol")
    observed_samples = metadata["samples"]
    if len(observed_samples) != len(config["samples"]) or len(classnames) != len(config["samples"]):
        raise ValueError("Attribution metadata does not contain every selected sample")
    for prepared, measured, classname in zip(config["samples"], observed_samples, classnames):
        if Path(prepared["impath"]).resolve() != Path(measured["impath"]).resolve() or prepared["classname"] != classname:
            raise ValueError("Measured sample order differs from the prepared audit")
        if measured.get("image_sha256") != receipt["sample_image_sha256"][str(Path(prepared["impath"]).resolve())]:
            raise ValueError("Measured image bytes differ from predeclared image bytes")
    # Keep the standard native localization output, then add the stricter audit
    # validity/pointing policy and class-balanced paired uncertainty below.
    analyze(bundle_path, data_root, output_dir/"native_localization")
    boxes, box_source = annotations(dataset, data_root)
    rows = []
    labels = np.asarray([sample["label"] for sample in config["samples"]], dtype=np.int64)
    values = {metric: np.full(maps.shape[:2], np.nan, dtype=np.float64) for metric in METRICS}
    for index, sample in enumerate(config["samples"]):
        path = Path(sample["impath"])
        key = path.stem if dataset == "fgvc_aircraft" else path.parent.name+"/"+path.name
        if key not in boxes:
            raise ValueError("Official annotation is missing for " + str(path))
        with Image.open(path) as image:
            box = crop_box(boxes[key], image.size, resolution)
        for column, method in enumerate(METHODS):
            result = checked_box_metrics(maps[index, column], box, resolution)
            for metric in METRICS:
                if result[metric] is not None:
                    values[metric][index, column] = float(result[metric])
            rows.append({"sample_index": index, "impath": str(path), "label": sample["label"], "classname": sample["classname"], "source_split": "test", "method": method, **result})
    method_results = {}
    names = {"bbox_mass": "mean_bbox_mass", "background_mass": "mean_background_mass", "bbox_enrichment": "mean_bbox_enrichment", "pointing_inside_bbox": "pointing_accuracy"}
    for column, method in enumerate(METHODS):
        result = {"valid_counts": {}, "valid_class_counts": {}, "valid_counts_by_class": {}, "image_weighted_means": {}}
        for metric in METRICS:
            vector = values[metric][:, column]
            valid = np.isfinite(vector)
            valid_classes = np.unique(labels[valid])
            class_means = [float(vector[(labels == key) & valid].mean()) for key in valid_classes]
            result[names[metric]] = float(np.mean(class_means)) if class_means else None
            result["image_weighted_means"][metric] = float(vector[valid].mean()) if valid.any() else None
            result["valid_counts"][metric] = int(valid.sum())
            result["valid_class_counts"][metric] = len(valid_classes)
            result["valid_counts_by_class"][metric] = {str(key): int(((labels == key) & valid).sum()) for key in valid_classes}
        own_rows = [row for row in rows if row["method"] == method]
        result["validity_counts"] = {state: sum(row["validity"] == state for row in own_rows) for state in ("VALID_MASS_METRICS", "INVALID_RELEVANCE", "ZERO_RELEVANCE", "BBOX_OUTSIDE_CROP")}
        result["tied_pointing_maps"] = sum(row["pointing_tied"] for row in own_rows)
        method_results[method] = result
    paired = {}
    luna_index = METHODS.index("LUNA")
    for column, method in enumerate(METHODS):
        if method == "LUNA":
            continue
        paired[method] = {metric: paired_cluster_bootstrap(values[metric][:, luna_index], values[metric][:, column], labels) for metric in METRICS}
        # Complementary mass has the same support and exactly opposite paired
        # effect/interval; it is not a second independent statistical result.
        mass, background = paired[method]["bbox_mass"], paired[method]["background_mass"]
        if mass["valid_images_by_class"] != background["valid_images_by_class"]:
            raise ValueError("Foreground/background paired supports unexpectedly differ")
        if mass["estimate"] is not None:
            background["estimate"] = -mass["estimate"]
            background["paired_luna_mean"] = 1-mass["paired_luna_mean"]
            background["paired_rival_mean"] = 1-mass["paired_rival_mean"]
        if mass["ci95"] is not None:
            background["ci95"] = [-mass["ci95"][1], -mass["ci95"][0]]
        background["derived_as_bbox_mass_complement"] = True
    report = {
        "status": "COMPLETE_PREDICTION_INDEPENDENT_LOCALIZATION_AUDIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "attribution": "gradient_weighted_attention_rollout",
        "score": "target-margin",
        "negative_k": 5,
        "rollout_last_layers": 6,
        "selection": config["sample_selection"],
        "methods": method_results,
        "paired_luna_minus_rival": paired,
        "bootstrap": {
            "unit": "class", "seed": SEED, "resamples": BOOTSTRAP_RESAMPLES, "confidence": .95,
            "estimator": "mean of within-class paired-image mean differences",
            "pairing": "For each rival/metric, retain images valid for both methods; average within class; resample eligible class means as intact paired clusters",
            "uncertainty": "Descriptive across-class variation conditional on these sampled images and one fixed trained checkpoint per method; not training-seed uncertainty or exhaustive fixed-dataset performance uncertainty",
            "multiple_comparisons": "Intervals are descriptive and unadjusted; no p-values or blanket significance claim",
        },
        "aggregation": "Method means are equal-weight means of within-class valid-image means. Paired contrasts use their own common valid support; paired-support method means are also provided.",
        "validity_policy": "Nonfinite/negative/zero relevance or fully cropped-out boxes are excluded with counts. Positive uniform maps retain mass/enrichment; multiple exactly equal maxima have no pointing result, matching the shared localization function. No display threshold, gamma, min-max scaling or hand-drawn ROI is used.",
        "protocol_note": "All novel classes are included; images were selected by class-balanced Random(1) sampling without predictions. Official bounding boxes are coarse foreground proxies and include background. Cars has one sampled image per class, so within-class image variation is not estimated. Background mass is exactly 1 minus bbox mass and supplies no independent evidence.",
        "annotation_source": box_source,
        "annotation_files": receipt["annotation_sources"],
        "source_hashes": {str(path): sha256(path) for path in (Path(__file__).resolve(), TOOLS/"attention_localization.py", config_path, bundle_path)},
        "selection_receipt": str((config_path.parent/"selection_receipt.json").resolve()),
        "model_metadata": metadata.get("model_metadata", {}),
        "checkpoint_hashes": dict(metadata.get("checkpoint_hashes", {}), CLIP=metadata["clip_sha256"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/"audit_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(output_dir/"summary.json", report)
    print(json.dumps({"dataset": dataset, "status": report["status"], "summary": str(output_dir/"summary.json")}), flush=True)
    return report


def run(dataset, config_path, args):
    root = config_path.parent
    output = root/"rollout"
    command = [sys.executable, str(TOOLS/"visualize_attention_comparison.py"), "infer", "--config", str(config_path), "--output-dir", str(output), "--attribution", "rollout", "--score", "target-margin", "--negative-k", "5", "--layers", "6", "--device", args.device, "--no-render"]
    write_json(root/"execution_command.json", {"command": command, "started_utc": datetime.now(timezone.utc).isoformat()})
    subprocess.run(command, cwd=str(TOOLS.parent), check=True)
    return summarize(dataset, config_path, output/"attention_bundle.npz", args.data_root, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(SAMPLES_PER_CLASS), default=list(SAMPLES_PER_CLASS))
    parser.add_argument("--comparison-root", type=Path, default=COMPARISON_ROOT)
    parser.add_argument("--output-root", type=Path, default=COMPARISON_ROOT/"localization_audit")
    parser.add_argument("--data-root", type=Path, default=Path("E:/LUNA/data"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze image selection without loading a checkpoint or importing torch")
    args = parser.parse_args()
    for dataset in args.datasets:
        config = prepare(dataset, args.comparison_root, args.output_root, args.data_root)
        if not args.prepare_only:
            run(dataset, config, args)


if __name__ == "__main__":
    main()
