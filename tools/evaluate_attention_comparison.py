#!/usr/bin/env python3
"""Evaluate held-out base/new classes and declare qualitative selection rules.

Success examples are explicitly selected qualitative cases. A separate seeded
random control and LUNA failure set accompany them; none estimates accuracy.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys


METHODS = ("CLIP", "CoOp", "KgCoOp", "DeKg", "LUNA")
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def choose_distinct(indices, samples, count):
    chosen, seen = [], set()
    for index in indices:
        label = samples[index]["label"]
        if label in seen:
            continue
        chosen.append(index)
        seen.add(label)
        if len(chosen) == count:
            break
    return chosen


def make_selections(samples, classes, correct, margins, config, output_dir, seed=1, count=3):
    """Select named diagnostic sets without altering or hiding per-image data."""
    import numpy as np

    cutoff = math.ceil(len(classes) / 2)
    novel = [i for i, row in enumerate(samples) if row["label"] >= cutoff]
    luna_index = METHODS.index("LUNA")
    rivals = [i for i, name in enumerate(METHODS) if name != "LUNA"]
    wrong_count = (~correct[:, rivals]).sum(axis=1)
    # Minimum gain over rivals equals gain over the strongest rival margin.
    gain = margins[:, luna_index] - margins[:, rivals].max(axis=1)
    eligible = [i for i in novel if correct[i, luna_index] and wrong_count[i] > 0]
    ranked = sorted(eligible, key=lambda i: (-int(wrong_count[i]), -float(gain[i]), samples[i]["impath"]))
    success = choose_distinct(ranked, samples, count)
    rng = random.Random(seed)
    novel_classes = sorted({samples[i]["label"] for i in novel})
    control_classes = rng.sample(novel_classes, min(count, len(novel_classes)))
    controls = [rng.choice([i for i in novel if samples[i]["label"] == label]) for label in control_classes]
    failures = [i for i in novel if not correct[i, luna_index]]
    random.Random(seed).shuffle(failures)
    failures = choose_distinct(failures, samples, count)
    rules = {
        "selected_success": {
            "purpose": "selected qualitative success cases",
            "rule": "Novel test images; LUNA correct and at least one rival wrong; descending number of wrong rivals, then descending LUNA target-cosine-margin gain over strongest rival; at most one image per class",
            "eligible_image_count": len(eligible),
            "eligible_distinct_classes": len({samples[i]["label"] for i in eligible}),
            "indices": success,
        },
        "random_control": {
            "purpose": "prediction-independent random qualitative control",
            "rule": "Python random.Random(seed): sample novel class labels uniformly without replacement, then sample one test image uniformly within each selected class; no prediction filtering",
            "indices": controls,
        },
        "luna_failures": {
            "purpose": "selected LUNA failure cases",
            "rule": "Novel test images misclassified by LUNA; seeded shuffle, then first distinct classes; no rival correctness filtering",
            "eligible_image_count": sum(not bool(correct[i, luna_index]) for i in novel),
            "indices": failures,
        },
    }
    reports = {}
    figure_notes = {
        "selected_success": "Selected LUNA success cases · same display scale",
        "random_control": "Random class-balanced examples · seed {}".format(seed),
        "luna_failures": "Selected LUNA failure cases · same display scale",
    }
    for name, rule in rules.items():
        chosen = rule.pop("indices")
        report = dict(rule, seed=seed, requested_count=count, actual_count=len(chosen), source_split="test", class_group="novel", shortfall=count-len(chosen))
        rows = []
        for index in chosen:
            row = dict(samples[index])
            row["selection_diagnostics"] = {
                "luna_correct": bool(correct[index, luna_index]),
                "wrong_rivals": int(wrong_count[index]),
                "luna_margin_gain_over_strongest_rival": float(gain[index]),
                "test_index": index,
            }
            rows.append(row)
        selected_config = {
            "dataset": config["dataset"],
            "display_name": config.get("display_name", config.get("title", config["dataset"])),
            "title": config.get("title", config.get("display_name", config["dataset"])),
            "figure_note": figure_notes[name],
            "variant_note": config.get("variant_note", ""),
            "clip_path": config["clip_path"],
            "classnames": classes[cutoff:],
            "methods": config["methods"],
            "samples": rows,
            "sample_selection": report,
            "evaluation_scope": config.get("_evaluation_scope", "full heldout evaluation"),
            "note": "These are {}. Evaluation scope: {}.".format(rule["purpose"], config.get("_evaluation_scope", "full heldout evaluation")),
        }
        target = output_dir / (name + ".json")
        write_json(target, selected_config)
        reports[name] = dict(report, config_path=str(target), selected_images=[row["impath"] for row in rows])
    return reports


def evaluate(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from attention_models import load_attention_models
    from clip.clip import _transform
    from visualize_attention_comparison import read_config

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu")
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8-sig"))
    classes = dataset["classnames"]
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Dataset must provide at least two distinct classes")
    config = read_config(args.models_config)
    if config.get("classnames") and config["classnames"] != classes:
        raise ValueError("models-config classnames must match the full dataset's original label order")
    if config.get("dataset") and config["dataset"] != dataset["dataset"]:
        raise ValueError("Dataset mismatch between manifests")
    config["dataset"] = dataset["dataset"]
    by_name = {spec["name"]: spec for spec in config["methods"]}
    if len(by_name) != len(config["methods"]) or set(by_name) != set(METHODS):
        raise ValueError("Evaluation requires exactly CLIP, CoOp, KgCoOp, DeKg, LUNA trained method specifications")
    config["methods"] = [by_name[name] for name in METHODS]
    samples = dataset["test"]
    if args.max_samples:
        # This mode is deliberately labelled a smoke check in every summary.
        samples = samples[:args.max_samples]
    if not samples:
        raise ValueError("No held-out test images")
    for row in samples:
        if not isinstance(row["label"], int) or not 0 <= row["label"] < len(classes) or classes[row["label"]] != row["classname"]:
            raise ValueError("Test sample class order mismatch: " + repr(row))
    loaded = load_attention_models(classes, dataset["dataset"], config["methods"], config["clip_path"], device=args.device)
    model = loaded.clip_model.float().eval()
    preprocess = _transform(model.visual.input_resolution)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scale = model.logit_scale.detach().float().exp()
    text_features = {name: loaded.text_features[name].detach().float() for name in METHODS}
    cutoff = math.ceil(len(classes)/2)
    labels = np.asarray([row["label"] for row in samples], dtype=np.int64)
    shape = (len(samples), len(METHODS))
    predictions = np.empty(shape, dtype=np.int64)
    probabilities = np.empty(shape, dtype=np.float32)
    cosines_array = np.empty(shape, dtype=np.float32)
    margins = np.empty(shape, dtype=np.float32)
    correct = np.zeros(shape, dtype=np.bool_)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for start in range(0, len(samples), args.batch_size):
            stop = min(start+args.batch_size, len(samples))
            images = []
            for row in samples[start:stop]:
                with Image.open(row["impath"]) as image:
                    images.append(preprocess(image.convert("RGB")))
            tensor = torch.stack(images).to(device=args.device, dtype=torch.float32)
            image_features = F.normalize(model.encode_image(tensor).float(), dim=-1)
            batch_labels = torch.as_tensor(labels[start:stop], device=args.device)
            for method_index, name in enumerate(METHODS):
                all_cosines = image_features @ text_features[name].T
                for low, high in ((0, cutoff), (cutoff, len(classes))):
                    positions = ((batch_labels >= low) & (batch_labels < high)).nonzero(as_tuple=True)[0]
                    if not len(positions):
                        continue
                    scores = all_cosines[positions, low:high]
                    target = batch_labels[positions] - low
                    target_cosine = scores.gather(1, target[:, None]).squeeze(1)
                    predicted = scores.argmax(dim=-1) + low
                    p_target = (scale*scores).softmax(dim=-1).gather(1, target[:, None]).squeeze(1)
                    competitors = scores.clone()
                    competitors.scatter_(1, target[:, None], -torch.inf)
                    margin = target_cosine-competitors.max(dim=-1).values
                    # A one-class group has no wrong-class margin to compare.
                    if high-low < 2:
                        margin = torch.zeros_like(target_cosine)
                    rows = start+positions.cpu().numpy()
                    predictions[rows, method_index] = predicted.cpu().numpy()
                    probabilities[rows, method_index] = p_target.cpu().numpy()
                    cosines_array[rows, method_index] = target_cosine.cpu().numpy()
                    margins[rows, method_index] = margin.cpu().numpy()
                    correct[rows, method_index] = (predicted == batch_labels[positions]).cpu().numpy()
            print("Evaluated {}/{} held-out images".format(stop, len(samples)), flush=True)
    csv_path = args.output_dir / "per_image.csv"
    fields = ("test_index", "impath", "classname", "label", "class_group", "source_split", "method", "predicted_classname", "predicted_label", "correct", "target_probability", "target_cosine", "target_margin")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(samples):
            for method_index, name in enumerate(METHODS):
                writer.writerow({
                    "test_index": index, "impath": row["impath"], "classname": row["classname"], "label": row["label"],
                    "class_group": "base" if row["label"] < cutoff else "novel", "source_split": "test", "method": name,
                    "predicted_classname": classes[predictions[index, method_index]], "predicted_label": int(predictions[index, method_index]),
                    "correct": bool(correct[index, method_index]), "target_probability": float(probabilities[index, method_index]),
                    "target_cosine": float(cosines_array[index, method_index]), "target_margin": float(margins[index, method_index]),
                })
    result = {}
    for method_index, name in enumerate(METHODS):
        summary = {}
        for group, mask in (("base", labels < cutoff), ("novel", labels >= cutoff)):
            count = int(mask.sum())
            hits = int(correct[mask, method_index].sum())
            summary[group] = {"count": count, "correct": hits, "accuracy_percent": 100.0*hits/count if count else None}
        base_acc, new_acc = summary["base"]["accuracy_percent"], summary["novel"]["accuracy_percent"]
        summary["harmonic_mean_percent"] = None if base_acc is None or new_acc is None else (2*base_acc*new_acc/(base_acc+new_acc) if base_acc+new_acc else 0.0)
        result[name] = summary
    config["_evaluation_scope"] = "explicit smoke subset, not full evaluation" if args.max_samples else "full heldout evaluation"
    selection = make_selections(samples, classes, correct, margins, config, args.output_dir / "selections", seed=args.selection_seed)
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset["dataset"],
        "status": "SMOKE_SUBSET_NOT_FULL_EVALUATION" if args.max_samples else "COMPLETE_HELDOUT_EVALUATION",
        "source_split": "test",
        "sample_count": len(samples),
        "full_test_count": len(dataset["test"]),
        "class_order": classes,
        "base_classes": classes[:cutoff],
        "novel_classes": classes[cutoff:],
        "protocol": "Separate base and novel evaluation: candidates restricted to the image's class group; base is first ceil(C/2) original labels",
        "precision": "float32, TF32 disabled; shared frozen CLIP image embedding computed once per image",
        "batch_size": args.batch_size,
        "device": args.device,
        "model_metadata": loaded.metadata,
        "dataset_manifest": {"path": str(args.dataset_manifest.resolve()), "sha256": sha256(args.dataset_manifest)},
        "models_config": {"path": str(args.models_config.resolve()), "sha256": sha256(args.models_config)},
        "source_sha256": sha256(__file__),
        "methods": result,
        "selections": selection,
        "interpretation": "Selected successes illustrate behavior and do not estimate average performance. Random controls are prediction-independent. Softmax pGT is conditional on each class group and is not calibrated across methods. Single-seed checkpoint results do not reproduce multi-seed paper averages.",
    }
    write_json(args.output_dir / "evaluation_summary.json", metadata)
    np.savez_compressed(args.output_dir / "evaluation_arrays.npz", labels=labels, methods=np.asarray(METHODS, dtype=str), classnames=np.asarray(classes, dtype=str), impaths=np.asarray([row["impath"] for row in samples], dtype=str), predictions=predictions, correct=correct, target_probabilities=probabilities, target_cosines=cosines_array, target_margins=margins, metadata=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=str))
    print(json.dumps({"status": metadata["status"], "results": result, "output_dir": str(args.output_dir)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--models-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=1)
    parser.add_argument("--max-samples", type=int, help="Explicit smoke subset; cannot produce a full-evaluation status")
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("Batch size, thread count and optional max samples must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
