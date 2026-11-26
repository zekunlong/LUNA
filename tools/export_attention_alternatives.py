#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only publication alternatives from unchanged measured attribution maps.

Both alternatives use exactly row zero of each dataset's existing ranked
selected_success set. There is no new ranking, image selection, model forward,
gradient computation, smoothing, or method-specific display processing here.
All three datasets and all five methods are mandatory for publication export.

Outputs: paper/alternatives/{gradcam,rollout}/main_comparison.{png,pdf,svg},
main_bundle.npz, caption.tex, provenance.json and byte-identical copies of the
three existing dataset figures. The primary contrastive package is untouched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DATASETS = ("fgvc_aircraft", "eurosat", "stanford_cars")
TITLES = {"fgvc_aircraft": "Aircraft", "eurosat": "EuroSAT", "stanford_cars": "Cars"}
METHODS = ("CLIP", "CoOp", "KgCoOp", "DeKg", "LUNA")
ATTRIBUTIONS = ("gradcam", "rollout")
ARRAY_KEYS = ("images", "maps", "classnames", "predictions", "target_probabilities", "correct", "target_cosines")
NORMALIZATION = "native unit sum; bilinear; pooled min-max across methods per image"
COMMON_SETTING_KEYS = ("attribution", "target_layer", "target", "score", "negative_k", "rollout_last_layers",
                       "rollout_power", "quantile_threshold", "raw_map_normalization", "shared_visual_encoder",
                       "precision", "tf32", "gradcam_formula", "gradcam_reference")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def required_files(root, datasets):
    files = []
    for dataset in datasets:
        files.extend([root / "evaluation" / dataset / "evaluation_summary.json",
                      root / "evaluation" / dataset / "selections/selected_success.json"])
        for attribution in ATTRIBUTIONS:
            directory = root / "figures" / dataset / "selected_success" / attribution
            files.extend(directory / filename for filename in ("attention_bundle.npz", "render_manifest.json",
                         "attention_comparison.png", "attention_comparison.pdf", "attention_comparison.svg"))
    return files


def load_source(root, dataset, attribution):
    """Validate actual all-method source data and its unchanged original selection."""
    import numpy as np
    from tools.attention_figure import load_bundle

    directory = root / "figures" / dataset / "selected_success" / attribution
    bundle_path = directory / "attention_bundle.npz"
    validated = load_bundle(bundle_path, allow_partial=False)
    with np.load(bundle_path, allow_pickle=False) as saved:
        if not set(ARRAY_KEYS) <= set(saved.files):
            raise ValueError(f"{bundle_path}: required measured prediction/relevance arrays missing")
        arrays = {key: np.array(saved[key], copy=True) for key in ARRAY_KEYS}
    metadata = validated["metadata"]
    render_path = directory / "render_manifest.json"
    render = read_json(render_path)
    selection_path = root / "evaluation" / dataset / "selections/selected_success.json"
    selected = read_json(selection_path)
    evaluation_path = root / "evaluation" / dataset / "evaluation_summary.json"
    evaluation = read_json(evaluation_path)
    if validated["methods"] != list(METHODS) or validated["available"] != list(METHODS):
        raise ValueError(f"{dataset}/{attribution}: five measured methods in canonical order required")
    if metadata.get("status") != "READY" or metadata.get("dataset") != dataset:
        raise ValueError(f"{dataset}/{attribution}: source inference is not a completed real comparison")
    if (evaluation.get("status") != "COMPLETE_HELDOUT_EVALUATION" or evaluation.get("dataset") != dataset or
            evaluation.get("sample_count") != evaluation.get("full_test_count")):
        raise ValueError(f"{dataset}: complete held-out evaluation required")
    if set(evaluation.get("methods", {})) != set(METHODS):
        raise ValueError(f"{dataset}: evaluation does not contain all five methods")
    if render.get("status") != "complete" or render.get("column_order") != ["Input", *METHODS] or render.get("missing_methods"):
        raise ValueError(f"{dataset}/{attribution}: source rendered figure is incomplete")
    if Path(render["source_bundle"]).resolve() != bundle_path.resolve() or render.get("source_sha256") != sha256(bundle_path):
        raise ValueError(f"{dataset}/{attribution}: source figure references a different bundle")
    if (render.get("normalization") != NORMALIZATION or render.get("quantile_clipping") is not False or
            render.get("method_specific_style") is not False):
        raise ValueError(f"{dataset}/{attribution}: source style is not the common display protocol")
    if render.get("colormap") != "jet" or not 0 < float(render.get("alpha", 0)) < 1:
        raise ValueError(f"{dataset}/{attribution}: expected the user's Jet reference style")
    settings = metadata["inference_settings"]
    expected_name = "gradcam" if attribution == "gradcam" else "gradient_weighted_attention_rollout"
    if (settings.get("attribution") != expected_name or settings.get("score") != "target" or
            settings.get("target") != "fixed ground-truth class from manifest for every method" or
            settings.get("shared_visual_encoder") is not True or settings.get("precision") != "float32"):
        raise ValueError(f"{dataset}/{attribution}: must use ground-truth target-cosine attribution")
    if not metadata.get("forward_parity", {}).get("passed"):
        raise ValueError(f"{dataset}/{attribution}: source forward parity did not pass")
    if attribution == "gradcam":
        if settings.get("target_layer") != "visual.transformer.resblocks[-1].ln_1":
            raise ValueError("Grad-CAM caption requires final visual block ln_1 patch activations")
    elif settings.get("rollout_last_layers") != 6 or settings.get("rollout_power") != 1.0:
        raise ValueError("Rollout caption requires final six blocks and unchanged rollout power 1")
    if settings.get("quantile_threshold") is not None:
        raise ValueError("Source relevance uses an unexpected quantile threshold")
    source_samples = metadata.get("samples", [])
    selected_samples = selected.get("samples", [])
    # The requested companion figures have three real rows per dataset.
    if len(source_samples) != 3 or len(selected_samples) != 3 or len(arrays["images"]) != 3:
        raise ValueError(f"{dataset}/{attribution}: exactly three genuine selected rows required; never duplicate rows")
    if metadata.get("sample_selection") != selected.get("sample_selection"):
        raise ValueError(f"{dataset}/{attribution}: selection receipt differs from the evaluator")
    selection = metadata["sample_selection"]
    if (selection.get("purpose") != "selected qualitative success cases" or selection.get("source_split") != "test" or
            selection.get("class_group") != "novel" or selection.get("actual_count") != 3):
        raise ValueError(f"{dataset}/{attribution}: source is not the declared selected-success set")
    if metadata.get("config_sha256") != sha256(selection_path):
        raise ValueError(f"{dataset}/{attribution}: selected-success config changed after inference")
    if arrays["correct"].dtype.kind != "b" or arrays["target_cosines"].shape != (3, 5) or not np.isfinite(arrays["target_cosines"]).all():
        raise ValueError(f"{dataset}/{attribution}: invalid measured correctness/cosine arrays")
    for index, (sample, declared) in enumerate(zip(source_samples, selected_samples)):
        for key in ("impath", "classname", "label"):
            if sample.get(key) != declared.get(key):
                raise ValueError(f"{dataset}/{attribution}: sample identity/order changed at row {index}")
        if arrays["classnames"][index] != sample["classname"]:
            raise ValueError("Bundle row class does not match its recorded sample")
        if not sample.get("image_sha256") or sha256(sample["impath"]) != sample["image_sha256"]:
            raise ValueError(f"{dataset}/{attribution}: original image bytes changed at row {index}")
        if not arrays["correct"][index, -1] or arrays["correct"][index, :-1].all():
            raise ValueError("A declared success row does not satisfy LUNA correct + at least one rival wrong")
    model_metadata = metadata["model_metadata"]
    evaluated_models = evaluation["model_metadata"]
    if model_metadata.get("clip_sha256") != evaluated_models.get("clip_sha256"):
        raise ValueError(f"{dataset}/{attribution}: visual CLIP differs from held-out evaluation")
    for method in METHODS[1:]:
        actual_hash = model_metadata["methods"][method].get("checkpoint_sha256")
        if not actual_hash or actual_hash != evaluated_models["methods"][method].get("checkpoint_sha256"):
            raise ValueError(f"{dataset}/{attribution}: checkpoint differs from held-out evaluation: {method}")
    for extension in ("png", "pdf", "svg"):
        target = directory / f"attention_comparison.{extension}"
        if not target.is_file() or target.stat().st_size == 0:
            raise ValueError(f"Missing real dataset export: {target}")
    source = {"dataset": dataset, "attribution": attribution, "bundle": str(bundle_path.resolve()),
              "bundle_sha256": sha256(bundle_path), "source_row_zero_based": 0,
              "selected_sample": source_samples[0], "class_order": metadata["class_order"],
              "sample_selection": selection, "inference_settings": settings,
              "model_metadata": model_metadata,
              "render_manifest": {"path": str(render_path.resolve()), "sha256": sha256(render_path)},
              "selection_config": {"path": str(selection_path.resolve()), "sha256": sha256(selection_path)},
              "evaluation_summary": {"path": str(evaluation_path.resolve()), "sha256": sha256(evaluation_path)}}
    return {"dataset": dataset, "attribution": attribution, "arrays": arrays, "metadata": metadata,
            "render": render, "source": source, "directory": directory}


def verify_same_samples(left, right):
    """The alternatives compare the identical already-selected three inputs."""
    import numpy as np
    if left["dataset"] != right["dataset"]:
        raise ValueError("Cannot pair alternative styles from different datasets")
    for key in ("images", "classnames", "predictions", "correct"):
        if not np.array_equal(left["arrays"][key], right["arrays"][key]):
            raise ValueError(f"{left['dataset']}: {key} differs between Grad-CAM and target rollout")
    for key in ("target_probabilities", "target_cosines"):
        if not np.allclose(left["arrays"][key], right["arrays"][key], rtol=1e-4, atol=1e-5):
            raise ValueError(f"{left['dataset']}: source predictions differ beyond forward-parity tolerance")
    for index in range(3):
        if left["metadata"]["samples"][index].get("image_sha256") != right["metadata"]["samples"][index].get("image_sha256"):
            raise ValueError(f"{left['dataset']}: source image hashes differ between attribution styles")


def short_caption(attribution):
    explanation = ("Grad-CAM uses patch activations at the final visual block's pre-attention layer norm." if attribution == "gradcam"
                   else "Gradient-weighted attention rollout propagates relevance through the final six visual blocks.")
    return ("% This caption is for the target-cosine " + attribution + " alternative only.\n" +
            "\\caption{Class-conditioned attribution for CLIP, CoOp, KgCoOp, DeKg, and LUNA. " +
            explanation + " All methods explain the same ground-truth cosine similarity with identical crops, " +
            "normalization, Jet colormap, and opacity. Red borders identify LUNA. Rows show the first " +
            "already-ranked novel-test success case per dataset: LUNA correct and at least one rival wrong. " +
            "No visual reranking is performed. These selected examples accompany random controls, " +
            "failures, and full-test results.}\n")


def validate_export_group(attribution, loaded):
    if tuple(source["dataset"] for source in loaded) != DATASETS:
        raise ValueError("Publication export requires exactly Aircraft, EuroSAT, Cars in that order")
    if any(source["attribution"] != attribution for source in loaded):
        raise ValueError("Cannot combine different attribution algorithms")
    alphas = {float(source["render"]["alpha"]) for source in loaded}
    if len(alphas) != 1:
        raise ValueError("Source datasets do not share the same overlay opacity")
    shapes = {(source["arrays"]["images"].shape[1:], source["arrays"]["maps"].shape[1:]) for source in loaded}
    if len(shapes) != 1:
        raise ValueError("Source image/patch shapes differ across datasets")
    settings = {key: loaded[0]["metadata"]["inference_settings"][key] for key in COMMON_SETTING_KEYS}
    for source in loaded[1:]:
        if settings != {key: source["metadata"]["inference_settings"][key] for key in COMMON_SETTING_KEYS}:
            raise ValueError("Source datasets do not share the same attribution settings")
    return settings, alphas.pop()


def export_style(root, attribution, loaded, dpi):
    import numpy as np
    from tools.attention_figure import render_bundle

    settings, alpha = validate_export_group(attribution, loaded)
    combined = {}
    for key in ARRAY_KEYS:
        values = []
        for source in loaded:
            value = source["arrays"][key][0]
            if key == "classnames":
                value = TITLES[source["dataset"]] + " · " + str(value)
            values.append(value)
        combined[key] = np.stack(values)
    # Strictly preserve native numerical maps and input pixels; only labels gain
    # a dataset prefix for the combined figure.
    for index, source in enumerate(loaded):
        for key in ("images", "maps", "predictions", "correct", "target_probabilities", "target_cosines"):
            if not np.array_equal(combined[key][index], source["arrays"][key][0]):
                raise ValueError("Unexpected mutation while stacking first source rows")
    name = "Grad-CAM" if attribution == "gradcam" else "Target-cosine gradient-weighted rollout"
    source_records = [source["source"] for source in loaded]
    metadata = {"schema_version": 1, "dataset": "combined", "datasets": list(DATASETS),
                "display_name": "Class-conditioned " + name,
                "figure_note": "Selected LUNA success cases · first ranked case per dataset",
                "status": "READY", "column_order": ["Input", *METHODS],
                "sample_selection": {"purpose": "selected qualitative success cases",
                    "combination_rule": "Exactly source row 0 from each existing selected_success bundle; no reselection or visual reranking",
                    "dataset_order": list(DATASETS), "source_selection_receipts": [source["sample_selection"] for source in source_records]},
                "sources": source_records, "inference_settings": settings,
                "samples": [dict(source["selected_sample"], dataset=source["dataset"]) for source in source_records],
                "class_order_note": "Each row has a dataset-specific novel class space recorded in sources; there is no shared cross-dataset classifier",
                "raw_map_preservation": "Input pixels and native maps copied exactly from measured source row 0",
                "generator_sha256": sha256(__file__)}
    output = root / "paper/alternatives" / attribution
    output.mkdir(parents=True, exist_ok=True)
    bundle_path = output / "main_bundle.npz"
    temporary = output / "main_bundle.tmp.npz"
    np.savez_compressed(temporary, **combined, methods=np.asarray(METHODS, dtype=str),
                        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=str))
    os.replace(temporary, bundle_path)
    rendered = render_bundle(bundle_path, output, allow_partial=False, cmap="jet", alpha=alpha, dpi=dpi)
    if rendered["status"] != "complete":
        raise ValueError("Alternative renderer did not produce a complete figure")
    copied = []
    for extension in ("png", "pdf", "svg"):
        shutil.copyfile(output / f"attention_comparison.{extension}", output / f"main_comparison.{extension}")
    for source in loaded:
        for extension in ("png", "pdf", "svg"):
            origin = source["directory"] / f"attention_comparison.{extension}"
            destination = output / f"{source['dataset']}.{extension}"
            shutil.copyfile(origin, destination)
            original_hash, copied_hash = sha256(origin), sha256(destination)
            if original_hash != copied_hash:
                raise ValueError("Dataset figure changed while copying")
            copied.append({"dataset": source["dataset"], "source": str(origin),
                           "destination": str(destination), "sha256": original_hash})
    (output / "caption.tex").write_text(short_caption(attribution), encoding="utf-8")
    artifact_names = ["main_bundle.npz", "caption.tex", "render_manifest.json", "sample_metrics.csv"] + [
        f"{stem}.{extension}" for stem in ("main_comparison", *DATASETS) for extension in ("png", "pdf", "svg")]
    receipt = {"status": "COMPLETE_REAL_ALTERNATIVE_EXPORT", "created_utc": datetime.now(timezone.utc).isoformat(),
               "attribution": attribution, "score": "target", "datasets": list(DATASETS),
               "source_rows": [0, 0, 0], "column_order": ["Input", *METHODS],
               "row_count": 3, "generator_sha256": sha256(__file__),
               "sources": source_records, "copied_dataset_figures": copied,
               "artifacts": [{"path": str(output / filename), "sha256": sha256(output / filename)} for filename in artifact_names],
               "no_reselection": True, "no_model_or_gradient_execution": True,
               "note": "Alternative target-cosine attribution style; selected examples do not estimate aggregate accuracy."}
    write_json(output / "provenance.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/LUNA/attention_comparison"))
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--check-only", action="store_true", help="Validate real source bundles without creating publication products")
    parser.add_argument("--dataset", choices=DATASETS, help="Only with --check-only: validate one actually available dataset")
    args = parser.parse_args()
    if args.dataset and not args.check_only:
        parser.error("--dataset is validation-only; exported main figures always require all three datasets")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    root = args.root.expanduser().resolve()
    datasets = (args.dataset,) if args.dataset else DATASETS
    missing = [str(path) for path in required_files(root, datasets) if not path.is_file() or path.stat().st_size == 0]
    if missing:
        print(json.dumps({"status": "PENDING_SOURCES", "missing": missing}, indent=2))
        raise SystemExit(2)
    # All source validation happens before creating either publication directory.
    loaded = {}
    for dataset in datasets:
        for attribution in ATTRIBUTIONS:
            loaded[(dataset, attribution)] = load_source(root, dataset, attribution)
        verify_same_samples(loaded[(dataset, "gradcam")], loaded[(dataset, "rollout")])
    if datasets == DATASETS:
        for attribution in ATTRIBUTIONS:
            validate_export_group(attribution, [loaded[(dataset, attribution)] for dataset in DATASETS])
    if args.check_only:
        print(json.dumps({"status": "VALIDATED_REAL_SOURCES", "datasets": list(datasets),
                          "source_rows": {dataset: loaded[(dataset, "gradcam")]["source"]["selected_sample"] for dataset in datasets},
                          "publication_products_written": False}, indent=2))
        return
    for attribution in ATTRIBUTIONS:
        receipt = export_style(root, attribution, [loaded[(dataset, attribution)] for dataset in DATASETS], args.dpi)
        print(json.dumps({"status": receipt["status"], "attribution": attribution,
                          "output_dir": str(root / "paper/alternatives" / attribution)}, indent=2))


if __name__ == "__main__":
    main()
