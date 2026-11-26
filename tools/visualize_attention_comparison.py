#!/usr/bin/env python3
"""Reproducible, fixed-sample CLIP/CoOp/KgCoOp/DeKg/LUNA attribution figures.

The preflight command and --help use only Python's standard library. Inference
loads actual trained text prompts; absent checkpoints are never synthesized.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


METHOD_ORDER = ("CLIP", "CoOp", "KgCoOp", "DeKg", "LUNA")
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value, base):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    config = dict(config)
    config["_config_path"] = str(path)
    config["_config_sha256"] = sha256(path)
    if config.get("clip_path"):
        config["clip_path"] = str(resolve_path(config["clip_path"], path.parent))
    for sample in config.get("samples", []):
        if sample.get("impath"):
            sample["impath"] = str(resolve_path(sample["impath"], path.parent))
    for method in config.get("methods", []):
        for key in ("checkpoint", "text_features"):
            if method.get(key):
                method[key] = str(resolve_path(method[key], path.parent))
    return config


def preflight(config, allow_partial=False):
    """Check declared files and comparison structure without importing torch."""
    errors, missing, available = [], [], []
    classnames = config.get("classnames", [])
    if not isinstance(classnames, list) or not classnames or any(
        not isinstance(name, str) or not name for name in classnames
    ):
        errors.append("classnames must be a non-empty list of class names in model evaluation order")
        classnames = []
    elif len(set(classnames)) != len(classnames):
        errors.append("classnames contains duplicates")
    clip_path = config.get("clip_path")
    if not clip_path or not Path(clip_path).is_file():
        errors.append("Shared CLIP base checkpoint is missing: " + str(clip_path))
    samples = config.get("samples", [])
    if not samples:
        errors.append("samples must contain fixed image paths and target class names")
    for index, sample in enumerate(samples):
        if sample.get("classname") not in classnames:
            errors.append("Sample {} target is absent from classnames: {}".format(index, sample.get("classname")))
        if not sample.get("impath") or not Path(sample["impath"]).is_file():
            errors.append("Sample {} image is missing: {}".format(index, sample.get("impath")))
    specs = {}
    for spec in config.get("methods", []):
        name = spec.get("name")
        if name not in METHOD_ORDER:
            errors.append("Unknown method name {!r}; use {}".format(name, ", ".join(METHOD_ORDER)))
        elif name in specs:
            errors.append("Duplicate method: " + name)
        else:
            specs[name] = spec
    for name in METHOD_ORDER:
        spec = specs.get(name)
        reason = None
        if spec is None:
            reason = "Method is not configured"
        elif name != "CLIP" and not (spec.get("checkpoint") or spec.get("text_features")):
            reason = "Trained prompt checkpoint is not configured"
        elif name != "CLIP" and not Path(spec.get("checkpoint") or spec["text_features"]).is_file():
            reason = "Trained prompt source does not exist: " + str(spec.get("checkpoint") or spec["text_features"])
        if reason:
            missing.append({"name": name, "reason": reason})
        else:
            available.append(name)
    if not available:
        errors.append("No method has all required declared assets")
    if missing and not allow_partial:
        errors.append("Full comparison needs every method checkpoint; --allow-partial explicitly permits a labeled draft")
    return {
        "status": "READY_PARTIAL" if missing and not errors else ("BLOCKED" if errors else "READY"),
        "config_path": config["_config_path"],
        "dataset": config.get("dataset", ""),
        "sample_count": len(samples),
        "class_count": len(classnames),
        "available_methods": available,
        "missing_methods": missing,
        "errors": errors,
        "validation_scope": "File presence and manifest structure; tensor compatibility is checked during inference",
    }


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight", help="Check assets without importing torch")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--allow-partial", action="store_true")
    check.add_argument("--output", type=Path, help="Optional JSON report path")
    infer = commands.add_parser("infer", help="Compute paired ground-truth class-conditioned attribution")
    infer.add_argument("--config", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--allow-partial", action="store_true")
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--cpu-threads", type=int, default=4)
    infer.add_argument("--attribution", choices=("rollout", "gradcam"), default="rollout")
    infer.add_argument("--score", choices=("target", "target-margin"), default="target")
    infer.add_argument("--negative-k", type=int, default=5)
    infer.add_argument("--layers", type=int, default=6)
    infer.add_argument("--cmap", default="jet")
    infer.add_argument("--alpha", type=float, default=0.48)
    infer.add_argument("--no-render", action="store_true")
    render = commands.add_parser("render", help="Render a saved raw attribution bundle without inference")
    render.add_argument("--bundle", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--allow-partial", action="store_true")
    render.add_argument("--cmap", default="jet")
    render.add_argument("--alpha", type=float, default=0.48)
    return root


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class LayerActivationCapture:
    """Observe a layer output without changing the model's forward function."""

    def __init__(self, layer):
        self.layer = layer
        self.activation = None
        self.handle = None

    def __enter__(self):
        def capture(module, inputs, output):
            self.activation = output
        self.handle = self.layer.register_forward_hook(capture)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.remove()


def gradcam_relevance(activation, gradient, patch_count):
    """ViT Grad-CAM: spatially averaged channel gradients, excluding CLS."""
    if activation.ndim != 3 or activation.shape[0] != patch_count + 1:
        raise ValueError("Grad-CAM requires unprompted [CLS + spatial patches, batch, channels]")
    patches = activation[1:].permute(1, 0, 2).float()
    spatial_gradients = gradient[1:].permute(1, 0, 2).float()
    channel_weights = spatial_gradients.mean(dim=1, keepdim=True)
    relevance = (patches * channel_weights).sum(dim=-1).clamp_min(0.0)
    return relevance / relevance.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def run_inference(config, args):
    # Heavy imports belong exclusively to inference, keeping asset checks usable
    # on machines that do not yet have the ML environment installed.
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from attention_models import load_attention_models
    from clip.clip import _transform
    from visualize_luna_attention import (
        VisualAttentionCapture,
        encode_image_for_rollout,
        gradient_attention_rollout,
    )

    if args.layers < 1 or args.negative_k < 1 or args.cpu_threads < 1:
        raise ValueError("--layers, --negative-k and --cpu-threads must be positive")
    if args.score == "target-margin" and len(config["classnames"]) < 2:
        raise ValueError("target-margin requires at least two classes")
    report = preflight(config, args.allow_partial)
    if report["errors"]:
        raise ValueError("; ".join(report["errors"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "preflight.json", report)
    torch.manual_seed(0)
    torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    method_specs = [
        next(spec for spec in config["methods"] if spec["name"] == name)
        for name in report["available_methods"]
    ]
    loaded = load_attention_models(config["classnames"], config.get("dataset", ""), method_specs, config["clip_path"], device=str(device))
    clip_model = loaded.clip_model.float().eval()
    preprocess = _transform(clip_model.visual.input_resolution)
    text_by_method = loaded.text_features
    visual = clip_model.visual
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)
    if args.layers > len(visual.transformer.resblocks):
        raise ValueError("Requested rollout layer count exceeds the shared encoder's depth")
    methods = report["available_methods"]
    classnames = config["classnames"]
    features = {
        name: F.normalize(text_by_method[name].detach().to(device=device, dtype=torch.float32), dim=-1)
        for name in methods
    }
    for name, value in features.items():
        if value.shape[0] != len(classnames) or not torch.isfinite(value).all():
            raise ValueError("Invalid class embeddings for " + name)
    logit_scale = clip_model.logit_scale.detach().float().exp()
    arrays = {key: [] for key in ("images", "maps", "predictions", "target_probabilities", "correct", "class_probabilities", "target_cosines")}
    parity = []
    sample_metadata = []
    mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    for index, sample in enumerate(config["samples"]):
        print("Attributing {}/{}: {}".format(index + 1, len(config["samples"]), sample["classname"]), flush=True)
        with Image.open(sample["impath"]) as image:
            tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device=device, dtype=torch.float32)
        displayed = tensor[0].detach().cpu().permute(1, 2, 0).numpy() * std + mean
        arrays["images"].append(np.clip(displayed, 0, 1).astype(np.float32))
        target = classnames.index(sample["classname"])
        with torch.no_grad():
            reference = clip_model.encode_image(tensor).float()
        tensor = tensor.detach().requires_grad_(True)
        row_maps, row_predictions, row_probabilities, row_correct, row_full_probabilities, row_cosines = [], [], [], [], [], []
        capture_context = VisualAttentionCapture(visual) if args.attribution == "rollout" else LayerActivationCapture(visual.transformer.resblocks[-1].ln_1)
        with capture_context as capture:
            if args.attribution == "rollout":
                embedding, grid_shape = encode_image_for_rollout(visual, tensor)
                gradient_inputs = capture.attentions
            else:
                embedding = clip_model.encode_image(tensor).float()
                patch_size = visual.conv1.kernel_size
                grid_shape = (tensor.shape[-2] // patch_size[0], tensor.shape[-1] // patch_size[1])
                if capture.activation is None:
                    raise RuntimeError("Last-block ln_1 hook did not observe an activation")
                gradient_inputs = (capture.activation,)
            torch.testing.assert_close(embedding.detach(), reference, rtol=1e-4, atol=1e-5)
            max_error = float((embedding.detach() - reference).abs().max().cpu())
            parity.append(max_error)
            embedding = F.normalize(embedding.float(), dim=-1)
            for method_index, name in enumerate(methods):
                cosines = embedding @ features[name].T
                probabilities = (logit_scale * cosines).softmax(dim=-1)
                target_score = cosines[:, target]
                if args.score == "target-margin":
                    negatives = cosines.clone()
                    negatives[:, target] = -torch.inf
                    target_score = target_score - negatives.topk(min(args.negative_k, len(classnames) - 1), dim=-1).values.mean(dim=-1)
                gradients = torch.autograd.grad(target_score.sum(), gradient_inputs, retain_graph=method_index < len(methods) - 1)
                if args.attribution == "rollout":
                    relevance = gradient_attention_rollout(capture.attentions, gradients, grid_shape[0] * grid_shape[1], args.layers, 1.0)
                else:
                    relevance = gradcam_relevance(capture.activation, gradients[0], grid_shape[0] * grid_shape[1])
                row_maps.append(relevance.detach().cpu().reshape(*grid_shape).numpy().astype(np.float32))
                predicted = int(probabilities.detach().argmax(dim=-1).cpu())
                row_predictions.append(classnames[predicted])
                row_probabilities.append(float(probabilities.detach()[0, target].cpu()))
                row_correct.append(predicted == target)
                row_full_probabilities.append(probabilities.detach()[0].cpu().numpy().astype(np.float32))
                row_cosines.append(float(cosines.detach()[0, target].cpu()))
        arrays["maps"].append(np.stack(row_maps))
        arrays["predictions"].append(row_predictions)
        arrays["target_probabilities"].append(row_probabilities)
        arrays["correct"].append(row_correct)
        arrays["class_probabilities"].append(np.stack(row_full_probabilities))
        arrays["target_cosines"].append(row_cosines)
        sample_metadata.append(dict(sample, image_sha256=sha256(sample["impath"]), target_index=target))
        del embedding, reference, gradients, relevance, capture, capture_context, gradient_inputs, tensor, cosines, probabilities, target_score
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": config.get("dataset", ""),
        "title": config.get("title", config.get("dataset", "")),
        "display_name": config.get("title", config.get("display_name", config.get("dataset", ""))),
        "figure_note": config.get("figure_note", ""),
        "note": config.get("note", ""),
        "variant_note": config.get("variant_note", ""),
        "status": report["status"],
        "required_methods": list(METHOD_ORDER),
        "available_methods": methods,
        "missing_methods": report["missing_methods"],
        "class_order": classnames,
        "samples": sample_metadata,
        "model_metadata": loaded.metadata,
        "method_specs": method_specs,
        "checkpoint_hashes": {name: sha256(spec["checkpoint"]) for name, spec in zip(methods, method_specs) if spec.get("checkpoint")},
        "clip_path": config["clip_path"],
        "clip_sha256": sha256(config["clip_path"]),
        "config_path": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "inference_settings": {
            "attribution": "gradient_weighted_attention_rollout" if args.attribution == "rollout" else "gradcam",
            "target_layer": "visual.transformer.resblocks[-1].ln_1" if args.attribution == "gradcam" else "last visual self-attention blocks",
            "gradcam_formula": "ReLU(sum_channels(activation * mean_spatial_gradient)); CLS excluded from spatial gradients and activation map" if args.attribution == "gradcam" else None,
            "gradcam_reference": "https://github.com/jacobgil/pytorch-grad-cam/blob/master/usage_examples/clip_example.py" if args.attribution == "gradcam" else None,
            "target": "fixed ground-truth class from manifest for every method",
            "score": args.score,
            "negative_k": args.negative_k if args.score == "target-margin" else None,
            "rollout_last_layers": args.layers if args.attribution == "rollout" else None,
            "rollout_power": 1.0 if args.attribution == "rollout" else None,
            "quantile_threshold": None,
            "raw_map_normalization": "nonnegative patch relevance divided by per-map sum; zero maps remain zero",
            "shared_visual_encoder": True,
            "precision": "float32",
            "tf32": False,
            "device": str(device),
            "logit_scale": float(logit_scale.cpu()),
            "probability_interpretation": "softmax over declared class_order; not calibrated confidence",
            "torch_version": torch.__version__,
        },
        "forward_parity": {"rtol": 1e-4, "atol": 1e-5, "max_absolute_error_per_image": parity, "passed": True},
        "source_sha256": {str(path.relative_to(REPO_ROOT)): sha256(path) for path in [Path(__file__).resolve(), REPO_ROOT / "tools/attention_models.py", REPO_ROOT / "tools/visualize_luna_attention.py"]},
        "sample_selection": config.get("sample_selection", "Fixed explicit manifest; no search by method performance in this command"),
    }
    bundle = args.output_dir / "attention_bundle.npz"
    np.savez_compressed(
        bundle,
        images=np.stack(arrays["images"]),
        maps=np.stack(arrays["maps"]),
        methods=np.asarray(methods, dtype=str),
        classnames=np.asarray([sample["classname"] for sample in config["samples"]], dtype=str),
        predictions=np.asarray(arrays["predictions"], dtype=str),
        target_probabilities=np.asarray(arrays["target_probabilities"], dtype=np.float32),
        correct=np.asarray(arrays["correct"], dtype=np.bool_),
        class_probabilities=np.stack(arrays["class_probabilities"]),
        target_cosines=np.asarray(arrays["target_cosines"], dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=str),
    )
    write_json(args.output_dir / "inference_metadata.json", metadata)
    print("Saved raw attribution bundle: " + str(bundle), flush=True)
    if not args.no_render:
        from attention_figure import render_bundle
        render_bundle(bundle, args.output_dir, allow_partial=args.allow_partial, cmap=args.cmap, alpha=args.alpha)
    return bundle


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "render":
            from attention_figure import render_bundle
            render_bundle(args.bundle, args.output_dir, allow_partial=args.allow_partial, cmap=args.cmap, alpha=args.alpha)
            return 0
        config = read_config(args.config)
        report = preflight(config, args.allow_partial)
        if args.command == "preflight":
            print(json.dumps(report, indent=2, ensure_ascii=False))
            if args.output:
                write_json(args.output, report)
            return 2 if report["errors"] else 0
        if report["errors"]:
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 2
        run_inference(config, args)
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
