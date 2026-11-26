#!/usr/bin/env python3

"""Create low-noise, text-conditioned CLIP/LUNA attention comparisons.

The CLIP and LUNA panels share the same frozen image features; only their text
prompts differ.  The target-class similarity minus the image's closest
wrong-class similarities seeds a gradient-weighted attention rollout through
the visual Transformer.  This suppresses generic responses while preserving
the image-text interpretation used by CLIP prompt-learning methods.
"""

import argparse
import csv
import math
import os
import random
import re
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DASSL_ROOT = REPO_ROOT / "Dassl.pytorch"
if str(DASSL_ROOT) not in sys.path:
    sys.path.insert(0, str(DASSL_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dassl.config import get_cfg_default
from dassl.data.datasets import build_dataset
from PIL import Image

from clip import clip
from train import extend_cfg
from trainers.luna import CustomCLIP, load_clip_to_cpu


CFG_NAME = "vit_b16_c2_ep20_batch8_4+4ctx"
DATASET_CONFIGS = {
    "caltech101": "caltech101.yaml",
    "dtd": "dtd.yaml",
    "eurosat": "eurosat.yaml",
    "fgvc_aircraft": "fgvc_aircraft.yaml",
    "food101": "food101.yaml",
    "oxford_flowers": "oxford_flowers.yaml",
    "oxford_pets": "oxford_pets.yaml",
    "stanford_cars": "stanford_cars.yaml",
    "sun397": "sun397.yaml",
    "ucf101": "ucf101.yaml",
}
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
MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare original CLIP and trained LUNA text-conditioned attention."
    )
    parser.add_argument("dataset", choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--overlay-alpha", type=float, default=0.50)
    parser.add_argument("--hard-negative-k", type=int, default=5)
    parser.add_argument("--visible-quantile", type=float, default=0.65)
    parser.add_argument("--rollout-layers", type=int, default=6)
    parser.add_argument("--rollout-power", type=float, default=1.5)
    parser.add_argument("--output-root", default="luna_attention_gradrollout")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ["LUNA_DATA_ROOT"]).expanduser()
        if os.environ.get("LUNA_DATA_ROOT")
        else None,
        help="Dataset root (defaults to the LUNA_DATA_ROOT environment variable).",
    )
    parser.add_argument("--workdir", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    if args.root is None:
        parser.error("--root is required unless LUNA_DATA_ROOT is set")
    return args


def build_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(str(args.workdir / "configs/datasets" / DATASET_CONFIGS[args.dataset]))
    cfg.merge_from_file(
        str(args.workdir / "configs/trainers/LUNA" / f"{CFG_NAME}.yaml")
    )
    cfg.DATASET.ROOT = str(args.root)
    cfg.DATASET.NUM_SHOTS = 0
    cfg.SEED = args.seed
    cfg.TRAINER.NAME = "LUNA"
    cfg.TRAINER.MODAL = "base2novel"
    cfg.freeze()
    return cfg


def checkpoint_path(args):
    return (
        args.workdir
        / "output"
        / "base2new"
        / "train_base"
        / args.dataset
        / "shots_0"
        / "LUNA"
        / CFG_NAME
        / f"seed_{args.seed}"
        / "VLPromptLearner"
        / "model-best.pth.tar"
    )


def load_luna_context(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["state_dict"]
    candidates = (
        "prompt_learner.ctx",
        "module.prompt_learner.ctx",
    )
    ctx = next((state[key] for key in candidates if key in state), None)
    if ctx is None:
        raise KeyError(f"No trained prompt_learner.ctx found in {path}")
    if tuple(ctx.shape) != tuple(model.prompt_learner.ctx.shape):
        raise ValueError(
            f"Context shape mismatch: checkpoint={tuple(ctx.shape)}, "
            f"model={tuple(model.prompt_learner.ctx.shape)}"
        )
    with torch.no_grad():
        model.prompt_learner.ctx.copy_(ctx)
    return int(checkpoint.get("epoch", -1))


@torch.no_grad()
def encode_image_with_patches(visual, images):
    """Mirror CLIP ViT forward while retaining all projected patch tokens."""
    x = visual.conv1(images)
    batch, width, grid_h, grid_w = x.shape
    x = x.reshape(batch, width, grid_h * grid_w).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype).expand(batch, 1, -1)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    if getattr(visual, "VPT_shallow", False):
        visual_ctx = visual.VPT.expand(batch, -1, -1).to(x.dtype)
        x = torch.cat([x, visual_ctx], dim=1)
    x = visual.ln_pre(x)
    x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj
    patch_count = grid_h * grid_w
    return x[:, 0], x[:, 1 : 1 + patch_count], (grid_h, grid_w)


def encode_image_for_rollout(visual, images):
    """Run the visual encoder with gradients and return its CLS embedding."""
    x = visual.conv1(images)
    batch, width, grid_h, grid_w = x.shape
    x = x.reshape(batch, width, grid_h * grid_w).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype).expand(batch, 1, -1)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    if getattr(visual, "VPT_shallow", False):
        visual_ctx = visual.VPT.expand(batch, -1, -1).to(x.dtype)
        x = torch.cat([x, visual_ctx], dim=1)
    x = visual.ln_pre(x)
    x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    cls_feature = visual.ln_post(x[:, 0])
    if visual.proj is not None:
        cls_feature = cls_feature @ visual.proj
    return cls_feature, (grid_h, grid_w)


class VisualAttentionCapture:
    """Temporarily retain per-head attention tensors from every visual block."""

    def __init__(self, visual):
        self.blocks = list(visual.transformer.resblocks)
        self.original_attention = []
        self.attentions = []

    def __enter__(self):
        for block in self.blocks:
            self.original_attention.append((block, block.attention))

            def attention_with_capture(block_self, x, capture=self):
                block_self.attn_mask = (
                    block_self.attn_mask.to(dtype=x.dtype, device=x.device)
                    if block_self.attn_mask is not None
                    else None
                )
                attention = block_self.attn
                if not attention._qkv_same_embed_dim:
                    raise RuntimeError("Only CLIP self-attention is supported")
                length, batch_size, embed_dim = x.shape
                head_count = attention.num_heads
                head_dim = embed_dim // head_count
                qkv = F.linear(x, attention.in_proj_weight, attention.in_proj_bias)
                query, key, value = qkv.chunk(3, dim=-1)

                def split_heads(tensor):
                    return tensor.reshape(
                        length, batch_size, head_count, head_dim
                    ).permute(1, 2, 0, 3)

                query = split_heads(query) * (head_dim ** -0.5)
                key = split_heads(key)
                value = split_heads(value)
                logits = torch.matmul(query, key.transpose(-2, -1))
                if block_self.attn_mask is not None:
                    logits = logits + block_self.attn_mask[None, None]
                weights = F.softmax(logits.float(), dim=-1).to(value.dtype)
                capture.attentions.append(weights)
                output = torch.matmul(weights, value)
                output = output.permute(2, 0, 1, 3).reshape(
                    length, batch_size, embed_dim
                )
                return F.linear(
                    output, attention.out_proj.weight, attention.out_proj.bias
                )

            block.attention = types.MethodType(attention_with_capture, block)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for block, original in self.original_attention:
            block.attention = original


def target_margin_and_negatives(cls_features, text_features, target_positions, k):
    similarity = cls_features @ text_features.T
    k = min(k, text_features.shape[0] - 1)
    masked = similarity.clone()
    masked.scatter_(1, target_positions[:, None], float("-inf"))
    negative_similarity, negative_positions = masked.topk(k, dim=-1)
    target_similarity = similarity.gather(1, target_positions[:, None]).squeeze(-1)
    margin = target_similarity - negative_similarity.mean(dim=-1)
    return similarity, margin, negative_positions


def gradient_attention_rollout(attentions, gradients, patch_count, layers, power):
    if layers < 1 or power <= 0:
        raise ValueError("rollout_layers and rollout_power must be positive")
    pairs = list(zip(attentions, gradients))[-min(layers, len(attentions)) :]
    batch_size, _, token_count, _ = pairs[0][0].shape
    identity = torch.eye(token_count, device=pairs[0][0].device).expand(
        batch_size, token_count, token_count
    )
    joint = identity
    for attention, gradient in pairs:
        weighted = (attention * gradient).clamp_min(0.0).mean(dim=1).float()
        weighted = weighted + identity
        weighted = weighted / weighted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        joint = torch.bmm(weighted, joint)
    relevance = joint[:, 0, 1 : 1 + patch_count].clamp_min(0.0)
    relevance = relevance.pow(power)
    return relevance / relevance.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def paired_gradient_rollout(
    visual,
    images,
    clip_text,
    luna_text,
    target_positions,
    hard_negative_k,
    rollout_layers,
    rollout_power,
):
    images = images.detach().requires_grad_(True)
    with VisualAttentionCapture(visual) as capture:
        cls_features, grid_shape = encode_image_for_rollout(visual, images)
        cls_features = F.normalize(cls_features.float(), dim=-1)
        clip_similarity, clip_margin, clip_negatives = target_margin_and_negatives(
            cls_features, clip_text, target_positions, hard_negative_k
        )
        luna_similarity, luna_margin, luna_negatives = target_margin_and_negatives(
            cls_features, luna_text, target_positions, hard_negative_k
        )
        clip_gradients = torch.autograd.grad(
            clip_margin.sum(), capture.attentions, retain_graph=True
        )
        luna_gradients = torch.autograd.grad(
            luna_margin.sum(), capture.attentions, retain_graph=False
        )

    patch_count = grid_shape[0] * grid_shape[1]
    clip_spatial = gradient_attention_rollout(
        capture.attentions,
        clip_gradients,
        patch_count,
        rollout_layers,
        rollout_power,
    )
    luna_spatial = gradient_attention_rollout(
        capture.attentions,
        luna_gradients,
        patch_count,
        rollout_layers,
        rollout_power,
    )
    return (
        cls_features.detach(),
        clip_similarity.detach(),
        luna_similarity.detach(),
        clip_spatial.detach(),
        luna_spatial.detach(),
        clip_margin.detach(),
        luna_margin.detach(),
        clip_negatives.detach(),
        luna_negatives.detach(),
        grid_shape,
    )


@torch.no_grad()
def build_text_features(model, clip_model, classnames, device):
    # Original CLIP: the standard handcrafted prompt used in prompt-learning baselines.
    clip_prompts = [f"a photo of a {name.replace('_', ' ')}." for name in classnames]
    tokens = torch.cat([clip.tokenize(text) for text in clip_prompts]).to(device)
    clip_features = clip_model.encode_text(tokens)
    clip_features = F.normalize(clip_features.float(), dim=-1)

    luna_prompts = model.prompt_learner().to(device=device, dtype=model.dtype)
    luna_features = model.text_encoder(luna_prompts, model.tokenized_prompts.to(device))
    luna_features = F.normalize(luna_features.float(), dim=-1)
    return clip_features, luna_features


def contrastive_patch_attention(
    cls_features,
    patch_features,
    text_features,
    target_positions,
    hard_negative_k,
    spatial_temperature,
):
    """Return native-grid probabilities and target-vs-hard-negative margins."""
    if text_features.shape[0] < 2:
        raise ValueError("Class-contrastive attention needs at least two classes")
    hard_negative_k = min(hard_negative_k, text_features.shape[0] - 1)
    if hard_negative_k < 1:
        raise ValueError("hard_negative_k must be positive")
    if spatial_temperature <= 0:
        raise ValueError("spatial_temperature must be positive")

    patch_class_similarity = torch.einsum(
        "bpd,cd->bpc", patch_features, text_features
    )
    cls_class_similarity = cls_features @ text_features.T
    masked_cls_similarity = cls_class_similarity.clone()
    masked_cls_similarity.scatter_(1, target_positions[:, None], float("-inf"))
    negative_cls_similarity, negative_positions = masked_cls_similarity.topk(
        hard_negative_k, dim=-1
    )

    batch_size, patch_count, _ = patch_class_similarity.shape
    target_patch_similarity = patch_class_similarity.gather(
        2,
        target_positions[:, None, None].expand(batch_size, patch_count, 1),
    ).squeeze(-1)
    negative_patch_similarity = patch_class_similarity.gather(
        2,
        negative_positions[:, None, :].expand(
            batch_size, patch_count, hard_negative_k
        ),
    )
    patch_margin = target_patch_similarity - negative_patch_similarity.mean(dim=-1)
    spatial_probability = F.softmax(
        patch_margin.float() / spatial_temperature, dim=-1
    )

    target_cls_similarity = cls_class_similarity.gather(
        1, target_positions[:, None]
    ).squeeze(-1)
    global_margin = target_cls_similarity - negative_cls_similarity.mean(dim=-1)
    return spatial_probability, global_margin, negative_positions


def render_heatmap(spatial_probability, grid_shape, visible_quantile):
    """Upsample native probabilities and hide weak responses for rendering only."""
    if not 0.0 <= visible_quantile < 1.0:
        raise ValueError("visible_quantile must be in [0, 1)")
    attention = spatial_probability.reshape(-1, 1, *grid_shape)
    attention = F.interpolate(
        attention, size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(1)
    flat = attention.flatten(1)
    lower = torch.quantile(flat, visible_quantile, dim=1, keepdim=True)
    upper = torch.quantile(flat, 0.995, dim=1, keepdim=True)
    rendered = (
        (flat - lower) / (upper - lower).clamp_min(1e-12)
    ).clamp(0.0, 1.0)
    return rendered.reshape_as(attention)


def render_gain_heatmap(luna_probability, clip_probability, grid_shape, visible_quantile):
    positive_gain = (luna_probability - clip_probability).clamp_min(0.0)
    return render_heatmap(positive_gain, grid_shape, visible_quantile)


def focus_metrics(spatial_probability):
    probability = np.asarray(spatial_probability, dtype=np.float64).reshape(-1)
    probability = np.clip(probability, 1e-12, None)
    probability = probability / probability.sum()
    entropy = float(
        -(probability * np.log(probability)).sum() / math.log(len(probability))
    )
    count = max(1, int(math.ceil(0.20 * len(probability))))
    top_mass = float(np.partition(probability, -count)[-count:].sum())
    return entropy, top_mass


def denormalize(images):
    array = images.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
    return np.clip(array * STD + MEAN, 0.0, 1.0)


def overlay(image, heatmap, alpha):
    strength = np.clip(heatmap, 0.0, 1.0) ** 0.75
    color = plt.get_cmap("magma")(strength)[..., :3]
    local_alpha = (alpha * strength)[..., None]
    return np.clip((1.0 - local_alpha) * image + local_alpha * color, 0.0, 1.0)


def safe_slug(text):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or "sample"


def save_panel(records, output_path, dataset_name, alpha, include_gain=False):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
        }
    )
    rows = len(records)
    columns = 4 if include_gain else 3
    fig, axes = plt.subplots(rows, columns, figsize=(2.32 * columns, 2.25 * rows + 0.45))
    if rows == 1:
        axes = np.asarray([axes])
    headers = ("Input", "CLIP", "LUNA", "LUNA gain") if include_gain else ("Input", "CLIP", "LUNA")
    for column, header in enumerate(headers):
        axes[0, column].set_title(header, fontweight="semibold", pad=7)
    for row, record in enumerate(records):
        panels = [
            record["image"],
            overlay(record["image"], record["clip_heatmap"], alpha),
            overlay(record["image"], record["luna_heatmap"], alpha),
        ]
        if include_gain:
            panels.append(overlay(record["image"], record["gain_heatmap"], alpha))
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(
            record["classname"].replace("_", " "),
            rotation=90,
            labelpad=8,
            fontsize=9,
        )
    fig.suptitle(dataset_name, y=0.995, fontsize=13, fontweight="semibold")
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.01, top=0.94, wspace=0.025, hspace=0.055)
    fig.savefig(output_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    if args.num_samples < 3:
        raise ValueError("At least three comparison images are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this visualization run")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = build_cfg(args)
    dataset = build_dataset(cfg)
    classnames = list(dataset.classnames)
    class_to_full_index = {name: index for index, name in enumerate(classnames)}
    checkpoint = checkpoint_path(args)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Trained LUNA checkpoint not found: {checkpoint}")

    clip_model = load_clip_to_cpu(cfg).eval()
    model = CustomCLIP(cfg, classnames, clip_model).eval()
    best_epoch = load_luna_context(model, checkpoint)
    model.to(device)
    clip_model.to(device)
    preprocess = clip._transform(clip_model.visual.input_resolution)
    clip_text, luna_text = build_text_features(model, clip_model, classnames, device)
    logit_scale = clip_model.logit_scale.exp().float()

    novel_items = sorted(dataset.test, key=lambda item: item.impath)
    sampler = random.Random(args.seed)
    sampler.shuffle(novel_items)
    novel_items = novel_items[: min(args.candidate_limit, len(novel_items))]
    novel_full_indices = sorted({class_to_full_index[item.classname] for item in dataset.test})
    novel_index_tensor = torch.tensor(novel_full_indices, device=device)
    novel_position_by_full_index = {
        full_index: position for position, full_index in enumerate(novel_full_indices)
    }
    clip_novel_text = clip_text[novel_index_tensor]
    luna_novel_text = luna_text[novel_index_tensor]

    candidates = []
    for start in range(0, len(novel_items), args.batch_size):
        batch_items = novel_items[start : start + args.batch_size]
        image_tensors = []
        valid_items = []
        for item in batch_items:
            try:
                with Image.open(item.impath) as image:
                    image_tensors.append(preprocess(image.convert("RGB")))
                valid_items.append(item)
            except (OSError, ValueError) as error:
                print(f"Skipping unreadable image {item.impath}: {error}")
        if not valid_items:
            continue

        images = torch.stack(image_tensors).to(device=device, dtype=model.dtype)
        targets = torch.tensor(
            [class_to_full_index[item.classname] for item in valid_items], device=device
        )
        batch_indices = torch.arange(len(valid_items), device=device)
        target_positions = torch.tensor(
            [novel_position_by_full_index[int(index)] for index in targets.cpu()],
            device=device,
        )
        (
            cls_features,
            clip_class_similarity,
            luna_class_similarity,
            clip_spatial,
            luna_spatial,
            clip_margin,
            luna_margin,
            clip_negative_positions,
            luna_negative_positions,
            grid_shape,
        ) = paired_gradient_rollout(
            model.image_encoder,
            images,
            clip_novel_text,
            luna_novel_text,
            target_positions,
            args.hard_negative_k,
            args.rollout_layers,
            args.rollout_power,
        )
        clip_logits = logit_scale * clip_class_similarity
        luna_logits = logit_scale * luna_class_similarity
        clip_probability = clip_logits.softmax(dim=-1)
        luna_probability = luna_logits.softmax(dim=-1)
        clip_spatial = clip_spatial.cpu().numpy()
        luna_spatial = luna_spatial.cpu().numpy()
        clip_margin = clip_margin.cpu().tolist()
        luna_margin = luna_margin.cpu().tolist()
        clip_negative_positions = clip_negative_positions.cpu().tolist()
        luna_negative_positions = luna_negative_positions.cpu().tolist()
        clip_predictions = novel_index_tensor[clip_logits.argmax(dim=-1)].cpu().tolist()
        luna_predictions = novel_index_tensor[luna_logits.argmax(dim=-1)].cpu().tolist()
        clip_target_prob = clip_probability[batch_indices, target_positions].cpu().tolist()
        luna_target_prob = luna_probability[batch_indices, target_positions].cpu().tolist()

        for index, item in enumerate(valid_items):
            clip_entropy, clip_top_mass = focus_metrics(clip_spatial[index])
            luna_entropy, luna_top_mass = focus_metrics(luna_spatial[index])
            focus_gain = clip_entropy - luna_entropy
            mass_gain = luna_top_mass - clip_top_mass
            margin_gain = luna_margin[index] - clip_margin[index]
            target = int(targets[index].item())
            candidates.append(
                {
                    "impath": item.impath,
                    "classname": item.classname,
                    "target_index": target,
                    "clip_pred": classnames[clip_predictions[index]],
                    "luna_pred": classnames[luna_predictions[index]],
                    "clip_correct": clip_predictions[index] == target,
                    "luna_correct": luna_predictions[index] == target,
                    "clip_target_prob": clip_target_prob[index],
                    "luna_target_prob": luna_target_prob[index],
                    "clip_global_margin": clip_margin[index],
                    "luna_global_margin": luna_margin[index],
                    "global_margin_gain": margin_gain,
                    "clip_hard_negatives": ";".join(
                        classnames[novel_full_indices[position]]
                        for position in clip_negative_positions[index]
                    ),
                    "luna_hard_negatives": ";".join(
                        classnames[novel_full_indices[position]]
                        for position in luna_negative_positions[index]
                    ),
                    "clip_entropy": clip_entropy,
                    "luna_entropy": luna_entropy,
                    "entropy_gain": focus_gain,
                    "clip_top20_mass": clip_top_mass,
                    "luna_top20_mass": luna_top_mass,
                    "top20_mass_gain": mass_gain,
                    "selection_score": 2.0 * focus_gain + mass_gain + margin_gain,
                    "_clip_spatial": clip_spatial[index],
                    "_luna_spatial": luna_spatial[index],
                    "_grid_shape": grid_shape,
                }
            )

    eligible = [
        item
        for item in candidates
        if item["luna_correct"]
        and item["entropy_gain"] > 0.0
        and item["top20_mass_gain"] > 0.0
        and item["global_margin_gain"] > 0.0
    ]
    eligible.sort(
        key=lambda item: (
            item["selection_score"],
            item["luna_target_prob"] - item["clip_target_prob"],
        ),
        reverse=True,
    )
    selected = []
    used_classes = set()
    for item in eligible:
        if item["classname"] in used_classes:
            continue
        selected.append(item)
        used_classes.add(item["classname"])
        if len(selected) == args.num_samples:
            break
    if len(selected) < args.num_samples:
        raise RuntimeError(
            f"Only {len(selected)} distinct-class candidates passed the focus criteria; "
            f"requested {args.num_samples}. No misleading fallback was produced."
        )

    records = []
    for item in selected:
        with Image.open(item["impath"]) as image:
            tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(
                device=device, dtype=model.dtype
            )
        grid_shape = item["_grid_shape"]
        clip_spatial = torch.from_numpy(item["_clip_spatial"])[None].to(device)
        luna_spatial = torch.from_numpy(item["_luna_spatial"])[None].to(device)
        item["image"] = denormalize(tensor)[0]
        item["clip_spatial_probability"] = clip_spatial[0].cpu().numpy()
        item["luna_spatial_probability"] = luna_spatial[0].cpu().numpy()
        item["clip_heatmap"] = render_heatmap(
            clip_spatial, grid_shape, args.visible_quantile
        )[0].cpu().numpy()
        item["luna_heatmap"] = render_heatmap(
            luna_spatial, grid_shape, args.visible_quantile
        )[0].cpu().numpy()
        item["gain_heatmap"] = render_gain_heatmap(
            luna_spatial, clip_spatial, grid_shape, args.visible_quantile
        )[0].cpu().numpy()
        records.append(item)

    output_dir = args.workdir / "output" / args.output_root / f"seed_{args.seed}" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    save_panel(
        records,
        output_dir / "attention_comparison.png",
        DISPLAY_NAMES[args.dataset],
        args.overlay_alpha,
    )
    save_panel(
        records,
        output_dir / "attention_gain.png",
        DISPLAY_NAMES[args.dataset],
        args.overlay_alpha,
        include_gain=True,
    )
    for rank, record in enumerate(records, start=1):
        save_panel(
            [record],
            output_dir / f"sample_{rank:02d}_{safe_slug(record['classname'])}.png",
            DISPLAY_NAMES[args.dataset],
            args.overlay_alpha,
        )

    manifest_fields = [
        "rank",
        "dataset",
        "seed",
        "checkpoint",
        "checkpoint_epoch",
        "impath",
        "classname",
        "target_index",
        "clip_pred",
        "luna_pred",
        "clip_correct",
        "luna_correct",
        "clip_target_prob",
        "luna_target_prob",
        "clip_global_margin",
        "luna_global_margin",
        "global_margin_gain",
        "clip_hard_negatives",
        "luna_hard_negatives",
        "clip_entropy",
        "luna_entropy",
        "entropy_gain",
        "clip_top20_mass",
        "luna_top20_mass",
        "top20_mass_gain",
        "selection_score",
    ]
    with (output_dir / "selection_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=manifest_fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for rank, record in enumerate(records, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_epoch": best_epoch,
                    **{field: record[field] for field in manifest_fields[5:]},
                }
            )

    np.savez_compressed(
        output_dir / "selected_heatmaps.npz",
        clip=np.stack([record["clip_heatmap"] for record in records]),
        luna=np.stack([record["luna_heatmap"] for record in records]),
        gain=np.stack([record["gain_heatmap"] for record in records]),
        clip_probability=np.stack(
            [record["clip_spatial_probability"] for record in records]
        ),
        luna_probability=np.stack(
            [record["luna_spatial_probability"] for record in records]
        ),
    )
    (output_dir / "candidate_limit.txt").write_text(
        f"{args.candidate_limit}\n",
        encoding="utf-8",
    )
    (output_dir / "visualization_config.txt").write_text(
        "method=class_contrastive_gradient_attention_rollout\n"
        f"hard_negative_k={args.hard_negative_k}\n"
        f"rollout_layers={args.rollout_layers}\n"
        f"rollout_power={args.rollout_power}\n"
        f"visible_quantile={args.visible_quantile}\n",
        encoding="utf-8",
    )
    print(f"Saved {len(records)} CLIP/LUNA comparisons to {output_dir}")


if __name__ == "__main__":
    main()
