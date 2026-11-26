#!/usr/bin/env python3
"""Create paired zero-shot CLIP/LUNA t-SNE plots from classification logits.

The implementation reconstructs the complete held-out set from the base
validation and novel-test portions exposed by the LUNA trainer. Both methods
use exactly the same images, class colors, t-SNE settings, seed, and initialization.
"""

import argparse
import csv
import inspect
import io
import json
import math
import os
import pickle
import random
import sys
import zipfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired t-SNE visualization of zero-shot CLIP and LUNA logits."
    )
    parser.add_argument("dataset", choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-classes", type=int, default=15)
    parser.add_argument("--max-classes", type=int, default=20)
    parser.add_argument("--samples-per-class", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/tsne_logits_discriminative_seed1"),
    )
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
    cfg.merge_from_file(
        str(args.workdir / "configs/datasets" / DATASET_CONFIGS[args.dataset])
    )
    cfg.merge_from_file(
        str(args.workdir / "configs/trainers/LUNA" / f"{CFG_NAME}.yaml")
    )
    cfg.DATASET.ROOT = str(args.root)
    cfg.DATASET.NUM_SHOTS = 0
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
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


@dataclass(frozen=True)
class StorageRef:
    storage_type: str
    key: str
    size: int


@dataclass(frozen=True)
class TensorRef:
    storage: StorageRef
    offset: int
    size: tuple
    stride: tuple


def _rebuild_tensor(storage, offset, size, stride, *unused):
    return TensorRef(storage, int(offset), tuple(size), tuple(stride))


def _rebuild_parameter(tensor, *unused):
    return tensor


class CheckpointMetadataUnpickler(pickle.Unpickler):
    """Read tensor metadata without materializing every optimizer storage."""

    def find_class(self, module, name):
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name.startswith("_rebuild_parameter"):
            return _rebuild_parameter
        if module == "torch" and name.endswith("Storage"):
            return name
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        return super().find_class(module, name)

    def persistent_load(self, saved_id):
        if not isinstance(saved_id, tuple) or saved_id[0] != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id: {saved_id}")
        _, storage_type, key, _location, size = saved_id
        return StorageRef(str(storage_type), str(key), int(size))


def _dtype_from_storage(storage_type):
    mapping = {
        "HalfStorage": np.float16,
        "FloatStorage": np.float32,
        "DoubleStorage": np.float64,
        "LongStorage": np.int64,
        "IntStorage": np.int32,
        "ShortStorage": np.int16,
        "CharStorage": np.int8,
        "ByteStorage": np.uint8,
        "BoolStorage": np.bool_,
        "BFloat16Storage": None,
    }
    short_name = storage_type.rsplit(".", 1)[-1]
    if short_name not in mapping or mapping[short_name] is None:
        raise TypeError(f"Unsupported checkpoint storage type: {storage_type}")
    return mapping[short_name]


def load_luna_context(model, path):
    """Load only the learned class-specific context storage from the checkpoint."""
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith("/data.pkl")
        )
        metadata = CheckpointMetadataUnpickler(
            io.BytesIO(archive.read(metadata_name))
        ).load()
        state = metadata["state_dict"]
        key = next(
            candidate
            for candidate in ("prompt_learner.ctx", "module.prompt_learner.ctx")
            if candidate in state
        )
        tensor_ref = state[key]
        if not isinstance(tensor_ref, TensorRef):
            raise TypeError(f"Unexpected context descriptor: {type(tensor_ref)}")
        root = metadata_name[: -len("data.pkl")]
        raw = archive.read(f"{root}data/{tensor_ref.storage.key}")

    dtype = _dtype_from_storage(tensor_ref.storage.storage_type)
    itemsize = np.dtype(dtype).itemsize
    count = math.prod(tensor_ref.size)
    offset_bytes = tensor_ref.offset * itemsize
    values = np.frombuffer(raw, dtype=dtype, count=count, offset=offset_bytes).copy()
    context = torch.from_numpy(values.reshape(tensor_ref.size))
    expected = tuple(model.prompt_learner.ctx.shape)
    if tuple(context.shape) != expected:
        raise ValueError(
            f"Context shape mismatch: checkpoint={tuple(context.shape)}, model={expected}"
        )
    with torch.no_grad():
        model.prompt_learner.ctx.copy_(
            context.to(dtype=model.prompt_learner.ctx.dtype)
        )


def choose_candidate_samples(dataset, classnames, args):
    class_to_full_index = {name: index for index, name in enumerate(classnames)}
    grouped = defaultdict(list)
    # LUNA exposes base held-out images as val and novel held-out images as
    # test. Combining them reconstructs the complete held-out class set.
    for item in list(dataset.val) + list(dataset.test):
        grouped[class_to_full_index[item.classname]].append(item)

    candidate_indices = sorted(grouped)
    records = []
    for full_index in candidate_indices:
        items = sorted(grouped[full_index], key=lambda item: item.impath)
        sampler = random.Random(f"{args.seed}:{args.dataset}:{full_index}:images")
        sampler.shuffle(items)
        for item in items[: args.samples_per_class]:
            records.append(
                {
                    "item": item,
                    "full_index": full_index,
                    "classname": classnames[full_index],
                }
            )
    records.sort(key=lambda row: (row["full_index"], row["item"].impath))
    return records, candidate_indices


@torch.no_grad()
def build_text_features(model, clip_model, classnames, device):
    zero_shot_prompts = [
        f"a photo of a {classname.replace('_', ' ')}." for classname in classnames
    ]
    tokenized = torch.cat([clip.tokenize(text) for text in zero_shot_prompts]).to(
        device
    )
    zero_shot_text = F.normalize(
        clip_model.encode_text(tokenized).float(), dim=-1
    )

    luna_prompts = model.prompt_learner().to(device=device, dtype=model.dtype)
    luna_text = model.text_encoder(
        luna_prompts, model.tokenized_prompts.to(device)
    )
    luna_text = F.normalize(luna_text.float(), dim=-1)
    return zero_shot_text, luna_text


@torch.no_grad()
def collect_logits(records, model, zero_shot_text, luna_text, preprocess, args):
    zero_shot_logits = []
    luna_logits = []
    kept = []
    device = next(model.parameters()).device
    logit_scale = model.logit_scale.exp().float()

    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        images = []
        valid_records = []
        for record in batch_records:
            try:
                with Image.open(record["item"].impath) as image:
                    images.append(preprocess(image.convert("RGB")))
                valid_records.append(record)
            except (OSError, ValueError) as error:
                print(f"Skipping unreadable image {record['item'].impath}: {error}")
        if not images:
            continue

        image_batch = torch.stack(images).to(device=device, dtype=model.dtype)
        image_features = model.image_encoder(image_batch)
        image_features = F.normalize(image_features.float(), dim=-1)
        zero_shot_logits.append(
            (logit_scale * image_features @ zero_shot_text.T).cpu().numpy()
        )
        luna_logits.append(
            (logit_scale * image_features @ luna_text.T).cpu().numpy()
        )
        kept.extend(valid_records)

    if not kept:
        raise RuntimeError("No readable images were available for visualization")
    return (
        np.concatenate(zero_shot_logits, axis=0),
        np.concatenate(luna_logits, axis=0),
        kept,
    )


def class_separability_matrix(logits, labels, class_indices):
    """Fisher-style pairwise separation: centroid distance / within scatter."""
    centers = []
    scatter = []
    for class_index in class_indices:
        class_logits = logits[labels == class_index]
        center = class_logits.mean(axis=0)
        centers.append(center)
        scatter.append(
            np.sqrt(np.mean(np.sum((class_logits - center) ** 2, axis=1)))
        )
    centers = np.asarray(centers, dtype=np.float64)
    scatter = np.asarray(scatter, dtype=np.float64)
    delta = centers[:, None, :] - centers[None, :, :]
    between = np.sqrt(np.sum(delta * delta, axis=-1))
    within = np.sqrt(scatter[:, None] ** 2 + scatter[None, :] ** 2)
    separation = between / np.maximum(within, 1e-12)
    np.fill_diagonal(separation, 0.0)
    positive = separation[separation > 0]
    scale = np.median(positive) if positive.size else 1.0
    return separation / max(scale, 1e-12)


def greedy_class_order(combined_separation, class_indices, target_count):
    """Select a diverse class set by maximizing worst-case pair separation."""
    count = len(class_indices)
    masked = combined_separation.copy()
    np.fill_diagonal(masked, np.nan)
    lower_quartile = np.nanquantile(masked, 0.25, axis=1)
    selected_positions = [int(np.nanargmax(lower_quartile))]

    while len(selected_positions) < min(target_count, count):
        remaining = [i for i in range(count) if i not in selected_positions]
        scores = []
        for position in remaining:
            distances = combined_separation[position, selected_positions]
            # The minimum distance prevents one visually confusable class from
            # entering the subset; the mean term breaks ties smoothly.
            scores.append(0.78 * distances.min() + 0.22 * distances.mean())
        selected_positions.append(remaining[int(np.argmax(scores))])

    return [class_indices[position] for position in selected_positions]


def choose_discriminative_classes(
    zero_shot_logits, luna_logits, records, candidate_indices, args
):
    full_labels = np.asarray([record["full_index"] for record in records], dtype=np.int64)
    clip_separation = class_separability_matrix(
        zero_shot_logits, full_labels, candidate_indices
    )
    luna_separation = class_separability_matrix(
        luna_logits, full_labels, candidate_indices
    )
    # Geometric mean makes the selection method-balanced: a class pair must be
    # separable for both CLIP and LUNA rather than being cherry-picked for one.
    combined = np.sqrt(np.maximum(clip_separation, 0) * np.maximum(luna_separation, 0))

    minimum = min(args.min_classes, len(candidate_indices))
    maximum = min(args.max_classes, len(candidate_indices))
    order = greedy_class_order(combined, candidate_indices, maximum)
    trials = []
    for class_count in range(minimum, maximum + 1):
        subset = order[:class_count]
        mask = np.isin(full_labels, subset)
        subset_labels = full_labels[mask]
        clip_silhouette = float(
            silhouette_score(zero_shot_logits[mask], subset_labels, metric="euclidean")
        )
        luna_silhouette = float(
            silhouette_score(luna_logits[mask], subset_labels, metric="euclidean")
        )
        balanced_score = 0.5 * (clip_silhouette + luna_silhouette)
        # Separation is primary. This tiny term only favors retaining more
        # classes when the measured separability is effectively tied.
        objective = balanced_score + 0.002 * (class_count - minimum)
        trials.append(
            {
                "class_count": class_count,
                "clip_silhouette": clip_silhouette,
                "luna_silhouette": luna_silhouette,
                "balanced_score": balanced_score,
                "objective": objective,
                "classes": subset,
            }
        )

    best = max(trials, key=lambda trial: (trial["objective"], trial["class_count"]))
    selected_indices = sorted(best["classes"])
    mask = np.isin(full_labels, selected_indices)
    selected_records = [record for record, keep in zip(records, mask) if keep]
    selected_zero = zero_shot_logits[mask]
    selected_luna = luna_logits[mask]
    display_map = {
        full_index: display_label
        for display_label, full_index in enumerate(selected_indices)
    }
    for record in selected_records:
        record["display_label"] = display_map[record["full_index"]]

    class_quality = {}
    position_by_class = {
        full_index: position for position, full_index in enumerate(candidate_indices)
    }
    selected_positions = [position_by_class[index] for index in selected_indices]
    for full_index in selected_indices:
        position = position_by_class[full_index]
        peers = [peer for peer in selected_positions if peer != position]
        class_quality[full_index] = float(combined[position, peers].min())

    return (
        selected_zero,
        selected_luna,
        selected_records,
        selected_indices,
        trials,
        class_quality,
    )


def common_initialization(zero_shot_logits, luna_logits, seed):
    midpoint = 0.5 * (zero_shot_logits + luna_logits)
    init = PCA(n_components=2, random_state=seed).fit_transform(midpoint)
    scale = np.std(init[:, 0])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (init / scale * 1e-4).astype(np.float32)


def fit_tsne(logits, init, args):
    if not np.isfinite(logits).all():
        raise ValueError("Classification logits contain NaN or infinite values")
    n_samples = logits.shape[0]
    perplexity = min(args.perplexity, max(5.0, (n_samples - 1) / 3.0))
    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "early_exaggeration": 12.0,
        "learning_rate": "auto",
        "metric": "euclidean",
        "init": init.copy(),
        "random_state": args.seed,
        "method": "barnes_hut",
        "angle": 0.5,
        "verbose": 1,
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = args.iterations
    else:
        kwargs["n_iter"] = args.iterations
    return TSNE(**kwargs).fit_transform(logits.astype(np.float32))


def embedding_limits(embedding):
    x0, x1 = embedding[:, 0].min(), embedding[:, 0].max()
    y0, y1 = embedding[:, 1].min(), embedding[:, 1].max()
    xpad = max(1e-6, (x1 - x0) * 0.07)
    ypad = max(1e-6, (y1 - y0) * 0.07)
    return (x0 - xpad, x1 + xpad), (y0 - ypad, y1 + ypad)


def render_pair(zero_embedding, luna_embedding, labels, dataset_name, output_path):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
        }
    )
    class_count = int(labels.max()) + 1
    palette = np.asarray(
        [
            "#4C78A8", "#F58518", "#54A24B", "#E45756", "#8E6CBE",
            "#9D755D", "#D85C9A", "#6B6B6B", "#B79A20", "#1DA7B8",
            "#2F5597", "#E07B39", "#2E8B57", "#B84A62", "#725A9E",
            "#7A5548", "#C24E78", "#4F7C7A", "#9B8F2F", "#3C91C7",
        ]
    )
    colors = palette[:class_count]
    fig, axes = plt.subplots(1, 2, figsize=(6.55, 3.08), dpi=180)

    for axis, embedding, title in zip(
        axes,
        (zero_embedding, luna_embedding),
        ("Zero-shot CLIP", "LUNA"),
    ):
        for class_id in range(class_count):
            selected = labels == class_id
            axis.scatter(
                embedding[selected, 0],
                embedding[selected, 1],
                s=17,
                c=[colors[class_id]],
                alpha=0.88,
                edgecolors="white",
                linewidths=0.42,
                rasterized=True,
            )
        axis.set_title(title, fontweight="semibold", pad=7)
        xlim, ylim = embedding_limits(embedding)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("#fbfbfc")
        for spine in axis.spines.values():
            spine.set_color("#c9ced6")
            spine.set_linewidth(0.8)

    fig.suptitle(dataset_name, fontsize=13, fontweight="semibold", y=0.992)
    fig.subplots_adjust(left=0.012, right=0.996, bottom=0.018, top=0.84, wspace=0.052)
    fig.savefig(output_path, dpi=450, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    plt.close(fig)


def save_metadata(
    output_dir,
    records,
    candidate_indices,
    selected_indices,
    selection_trials,
    class_quality,
    classnames,
    zero_shot_logits,
    luna_logits,
    zero_embedding,
    luna_embedding,
    args,
):
    labels = np.asarray([record["display_label"] for record in records], dtype=np.int64)
    paths = np.asarray([record["item"].impath for record in records])
    np.savez_compressed(
        output_dir / "logits_and_embeddings.npz",
        zero_shot_logits=zero_shot_logits,
        luna_logits=luna_logits,
        zero_shot_tsne=zero_embedding,
        luna_tsne=luna_embedding,
        display_labels=labels,
        image_paths=paths,
        candidate_full_indices=np.asarray(candidate_indices, dtype=np.int64),
        selected_full_indices=np.asarray(selected_indices, dtype=np.int64),
    )

    with (output_dir / "selected_classes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "display_label",
                "full_class_index",
                "classname",
                "min_balanced_pair_separability",
            ]
        )
        for display_label, full_index in enumerate(selected_indices):
            writer.writerow(
                [
                    display_label,
                    full_index,
                    classnames[full_index],
                    f"{class_quality[full_index]:.8f}",
                ]
            )

    with (output_dir / "selection_trials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class_count",
                "clip_silhouette",
                "luna_silhouette",
                "balanced_score",
                "objective",
                "selected",
            ]
        )
        for trial in selection_trials:
            writer.writerow(
                [
                    trial["class_count"],
                    f"{trial['clip_silhouette']:.8f}",
                    f"{trial['luna_silhouette']:.8f}",
                    f"{trial['balanced_score']:.8f}",
                    f"{trial['objective']:.8f}",
                    int(set(trial["classes"]) == set(selected_indices)),
                ]
            )

    with (output_dir / "selected_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "display_label", "full_class_index", "classname"])
        for record in records:
            writer.writerow(
                [
                    record["item"].impath,
                    record["display_label"],
                    record["full_index"],
                    record["classname"],
                ]
            )

    metadata = {
        "dataset": args.dataset,
        "display_name": DISPLAY_NAMES[args.dataset],
        "seed": args.seed,
        "split": "complete_held_out_test_from_base_val_plus_novel_test",
        "feature": "classification_logits_over_all_dataset_classes",
        "zero_shot_prompt": "a photo of a {class}.",
        "min_classes": args.min_classes,
        "max_classes": args.max_classes,
        "samples_per_class": args.samples_per_class,
        "selected_class_count": len(selected_indices),
        "sample_count": len(records),
        "candidate_class_count": len(candidate_indices),
        "logit_dimension": len(candidate_indices),
        "class_selection": (
            "greedy max-min Fisher separation using the geometric mean of "
            "CLIP and LUNA separation; class count chosen by balanced silhouette"
        ),
        "tsne_perplexity_requested": args.perplexity,
        "tsne_iterations": args.iterations,
        "tsne_random_state": args.seed,
        "shared_initialization": True,
        "checkpoint": str(checkpoint_path(args)),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    if args.min_classes < 2 or args.max_classes < args.min_classes:
        raise ValueError("Require 2 <= min_classes <= max_classes")
    if args.samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this visualization run")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")

    cfg = build_cfg(args)
    dataset = build_dataset(cfg)
    classnames = list(dataset.classnames)
    records, candidate_indices = choose_candidate_samples(dataset, classnames, args)
    if len(candidate_indices) < 2:
        raise RuntimeError("At least two classes are required for t-SNE")

    checkpoint = checkpoint_path(args)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Trained LUNA checkpoint not found: {checkpoint}")

    print(f"[{args.dataset}] Loading CLIP and LUNA context")
    clip_model = load_clip_to_cpu(cfg).eval()
    model = CustomCLIP(cfg, classnames, clip_model).eval()
    load_luna_context(model, checkpoint)
    model.to(device)
    clip_model.to(device)

    preprocess = clip._transform(clip_model.visual.input_resolution)
    zero_shot_text, luna_text = build_text_features(model, clip_model, classnames, device)
    zero_shot_logits, luna_logits, records = collect_logits(
        records, model, zero_shot_text, luna_text, preprocess, args
    )
    print(
        f"[{args.dataset}] Selecting {args.min_classes}-{args.max_classes} "
        "discriminative classes"
    )
    (
        zero_shot_logits,
        luna_logits,
        records,
        selected_indices,
        selection_trials,
        class_quality,
    ) = choose_discriminative_classes(
        zero_shot_logits, luna_logits, records, candidate_indices, args
    )
    labels = np.asarray([record["display_label"] for record in records], dtype=np.int64)

    init = common_initialization(zero_shot_logits, luna_logits, args.seed)
    print(f"[{args.dataset}] Fitting zero-shot CLIP t-SNE first")
    zero_embedding = fit_tsne(zero_shot_logits, init, args)
    print(f"[{args.dataset}] Fitting LUNA t-SNE second")
    luna_embedding = fit_tsne(luna_logits, init, args)

    output_dir = (args.workdir / args.output_root / args.dataset).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"{args.dataset}_clip_luna_logits_tsne.png"
    render_pair(
        zero_embedding,
        luna_embedding,
        labels,
        DISPLAY_NAMES[args.dataset],
        figure_path,
    )
    save_metadata(
        output_dir,
        records,
        candidate_indices,
        selected_indices,
        selection_trials,
        class_quality,
        classnames,
        zero_shot_logits,
        luna_logits,
        zero_embedding,
        luna_embedding,
        args,
    )
    print(f"[{args.dataset}] Saved {figure_path}")


if __name__ == "__main__":
    main()
