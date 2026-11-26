#!/usr/bin/env python3
"""Search class subsets that visibly contrast zero-shot CLIP and LUNA logits.

This is a diagnostic companion to ``visualize_logits_tsne.py``. It deliberately
selects *classes*, never individual images, and uses the same fixed held-out
images for both methods. For every selected class subset, both the sample rows
and the classification-logit columns are restricted to that subset.

The final embedding follows the implementation in ``trainers/luna.py`` exactly:
raw scaled classification logits are fed
directly to sklearn TSNE with only n_components, random_state and perplexity
specified. There is no softmax, feature standardisation, PCA preprocessing or
shared custom initialisation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

import visualize_logits_tsne as base


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Manually inspectable class-subset search for paired zero-shot "
            "CLIP/LUNA logit t-SNE plots."
        )
    )
    parser.add_argument("dataset", choices=sorted(base.DATASET_CONFIGS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--class-count",
        type=int,
        default=15,
        help="Requested classes; datasets with fewer held-out classes use all.",
    )
    parser.add_argument("--samples-per-class", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Parallel image-loading workers used while constructing the cache.",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--search-trials", type=int, default=4000)
    parser.add_argument(
        "--shortlist",
        type=int,
        default=30,
        help="Only this many subsets receive O(N^2) raw-logit metrics.",
    )
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--local-seeds", type=int, default=12)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--local-proposals", type=int, default=64)
    parser.add_argument(
        "--luna-quality-quantile",
        type=float,
        default=0.55,
        help="Absolute LUNA-quality gate within the exact-metric shortlist.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/tsne_logits_manual_seed1"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ["LUNA_DATA_ROOT"]).expanduser()
        if os.environ.get("LUNA_DATA_ROOT")
        else None,
        help="Dataset root (defaults to the LUNA_DATA_ROOT environment variable).",
    )
    parser.add_argument("--workdir", type=Path, default=base.REPO_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--force-cache",
        action="store_true",
        help="Recompute the full fixed-image logit cache.",
    )
    args = parser.parse_args()
    if args.root is None:
        parser.error("--root is required unless LUNA_DATA_ROOT is set")
    return args


def stable_rng(seed: int, dataset: str) -> np.random.Generator:
    dataset_offset = sum((index + 1) * byte for index, byte in enumerate(dataset.encode()))
    return np.random.default_rng(seed * 1000003 + dataset_offset)


def output_directory(args) -> Path:
    return (args.workdir / args.output_root / args.dataset).resolve()


def cache_path(args) -> Path:
    return output_directory(args) / (
        f"full_heldout_raw_logits_seed{args.seed}_n{args.samples_per_class}.npz"
    )


def _scalar(array):
    value = np.asarray(array)
    return value.item() if value.ndim == 0 else value.reshape(-1)[0].item()


def load_cache(path: Path, args):
    with np.load(path, allow_pickle=False) as archive:
        cache = {key: archive[key] for key in archive.files}
    expected = {
        "dataset": args.dataset,
        "seed": args.seed,
        "samples_per_class": args.samples_per_class,
    }
    actual = {
        "dataset": str(_scalar(cache["dataset"])),
        "seed": int(_scalar(cache["seed"])),
        "samples_per_class": int(_scalar(cache["samples_per_class"])),
    }
    if actual != expected:
        raise ValueError(f"Cache metadata mismatch: expected={expected}, actual={actual}")
    return cache


class FixedRecordImageDataset(Dataset):
    """Load the fixed record list without reordering or silently dropping data."""

    def __init__(self, records, preprocess):
        self.records = records
        self.preprocess = preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path = self.records[index]["item"].impath
        try:
            with Image.open(path) as image:
                tensor = self.preprocess(image.convert("RGB"))
        except Exception as error:
            raise RuntimeError(
                f"Failed to load fixed visualization image at index {index}: {path}"
            ) from error
        return tensor, index


def seed_loader_worker(worker_id):
    """Give each Linux worker a deterministic seed without changing ordering."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def collect_logits_parallel(
    records,
    model,
    zero_shot_text,
    luna_text,
    preprocess,
    args,
):
    """Batch raw-logit extraction with strict end-to-end order validation."""
    dataset = FixedRecordImageDataset(records, preprocess)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_loader_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )

    zero_batches = []
    luna_batches = []
    returned_indices = []
    device = next(model.parameters()).device
    logit_scale = model.logit_scale.exp().float()
    next_expected_index = 0

    for image_batch, batch_indices in loader:
        indices = batch_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        expected = np.arange(
            next_expected_index,
            next_expected_index + len(indices),
            dtype=np.int64,
        )
        if not np.array_equal(indices, expected):
            raise RuntimeError(
                "Parallel image loader changed fixed record order: "
                f"expected {expected.tolist()}, received {indices.tolist()}"
            )
        next_expected_index += len(indices)
        returned_indices.append(indices.copy())

        image_batch = image_batch.to(
            device=device,
            dtype=model.dtype,
            non_blocking=True,
        )
        image_features = model.image_encoder(image_batch)
        image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
        zero_batches.append(
            (logit_scale * image_features @ zero_shot_text.T).cpu().numpy()
        )
        luna_batches.append(
            (logit_scale * image_features @ luna_text.T).cpu().numpy()
        )

    if not returned_indices:
        raise RuntimeError("No images were returned by the parallel image loader")
    returned_indices = np.concatenate(returned_indices)
    expected_indices = np.arange(len(records), dtype=np.int64)
    if not np.array_equal(returned_indices, expected_indices):
        raise RuntimeError(
            "Parallel image loader did not return every fixed record exactly once"
        )
    if len(zero_batches) == 0 or len(luna_batches) == 0:
        raise RuntimeError("No classification logits were produced")

    return (
        np.concatenate(zero_batches, axis=0),
        np.concatenate(luna_batches, axis=0),
        records,
    )


def build_cache(path: Path, args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to construct the logit cache")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = base.build_cfg(args)
    dataset = base.build_dataset(cfg)
    classnames = list(dataset.classnames)
    records, candidate_indices = base.choose_candidate_samples(dataset, classnames, args)
    if len(candidate_indices) < 2:
        raise RuntimeError("At least two held-out classes are required")

    checkpoint = base.checkpoint_path(args)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Trained LUNA checkpoint not found: {checkpoint}")

    print(f"[{args.dataset}] Building full fixed-image raw-logit cache")
    clip_model = base.load_clip_to_cpu(cfg).eval()
    model = base.CustomCLIP(cfg, classnames, clip_model).eval()
    base.load_luna_context(model, checkpoint)
    model.to(device)
    clip_model.to(device)

    preprocess = base.clip._transform(clip_model.visual.input_resolution)
    zero_text, luna_text = base.build_text_features(
        model, clip_model, classnames, device
    )
    zero_logits, luna_logits, records = collect_logits_parallel(
        records, model, zero_text, luna_text, preprocess, args
    )
    labels = np.asarray([record["full_index"] for record in records], dtype=np.int64)
    image_paths = np.asarray([record["item"].impath for record in records], dtype=np.str_)

    # Save atomically so an interrupted GPU run cannot leave a valid-looking cache.
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            dataset=np.asarray(args.dataset),
            seed=np.asarray(args.seed, dtype=np.int64),
            samples_per_class=np.asarray(args.samples_per_class, dtype=np.int64),
            split=np.asarray("base_val_plus_novel_test_complete_held_out"),
            feature=np.asarray("raw_scaled_classification_logits"),
            checkpoint=np.asarray(str(checkpoint)),
            zero_shot_logits=zero_logits.astype(np.float32),
            luna_logits=luna_logits.astype(np.float32),
            full_labels=labels,
            image_paths=image_paths,
            classnames=np.asarray(classnames, dtype=np.str_),
            candidate_full_indices=np.asarray(candidate_indices, dtype=np.int64),
        )
    temporary.replace(path)
    print(
        f"[{args.dataset}] Cached {len(records)} images, "
        f"{len(candidate_indices)} held-out classes -> {path}"
    )
    return load_cache(path, args)


def obtain_cache(args):
    destination = cache_path(args)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not args.force_cache:
        print(f"[{args.dataset}] Reusing cache {destination}")
        return load_cache(destination, args)
    return build_cache(destination, args)


def rows_by_class(labels, available):
    return {
        int(class_index): np.flatnonzero(labels == class_index)
        for class_index in available
    }


def restricted_view(cache, subset, grouped_rows):
    """Crop both samples (rows) and class-logit dimensions (columns)."""
    subset = tuple(sorted(int(value) for value in subset))
    row_indices = np.concatenate([grouped_rows[index] for index in subset])
    zero = cache["zero_shot_logits"][np.ix_(row_indices, subset)]
    luna = cache["luna_logits"][np.ix_(row_indices, subset)]
    full_labels = cache["full_labels"][row_indices]
    local_lookup = {full_index: local for local, full_index in enumerate(subset)}
    local_labels = np.fromiter(
        (local_lookup[int(label)] for label in full_labels),
        dtype=np.int64,
        count=len(full_labels),
    )
    return zero, luna, local_labels, row_indices


def centroid_metrics(features, labels, class_count):
    centers = []
    within_sum = 0.0
    within_count = 0
    for class_id in range(class_count):
        group = features[labels == class_id].astype(np.float64, copy=False)
        center = group.mean(axis=0)
        centers.append(center)
        residual = group - center
        within_sum += float(np.sum(residual * residual))
        within_count += len(group)
    centers = np.asarray(centers)
    within = within_sum / max(within_count, 1)
    delta = centers[:, None, :] - centers[None, :, :]
    squared = np.sum(delta * delta, axis=-1)
    triangle = squared[np.triu_indices(class_count, 1)]
    between = float(np.mean(triangle)) if triangle.size else 0.0
    ratio = between / max(within, 1e-12)

    sample_sq = np.sum(features.astype(np.float64) ** 2, axis=1, keepdims=True)
    center_sq = np.sum(centers**2, axis=1)[None, :]
    distances = sample_sq + center_sq - 2.0 * features.astype(np.float64) @ centers.T
    centroid_purity = float(np.mean(np.argmin(distances, axis=1) == labels))
    return ratio, centroid_purity


def restricted_accuracy(logits, labels):
    return float(np.mean(np.argmax(logits, axis=1) == labels))


def quick_metrics(cache, subset, grouped_rows):
    zero, luna, labels, _ = restricted_view(cache, subset, grouped_rows)
    class_count = len(subset)
    clip_ratio, clip_centroid = centroid_metrics(zero, labels, class_count)
    luna_ratio, luna_centroid = centroid_metrics(luna, labels, class_count)
    clip_accuracy = restricted_accuracy(zero, labels)
    luna_accuracy = restricted_accuracy(luna, labels)

    ratio_gain = math.log1p(luna_ratio) - math.log1p(clip_ratio)
    accuracy_gain = luna_accuracy - clip_accuracy
    centroid_gain = luna_centroid - clip_centroid
    luna_quality = (
        0.46 * luna_centroid
        + 0.38 * luna_accuracy
        + 0.16 * math.tanh(math.log1p(luna_ratio) / 2.0)
    )
    contrast_gain = 0.43 * centroid_gain + 0.39 * accuracy_gain + 0.18 * math.tanh(
        ratio_gain
    )
    # Contrast is primary, while the absolute term rejects two equally poor views.
    objective = 1.60 * contrast_gain + 0.32 * luna_quality
    return {
        "clip_ratio": clip_ratio,
        "luna_ratio": luna_ratio,
        "ratio_gain": ratio_gain,
        "clip_accuracy": clip_accuracy,
        "luna_accuracy": luna_accuracy,
        "accuracy_gain": accuracy_gain,
        "clip_centroid_purity": clip_centroid,
        "luna_centroid_purity": luna_centroid,
        "centroid_purity_gain": centroid_gain,
        "luna_quality": luna_quality,
        "contrast_gain": contrast_gain,
        "quick_objective": objective,
    }


def class_pair_scores(cache, available, grouped_rows):
    """Binary class-pair repair gain used to seed difficult class subsets."""
    available = list(map(int, available))
    count = len(available)
    score = np.full((count, count), -np.inf, dtype=np.float32)
    zero = cache["zero_shot_logits"]
    luna = cache["luna_logits"]
    for left in range(count):
        class_left = available[left]
        rows_left = grouped_rows[class_left]
        for right in range(left + 1, count):
            class_right = available[right]
            rows_right = grouped_rows[class_right]
            clip_accuracy = 0.5 * (
                np.mean(zero[rows_left, class_left] > zero[rows_left, class_right])
                + np.mean(zero[rows_right, class_right] > zero[rows_right, class_left])
            )
            luna_accuracy = 0.5 * (
                np.mean(luna[rows_left, class_left] > luna[rows_left, class_right])
                + np.mean(luna[rows_right, class_right] > luna[rows_right, class_left])
            )
            value = float(luna_accuracy - clip_accuracy + 0.16 * luna_accuracy)
            score[left, right] = value
            score[right, left] = value
    np.fill_diagonal(score, 0.0)
    return score


def pair_seed_subsets(available, pair_scores, class_count, seed_count):
    available = list(map(int, available))
    position = {value: index for index, value in enumerate(available)}
    upper = []
    for left in range(len(available)):
        for right in range(left + 1, len(available)):
            upper.append((float(pair_scores[left, right]), left, right))
    upper.sort(reverse=True)

    seeds = []
    for _, left, right in upper:
        selected = [available[left], available[right]]
        while len(selected) < class_count:
            selected_positions = [position[value] for value in selected]
            best_class = None
            best_score = -np.inf
            for candidate in available:
                if candidate in selected:
                    continue
                values = pair_scores[position[candidate], selected_positions]
                value = 0.70 * float(np.mean(values)) + 0.30 * float(np.min(values))
                if value > best_score:
                    best_score = value
                    best_class = candidate
            selected.append(best_class)
        subset = tuple(sorted(selected))
        if subset not in seeds:
            seeds.append(subset)
        if len(seeds) >= seed_count:
            break
    return seeds


def search_subsets(cache, args):
    labels = cache["full_labels"].astype(np.int64)
    available = [
        int(value)
        for value in cache["candidate_full_indices"]
        if np.sum(labels == value) >= 2
    ]
    class_count = min(args.class_count, len(available))
    if class_count < 2:
        raise RuntimeError("At least two populated classes are required")
    grouped_rows = rows_by_class(labels, available)
    rng = stable_rng(args.seed, args.dataset)

    print(f"[{args.dataset}] Computing class-pair repair gains")
    pair_scores = class_pair_scores(cache, available, grouped_rows)
    initial_pair_seeds = pair_seed_subsets(
        available,
        pair_scores,
        class_count,
        max(args.local_seeds * 2, args.candidates),
    )

    candidate_map = {}

    def add_candidate(subset, source):
        key = tuple(sorted(int(value) for value in subset))
        if len(key) != class_count or len(set(key)) != class_count:
            return None
        if key not in candidate_map:
            metrics = quick_metrics(cache, key, grouped_rows)
            metrics.update({"classes": key, "sources": {source}})
            candidate_map[key] = metrics
        else:
            candidate_map[key]["sources"].add(source)
        return candidate_map[key]

    for subset in initial_pair_seeds:
        add_candidate(subset, "pair_repair")

    # Half uniform and half pair-priority-weighted trials keeps broad coverage.
    finite_pairs = np.where(np.isfinite(pair_scores), pair_scores, 0.0)
    priority = np.mean(np.sort(finite_pairs, axis=1)[:, -min(8, len(available)) :], axis=1)
    priority = priority - priority.min() + 1e-3
    probability = 0.5 / len(available) + 0.5 * priority / priority.sum()
    for trial in range(args.search_trials):
        if trial % 2 == 0:
            subset = rng.choice(available, size=class_count, replace=False)
        else:
            subset = rng.choice(
                available, size=class_count, replace=False, p=probability
            )
        add_candidate(subset, "random")

    # Local swap repair starts from the strongest diverse quick candidates.
    ranked = sorted(
        candidate_map.values(), key=lambda row: row["quick_objective"], reverse=True
    )
    local_starts = []
    for row in ranked:
        subset = set(row["classes"])
        if all(
            len(subset & set(existing)) / len(subset | set(existing)) <= 0.80
            for existing in local_starts
        ):
            local_starts.append(row["classes"])
        if len(local_starts) >= args.local_seeds:
            break
    for start in local_starts:
        current = candidate_map[start]
        for _ in range(args.local_steps):
            best = current
            current_set = set(current["classes"])
            outside = [value for value in available if value not in current_set]
            # When every available class is selected (EuroSAT: 10/10), there
            # is exactly one valid subset and therefore nothing to swap.
            if not outside:
                break
            for _ in range(args.local_proposals):
                removed = int(rng.choice(list(current_set)))
                added = int(rng.choice(outside))
                proposed = tuple(sorted((current_set - {removed}) | {added}))
                row = add_candidate(proposed, "local_swap")
                if row["quick_objective"] > best["quick_objective"]:
                    best = row
            if best is current:
                break
            current = best

    rows = list(candidate_map.values())
    quality_threshold = float(
        np.quantile([row["luna_quality"] for row in rows], 0.60)
    )
    eligible = [row for row in rows if row["luna_quality"] >= quality_threshold]
    if len(eligible) < args.shortlist:
        eligible = rows

    # Round-robin objectives make the expensive shortlist less myopic.
    ranking_keys = (
        "quick_objective",
        "contrast_gain",
        "luna_quality",
        "ratio_gain",
        "accuracy_gain",
        "centroid_purity_gain",
    )
    rankings = [
        sorted(eligible, key=lambda row, key=key: row[key], reverse=True)
        for key in ranking_keys
    ]
    shortlist = []
    seen = set()
    depth = 0
    target = min(args.shortlist, len(eligible))
    while len(shortlist) < target:
        progressed = False
        for ranking in rankings:
            if depth >= len(ranking):
                continue
            row = ranking[depth]
            if row["classes"] not in seen:
                seen.add(row["classes"])
                shortlist.append(row)
                progressed = True
                if len(shortlist) >= target:
                    break
        depth += 1
        if not progressed and depth >= max(len(ranking) for ranking in rankings):
            break

    print(
        f"[{args.dataset}] {len(candidate_map)} unique quick trials; "
        f"computing exact raw metrics for {len(shortlist)} subsets"
    )
    for index, row in enumerate(shortlist, 1):
        zero, luna, local_labels, _ = restricted_view(
            cache, row["classes"], grouped_rows
        )
        row["clip_raw_silhouette"] = safe_silhouette(zero, local_labels)
        row["luna_raw_silhouette"] = safe_silhouette(luna, local_labels)
        row["raw_silhouette_gain"] = (
            row["luna_raw_silhouette"] - row["clip_raw_silhouette"]
        )
        row["clip_raw_knn_purity"] = knn_purity(zero, local_labels)
        row["luna_raw_knn_purity"] = knn_purity(luna, local_labels)
        row["raw_knn_gain"] = (
            row["luna_raw_knn_purity"] - row["clip_raw_knn_purity"]
        )
        row["exact_objective"] = (
            row["raw_silhouette_gain"]
            + 0.34 * row["luna_raw_silhouette"]
            + 0.16 * row["raw_knn_gain"]
            + 0.05 * row["luna_raw_knn_purity"]
        )
        print(
            f"[{args.dataset}] exact {index:02d}/{len(shortlist):02d}: "
            f"sil {row['clip_raw_silhouette']:+.3f} -> "
            f"{row['luna_raw_silhouette']:+.3f}"
        )

    selected = diversify_exact(shortlist, args)
    return selected, shortlist, grouped_rows, class_count, len(candidate_map)


def safe_silhouette(features, labels):
    if len(np.unique(labels)) < 2 or len(features) <= len(np.unique(labels)):
        return float("nan")
    return float(silhouette_score(features, labels, metric="euclidean"))


def knn_purity(features, labels, neighbors=5):
    count = len(features)
    if count < 3:
        return float("nan")
    k = min(neighbors, count - 1)
    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    model.fit(features)
    indices = model.kneighbors(features, return_distance=False)
    # The first neighbor is normally the point itself; remove it robustly.
    neighbor_labels = []
    for row_index, row in enumerate(indices):
        usable = [index for index in row if index != row_index][:k]
        neighbor_labels.append(labels[usable])
    neighbor_labels = np.asarray(neighbor_labels)
    return float(np.mean(neighbor_labels == labels[:, None]))


def diversify_exact(shortlist, args):
    luna_silhouettes = np.asarray(
        [row["luna_raw_silhouette"] for row in shortlist], dtype=np.float64
    )
    luna_knn = np.asarray(
        [row["luna_raw_knn_purity"] for row in shortlist], dtype=np.float64
    )
    silhouette_gate = float(
        np.nanquantile(luna_silhouettes, args.luna_quality_quantile)
    )
    knn_gate = float(np.nanquantile(luna_knn, 0.40))
    eligible = [
        row
        for row in shortlist
        if row["luna_raw_silhouette"] >= silhouette_gate
        and row["luna_raw_knn_purity"] >= knn_gate
    ]
    if len(eligible) < args.candidates:
        eligible = [
            row for row in shortlist if row["luna_raw_silhouette"] >= silhouette_gate
        ]
    if len(eligible) < args.candidates:
        eligible = shortlist
    ranked = sorted(eligible, key=lambda row: row["exact_objective"], reverse=True)

    selected = []
    for maximum_jaccard in (0.52, 0.62, 0.72, 0.82, 0.92, 1.01):
        for row in ranked:
            if row in selected:
                continue
            classes = set(row["classes"])
            if all(
                len(classes & set(other["classes"]))
                / len(classes | set(other["classes"]))
                <= maximum_jaccard
                for other in selected
            ):
                selected.append(row)
                if len(selected) >= min(args.candidates, len(ranked)):
                    return selected
    return selected


def reference_tsne(logits, seed, requested_perplexity):
    if not np.isfinite(logits).all():
        raise ValueError("Classification logits contain NaN or infinite values")
    perplexity = min(float(requested_perplexity), float(len(logits) - 1))
    # Intentionally only the arguments used by luna.py are supplied.
    estimator = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
    )
    return estimator.fit_transform(logits.astype(np.float32, copy=False))


def fit_candidate_embeddings(cache, selected, grouped_rows, args):
    results = []
    for index, row in enumerate(selected, 1):
        zero, luna, labels, row_indices = restricted_view(
            cache, row["classes"], grouped_rows
        )
        print(
            f"[{args.dataset}] candidate {index}/{len(selected)}: "
            "zero-shot CLIP t-SNE"
        )
        zero_embedding = reference_tsne(zero, args.seed, args.perplexity)
        print(
            f"[{args.dataset}] candidate {index}/{len(selected)}: LUNA t-SNE"
        )
        luna_embedding = reference_tsne(luna, args.seed, args.perplexity)

        result = dict(row)
        result.update(
            {
                "zero_logits": zero,
                "luna_logits": luna,
                "labels": labels,
                "row_indices": row_indices,
                "zero_embedding": zero_embedding,
                "luna_embedding": luna_embedding,
                "clip_2d_silhouette": safe_silhouette(zero_embedding, labels),
                "luna_2d_silhouette": safe_silhouette(luna_embedding, labels),
                "clip_2d_knn_purity": knn_purity(zero_embedding, labels),
                "luna_2d_knn_purity": knn_purity(luna_embedding, labels),
            }
        )
        result["2d_silhouette_gain"] = (
            result["luna_2d_silhouette"] - result["clip_2d_silhouette"]
        )
        result["2d_knn_gain"] = (
            result["luna_2d_knn_purity"] - result["clip_2d_knn_purity"]
        )
        result["visual_objective"] = (
            result["2d_silhouette_gain"]
            + 0.30 * result["luna_2d_silhouette"]
            + 0.15 * result["2d_knn_gain"]
            + 0.04 * result["luna_2d_knn_purity"]
        )
        results.append(result)
    return sorted(results, key=lambda row: row["visual_objective"], reverse=True)


def save_candidate_outputs(cache, results, shortlist, class_count, trial_count, args):
    destination = output_directory(args)
    candidate_dir = destination / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    classnames = [str(value) for value in cache["classnames"]]
    image_paths = cache["image_paths"]

    for rank, row in enumerate(results, 1):
        stem = f"candidate_{rank:02d}"
        figure_path = candidate_dir / f"{stem}.png"
        base.render_pair(
            row["zero_embedding"],
            row["luna_embedding"],
            row["labels"],
            base.DISPLAY_NAMES[args.dataset],
            figure_path,
        )
        with (candidate_dir / f"{stem}.npz").open("wb") as handle:
            np.savez_compressed(
                handle,
                zero_shot_logits=row["zero_logits"],
                luna_logits=row["luna_logits"],
                zero_shot_tsne=row["zero_embedding"],
                luna_tsne=row["luna_embedding"],
                display_labels=row["labels"],
                selected_full_indices=np.asarray(row["classes"], dtype=np.int64),
                image_paths=image_paths[row["row_indices"]],
            )
        with (candidate_dir / f"{stem}_classes.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["display_label", "full_class_index", "classname"])
            for display_label, full_index in enumerate(row["classes"]):
                writer.writerow(
                    [display_label, full_index, classnames[int(full_index)]]
                )
        row["rank"] = rank
        row["figure_path"] = str(figure_path)

    save_metrics(destination, results, classnames)
    save_selected_classes(destination, results, classnames)
    save_shortlist(destination, shortlist, classnames)
    save_contact_sheet(destination, results, args)

    metadata = {
        "dataset": args.dataset,
        "display_name": base.DISPLAY_NAMES[args.dataset],
        "seed": args.seed,
        "split": "complete held-out set reconstructed from base val plus novel test",
        "sample_policy": (
            f"fixed seed-{args.seed} path-level sample, at most "
            f"{args.samples_per_class} images per class; no result-based image selection"
        ),
        "class_count": class_count,
        "feature": "raw scaled classification logits restricted to selected classes",
        "zero_shot_prompt": "a photo of a {class}.",
        "quick_trial_count": trial_count,
        "exact_shortlist_count": len(shortlist),
        "rendered_candidate_count": len(results),
        "selection_goal": (
            "maximize LUNA-minus-CLIP separation subject to an absolute LUNA "
            "quality gate, followed by diverse candidate inspection"
        ),
        "tsne": {
            "constructor": (
                "TSNE(n_components=2, random_state=seed, "
                "perplexity=min(requested,n_samples-1))"
            ),
            "perplexity": args.perplexity,
            "softmax": False,
            "standardization": False,
            "external_pca": False,
            "shared_custom_initialization": False,
        },
        "auto_ranking_note": (
            "Candidate numbering is a screening rank only; inspect contact_sheet.png "
            "before choosing the paper figure."
        ),
        "cache": str(cache_path(args)),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[{args.dataset}] Saved candidate contact sheet -> {destination / 'contact_sheet.png'}")


METRIC_FIELDS = (
    "rank",
    "clip_raw_silhouette",
    "luna_raw_silhouette",
    "raw_silhouette_gain",
    "clip_raw_knn_purity",
    "luna_raw_knn_purity",
    "raw_knn_gain",
    "clip_accuracy",
    "luna_accuracy",
    "accuracy_gain",
    "clip_ratio",
    "luna_ratio",
    "ratio_gain",
    "clip_centroid_purity",
    "luna_centroid_purity",
    "centroid_purity_gain",
    "clip_2d_silhouette",
    "luna_2d_silhouette",
    "2d_silhouette_gain",
    "clip_2d_knn_purity",
    "luna_2d_knn_purity",
    "2d_knn_gain",
    "quick_objective",
    "exact_objective",
    "visual_objective",
    "sources",
    "selected_full_indices",
    "selected_classnames",
    "figure_path",
)


def serialisable_metric(value):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8f}"
    if isinstance(value, set):
        return ";".join(sorted(map(str, value)))
    return value


def save_metrics(destination, results, classnames):
    with (destination / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in results:
            output = {}
            for field in METRIC_FIELDS:
                if field == "selected_full_indices":
                    value = json.dumps(list(row["classes"]))
                elif field == "selected_classnames":
                    value = json.dumps(
                        [classnames[int(index)] for index in row["classes"]],
                        ensure_ascii=False,
                    )
                else:
                    value = row.get(field, "")
                output[field] = serialisable_metric(value)
            writer.writerow(output)


def save_selected_classes(destination, results, classnames):
    with (destination / "selected_classes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_rank", "display_label", "full_class_index", "classname"])
        for row in results:
            for display_label, full_index in enumerate(row["classes"]):
                writer.writerow(
                    [row["rank"], display_label, full_index, classnames[int(full_index)]]
                )


def save_shortlist(destination, shortlist, classnames):
    fields = [
        "clip_raw_silhouette",
        "luna_raw_silhouette",
        "raw_silhouette_gain",
        "clip_raw_knn_purity",
        "luna_raw_knn_purity",
        "raw_knn_gain",
        "quick_objective",
        "exact_objective",
        "selected_full_indices",
        "selected_classnames",
    ]
    ranked = sorted(shortlist, key=lambda row: row["exact_objective"], reverse=True)
    with (destination / "exact_shortlist.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            output = {field: serialisable_metric(row.get(field, "")) for field in fields}
            output["selected_full_indices"] = json.dumps(list(row["classes"]))
            output["selected_classnames"] = json.dumps(
                [classnames[int(index)] for index in row["classes"]],
                ensure_ascii=False,
            )
            writer.writerow(output)


def save_contact_sheet(destination, results, args):
    column_count = 2
    row_count = int(math.ceil(len(results) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(13.2, 3.75 * row_count),
        dpi=150,
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, results):
        with Image.open(row["figure_path"]) as image:
            axis.imshow(np.asarray(image.convert("RGB")))
        axis.set_title(
            (
                f"Candidate {row['rank']:02d} | raw silhouette "
                f"{row['clip_raw_silhouette']:+.3f} → "
                f"{row['luna_raw_silhouette']:+.3f} "
                f"(Δ {row['raw_silhouette_gain']:+.3f})\n"
                f"2D silhouette {row['clip_2d_silhouette']:+.3f} → "
                f"{row['luna_2d_silhouette']:+.3f} "
                f"(Δ {row['2d_silhouette_gain']:+.3f}); 2D kNN "
                f"{row['clip_2d_knn_purity']:.3f} → "
                f"{row['luna_2d_knn_purity']:.3f}"
            ),
            fontsize=10.0,
            fontweight="semibold",
            pad=5,
        )
    fig.suptitle(
        f"{base.DISPLAY_NAMES[args.dataset]} — class-subset screening",
        fontsize=14,
        fontweight="semibold",
        y=0.998,
    )
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.955, hspace=0.18, wspace=0.02)
    fig.savefig(destination / "contact_sheet.png", dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def validate_args(args):
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.class_count < 2:
        raise ValueError("class-count must be at least 2")
    if args.samples_per_class < 2:
        raise ValueError("samples-per-class must be at least 2")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.search_trials < 1 or args.shortlist < 1 or args.candidates < 1:
        raise ValueError("search-trials, shortlist, and candidates must be positive")
    if args.candidates > args.shortlist:
        raise ValueError("candidates cannot exceed shortlist")
    if not 0.0 <= args.luna_quality_quantile <= 1.0:
        raise ValueError("luna-quality-quantile must be in [0, 1]")
    if args.perplexity <= 0:
        raise ValueError("perplexity must be positive")


def main():
    args = parse_args()
    validate_args(args)
    cache = obtain_cache(args)
    selected, shortlist, grouped_rows, class_count, trial_count = search_subsets(
        cache, args
    )
    results = fit_candidate_embeddings(cache, selected, grouped_rows, args)
    save_candidate_outputs(
        cache, results, shortlist, class_count, trial_count, args
    )


if __name__ == "__main__":
    main()
