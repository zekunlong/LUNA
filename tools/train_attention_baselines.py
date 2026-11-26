#!/usr/bin/env python3
"""Train real CoOp/KgCoOp/DeKg checkpoints for the attention comparison.

The default protocol follows the authors' ViT-B/16 ctxv1 configs: 4 shared
tokens initialized with 'a photo of a', base-class 16-shot seed 1, 100 epochs,
SGD .002/momentum .9/weight decay .0005, one constant warmup epoch, cosine LR.
CoOp uses batch 32; KgCoOp and DeKg use batch 128. KgCoOp weight 8 and DeKg
weights 8/6 follow the authors' scripts/run.sh. The final epoch is selected;
novel/test data are never used for gradient updates or checkpoint selection.

The manifest contains dataset, classnames in original label order, and
train/val/test lists of {impath, classname, label}. Original ordering is kept
when sampling all classes, before selecting the first ceil(C/2) base classes,
as in the upstream dataset loaders. No stored/cached image features replace
random resized crops during training.

Example:
  python tools/train_attention_baselines.py --manifest dataset.json \
      --method DeKg --clip-path ViT-B-16.pt --output-dir output/DeKg
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "Dassl.pytorch"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.attention_models import TEMPLATES, _load_clip, file_sha256

SOURCES = {
    "CoOp": ("KaiyangZhou/CoOp", ["trainers/coop.py",
                                  "configs/trainers/CoOp/vit_b16_ep100_ctxv1.yaml"]),
    "KgCoOp": ("htyao89/KgCoOp", ["KgCoOp/trainers/kgcoop.py",
                                    "KgCoOp/configs/trainers/KgCoOp/vit_b16_ep100_ctxv1.yaml",
                                    "KgCoOp/scripts/run.sh", "KgCoOp/scripts/base2new_train.sh"]),
    "DeKg": ("cnunlp/DeKg", ["trainers/DeKg.py", "trainers/loss.py",
                              "configs/trainers/DeKg/vit_b16_ep100_ctxv1.yaml",
                              "scripts/run.sh", "scripts/base2new_train.sh"]),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def archive_sources(method, source_root):
    """Pin the upstream commit and retain exact source bytes for an audit."""
    root = Path(source_root) / method
    receipt_path = root / "source_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for row in receipt["files"]:
            if file_sha256(root / row["relative_path"]) != row["sha256"]:
                raise ValueError(f"Archived upstream source has changed: {row['relative_path']}")
        return receipt
    repo, files = SOURCES[method]
    request = urllib.request.Request(f"https://api.github.com/repos/{repo}/commits/main",
                                     headers={"User-Agent": "LUNA-attention-reproduction"})
    with urllib.request.urlopen(request, timeout=60) as response:
        commit = json.load(response)["sha"]
    rows = []
    for relative in files:
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{relative}"
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        local = root / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(payload)
        rows.append({"relative_path": relative, "url": url,
                     "sha256": hashlib.sha256(payload).hexdigest()})
    receipt = {"method": method, "repository": f"https://github.com/{repo}",
               "commit": commit, "files": rows}
    write_json(receipt_path, receipt)
    return receipt


def read_manifest(path, shots, seed):
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    names = manifest["classnames"]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Manifest needs distinct classnames in official label order")
    for split in ("train", "val", "test"):
        if not manifest.get(split):
            raise ValueError(f"Manifest missing original {split} split")
        for row in manifest[split]:
            label = int(row["label"])
            if label < 0 or label >= len(names) or row["classname"] != names[label]:
                raise ValueError(f"Label/classname mismatch in {split}: {row}")
            if not Path(row["impath"]).is_file():
                raise FileNotFoundError(row["impath"])
    # Split manifests must not accidentally leak a test file into train/val.
    path_sets = [{str(Path(row["impath"]).resolve()).lower() for row in manifest[s]}
                 for s in ("train", "val", "test")]
    if any(path_sets[i] & path_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Original train/val/test image paths overlap")
    rng = random.Random(seed)
    def sample(rows, count):
        groups = defaultdict(list)
        for row in rows:
            groups[int(row["label"])].append(row)
        if set(groups) != set(range(len(names))):
            raise ValueError("Original split does not cover all class labels")
        sampled = []
        for label, rows in groups.items():
            if len(rows) < count:
                raise ValueError(f"Class {label} has only {len(rows)} samples, fewer than {count}")
            sampled.extend(rng.sample(rows, count))
        return sampled
    train_all = sample(manifest["train"], shots)
    val_all = sample(manifest["val"], min(shots, 4))
    base_count = math.ceil(len(names) / 2)
    train = [row for row in train_all if int(row["label"]) < base_count]
    val = [row for row in val_all if int(row["label"]) < base_count]
    return manifest, train, val, base_count


def hsic_loss(learned, crafted):
    """Exact published DeKg HSICLoss, including global Frobenius normalization."""
    import torch
    learned = learned / torch.norm(learned)
    crafted = crafted / torch.norm(crafted)
    count = learned.size(0)
    h = (torch.eye(count, device=learned.device, dtype=learned.dtype) -
         (1 / count) * torch.ones(count, count, device=learned.device, dtype=learned.dtype))
    k_learned = torch.mm(learned, learned.t())
    k_crafted = torch.mm(crafted, crafted.t())
    return torch.trace(torch.mm(h, torch.mm(k_learned, torch.mm(h, k_crafted))))


def make_prompt_encoder(clip_model, classnames):
    """A shared shallow CoOp ctx; only this parameter receives gradients."""
    import torch
    from torch import nn
    from clip import clip

    class PromptEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.clip_model = clip_model
            self.n_ctx = 4
            names = [name.replace("_", " ") for name in classnames]
            device = clip_model.token_embedding.weight.device
            tokens = clip.tokenize([f"a photo of a {name}." for name in names]).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokens).type(clip_model.dtype)
                initial = clip_model.token_embedding(clip.tokenize("a photo of a").to(device))
            self.ctx = nn.Parameter(initial[0, 1:5].type(clip_model.dtype).clone())
            self.register_buffer("tokens", tokens)
            self.register_buffer("prefix", embedding[:, :1].clone())
            self.register_buffer("suffix", embedding[:, 5:].clone())

        def forward(self, indices=None):
            if indices is None:
                tokens, prefix, suffix = self.tokens, self.prefix, self.suffix
            else:
                tokens, prefix, suffix = self.tokens[indices], self.prefix[indices], self.suffix[indices]
            ctx = self.ctx.unsqueeze(0).expand(len(tokens), -1, -1)
            x = torch.cat((prefix, ctx, suffix), dim=1)
            model = self.clip_model
            x = x + model.positional_embedding.type(model.dtype)
            x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = model.ln_final(x).type(model.dtype)
            x = x[torch.arange(len(tokens), device=tokens.device), tokens.argmax(dim=-1)]
            features = x @ model.text_projection
            return features / features.norm(dim=-1, keepdim=True)

    return PromptEncoder()


def verify_implementation(encoder, method, source_root):
    """Check the prompt forward against native CLIP and HSIC value AND gradient."""
    import torch
    import torch.nn.functional as F
    from clip import clip
    model = encoder.clip_model
    with torch.no_grad():
        actual = encoder()[:min(8, len(encoder.tokens))]
        native = model.encode_text(encoder.tokens[:len(actual)])
        native = native / native.norm(dim=-1, keepdim=True)
        error = (actual - native).abs().max().item()
    tolerance = 2e-3 if model.dtype == torch.float16 else 1e-6
    if error > tolerance:
        raise ValueError(f"Prompt inference differs from native CLIP: {error}")
    result = {"native_clip_initialization_max_abs_error": error,
              "native_clip_tolerance": tolerance}
    if method == "DeKg":
        path = Path(source_root) / "DeKg/trainers/loss.py"
        spec = importlib.util.spec_from_file_location("official_dekg_losses", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        generator = torch.Generator(device=model.token_embedding.weight.device).manual_seed(418)
        left = torch.randn((7, 19), device=model.token_embedding.weight.device,
                           dtype=model.dtype, generator=generator).requires_grad_()
        right = torch.randn((7, 19), device=left.device, dtype=left.dtype, generator=generator)
        own = hsic_loss(left, right)
        reference = module.HSICLoss()(left, right)
        own_grad, = torch.autograd.grad(own, left, retain_graph=True)
        reference_grad, = torch.autograd.grad(reference, left)
        value_error = abs(float(own) - float(reference))
        gradient_error = (own_grad - reference_grad).abs().max().item()
        if value_error != 0 or gradient_error != 0:
            raise ValueError("DeKg HSIC differs from the pinned official implementation")
        result.update(hsic_value_error=value_error, hsic_gradient_max_error=gradient_error)
    return result


class ImageRecords:
    def __init__(self, rows, transform, cache_images=False):
        self.rows = rows
        self.transform = transform
        self.cached_images = None
        if cache_images:
            from PIL import Image
            self.cached_images = []
            for row in rows:
                with Image.open(row["impath"]) as source:
                    decoded = source.convert("RGB")
                    decoded.load()
                self.cached_images.append(decoded)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        row = self.rows[index]
        if self.cached_images is not None:
            # Only decode/I/O is cached. Every call still draws a fresh random
            # crop/flip, and a copy isolates the immutable decoded original.
            image = self.transform(self.cached_images[index].copy())
        else:
            with Image.open(row["impath"]) as source:
                image = self.transform(source.convert("RGB"))
        return image, int(row["label"])


def atomic_checkpoint(path, checkpoint):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def _rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _scheduler_state(scheduler):
    state = scheduler.state_dict()
    # Dassl's wrapper puts the live successor scheduler (and its optimizer) in
    # state_dict. Save scalar state separately to preserve the optimizer binding
    # when resuming in a new process.
    state.pop("successor", None)
    return {"wrapper": state, "successor": scheduler.successor.state_dict()}


def _restore_scheduler(scheduler, state):
    scheduler.load_state_dict(state["wrapper"])
    scheduler.successor.load_state_dict(state["successor"])


def evaluate(encoder, rows, base_count, batch_size, device, transform, output_dir):
    import torch
    from torch.utils.data import DataLoader
    from torch.nn import functional as F
    model = encoder.clip_model
    loader = DataLoader(ImageRecords(rows, transform), batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=str(device).startswith("cuda"))
    base_correct = new_correct = base_total = new_total = 0
    predictions = []
    with torch.no_grad():
        text = encoder()
        for images, labels in loader:
            feature = model.encode_image(images.to(device=device, dtype=model.dtype))
            feature = feature / feature.norm(dim=-1, keepdim=True)
            similarity = feature @ text.T
            for sim, label in zip(similarity, labels.tolist()):
                is_base = label < base_count
                predicted = int(sim[:base_count].argmax()) if is_base else base_count + int(sim[base_count:].argmax())
                predictions.append({"sample_index": len(predictions), "label": label,
                                    "prediction": predicted, "correct": predicted == label,
                                    "class_split": "base" if is_base else "new"})
                if is_base:
                    base_total += 1
                    base_correct += predicted == label
                else:
                    new_total += 1
                    new_correct += predicted == label
    base = 100.0 * base_correct / base_total
    new = 100.0 * new_correct / new_total
    metrics = {"base_accuracy": base, "new_accuracy": new,
               "harmonic_mean": 2 * base * new / (base + new) if base + new else 0.0,
               "base_samples": base_total, "new_samples": new_total,
               "checkpoint_selection": "final_epoch", "test_used_for_training": False}
    write_json(Path(output_dir) / "final_metrics.json", metrics)
    with (Path(output_dir) / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(SOURCES), required=True)
    parser.add_argument("--clip-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=REPO.parents[1] / "external/attention_baselines")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-images", action="store_true",
                        help="Cache decoded training originals in RAM; every access still applies fresh augmentation")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Save the final trained checkpoint and defer test evaluation to the comparison runner")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--eval-batch-size", type=int, default=100)
    parser.add_argument("--kg-weight", type=float, default=8.0)
    parser.add_argument("--hsic-weight", type=float, default=6.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Stop at this epoch for a resumability smoke check; status remains INCOMPLETE")
    return parser.parse_args()


def main(args=None):
    args = args or parse_args()
    if args.epochs < 1 or args.shots < 1 or args.workers < 0:
        raise ValueError("Epochs/shots must be positive and workers nonnegative")
    if args.cache_images and args.workers:
        raise ValueError("--cache-images requires --workers 0 to avoid duplicating the RAM cache in Windows workers")
    batch_size = args.batch_size or (32 if args.method == "CoOp" else 128)
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest, train_rows, val_rows, base_count = read_manifest(args.manifest, args.shots, args.seed)
    source = archive_sources(args.method, args.source_root)
    protocol = {"method": args.method, "dataset": manifest["dataset"],
                "backbone": "ViT-B/16", "clip_path": str(args.clip_path.resolve()),
                "clip_sha256": file_sha256(args.clip_path), "seed": args.seed,
                "shots": args.shots, "epochs": args.epochs, "batch_size": batch_size,
                "precision": args.precision, "workers": args.workers,
                "cache_decoded_training_images": args.cache_images,
                "defer_final_test_evaluation": args.skip_eval,
                "learning_rate": .002, "momentum": .9, "weight_decay": .0005,
                "warmup_epoch": 1, "warmup_lr": 1e-5, "schedule": "Dassl_constant_warmup_cosine",
                "context_tokens": 4, "ctx_init": "a photo of a", "class_token_position": "end",
                "class_specific_context": False, "base_count": base_count,
                "classnames": manifest["classnames"], "kg_weight": 0 if args.method == "CoOp" else args.kg_weight,
                "hsic_weight": args.hsic_weight if args.method == "DeKg" else 0,
                "manifest_sha256": file_sha256(args.manifest), "training_samples": train_rows,
                "sampled_validation_samples": val_rows, "training_selection": "upstream_all_classes_then_base",
                "test_used_for_training": False, "checkpoint_selection": "final_epoch",
                "source": source, "implementation_sha256": file_sha256(__file__),
                "attention_models_sha256": file_sha256(REPO / "tools/attention_models.py")}
    fingerprint = json_hash(protocol)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old.get("protocol_sha256") != fingerprint:
            raise ValueError("Existing output has a different protocol; use a distinct output directory")
    write_json(protocol_path, {"protocol_sha256": fingerprint, **protocol})
    if args.prepare_only:
        print(json.dumps({"status": "PREPARED", "protocol": str(protocol_path),
                          "train_samples": len(train_rows), "base_classes": base_count}), flush=True)
        return

    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    from dassl.config import get_cfg_default
    from dassl.optim import build_lr_scheduler
    from clip import clip
    from clip.model import convert_weights
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    model = _load_clip(args.clip_path.resolve(), "ViT-B/16", args.device)
    if args.precision == "fp16":
        if not str(args.device).startswith("cuda"):
            raise ValueError("fp16 training requires a CUDA device")
        convert_weights(model)
    encoder = make_prompt_encoder(model, manifest["classnames"]).to(args.device)
    validation = verify_implementation(encoder, args.method, args.source_root)
    write_json(output / "implementation_validation.json", validation)
    if args.verify_only:
        print(json.dumps({"status": "VERIFIED", **validation}), flush=True)
        return
    mean, std = (.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711)
    training_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(.08, 1.), interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    # Both upstream test transforms and CLIP use smaller-edge resize + center crop.
    test_transform = clip._transform(224)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(ImageRecords(train_rows, training_transform, args.cache_images),
                              batch_size=batch_size, shuffle=True,
                              drop_last=len(train_rows) >= batch_size,
                              num_workers=args.workers, pin_memory=str(args.device).startswith("cuda"),
                              generator=loader_generator)
    optimizer = torch.optim.SGD([encoder.ctx], lr=.002, momentum=.9, weight_decay=.0005)
    optim_cfg = get_cfg_default().OPTIM.clone()
    optim_cfg.LR_SCHEDULER = "cosine"
    optim_cfg.MAX_EPOCH = args.epochs
    optim_cfg.WARMUP_EPOCH = 1
    optim_cfg.WARMUP_TYPE = "constant"
    optim_cfg.WARMUP_CONS_LR = 1e-5
    scheduler = build_lr_scheduler(optimizer, optim_cfg)
    with torch.no_grad():
        template = TEMPLATES[manifest["dataset"]]
        # Official KgCoOp/DeKg differ from ZeroshotCLIP for DTD specifically.
        if manifest["dataset"] == "dtd":
            template = "a photo of a {}, a type of texture."
        texts = [template.format(name.replace("_", " ")) for name in manifest["classnames"][:base_count]]
        crafted = model.encode_text(clip.tokenize(texts).to(args.device))
        crafted = crafted / crafted.norm(dim=-1, keepdim=True)
    base_indices = torch.arange(base_count, device=args.device)
    last_checkpoint = output / "checkpoint_last.pth"
    start_epoch = 0
    elapsed_before = 0.0
    if last_checkpoint.exists():
        if not args.resume:
            raise ValueError("Existing training checkpoint; pass --resume to continue it")
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        if checkpoint["protocol_sha256"] != fingerprint:
            raise ValueError("Resume checkpoint protocol mismatch")
        encoder.ctx.data.copy_(checkpoint["state_dict"]["ctx"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        _restore_scheduler(scheduler, checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        elapsed_before = checkpoint.get("training_seconds", 0.0)
        loader_generator.set_state(checkpoint["loader_rng"])
        _restore_rng(checkpoint["rng"])
    start_time = time.monotonic()
    scalar_path = output / "epochs.csv"
    scalar_fields = ["epoch", "lr", "loss", "ce_loss", "kg_loss", "hsic_loss", "train_accuracy",
                     "samples", "batches", "ctx_gradient_norm", "epoch_seconds", "training_seconds"]
    if start_epoch and scalar_path.exists():
        with scalar_path.open(newline="", encoding="utf-8") as handle:
            existing = [row for row in csv.DictReader(handle) if int(row["epoch"]) <= start_epoch]
        with scalar_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_fields)
            writer.writeheader()
            writer.writerows(existing)
    write_json(output / "status.json", {"status": "RUNNING", "epoch": start_epoch,
                                       "target_epochs": args.epochs, "protocol_sha256": fingerprint})
    stop = min(args.epochs, args.stop_after_epoch) if args.stop_after_epoch else args.epochs
    for epoch in range(start_epoch, stop):
        epoch_start = time.monotonic()
        # Frozen CLIP remains in eval mode; ctx has no stochastic modules.
        model.eval()
        totals = np.zeros(4, dtype=np.float64)
        samples = correct = batches = 0
        grad_norm = 0.0
        lr = optimizer.param_groups[0]["lr"]
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            with torch.no_grad():
                image_features = model.encode_image(images.type(model.dtype))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            learned = encoder(base_indices)
            logits = model.logit_scale.exp() * image_features @ learned.T
            ce = F.cross_entropy(logits, labels)
            kg = 1.0 - torch.nn.CosineSimilarity(dim=1, eps=1e-7)(learned, crafted).mean()
            hsic = hsic_loss(learned, crafted) if args.method == "DeKg" else ce.new_zeros(())
            loss = ce + protocol["kg_weight"] * kg + protocol["hsic_weight"] * hsic
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {args.method} loss at epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if encoder.ctx.grad is None or not torch.isfinite(encoder.ctx.grad).all():
                raise FloatingPointError("Missing/nonfinite context gradient")
            grad_norm += encoder.ctx.grad.float().norm().item()
            optimizer.step()
            size = labels.numel()
            totals += np.array([float(loss), float(ce), float(kg), float(hsic)]) * size
            correct += int((logits.argmax(dim=-1) == labels).sum())
            samples += size
            batches += 1
        scheduler.step()
        elapsed = elapsed_before + time.monotonic() - start_time
        row = dict(zip(("loss", "ce_loss", "kg_loss", "hsic_loss"), (totals / samples).tolist()))
        row.update(epoch=epoch + 1, lr=lr, train_accuracy=100.0 * correct / samples,
                   samples=samples, batches=batches, ctx_gradient_norm=grad_norm / batches,
                   epoch_seconds=time.monotonic() - epoch_start, training_seconds=elapsed)
        fresh = not scalar_path.exists()
        with scalar_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_fields)
            if fresh:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
        checkpoint = {"state_dict": {"ctx": encoder.ctx.detach().cpu()},
                      "epoch": epoch + 1, "method": args.method, "classnames": manifest["classnames"],
                      "optimizer": optimizer.state_dict(), "scheduler": _scheduler_state(scheduler),
                      "rng": _rng_state(), "loader_rng": loader_generator.get_state(),
                      "protocol_sha256": fingerprint, "training_seconds": elapsed,
                      "configuration": protocol}
        atomic_checkpoint(last_checkpoint, checkpoint)
        write_json(output / "status.json", {"status": "RUNNING", "epoch": epoch + 1,
                                           "target_epochs": args.epochs, "latest": row,
                                           "protocol_sha256": fingerprint})
        print(json.dumps({"method": args.method, "dataset": manifest["dataset"], **row}), flush=True)
    if stop < args.epochs:
        write_json(output / "status.json", {"status": "INCOMPLETE", "epoch": stop,
                                           "target_epochs": args.epochs, "protocol_sha256": fingerprint})
        return
    compact_path = output / "attention_checkpoint.pth"
    atomic_checkpoint(compact_path, {"state_dict": {"ctx": encoder.ctx.detach().cpu()},
                                    "method": args.method, "epoch": args.epochs,
                                    "classnames": manifest["classnames"], "configuration": protocol,
                                    "protocol_sha256": fingerprint, "ctx_init": "a photo of a"})
    metrics = ({"evaluation_status": "DEFERRED_TO_COMPARISON_RUNNER"} if args.skip_eval else
               evaluate(encoder, manifest["test"], base_count, args.eval_batch_size,
                        args.device, test_transform, output))
    write_json(output / "status.json", {"status": "COMPLETED_REAL_TRAINING",
                                       "method": args.method, "dataset": manifest["dataset"],
                                       "epoch": args.epochs, "protocol_sha256": fingerprint,
                                       "checkpoint": str(compact_path),
                                       "checkpoint_sha256": file_sha256(compact_path),
                                       "metrics": metrics, "implementation_validation": validation})
    print(json.dumps({"status": "COMPLETED_REAL_TRAINING", "checkpoint": str(compact_path),
                      **metrics}), flush=True)


if __name__ == "__main__":
    main()
