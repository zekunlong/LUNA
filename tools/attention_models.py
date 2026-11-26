"""Read trained prompt checkpoints without invoking their training pipelines.

All supported methods use the same frozen CLIP ViT. Differences in these
class-conditioned explanations therefore come from their *trained* text
contexts, not from raw image-only self-attention. No checkpoint is synthesized.

This module has no third-party imports at module scope, so preflight/rendering
can run in environments without PyTorch. ``load_attention_models`` returns a
shared CLIP model and unit-normalized, class-aligned text features.

Checkpoint specs are dictionaries, e.g.::

    {"name": "CLIP", "template": "a photo of a {}."}
    {"name": "CoOp", "checkpoint": ".../prompt_learner/model-best.pth.tar",
     "class_token_position": "end", "ctx_init": "a_photo_of_a"}
    {"name": "LUNA", "checkpoint": ".../VLPromptLearner/model-best.pth.tar",
     "checkpoint_classnames": ["class in row 0", "class in row 1"]}

Class-specific contexts require checkpoint_classnames in the spec or checkpoint
metadata; their rows are never inferred from the number of evaluation classes.
An exported .npz can instead supply text_features, classnames and metadata_json.
Its metadata must identify the method, source_checkpoint_sha256, backbone and
clip_sha256, and assert image_encoder='frozen_clip' and
inference='normalized_text_features'.

Inference references (inspected 2026-09-07):
https://github.com/KaiyangZhou/CoOp/blob/main/trainers/coop.py
https://github.com/htyao89/KgCoOp/blob/main/KgCoOp/trainers/kgcoop.py
https://github.com/cnunlp/DeKg/blob/main/trainers/DeKg.py
LUNA: trainers/luna.py, VLPromptLearner and CustomCLIP.
KgCoOp/DeKg regularizers are training-only; their official forward computes
logits directly from the learned ctx features. DeKg_TCP is a different model
and is intentionally unsupported by this checkpoint adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


METHODS = ("CLIP", "CoOp", "KgCoOp", "DeKg", "LUNA")
TEMPLATES = {
    "oxford_pets": "a photo of a {}, a type of pet.",
    "oxford_flowers": "a photo of a {}, a type of flower.",
    "fgvc_aircraft": "a photo of a {}, a type of aircraft.",
    "dtd": "{} texture.",
    "eurosat": "a centered satellite photo of {}.",
    "stanford_cars": "a photo of a {}.",
    "food101": "a photo of {}, a type of food.",
    "sun397": "a photo of a {}.",
    "caltech101": "a photo of a {}.",
    "ucf101": "a photo of a person doing {}.",
    "imagenet": "a photo of a {}.",
}
INFERENCE_SOURCES = {
    "CLIP": "https://github.com/KaiyangZhou/CoOp/blob/main/trainers/zsclip.py",
    "CoOp": "https://github.com/KaiyangZhou/CoOp/blob/main/trainers/coop.py",
    "KgCoOp": "https://github.com/htyao89/KgCoOp/blob/main/KgCoOp/trainers/kgcoop.py",
    "DeKg": "https://github.com/cnunlp/DeKg/blob/main/trainers/DeKg.py",
    "LUNA": "trainers/luna.py:VLPromptLearner,CustomCLIP",
}


@dataclass
class AttentionModels:
    clip_model: Any
    image_encoder: Any
    text_features: dict[str, Any]
    metadata: dict[str, Any]
    classnames: tuple[str, ...]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_method(value: str) -> str:
    matches = {name.lower(): name for name in METHODS}
    try:
        return matches[value.lower()]
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"Unsupported method {value!r}; expected {METHODS}") from exc


def preflight_specs(specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report missing inputs without loading torch or substituting a baseline."""
    result = []
    seen = set()
    for spec in specs:
        name = canonical_method(spec.get("name", ""))
        if name in seen:
            raise ValueError(f"Duplicate method specification: {name}")
        seen.add(name)
        if name == "CLIP":
            result.append({"name": name, "status": "ready", "source": "frozen_clip"})
            continue
        path_value = spec.get("checkpoint") or spec.get("text_features")
        path = Path(path_value).expanduser() if path_value else None
        result.append({"name": name,
                       "status": "ready" if path is not None and path.is_file() else "missing",
                       "source": str(path) if path is not None else None})
    return result


def _classes(names: Sequence[str], description: str) -> list[str]:
    if isinstance(names, (str, bytes)):
        raise ValueError(f"{description} must be an ordered list, not a string")
    values = [str(name).replace("_", " ") for name in names]
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{description} must contain distinct, nonempty class names")
    if any(not name.strip() for name in values):
        raise ValueError(f"{description} contains an empty class name")
    return values


def _load_checkpoint(path: Path):
    import torch

    # Dassl training checkpoints include scheduler objects as well as tensors.
    # This entry point is for trusted local experiment files, like the existing
    # trainer loader; it must not be used for untrusted uploaded pickle files.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch versions predating the weights_only parameter
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a state dictionary: {path}")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid state_dict in {path}")
    # DataParallel only changes this single known wrapper.
    state = {key.removeprefix("module."): value for key, value in state.items()}
    return checkpoint, state


def _load_clip(path: Path, backbone: str, device: str):
    import torch
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from clip import clip

    try:
        jit = torch.jit.load(str(path), map_location="cpu").eval()
        state = jit.state_dict()
    except RuntimeError:
        _, state = _load_checkpoint(path)
    if "visual.proj" not in state:
        raise ValueError("Attention comparisons currently require a CLIP ViT checkpoint")
    expected = {"ViT-B/16": (768, 16, 512, 12),
                "ViT-B/32": (768, 32, 512, 12),
                "ViT-L/14": (1024, 14, 768, 24)}
    if backbone not in expected:
        raise ValueError(f"Unsupported backbone: {backbone}")
    actual = (state["visual.conv1.weight"].shape[0],
              state["visual.conv1.weight"].shape[-1],
              state["visual.proj"].shape[-1],
              sum(key.startswith("visual.transformer.resblocks.") and
                  key.endswith("attn.in_proj_weight") for key in state))
    if actual != expected[backbone]:
        raise ValueError(f"{path} architecture {actual} does not match {backbone}")
    design = {"trainer": "CoOp", "vision_depth": 0, "language_depth": 0,
              "vision_ctx": 0, "language_ctx": 0}
    model = clip.build_model(dict(state), design)
    # build_model has a permissive fallback; enforce strictness independently.
    known = {key: value for key, value in state.items()
             if key not in ("input_resolution", "context_length", "vocab_size")}
    model.load_state_dict(known, strict=True)
    model.float().eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _check_frozen_encoder(state, clip_model, method):
    """Reject full checkpoints whose backbone differs from the shared CLIP."""
    import torch
    clip_state = clip_model.state_dict()
    checked = 0
    for key, value in state.items():
        reference_key = None
        if key.startswith("image_encoder."):
            reference_key = "visual." + key[len("image_encoder."):]
        elif key.startswith("text_encoder."):
            reference_key = key[len("text_encoder."):]
        elif key == "logit_scale":
            reference_key = key
        if reference_key is None:
            continue
        if reference_key not in clip_state:
            raise ValueError(f"{method} contains unsupported encoder parameter {key}")
        reference = clip_state[reference_key].detach().cpu().float()
        supplied = value.detach().cpu().float()
        # CLIP leaves layer norms in fp32; whole-model half conversion may round
        # them. Comparison in fp16 accepts only that normal precision change.
        if (reference.shape != supplied.shape or
                not torch.equal(reference.half(), supplied.half())):
            raise ValueError(f"{method} frozen backbone differs from supplied CLIP at {key}")
        checked += 1
    for key in state:
        if any(part in key.lower() for part in ("vpt", "compound_prompts", "prompt_deep")):
            raise ValueError(f"{method} uses unsupported deep/visual prompts: {key}")
    return checked


def _context_rows(ctx, classnames, declared_names, method):
    import torch
    if not torch.is_tensor(ctx) or ctx.ndim not in (2, 3):
        raise ValueError(f"{method} ctx must have shape [tokens,width] or [classes,tokens,width]")
    if not torch.isfinite(ctx).all():
        raise ValueError(f"{method} ctx contains nonfinite values")
    if ctx.ndim == 2:
        return ctx, "shared_context"
    if declared_names is None:
        raise ValueError(f"{method} has class-specific ctx; supply checkpoint_classnames in original row order")
    source_names = _classes(declared_names, f"{method} checkpoint_classnames")
    if len(source_names) != ctx.shape[0]:
        raise ValueError(f"{method} checkpoint_classnames length differs from ctx rows")
    lookup = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in classnames if name not in lookup]
    if missing:
        raise ValueError(f"{method} has no trained class-specific context for {missing[:8]}")
    return ctx[[lookup[name] for name in classnames]], "explicit_classnames"


def _encode_prompt_features(clip_model, classnames, ctx, position, ctx_init, batch_size):
    import torch
    from torch.nn import functional as F
    from clip import clip
    from clip.simple_tokenizer import SimpleTokenizer

    n_ctx = ctx.shape[-2]
    if n_ctx < 1 or ctx.shape[-1] != clip_model.ln_final.weight.shape[0]:
        raise ValueError("Trained context shape is incompatible with the CLIP text encoder")
    tokenizer = SimpleTokenizer()
    prefix = str(ctx_init).replace("_", " ") if ctx_init else " ".join(["X"] * n_ctx)
    if len(tokenizer.encode(prefix)) != n_ctx:
        raise ValueError("ctx_init BPE token count must equal saved context length")
    if position not in ("end", "middle", "front"):
        raise ValueError(f"Unsupported class_token_position: {position}")
    device = clip_model.token_embedding.weight.device
    outputs = []
    with torch.no_grad():
        for start in range(0, len(classnames), batch_size):
            names = classnames[start:start + batch_size]
            tokenized = clip.tokenize([f"{prefix} {name}." for name in names]).to(device)
            embedding = clip_model.token_embedding(tokenized)
            contexts = ctx if ctx.ndim == 2 else ctx[start:start + len(names)]
            contexts = contexts.to(device=device, dtype=embedding.dtype)
            if contexts.ndim == 2:
                contexts = contexts.unsqueeze(0).expand(len(names), -1, -1)
            prompts = []
            for i, name in enumerate(names):
                begin = embedding[i, :1]
                suffix = embedding[i, 1 + n_ctx:]
                name_len = len(tokenizer.encode(name))
                class_tokens, ending = suffix[:name_len], suffix[name_len:]
                context = contexts[i]
                if position == "end":
                    pieces = (begin, context, suffix)
                elif position == "front":
                    pieces = (begin, class_tokens, context, ending)
                else:
                    half = n_ctx // 2
                    pieces = (begin, context[:half], class_tokens, context[half:], ending)
                prompts.append(torch.cat(pieces, dim=0))
            x = torch.stack(prompts) + clip_model.positional_embedding
            x = clip_model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = clip_model.ln_final(x)
            features = x[torch.arange(len(names), device=device), tokenized.argmax(dim=-1)]
            outputs.append(F.normalize(features @ clip_model.text_projection, dim=-1))
    return torch.cat(outputs)


def _exported_features(spec, classnames, clip_hash, backbone, device):
    import numpy as np
    import torch
    from torch.nn import functional as F
    path = Path(spec["text_features"]).expanduser()
    with np.load(path, allow_pickle=False) as archive:
        required = {"text_features", "classnames", "metadata_json"}
        if not required <= set(archive.files):
            raise ValueError(f"{path} must include {sorted(required)}")
        names = _classes(archive["classnames"].tolist(), "export classnames")
        features = np.array(archive["text_features"], dtype=np.float32, copy=True)
        metadata = json.loads(str(archive["metadata_json"].item()))
    expected = {"method": canonical_method(spec["name"]), "backbone": backbone,
                "clip_sha256": clip_hash, "image_encoder": "frozen_clip",
                "inference": "normalized_text_features"}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{path}: metadata {key} must be {value!r}")
    source_hash = metadata.get("source_checkpoint_sha256", "")
    if len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
        raise ValueError(f"{path}: missing valid source_checkpoint_sha256 provenance")
    if features.ndim != 2 or features.shape[0] != len(names) or not np.isfinite(features).all():
        raise ValueError(f"{path}: invalid exported text features")
    if (np.linalg.norm(features, axis=-1) <= 1e-8).any():
        raise ValueError(f"{path}: zero text feature rows")
    lookup = {name: i for i, name in enumerate(names)}
    missing = [name for name in classnames if name not in lookup]
    if missing:
        raise ValueError(f"{path}: missing class features for {missing[:8]}")
    tensor = torch.from_numpy(features[[lookup[name] for name in classnames]]).to(device)
    metadata = {**metadata, "text_feature_file": str(path.resolve()),
                "text_feature_sha256": file_sha256(path), "source": "exported_features"}
    return F.normalize(tensor, dim=-1), metadata


def load_attention_models(classnames: Sequence[str], dataset: str,
                          specs: Sequence[Mapping[str, Any]], clip_path: str | Path,
                          device: str = "cuda", backbone: str = "ViT-B/16",
                          text_batch_size: int = 64) -> AttentionModels:
    """Load genuine method features in exactly the requested class order.

    Subsets are supported by passing the evaluation classnames. For LUNA or CSC
    CoOp, also pass the complete saved row order as checkpoint_classnames. No
    base/new split is guessed here. Missing inputs always raise an exception.
    """
    names = _classes(classnames, "evaluation classnames")
    if text_batch_size < 1:
        raise ValueError("text_batch_size must be positive")
    checks = preflight_specs(specs)
    missing = [row for row in checks if row["status"] == "missing"]
    if missing:
        raise FileNotFoundError("Missing trained method inputs: " + json.dumps(missing))
    clip_path = Path(clip_path).expanduser().resolve()
    if not clip_path.is_file():
        raise FileNotFoundError(f"CLIP checkpoint missing: {clip_path}")

    import torch
    from torch.nn import functional as F
    clip_model = _load_clip(clip_path, backbone, device)
    from clip import clip
    clip_hash = file_sha256(clip_path)
    features_by_method = {}
    metadata = {"backbone": backbone, "clip_path": str(clip_path),
                "clip_sha256": clip_hash, "inference_precision": "float32",
                "image_encoder": "frozen_clip", "methods": {}}
    for spec in specs:
        method = canonical_method(spec["name"])
        if spec.get("variant", method) != method:
            raise ValueError(f"Unsupported {method} variant: {spec['variant']}")
        if method == "CLIP":
            if spec.get("checkpoint") or spec.get("text_features"):
                raise ValueError("CLIP uses clip_path and a template, not a trained prompt checkpoint")
            template = spec.get("template", TEMPLATES.get(dataset))
            if not template or template.count("{}") != 1:
                raise ValueError("CLIP template must contain exactly one '{}' class placeholder")
            batches = []
            with torch.no_grad():
                for start in range(0, len(names), text_batch_size):
                    tokens = clip.tokenize([template.format(name)
                                           for name in names[start:start + text_batch_size]]).to(device)
                    batches.append(F.normalize(clip_model.encode_text(tokens), dim=-1))
            features, info = torch.cat(batches), {"source": "zero_shot", "template": template}
        elif spec.get("text_features"):
            if spec.get("checkpoint"):
                raise ValueError(f"{method}: use either checkpoint or text_features")
            features, info = _exported_features(spec, names, clip_hash, backbone, device)
        else:
            path = Path(spec["checkpoint"]).expanduser().resolve()
            checkpoint, state = _load_checkpoint(path)
            declared_method = checkpoint.get("method")
            if declared_method and canonical_method(declared_method) != method:
                raise ValueError(f"{path}: checkpoint method differs from {method}")
            checkpoint_config = checkpoint.get("configuration", {})
            if isinstance(checkpoint_config, dict):
                saved_clip_hash = checkpoint_config.get("clip_sha256")
                if saved_clip_hash and saved_clip_hash != clip_hash:
                    raise ValueError(f"{path}: trained with a different CLIP checkpoint")
            matches = [key for key in ("ctx", "prompt_learner.ctx") if key in state]
            if len(matches) != 1:
                raise ValueError(f"{path}: expected exactly one ctx or prompt_learner.ctx tensor")
            checked = _check_frozen_encoder(state, clip_model, method)
            declared_names = spec.get("checkpoint_classnames", checkpoint.get("classnames"))
            saved_ctx = state[matches[0]]
            ctx, alignment = _context_rows(saved_ctx, names, declared_names, method)
            if method == "LUNA" and (saved_ctx.ndim != 3 or saved_ctx.shape[-2] != 4):
                raise ValueError("This LUNA adapter requires the repository's class-specific 4-token student")
            position = spec.get("class_token_position", "end")
            if method != "CoOp" and position != "end":
                raise ValueError(f"Official {method} inference places the class token at the end")
            ctx_init = spec.get("ctx_init", checkpoint.get("ctx_init", ""))
            if method in ("KgCoOp", "DeKg") and ctx_init:
                # Both official implementations force this fixed initializer.
                ctx_init = "a photo of a"
            if method == "LUNA":
                ctx_init = ""
            features = _encode_prompt_features(clip_model, names, ctx, position,
                                               ctx_init, text_batch_size)
            info = {"source": "trained_checkpoint", "checkpoint": str(path),
                    "checkpoint_sha256": file_sha256(path), "epoch": checkpoint.get("epoch"),
                    "context_key": matches[0], "context_shape": list(saved_ctx.shape),
                    "class_alignment": alignment, "class_token_position": position,
                    "ctx_init": ctx_init, "checked_frozen_encoder_tensors": checked}
            if declared_names is not None:
                info["checkpoint_classnames"] = list(declared_names)
        if features.shape != (len(names), clip_model.visual.output_dim):
            raise ValueError(f"{method}: text feature dimensions do not match shared CLIP")
        if not torch.isfinite(features).all():
            raise ValueError(f"{method}: nonfinite text feature values")
        features_by_method[method] = features.detach()
        metadata["methods"][method] = {**info, "inference_source": INFERENCE_SOURCES[method]}
    return AttentionModels(clip_model, clip_model.visual, features_by_method,
                           metadata, tuple(classnames))
