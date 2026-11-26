"""Read-only asset and environment checks for the separate journal experiments.

READY means that prerequisites passed preflight, not that a checkpoint's contents
have been authenticated. The trainer must still validate its saved epoch before
training. This module never deserializes checkpoints or regenerates data splits.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys


_DATASETS = {
    "dtd": ("dtd", "images", "split_zhou_DescribableTextures.json"),
    "fgvc_aircraft": ("fgvc_aircraft", "images", None),
    "eurosat": ("eurosat", "2750", "split_zhou_EuroSAT.json"),
}
_ALIASES = {
    "dtd": "dtd", "describabletextures": "dtd",
    "aircraft": "fgvc_aircraft", "fgvcaircraft": "fgvc_aircraft",
    "eurosat": "eurosat",
}
_PACKAGES = {
    "torch": ("torch", "2.0.1"),
    "torchvision": ("torchvision", "0.15.2"),
    "numpy": ("numpy", "1.24.2"),
    "ftfy": ("ftfy", "6.1.1"),
    "Pillow": ("PIL", None),
    "regex": ("regex", None),
    "tqdm": ("tqdm", None),
    "yacs": ("yacs", None),
    "gdown": ("gdown", None),
    "tabulate": ("tabulate", None),
    "scipy": ("scipy", None),
    "scikit-learn": ("sklearn", None),
    "matplotlib": ("matplotlib", None),
    "tensorboard": ("tensorboard", None),
    # Bundled Dassl imports its dataset registry even for DTD; this registry
    # imports the WILDS wrappers declared in Dassl.pytorch/requirements.txt.
    "wilds": ("wilds", "1.2.2"),
    "future": ("future", None),
}


def _canonical_dataset(value):
    if not isinstance(value, str):
        return None
    return _ALIASES.get(re.sub(r"[\s_-]", "", value).lower())


def _input(value, env_name):
    if value is not None:
        return value, "argument"
    if env_name in os.environ:
        return os.environ[env_name], "environment:" + env_name
    return None, "unset"


def _path_asset(value, env_name, kind, label, blockers):
    value, source = _input(value, env_name)
    result = {"path": None, "source": source, "exists": False, "kind": kind}
    if value is None or not str(value).strip():
        blockers.append(label + " is missing; supply an explicit path or " + env_name)
        return result
    try:
        path = Path(value).expanduser().resolve()
        result["path"] = str(path)
        result["exists"] = path.is_dir() if kind == "directory" else path.is_file()
        if not result["exists"]:
            blockers.append(label + " does not exist as a " + kind + ": " + str(path))
        elif kind == "file":
            result["size_bytes"] = path.stat().st_size
            if result["size_bytes"] == 0:
                blockers.append(label + " is empty: " + str(path))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        blockers.append(label + " has an invalid or inaccessible path: " + str(exc))
    return result


def _read_manifest(path, blockers):
    """Read existing text only; retain a hash so callers can record the split."""
    entry = {"path": str(path), "exists": path.is_file()}
    if not entry["exists"]:
        blockers.append("Required original split metadata is missing: " + str(path))
        return None, entry
    try:
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("split metadata exceeds the 64 MiB inspection limit")
        raw = path.read_bytes()
        entry.update(size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return raw.decode("utf-8-sig"), entry
    except (OSError, UnicodeError, ValueError) as exc:
        blockers.append("Cannot read original split metadata " + str(path) + ": " + str(exc))
        return None, entry


def _relative_image(image_root, reference):
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("image reference must be a nonempty relative path")
    portable = reference.replace("\\", "/")
    windows_path = PureWindowsPath(reference)
    if portable.startswith("/") or windows_path.drive or ".." in portable.split("/"):
        raise ValueError("image reference must stay inside the dataset image directory")
    candidate = (image_root / portable).resolve()
    if not candidate.is_relative_to(image_root.resolve()):
        raise ValueError("image reference resolves outside the dataset image directory")
    return candidate


def _inspect_records(records, image_root, blockers, entry):
    labels, classnames, seen, errors = {}, {}, {}, []
    missing_count, missing_examples, duplicate_count = 0, [], 0
    counts = {}
    for split in ("train", "val", "test"):
        rows = records.get(split) if isinstance(records, dict) else None
        if not isinstance(rows, list) or not rows:
            errors.append(split + " must contain a nonempty list")
            continue
        counts[split] = len(rows)
        for index, row in enumerate(rows):
            context = split + "[" + str(index) + "]"
            try:
                if not isinstance(row, (list, tuple)) or len(row) != 3:
                    raise ValueError("expected [relative_image_path, integer_label, classname]")
                reference, label, classname = row
                if isinstance(label, bool) or not isinstance(label, int) or label < 0:
                    raise ValueError("label must be a nonnegative integer")
                if not isinstance(classname, str) or not classname.strip():
                    raise ValueError("classname must be a nonempty string")
                if label in labels and labels[label] != classname:
                    raise ValueError("a label has conflicting classnames")
                if classname in classnames and classnames[classname] != label:
                    raise ValueError("a classname has conflicting labels")
                labels[label], classnames[classname] = classname, label
                candidate = _relative_image(image_root, reference)
                if str(candidate) in seen:
                    duplicate_count += 1
                else:
                    seen[str(candidate)] = split
                if not candidate.is_file():
                    missing_count += 1
                    if len(missing_examples) < 3:
                        missing_examples.append(str(candidate))
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                if len(errors) < 12:
                    errors.append(context + ": " + str(exc))
    entry.update(split_counts=counts, class_count=len(labels),
                 unique_image_count=len(seen), missing_image_count=missing_count,
                 duplicate_reference_count=duplicate_count)
    if errors:
        blockers.append("Invalid original split records: " + "; ".join(errors))
    if missing_count:
        blockers.append("Original split references " + str(missing_count) +
                        " missing image(s); examples: " + ", ".join(missing_examples))
    if duplicate_count:
        blockers.append("Original splits contain " + str(duplicate_count) +
                        " repeated image reference(s); no split repair was attempted")


def _inspect_dataset(dataset, root, blockers):
    directory, image_subdir, split_name = _DATASETS[dataset]
    dataset_root = root / directory
    image_root = dataset_root / image_subdir
    result = {"path": str(dataset_root), "image_root": str(image_root),
              "images_exist": image_root.is_dir(), "manifests": []}
    if not image_root.is_dir():
        blockers.append("Dataset image directory is missing: " + str(image_root))
    if split_name:
        raw, manifest = _read_manifest(dataset_root / split_name, blockers)
        result["manifests"].append(manifest)
        if raw is not None:
            try:
                records = json.loads(raw)
            except (ValueError, RecursionError) as exc:
                blockers.append("Invalid existing split JSON: " + str(exc))
            else:
                _inspect_records(records, image_root, blockers, result)
        return result

    raw, manifest = _read_manifest(dataset_root / "variants.txt", blockers)
    result["manifests"].append(manifest)
    variants = [] if raw is None else [line.strip() for line in raw.splitlines() if line.strip()]
    if raw is not None and (not variants or len(set(variants)) != len(variants)):
        blockers.append("Aircraft variants.txt must contain distinct nonempty class names")
    class_to_label = {name: index for index, name in enumerate(variants)}
    records = {}
    for split in ("train", "val", "test"):
        raw, manifest = _read_manifest(dataset_root / ("images_variant_" + split + ".txt"), blockers)
        result["manifests"].append(manifest)
        records[split] = []
        if raw is None:
            continue
        invalid_lines = []
        for index, line in enumerate(raw.splitlines(), 1):
            pieces = line.strip().split(maxsplit=1)
            if len(pieces) != 2 or pieces[1] not in class_to_label:
                if len(invalid_lines) < 3:
                    invalid_lines.append(str(index))
                continue
            image_id, classname = pieces
            if "/" in image_id or "\\" in image_id or ":" in image_id or image_id in (".", ".."):
                if len(invalid_lines) < 3:
                    invalid_lines.append(str(index))
                continue
            records[split].append([image_id + ".jpg", class_to_label[classname], classname])
        if invalid_lines:
            blockers.append("Invalid Aircraft " + split + " metadata at line(s): " + ", ".join(invalid_lines))
    _inspect_records(records, image_root, blockers, result)
    return result


def _inspect_environment(blockers):
    result = {"checked": True, "python": sys.version.split()[0],
              "python_required": "3.9.x", "packages": {}}
    if sys.version_info[:2] != (3, 9):
        blockers.append("Environment requires README Python 3.9.x; found " + result["python"])
    for distribution, (module, required) in _PACKAGES.items():
        item = {"module": module, "required_version": required}
        try:
            item["installed_version"] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            item["installed_version"] = None
            blockers.append("Required package is not installed: " + distribution)
        try:
            item["module_available"] = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError, AttributeError):
            item["module_available"] = False
        if not item["module_available"]:
            blockers.append("Required import is unavailable: " + module)
        # CUDA local version suffixes are valid builds of the same pinned release.
        installed = item["installed_version"]
        if required and installed and installed.split("+", 1)[0] != required:
            blockers.append(distribution + " must be " + required + "; found " + installed)
        result["packages"][distribution] = item

    bundled = Path(__file__).resolve().parents[1] / "Dassl.pytorch" / "dassl" / "__init__.py"
    result["bundled_dassl"] = {"path": str(bundled), "exists": bundled.is_file()}
    if not bundled.is_file():
        blockers.append("Bundled Dassl package is missing: " + str(bundled))

    # Actual imports run with bytecode writes disabled. Do not import plotting
    # backends here: their first import can create a user configuration directory.
    modules = [value[0] for key, value in _PACKAGES.items() if key != "matplotlib"]
    probe = (
        "import importlib,json,sys\n"
        "result={}\n"
        "for name in json.loads(sys.argv[1]):\n"
        " try:\n"
        # WILDS 1.2.2 starts an HTTP version-check thread at import. Suppress
        # only that optional hook in this disposable probe, not in training
        # or the installed package; preflight must not perform network I/O.
        "  if name=='wilds':\n"
        "   import outdated\n"
        "   outdated.check_outdated=None\n"
        "  module=importlib.import_module(name); result[name]={'ok':True}\n"
        "  if name=='torch': result[name]['cuda_available']=module.cuda.is_available()\n"
        " except Exception as exc: result[name]={'ok':False,'error':type(exc).__name__+': '+str(exc)[:300]}\n"
        "print('LUNA_PREFLIGHT_IMPORTS='+json.dumps(result))\n"
    )
    try:
        completed = subprocess.run([sys.executable, "-B", "-c", probe, json.dumps(modules)],
                                   capture_output=True, text=True, timeout=45)
        tagged = [line for line in completed.stdout.splitlines() if line.startswith("LUNA_PREFLIGHT_IMPORTS=")]
        if completed.returncode != 0 or not tagged:
            raise RuntimeError("import subprocess failed with exit code " + str(completed.returncode))
        imports = json.loads(tagged[-1].split("=", 1)[1])
        result["imports"] = imports
        result["wilds_update_check"] = "disabled only in isolated import probe; no package or training change"
        result["plotting_import_note"] = "matplotlib checked by metadata and module discovery; plotting backend not imported"
        for name, item in imports.items():
            if not item["ok"]:
                blockers.append("Import failed for " + name + ": " + item["error"])
        cuda = imports.get("torch", {}).get("cuda_available")
        result["cuda_available"] = cuda
        if cuda is False:
            blockers.append("CUDA is unavailable; the reused LUNA teacher requires CUDA")
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        result["import_probe_error"] = str(exc)
        blockers.append("Environment import validation did not complete: " + str(exc))
    return result


def inspect_assets(dataset, data_root=None, teacher_clip_path=None,
                   teacher_checkpoint=None, teacher_total_epochs=None,
                   check_environment=True, teacher_quality="strong"):
    """Return a JSON-serializable READY/BLOCKED report without changing assets.

    Explicit arguments win over LUNA_DATA_ROOT, LUNA_TEACHER_CLIP_PATH,
    LUNA_TEACHER_CHECKPOINT and LUNA_TEACHER_TOTAL_EPOCHS environment values.
    teacher_quality controls final/intermediate epoch validation only; choosing
    the checkpoint nearest a predefined training fraction is the runner's job.
    check_environment=False is intended for fixture tests and asset-only checks;
    callers must retain the returned environment.checked flag when reporting it.
    """
    blockers = []
    canonical = _canonical_dataset(dataset)
    result = {"status": "BLOCKED", "dataset": canonical, "requested_dataset": str(dataset),
              "blockers": blockers, "assets": {}, "environment": {"checked": False}}
    if canonical is None:
        blockers.append("Unsupported dataset; expected DTD, FGVCAircraft/Aircraft, or EuroSAT")
        return result
    assets = result["assets"]
    assets["data_root"] = _path_asset(data_root, "LUNA_DATA_ROOT", "directory", "Dataset root", blockers)
    assets["teacher_clip"] = _path_asset(teacher_clip_path, "LUNA_TEACHER_CLIP_PATH", "file", "Teacher CLIP base weights", blockers)
    checkpoint_label = "Explicit final teacher checkpoint" if teacher_quality == "strong" else "Explicit intermediate teacher checkpoint"
    assets["teacher_checkpoint"] = _path_asset(teacher_checkpoint, "LUNA_TEACHER_CHECKPOINT", "file", checkpoint_label, blockers)
    assets["teacher_checkpoint"]["quality"] = teacher_quality
    if teacher_quality not in ("weak", "medium", "strong"):
        blockers.append("teacher_quality must be weak, medium, or strong")
    assets["teacher_clip"]["metadata_status"] = "unvalidated_not_loaded"
    assets["teacher_checkpoint"]["metadata_status"] = "unvalidated_not_loaded"
    epochs, source = _input(teacher_total_epochs, "LUNA_TEACHER_TOTAL_EPOCHS")
    valid_epochs = isinstance(epochs, int) and not isinstance(epochs, bool) and epochs > 0
    if isinstance(epochs, str) and re.fullmatch(r"[1-9][0-9]*", epochs.strip()):
        epochs, valid_epochs = int(epochs.strip()), True
    assets["teacher_total_epochs"] = {"value": epochs if valid_epochs else None, "source": source}
    if not valid_epochs:
        blockers.append("teacher_total_epochs must explicitly declare a positive integer final teacher epoch")
    checkpoint_path = assets["teacher_checkpoint"]["path"]
    if checkpoint_path:
        filename = Path(checkpoint_path).name.lower()
        if re.search(r"(?:^|[._-])best(?:[._-]|$)", filename):
            blockers.append("Journal teachers require preselected training-stage checkpoints; model-best/best-selected files are forbidden")
        suffix = re.fullmatch(r"model\.pth\.tar-([0-9]+)", filename)
        if suffix and valid_epochs:
            selected_epoch = int(suffix.group(1))
            assets["teacher_checkpoint"]["filename_epoch"] = selected_epoch
            if teacher_quality == "strong" and selected_epoch != epochs:
                blockers.append("Teacher checkpoint filename epoch disagrees with teacher_total_epochs")
            elif teacher_quality in ("weak", "medium") and not 0 < selected_epoch < epochs:
                blockers.append("Weak/medium teacher checkpoint filename epoch must precede teacher_total_epochs and be positive")
    if assets["data_root"]["exists"]:
        assets["dataset"] = _inspect_dataset(canonical, Path(assets["data_root"]["path"]), blockers)
    if check_environment:
        result["environment"] = _inspect_environment(blockers)
    result["status"] = "BLOCKED" if blockers else "READY"
    return result
