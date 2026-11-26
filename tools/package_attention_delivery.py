"""Package a completed attention comparison with a verified, portable inventory.

Standard library only. --dry-run writes validation/package_dry_run.json and
never creates a ZIP. Source scripts and compact learned prompts are included;
dataset photographs, full pretrained models, credentials, logs and caches are
not copied. Scientific completion is checked before any final archive is made.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
from typing import Optional
from urllib.parse import unquote, urlsplit
import zipfile


DATASETS = ("eurosat", "fgvc_aircraft", "stanford_cars")
METHODS = ("CLIP", "CoOp", "KgCoOp", "DeKg", "LUNA")
ATTRIBUTIONS = ("rollout", "gradcam", "contrastive_rollout")
ROOT_FILES = ("README_zh.md", "REPORT_zh.md", "review.html", "report_manifest.json",
              "paper_caption.tex", "paper_caption_primary.tex")
SOURCE_SCRIPTS = (
    "attention_models.py", "attention_figure.py", "attention_localization.py",
    "visualize_attention_comparison.py", "visualize_luna_attention.py",
    "evaluate_attention_comparison.py", "prepare_attention_datasets.py",
    "make_attention_configs.py", "match_attention_inputs.py",
    "train_attention_baselines.py", "train_attention_luna.py",
    "run_attention_queue.py", "prepare_attention_queue.py",
    "audit_attention_localization.py", "finalize_attention_product.py",
    "export_attention_alternatives.py", "export_requested_originals.py", "build_attention_report.py",
    "package_attention_delivery.py",
)
BLOCKED_DIRS = {"checks", "logs", "cache", "caches", "__pycache__", ".git", ".ssh",
                "tensorboard", "fewshot_cache", "vlpromptlearner", "node_modules"}
BLOCKED_NAMES = {"package_manifest.json", "package_dry_run.json", "credentials.json",
                 "credentials", "secrets.json", "secrets", "token.json", "tokens.json"}
DOCUMENT_SUFFIXES = {".png", ".pdf", ".svg", ".npz", ".json", ".csv", ".tex", ".md", ".html", ".txt"}
SOURCE_NOTE = b"""# Included source and reproduction requirements

`source/tools/` preserves the attention comparison production scripts as shipped.
These scripts belong in the original LUNA repository; this archive is not a
self-contained Python environment or a copy of the whole training repository.
CLIP, Dassl, project trainers, their dependencies, the official datasets and full
pretrained/teacher models must be obtained separately. Training and inference
manifests preserve the original machine paths and hashes for provenance; update
paths in a copied configuration when reproducing on another machine.

The report, gallery, figures and downloadable local evidence retain their
relative paths, so README_zh.md and review.html work after extraction. Compact
attention_checkpoint.pth files contain the learned prompt parameters; the large
frozen backbones, complete teacher models and raw dataset photographs are excluded.
Teacher provenance records are retained under training/<dataset>/PromptSRC_teacher/.

Existing bundles can be rendered without loading any model (NumPy, Pillow and
Matplotlib are required):

    python source/tools/attention_figure.py paper/main_bundle.npz --output-dir rerender

package_manifest.json records the relative path, byte size and SHA256 of every
payload file. It excludes its own hash to avoid a circular checksum. The ZIP
contains no absolute member paths or parent-directory traversals.
"""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_link(path):
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) &
                                     getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def bounded(path, boundary):
    """Require an existing regular file/directory inside an explicit boundary."""
    boundary = boundary.resolve(strict=True)
    original = Path(os.path.abspath(str(path)))
    original.relative_to(boundary)
    resolved = original.resolve(strict=True)
    resolved.relative_to(boundary)
    current = original
    while current != boundary:
        if is_link(current):
            raise ValueError("Symlink or reparse point is not allowed: " + str(current))
        current = current.parent
    return resolved


def member_name(value):
    if "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("Unsafe archive member name: " + repr(value))
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in (".", "..") for p in path.parts):
        raise ValueError("Unsafe archive member name: " + repr(value))
    if path.as_posix() != value:
        raise ValueError("Non-canonical archive member name: " + repr(value))
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("Non-canonical archive member name: " + repr(value))
    return value


def excluded_name(path):
    name = path.name.lower()
    return (name in BLOCKED_NAMES or name.startswith(".env") or
            name.startswith("credentials.") or name.startswith("secrets.") or
            path.suffix.lower() in {".pem", ".key", ".pfx", ".p12", ".env"} or
            any(part.lower() in BLOCKED_DIRS for part in path.parts))


@dataclass
class Entry:
    name: str
    size: int
    sha256: str
    source: Optional[Path] = None
    content: Optional[bytes] = None

    def record(self):
        return {"path": self.name, "size_bytes": self.size, "sha256": self.sha256}


def add_file(entries, path, boundary, archive_name=None):
    path = bounded(path, boundary)
    if not path.is_file() or excluded_name(path):
        raise ValueError("Not an allowed payload file: " + str(path))
    name = member_name(archive_name or path.relative_to(boundary).as_posix())
    key = name.casefold()
    if any(existing.casefold() == key for existing in entries):
        raise ValueError("Duplicate archive member: " + name)
    if path.suffix.lower() == ".pth" and path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Compact checkpoint exceeds the 64 MiB safety limit: " + name)
    entries[name] = Entry(name, path.stat().st_size, digest_file(path), source=path)


def add_tree(entries, root, directory, suffixes):
    path = root / directory
    if not path.exists():
        return
    bounded(path, root)
    for current, dirs, files in os.walk(path, followlinks=False):
        current_path = Path(current)
        allowed = []
        for directory_name in sorted(dirs):
            child = current_path / directory_name
            if directory_name.lower() in BLOCKED_DIRS or directory_name.startswith("."):
                continue
            bounded(child, root)
            allowed.append(directory_name)
        dirs[:] = allowed
        for filename in sorted(files):
            file_path = current_path / filename
            if file_path.suffix.lower() in suffixes and not excluded_name(file_path):
                add_file(entries, file_path, root)


def collect(root):
    entries = {}
    for name in ROOT_FILES:
        if (root / name).exists():
            add_file(entries, root / name, root)
    for directory in ("requested_originals", "paper", "figures", "evaluation", "localization_audit"):
        add_tree(entries, root, directory, DOCUMENT_SUFFIXES)
    for directory in ("validation", "configs", "datasets"):
        add_tree(entries, root, directory, {".json", ".md"})
    for dataset in DATASETS:
        for method in METHODS[1:]:
            directory = root / "training" / dataset / method
            for name in ("attention_checkpoint.pth", "protocol.json", "training_manifest.json",
                         "status.json", "epochs.csv", "loss_trends.csv", "implementation_validation.json"):
                path = directory / name
                if path.exists():
                    add_file(entries, path, root)
        teacher_name = "training/" + dataset + "/PromptSRC_teacher/teacher_provenance.json"
        local_teacher = root / teacher_name
        if local_teacher.exists():
            add_file(entries, local_teacher, root)
        elif dataset in ("eurosat", "fgvc_aircraft"):
            # Only these two fixed, existing provenance files may come from the
            # sibling teacher directory. No manifest-controlled external paths.
            teacher_root = root.parent / "results_reliability_analysis" / "teacher_training"
            external_teacher = teacher_root / (dataset + "_promptsrc_l14_seed1_final20") / "teacher_provenance.json"
            if external_teacher.exists():
                bounded(teacher_root, root.parent)
                add_file(entries, external_teacher, teacher_root, teacher_name)
    source_root = Path(__file__).resolve().parent
    for filename in SOURCE_SCRIPTS:
        add_file(entries, source_root / filename, source_root, "source/tools/" + filename)
    note_name = "source/README.md"
    entries[note_name] = Entry(note_name, len(SOURCE_NOTE), hashlib.sha256(SOURCE_NOTE).hexdigest(), content=SOURCE_NOTE)
    return entries


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def npy_header(handle):
    if handle.read(6) != b"\x93NUMPY":
        raise ValueError("Bad NPY signature")
    version = handle.read(2)
    if version == b"\x01\x00":
        size = struct.unpack("<H", handle.read(2))[0]
        encoding = "latin1"
    elif version in (b"\x02\x00", b"\x03\x00"):
        size = struct.unpack("<I", handle.read(4))[0]
        encoding = "utf8" if version[0] == 3 else "latin1"
    else:
        raise ValueError("Unsupported NPY version")
    if size > 1024 * 1024:
        raise ValueError("Unreasonably large NPY header")
    value = ast.literal_eval(handle.read(size).decode(encoding).strip())
    if not isinstance(value, dict) or not isinstance(value.get("descr"), str):
        raise ValueError("Structured or invalid NPY dtype is not supported")
    if "O" in value["descr"] or not isinstance(value.get("shape"), tuple):
        raise ValueError("Object arrays or invalid NPY shapes are forbidden")
    if any(not isinstance(size, int) or size < 0 for size in value["shape"]):
        raise ValueError("Invalid NPY dimensions")
    if not isinstance(value.get("fortran_order"), bool):
        raise ValueError("Invalid NPY storage order")
    return value


def read_strings(handle, header):
    match = re.fullmatch(r"([<>=|])([US])(\d+)", header["descr"])
    if not match:
        raise ValueError("Expected a fixed-width string array")
    order, kind, width = match.groups()
    width = int(width)
    if kind == "U":
        width *= 4
        if order == "=":
            import sys
            order = "<" if sys.byteorder == "little" else ">"
        encoding = "utf-32-be" if order == ">" else "utf-32-le"
    else:
        encoding = "utf-8"
    count = 1
    for size in header["shape"]:
        count *= size
    if width <= 0 or count * width > 8 * 1024 * 1024:
        raise ValueError("Unreasonably large string array")
    body = handle.read(count * width + 1)
    if len(body) != count * width:
        raise ValueError("Incorrect string array length")
    return [body[i * width:(i + 1) * width].decode(encoding).rstrip("\x00") for i in range(count)]


def inspect_bundle(path, expected_rows=3):
    """Read small headers/metadata, not image tensors; no NumPy dependency."""
    headers = {}
    strings = {}
    input_array_sha256 = None
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate arrays in bundle")
        for name in names:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.npy", name):
                raise ValueError("Unexpected NPZ member: " + name)
            with archive.open(name) as handle:
                key = name[:-4]
                headers[key] = npy_header(handle)
                dtype = re.fullmatch(r"[<>=|]([buifcUS])(\d+)", headers[key]["descr"])
                if not dtype:
                    raise ValueError("Unsupported plain NPY dtype")
                width = int(dtype.group(2)) * (4 if dtype.group(1) == "U" else 1)
                count = 1
                for size in headers[key]["shape"]:
                    count *= size
                if width <= 0 or archive.getinfo(name).file_size - handle.tell() != width * count:
                    raise ValueError("NPY payload length does not match its header")
                if key in ("methods", "metadata", "classnames"):
                    strings[key] = read_strings(handle, headers[key])
                elif key == "images":
                    digest = hashlib.sha256()
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                    input_array_sha256 = digest.hexdigest()
    for key in ("images", "maps", "methods", "classnames", "metadata"):
        if key not in headers:
            raise ValueError("Bundle is missing " + key)
    if tuple(strings["methods"]) != METHODS or headers["methods"]["shape"] != (len(METHODS),):
        raise ValueError("Five methods in the required order are mandatory")
    shapes = {key: headers[key]["shape"] for key in headers}
    if any(not re.fullmatch(r"[<>=]f[48]", headers[key]["descr"]) for key in ("images", "maps")):
        raise ValueError("Image and attention tensors must have float32/float64 dtype")
    if (len(shapes["images"]) != 4 or shapes["images"][0] != expected_rows or
            shapes["images"][-1] != 3 or min(shapes["images"][1:3]) < 1):
        raise ValueError("Input image rows/dimensions are incomplete")
    if (len(shapes["maps"]) != 4 or shapes["maps"][:2] != (expected_rows, len(METHODS)) or
            min(shapes["maps"][2:]) < 1 or shapes["classnames"] != (expected_rows,)):
        raise ValueError("Attention rows/columns are incomplete")
    if shapes["metadata"] != () or len(strings["metadata"]) != 1:
        raise ValueError("Metadata must be a scalar JSON string")
    metadata = json.loads(strings["metadata"][0])
    if metadata.get("missing_methods"):
        raise ValueError("Bundle declares missing methods")
    return {"rows": expected_rows, "methods": list(METHODS), "classnames": strings["classnames"],
            "input_array_sha256": input_array_sha256, "metadata": metadata}


class LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ("src", "href") and value)


def unresolved_links(entries):
    failures = []
    for name in ("README_zh.md", "REPORT_zh.md", "review.html"):
        if name not in entries:
            continue
        text = entries[name].source.read_text(encoding="utf-8")
        parser = LinkCollector()
        parser.feed(text)
        links = list(parser.links)
        if name.endswith(".md"):
            links += re.findall(r"!?\[[^\]]*\]\(<?([^\s)>]+)>?(?:\s+[^)]*)?\)", text)
        for value in links:
            parsed = urlsplit(value)
            if parsed.scheme in ("http", "https", "mailto", "data") or not parsed.path:
                continue
            if parsed.scheme or parsed.netloc:
                failures.append({"document": name, "link": value, "reason": "nonportable local link"})
                continue
            decoded = unquote(parsed.path)
            target = (PurePosixPath(name).parent / decoded).as_posix()
            try:
                target = member_name(target.rstrip("/"))
            except ValueError:
                failures.append({"document": name, "link": value, "reason": "unsafe local link"})
                continue
            if target not in entries and not any(candidate.startswith(target + "/") for candidate in entries):
                failures.append({"document": name, "link": value, "reason": "target not in archive"})
    return failures


def validate(root, entries):
    blockers = []
    verified_figures = []
    input_identities = {}

    def require(name):
        if name not in entries:
            blockers.append("Missing payload file: " + name)
            return False
        if entries[name].size <= 0:
            blockers.append("Empty payload file: " + name)
            return False
        return True

    report = {}
    for name in ROOT_FILES:
        require(name)
    if "report_manifest.json" in entries:
        try:
            report = load_json(root / "report_manifest.json")
            if report.get("status") != "COMPLETE":
                blockers.append("report_manifest.status is not COMPLETE")
            for name in ROOT_FILES[1:]:
                if name == "report_manifest.json":
                    continue
                artifact = report.get("artifacts", {}).get(name, {})
                if name in entries and artifact.get("sha256") != entries[name].sha256:
                    blockers.append("Report artifact hash is absent or stale: " + name)
            by_source = {str(entry.source).casefold(): entry for entry in entries.values() if entry.source is not None}
            for source in report.get("source_files", []):
                source_path = Path(os.path.abspath(source["path"]))
                entry = by_source.get(str(source_path).casefold())
                if entry is not None and entry.sha256 != source.get("sha256"):
                    blockers.append("Report evidence hash is stale: " + entry.name)
                elif entry is None:
                    try:
                        relative = source_path.relative_to(root).as_posix()
                    except ValueError:
                        continue
                    blockers.append("Report evidence is absent from the payload: " + relative)
        except (OSError, ValueError, TypeError, KeyError) as error:
            blockers.append("Invalid report manifest: " + str(error))

    def figure(directory, basename, bundle_name, dataset):
        relative = directory.as_posix()
        required = [relative + "/" + basename + extension for extension in (".png", ".pdf", ".svg")]
        required += [relative + "/" + bundle_name, relative + "/render_manifest.json"]
        available = [require(name) for name in required]
        if not all(available):
            return
        try:
            bundle_path = root / directory / bundle_name
            info = inspect_bundle(bundle_path)
            metadata = info.pop("metadata")
            if metadata.get("dataset") != dataset:
                raise ValueError("Unexpected dataset in bundle metadata")
            if dataset != "combined" and len(metadata.get("samples", [])) != 3:
                raise ValueError("Expected exactly three original input samples")
            if dataset == "combined" and {item["dataset"] for item in metadata.get("sources", [])} != set(DATASETS):
                raise ValueError("Paper main figure must contain all three datasets")
            identity = (info["input_array_sha256"], tuple(info["classnames"]))
            if dataset in input_identities and identity != input_identities[dataset]:
                raise ValueError("The three attribution styles must use identical inputs and row labels")
            input_identities[dataset] = identity
            render = load_json(root / directory / "render_manifest.json")
            if (render.get("status") != "complete" or render.get("missing_methods") or
                    render.get("column_order") != ["Input"] + list(METHODS)):
                raise ValueError("Renderer did not complete all six columns")
            if render.get("source_sha256") != entries[relative + "/" + bundle_name].sha256:
                raise ValueError("Render manifest hash differs from the raw bundle")
            if render.get("method_specific_style") is not False:
                raise ValueError("Method-specific rendering is disallowed")
            verified_figures.append({"directory": relative, **info})
        except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile, struct.error) as error:
            blockers.append("Invalid required figure " + relative + ": " + str(error))

    figure(PurePosixPath("paper"), "main_comparison", "main_bundle.npz", "combined")
    for attribution in ("gradcam", "rollout"):
        figure(PurePosixPath("paper/alternatives") / attribution, "main_comparison", "main_bundle.npz", "combined")
        require("paper/alternatives/" + attribution + "/caption.tex")
    for dataset in DATASETS:
        for attribution in ATTRIBUTIONS:
            figure(PurePosixPath("figures") / dataset / "original" / attribution,
                   "attention_comparison", "attention_bundle.npz", dataset)
            rows = [item for item in report.get("figures", []) if item.get("dataset") == dataset and
                    item.get("group") == "original" and item.get("attribution") == attribution]
            if len(rows) != 1 or rows[0].get("status") != "COMPLETE":
                blockers.append("Report lacks a complete original figure: " + dataset + "/" + attribution)
        evaluation_name = "evaluation/" + dataset + "/evaluation_summary.json"
        evaluation = {}
        if require(evaluation_name):
            try:
                evaluation = load_json(root / evaluation_name)
                if evaluation.get("status") != "COMPLETE_HELDOUT_EVALUATION":
                    blockers.append("Held-out evaluation is incomplete: " + dataset)
            except (OSError, ValueError, TypeError) as error:
                blockers.append("Invalid evaluation: " + dataset + ": " + str(error))
        require("evaluation/" + dataset + "/per_image.csv")
        for method in METHODS[1:]:
            prefix = "training/" + dataset + "/" + method + "/"
            require(prefix + "attention_checkpoint.pth")
            checkpoint = entries.get(prefix + "attention_checkpoint.pth")
            evaluated_hash = evaluation.get("model_metadata", {}).get("methods", {}).get(method, {}).get("checkpoint_sha256")
            if checkpoint is not None and evaluation and checkpoint.sha256 != evaluated_hash:
                blockers.append("Packaged checkpoint does not match held-out evaluation: " + dataset + "/" + method)
            if method == "LUNA":
                require(prefix + "training_manifest.json")
            else:
                for name in ("protocol.json", "status.json", "epochs.csv"):
                    require(prefix + name)
        require("training/" + dataset + "/PromptSRC_teacher/teacher_provenance.json")
    for dataset in ("fgvc_aircraft", "stanford_cars"):
        name = "localization_audit/" + dataset + "/summary.json"
        if require(name):
            try:
                if load_json(root / name).get("status") != "COMPLETE_PREDICTION_INDEPENDENT_LOCALIZATION_AUDIT":
                    blockers.append("Prediction-independent localization audit is incomplete: " + dataset)
            except (OSError, ValueError, TypeError) as error:
                blockers.append("Invalid localization audit: " + dataset + ": " + str(error))
    broken_links = unresolved_links(entries)
    blockers.extend("Unresolved local link in " + link["document"] + ": " + link["link"] for link in broken_links)
    return {"report_status": report.get("status", "MISSING"), "blockers": blockers,
            "verified_required_figures": verified_figures, "unresolved_local_links": broken_links}


def manifest(entries, verification, dry_run):
    records = [entries[name].record() for name in sorted(entries)]
    return {"schema_version": 1,
            "status": ("BLOCKED_DRY_RUN" if dry_run else "BLOCKED") if verification["blockers"] else "READY_DRY_RUN" if dry_run else "COMPLETE",
            "created_utc": utc_now(), "dry_run": dry_run, "archive_name": "LUNA_attention_comparison.zip",
            "archive_layout": "relative to the extraction directory; no enclosing top-level folder",
            "file_count": len(records), "total_uncompressed_bytes": sum(item["size_bytes"] for item in records),
            "hash_algorithm": "SHA256", "manifest_self_hash": "excluded to avoid circular checksum",
            "excluded": ["full pretrained and teacher checkpoints", "dataset photographs", "checks", "logs",
                         "caches", "environment and credential files", "previous archives and packaging manifests"],
            "reproduction_note": "Production tools and compact learned prompts are included; original repository, dependencies, datasets and frozen backbones remain external.",
            "verification": verification, "files": records}


def safe_destination(path, root):
    path = Path(os.path.abspath(str(path)))
    path.relative_to(root.resolve(strict=True))
    ancestor = path.parent
    while not ancestor.exists() and not ancestor.is_symlink():
        ancestor = ancestor.parent
    bounded(ancestor, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    bounded(path.parent, root)
    if path.exists() or path.is_symlink():
        bounded(path, root)
        if not path.is_file():
            raise ValueError("Output is not a regular file: " + str(path))
    return path


def atomic_json(path, value, root):
    safe_destination(path, root)
    temporary = safe_destination(path.with_name(path.name + ".partial"), root)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_archive(root, entries, inventory):
    if inventory["verification"]["blockers"] or inventory["status"] != "COMPLETE":
        raise ValueError("Incomplete delivery cannot be packaged")
    destination = safe_destination(root / "LUNA_attention_comparison.zip", root)
    temporary = safe_destination(root / "LUNA_attention_comparison.zip.partial", root)
    body = (json.dumps(inventory, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for name in sorted(entries):
            entry = entries[name]
            hashed = hashlib.sha256()
            size = 0
            info = zipfile.ZipInfo(name, date_time=datetime.now().timetuple()[:6])
            info.compress_type = zipfile.ZIP_STORED if Path(name).suffix.lower() in {".png", ".npz", ".pdf"} else zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with archive.open(info, "w", force_zip64=True) as target:
                if entry.source is None:
                    blocks = [entry.content]
                    for block in blocks:
                        target.write(block); hashed.update(block); size += len(block)
                else:
                    with entry.source.open("rb") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            target.write(block); hashed.update(block); size += len(block)
            if hashed.hexdigest() != entry.sha256 or size != entry.size:
                raise ValueError("Payload changed during packaging: " + name)
        archive.writestr("package_manifest.json", body)
    with zipfile.ZipFile(temporary) as archive:
        if set(archive.namelist()) != set(entries) | {"package_manifest.json"}:
            raise ValueError("Archive inventory differs from the planned payload")
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError("ZIP CRC verification failed: " + corrupt)
    os.replace(temporary, destination)
    atomic_json(root / "package_manifest.json", inventory, root)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/LUNA/attention_comparison"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    if not root.is_dir() or root == Path(root.anchor):
        parser.error("--root must be an existing product directory, not a filesystem root")
    try:
        entries = collect(root)
        verification = validate(root, entries)
        inventory = manifest(entries, verification, args.dry_run)
        if args.dry_run:
            output = root / "validation" / "package_dry_run.json"
            atomic_json(output, inventory, root)
            result = {"status": inventory["status"], "manifest": str(output), "archive_created": False,
                      "file_count": inventory["file_count"], "total_uncompressed_bytes": inventory["total_uncompressed_bytes"],
                      "blocker_count": len(verification["blockers"]), "blockers": verification["blockers"]}
        elif verification["blockers"]:
            print(json.dumps({"status": "BLOCKED", "archive_created": False, "blockers": verification["blockers"]}, ensure_ascii=False))
            return 2
        else:
            destination = write_archive(root, entries, inventory)
            result = {"status": "COMPLETE", "archive": str(destination), "archive_sha256": digest_file(destination),
                      "file_count": inventory["file_count"], "archive_bytes": destination.stat().st_size}
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "ERROR", "archive_created": False, "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
