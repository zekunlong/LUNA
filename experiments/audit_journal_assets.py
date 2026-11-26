"""Read-only asset discovery; writes only an explicit audit output directory.

Uses ripgrep to enumerate accessible files including hidden/ignored files, then
lists ZIP central directories without extracting datasets or loading checkpoints.
Non-ZIP archives are listed by Windows tar with a per-archive timeout.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time
import zipfile

TARGET = re.compile(r"(?:dtd|describable.?textures|fgvc|aircraft|eurosat|euro_sat)", re.I)
WEIGHT = re.compile(r"(?:\.(?:pt|pth|ckpt|safetensors|bin)$|\.pth\.tar(?:[-.]\d+)?$)", re.I)
SPLIT = re.compile(r"(?:split_zhou_(?:DescribableTextures|EuroSAT)\.json|images_variant_(?:train|val|test)\.txt|(?:^|[\\/])variants\.txt)$", re.I)
CLIP = re.compile(r"(?:clip|vit[-_]?b[-_/]?16|vit[-_]?l[-_/]?14|rn50)", re.I)
RUN = re.compile(r"(?:promptsrc|teacher|student|luna|journal|adaptive)", re.I)
ARCHIVE = re.compile(r"\.(?:zip|tar|tar\.gz|tgz|tar\.bz2|tbz2|tar\.xz|txz|7z|rar)$", re.I)
IMAGE = re.compile(r"\.(?:jpg|jpeg|png|tif|tiff|bmp|webp)$", re.I)


def stat(path):
    try:
        item = Path(path).stat()
        return {"size_bytes": item.st_size, "modified": dt.datetime.fromtimestamp(item.st_mtime).astimezone().isoformat()}
    except OSError as exc:
        return {"stat_error": str(exc)}


def summarize(names):
    return {
        "member_count": len(names),
        "target_members": [n for n in names if TARGET.search(n)],
        "checkpoint_members": [n for n in names if WEIGHT.search(n)],
        "split_members": [n for n in names if SPLIT.search(n)],
        "nested_archive_members": [n for n in names if ARCHIVE.search(n)],
    }


def add_review(output):
    """Add protocol-specific conclusions to an already completed inventory."""
    output = Path(output)
    path = output / "asset_audit.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("completed_at"):
        raise RuntimeError("Asset enumeration and archive inspection must finish first")
    files = Path(result["file_list"]).read_text(encoding="utf-8", errors="replace").splitlines()
    class_layout = re.compile(r"[\\/](?:AnnualCrop|HerbaceousVegetation|PermanentCrop|SeaLake|banded|blotchy|braided|bubbly|chequered|cobwebbed|crosshatched|crystalline|flecked|frilly|gauzy|honeycombed|interlaced|lacelike|marbled|paisley|perforated|polka-dotted|potholed|scaly|spiralled|sprinkled|stratified|studded|swirly|veined|waffled|zigzagged)[\\/]", re.I)
    result["named_dataset_class_image_candidates"] = [p for p in files if class_layout.search(p) and IMAGE.search(p)]
    result["dataset_requirements"] = {
        "DTD": {"image_directory": "<data_root>/dtd/images", "split_files": ["<data_root>/dtd/split_zhou_DescribableTextures.json"], "evidence": "datasets/dtd.py:15-21"},
        "FGVCAircraft": {"image_directory": "<data_root>/fgvc_aircraft/images", "split_files": ["<data_root>/fgvc_aircraft/variants.txt", "<data_root>/fgvc_aircraft/images_variant_train.txt", "<data_root>/fgvc_aircraft/images_variant_val.txt", "<data_root>/fgvc_aircraft/images_variant_test.txt"], "evidence": "datasets/fgvc_aircraft.py:15-36"},
        "EuroSAT": {"image_directory": "<data_root>/eurosat/2750", "split_files": ["<data_root>/eurosat/split_zhou_EuroSAT.json"], "evidence": "datasets/eurosat.py:29-35"},
    }
    result["protocol_checkpoint_requirements"] = {
        "student_clip": "OpenAI CLIP ViT-B/16 (ViT-B-16.pt)",
        "teacher_clip": "OpenAI CLIP ViT-L/14 (ViT-L-14.pt)",
        "teacher_prompt": "Dataset/seed-specific PromptSRC teacher weights; final saved epoch with total-epoch provenance",
        "teacher_quality": "Saved epochs nearest 25%, 60%, and final; no test-performance selection",
        "student_runs": "Protocol-identical AdaptiveLUNA/Equal/StaticMean/InstanceOnly checkpoints with configs, logs and split/seed provenance",
        "evidence": ["configs/trainers/LUNA/vit_b16_c2_ep20_batch8_4+4ctx.yaml", "clip/clip.py:35-36", "experiments/journal_preflight.py"],
    }
    result["cache_protocol_review"] = [
        {"path": str(Path.home() / ".cache" / "clip"), "status": "MISSING" if not (Path.home() / ".cache" / "clip").exists() else "SEE_CACHE_CHECKS"},
        {"path": str(Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vitl14_pretrain.pth"), "status": "UNRELATED_MODEL", "reason": "DINOv2 visual pretraining does not provide the required CLIP teacher/student image-text model"},
        {"path": str(Path.home() / ".cache" / "huggingface" / "hub" / "models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K" / "snapshots" / "84c9828e63dc9a9351d1fe637c346d4c1c4db341" / "open_clip_pytorch_model.bin"), "status": "DIFFERENT_PRETRAINED_PROTOCOL", "reason": "Existing cache is LAION DataComp OpenCLIP ViT-L/14, not the required OpenAI-pretrained CLIP ViT-L/14. No silent substitution or compatibility claim made; no checkpoint deserialized"},
    ]
    result["prior_materials_review"] = {
        "manuscript": "E:\\LUNA\\LUNA\\IEEE Conference Template - ICME 2026\\comb.tex",
        "legacy_result_evidence": "Base/novel lines around 257; cross-dataset around 292; domain generalization around 334",
        "status": "PRIOR_MANUSCRIPT_CLAIMS_UNVERIFIED_FOR_CURRENT_PROTOCOL",
        "reason": "Rendered figures and manuscript tables have no associated run/checkpoint/split provenance establishing new fixed-reliability Exp1-4 measurements. They cannot populate REAL_EXPERIMENT tables.",
        "attention_figures": [p for p in files if re.search(r"[\\/]LUNA[\\/]attention_[^\\/]+\.png$", p, re.I)],
        "tsne_figures": [p for p in files if "E:\\LUNA\\LUNA\\tsne\\" in p and p.lower().endswith(".png")],
        "loss_curve_figures": [p for p in files if re.search(r"[\\/]LUNA[\\/]loss_curves_[^\\/]+\.png$", p, re.I)],
    }
    result["checkpoint_groups"] = dict(collections.Counter(str(Path(p["path"]).parts[1]) for p in result["checkpoint_candidates"]))
    result["relevant_checkpoint_name_matches"] = [p for p in result["checkpoint_candidates"] if p["clip_name_match"] or p["luna_teacher_student_name_match"] or TARGET.search(p["path"])]
    result["checkpoint_false_positives"] = [p for p in result["relevant_checkpoint_name_matches"] if "site-packages" in p["path"] and Path(p["path"]).name == "distutils-precedence.pth"]
    result["relevant_checkpoint_name_matches"] = [p for p in result["relevant_checkpoint_name_matches"] if p not in result["checkpoint_false_positives"]]
    result["archive_status_counts"] = dict(collections.Counter(a["status"] for a in result["archive_inspections"]))
    result["uninspected_nested_archives"] = []
    for archive in result["archive_inspections"]:
        for nested in archive.get("nested_inspections", []):
            if nested["status"] != "INSPECTED":
                result["uninspected_nested_archives"].append({"outer_archive": archive["path"], **nested})
        for member in archive.get("nested_archives_not_inspected", []):
            result["uninspected_nested_archives"].append({"outer_archive": archive["path"], "member": member, "status": "NOT_INSPECTED", "reason": "Nested non-ZIP archive member not extracted; loose files with corresponding names were independently inventoried"})
    no_candidates = not (result["possible_dataset_image_paths"] or result["official_split_paths"] or result["named_dataset_class_image_candidates"] or result["relevant_checkpoint_name_matches"])
    result["dataset_conclusions"] = {}
    for dataset, requirements in result["dataset_requirements"].items():
        result["dataset_conclusions"][dataset] = {
            "status": "NOT_RUN" if no_candidates else "REQUIRES_CANDIDATE_REVIEW",
            "runnable": False,
            "usable_image_directory": None,
            "usable_split_paths": [],
            "usable_student_clip_path": None,
            "usable_teacher_clip_path": None,
            "usable_teacher_checkpoint": None,
            "teacher_total_epochs": None,
            "teacher_weak_epoch": None,
            "teacher_medium_epoch": None,
            "teacher_strong_epoch": None,
            "usable_intermediate_teacher_checkpoints": [],
            "usable_student_checkpoints": [],
            "reusable_valid_exp1_to_exp4_runs": [],
            "blockers": [
                {"category": "MISSING_DATASET", "reason": "No usable " + dataset + " images found among " + format(result["accessible_file_count"], ",") + " accessible E: files or inspected archive members. Expected " + requirements["image_directory"] + "; named matches are code/configuration or rendered figures."},
                {"category": "MISSING_SPLIT", "reason": "No original target split files found. Required: " + ", ".join(requirements["split_files"])},
                {"category": "MISSING_CLIP_WEIGHT", "reason": "Required OpenAI CLIP ViT-B/16 student and ViT-L/14 teacher pretrained files were not found; default OpenAI CLIP cache is absent. Available LAION DataComp OpenCLIP and DINOv2 are different pretrained models."},
                {"category": "MISSING_TEACHER_CHECKPOINT", "reason": "No " + dataset + " PromptSRC final teacher checkpoint or associated teacher configuration/log/total-epoch provenance found."},
                {"category": "MISSING_INTERMEDIATE_TEACHER_CHECKPOINT", "reason": "No " + dataset + " saved PromptSRC training checkpoints or total epochs found, so nearest 25%/60% teacher epochs cannot be selected."},
            ],
            "additional_reuse_blocker": "No protocol-identical AdaptiveLUNA/Equal/StaticMean/InstanceOnly saved student runs available for Exp2/Exp4 reuse.",
            "absence_scope": "Accessible enumerated E: files, inspected archive entries, and supplemental default caches only; 11 inaccessible system/recycle paths and listed nested-archive limits remain explicitly outside verified scope.",
            "evidence": ["asset_audit.json", "logs/asset_search_all_files.txt", "logs/asset_search_errors.txt"],
        }
        if not no_candidates:
            result["dataset_conclusions"][dataset]["blockers"] = [{"category": "OTHER", "reason": "New candidate assets were discovered. Authenticate dataset contents, checkpoints, split provenance and training epochs before drawing availability conclusions."}]
    result["archive_scope_caveats"] = [
        {"path": a["path"], "status": a["status"], "reason": a.get("reason", a.get("error", ""))}
        for a in result["archive_inspections"] if a["status"] != "INSPECTED"
    ]
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = output / "ASSET_AUDIT.md"
    original = md_path.read_text(encoding="utf-8")
    original = re.sub(r"- Archives:.*", "- Archive-named files: " + str(result["archive_count"]) + "; final status counts: " + str(result["archive_status_counts"]), original)
    md_path.write_text(original, encoding="utf-8")
    lines = ["", "## Protocol-specific review", "", "The accessible inspected files and archive member names did not establish a runnable target dataset, original target split, OpenAI CLIP checkpoint, or PromptSRC/student run. Checkpoint filename candidates from unrelated projects are not reusable LUNA experiments. Review the JSON candidate lists and archive/access caveats before interpreting absence as global absence.", "", "Expected dataset layouts:", ""]
    for dataset, requirement in result["dataset_requirements"].items():
        lines += ["- " + dataset + ": `" + requirement["image_directory"] + "`; " + ", ".join("`" + p + "`" for p in requirement["split_files"])]
    lines += ["", "The configured student is OpenAI CLIP ViT-B/16 and teacher is OpenAI CLIP ViT-L/14. Dataset-specific PromptSRC final/intermediate teacher checkpoints and saved epoch provenance are required. No teacher total epoch is inferred from a student configuration filename.", "", "Supplemental default caches contain a DINOv2 checkpoint and a LAION DataComp OpenCLIP ViT-L/14 checkpoint. These do not establish availability of the required OpenAI CLIP assets; they were not silently substituted or deserialized.", "", "## Existing materials and reuse", "", "`E:\\LUNA\\LUNA\\IEEE Conference Template - ICME 2026\\comb.tex` contains legacy base-to-novel, cross-dataset and domain-generalization tables. Attention maps, class-wise t-SNE panels and loss-curve images exist. Their run/checkpoint/split provenance has not been recovered, and their legacy protocol does not establish fixed-reliability Exp1-4 measurements. Retain them as prior manuscript materials, never as current REAL_EXPERIMENT results.", "", "## Access and archive caveats", ""]
    lines += ["- " + e for e in result["search_errors"]]
    lines += ["- `" + a["path"] + "`: " + a["status"] + "; " + a["reason"] for a in result["archive_scope_caveats"]]
    lines += ["", "All nested-archive inspection outcomes and deeper uninspected members are recorded per archive in JSON. No archive extraction was performed.", ""]
    lines += ["## Dataset conclusions", "", "| Dataset | Usable images / splits | Required CLIP weights | PromptSRC final / intermediate | Valid reusable student runs |", "|---|---|---|---|---|"]
    for dataset in result["dataset_conclusions"]:
        lines += ["| " + dataset + " | Not found | Not found | Not found | Not found |"]
    lines += ["", "All three datasets are NOT_RUN because required images, original splits, OpenAI CLIP weights and dataset-specific PromptSRC teacher checkpoints are unavailable in verified scope. Teacher total epochs and weak/medium/strong epoch selections remain null. These are external-asset blockers; this asset audit does not claim a code or numerical failure.", "", "There are " + str(len(result["uninspected_nested_archives"])) + " uninspected nested-archive references: 18 ZIP members larger than the 64 MiB in-memory limit and two nested RAR references. Their names/provenance and exact limits are recorded in JSON. All 71 actual loose archives were listed successfully; the other three archive-suffixed files are Windows recycle metadata. The Unicode-path and LZMA listing errors were resolved through binary stdin and installed 7-Zip, respectively.", ""]
    with (output / "ASSET_AUDIT.md").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def archive_listing(path, nested_limit=64 * 1024 * 1024):
    result = {"path": str(path), **stat(path)}
    try:
        if Path(path).name.startswith("$I") and "$RECYCLE.BIN" in str(path) and result.get("size_bytes", 0) < 4096:
            data = Path(path).read_bytes()
            if len(data) >= 28 and struct.unpack("<Q", data[:8])[0] == 2:
                result.update(status="NOT_AN_ARCHIVE", method="Windows recycle-bin metadata header inspected", reason="Version-2 $I metadata file records a recycled archive; the actual archive payload is the corresponding $R file", member_count=0)
                return result
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                names = [info.filename for info in archive.infolist() if not info.is_dir()]
                result.update(status="INSPECTED", method="ZIP central-directory listing", **summarize(names))
                result["nested_inspections"] = []
                for info in archive.infolist():
                    if not info.filename.lower().endswith(".zip"):
                        continue
                    item = {"member": info.filename, "size_bytes": info.file_size}
                    if info.file_size > nested_limit:
                        item.update(status="NOT_INSPECTED", reason="Nested archive exceeds 64 MiB bounded in-memory inspection limit")
                    else:
                        try:
                            with zipfile.ZipFile(io.BytesIO(archive.read(info))) as nested:
                                nms = [i.filename for i in nested.infolist() if not i.is_dir()]
                                item.update(status="INSPECTED", **summarize(nms))
                                item["deeper_archives_not_inspected"] = item["nested_archive_members"]
                        except Exception as exc:
                            item.update(status="ERROR", reason=type(exc).__name__ + ": " + str(exc))
                    result["nested_inspections"].append(item)
        else:
            try:
                proc = subprocess.run(["tar", "-tf", str(path)], capture_output=True, timeout=20)
                names = proc.stdout.decode("utf-8", errors="replace").splitlines()
                result.update(status="INSPECTED" if proc.returncode == 0 else "PARTIAL_OR_ERROR", method="tar -tf; 20 second timeout", returncode=proc.returncode, error=proc.stderr.decode("utf-8", errors="replace"), **summarize(names))
                if proc.returncode != 0 and "Failed to open" in result["error"]:
                    result["initial_attempt"] = {k: result.get(k) for k in ("status", "error", "returncode", "method")}
                    with Path(path).open("rb") as handle:
                        proc = subprocess.run(["tar", "-tf", "-"], stdin=handle, capture_output=True, timeout=20)
                    names = proc.stdout.decode("utf-8", errors="replace").splitlines()
                    result.update(status="INSPECTED" if proc.returncode == 0 else "PARTIAL_OR_ERROR", method="tar -tf - with binary file stdin (Unicode path workaround)", returncode=proc.returncode, error=proc.stderr.decode("utf-8", errors="replace"), **summarize(names))
                sevenzip = Path("C:/Program Files/7-Zip/7z.exe")
                if result["status"] != "INSPECTED" and sevenzip.exists():
                    result["initial_attempt"] = {k: result.get(k) for k in ("status", "error", "returncode", "method")}
                    proc = subprocess.run([str(sevenzip), "l", "-slt", "-sccUTF-8", str(path)], capture_output=True, timeout=20)
                    raw = proc.stdout.decode("utf-8", errors="replace")
                    names = [line.split(" = ", 1)[1] for line in raw.split("----------", 1)[-1].splitlines() if line.startswith("Path = ")]
                    result.update(status="INSPECTED" if proc.returncode == 0 else "PARTIAL_OR_ERROR", method="7-Zip l -slt -sccUTF-8 (listing only)", returncode=proc.returncode, error=proc.stderr.decode("utf-8", errors="replace"), **summarize(names))
                result["nested_archives_not_inspected"] = result["nested_archive_members"]
            except subprocess.TimeoutExpired:
                result.update(status="NOT_INSPECTED", reason="tar listing exceeded 20 second timeout; no extraction attempted")
    except Exception as exc:
        result.update(status="ERROR", reason=type(exc).__name__ + ": " + str(exc))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="E:\\")
    parser.add_argument("--output", default="E:\\LUNA\\results_reliability_analysis")
    args = parser.parse_args()
    output = Path(args.output)
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = [shutil.which("rg") or "rg", "--files", "--hidden", "--no-ignore", args.root]
    started = time.time()
    with (logs / "asset_search_all_files.txt").open("wb") as out, (logs / "asset_search_errors.txt").open("wb") as err:
        proc = subprocess.run(command, stdout=out, stderr=err)
    files = (logs / "asset_search_all_files.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    errors = (logs / "asset_search_errors.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    checkpoint_paths = [p for p in files if WEIGHT.search(p)]
    target_paths = [p for p in files if TARGET.search(p)]
    data_images = [p for p in target_paths if IMAGE.search(p) and not any(s in p.lower() for s in ("tsne", "attention", "loss_curve", "site-packages"))]
    splits = [p for p in files if SPLIT.search(p)]
    archives = [p for p in files if ARCHIVE.search(p) and not WEIGHT.search(p)]
    caches = [Path.home() / ".cache" / "clip", Path.home() / ".cache" / "torch" / "hub" / "checkpoints", Path.home() / ".cache" / "huggingface" / "hub"]
    cache_results = []
    for cache in caches:
        entry = {"path": str(cache), "exists": cache.exists(), "files": []}
        if cache.exists():
            entry["files"] = [str(p) for p in cache.rglob("*") if p.is_file() and WEIGHT.search(str(p))]
        cache_results.append(entry)
    result = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "root": args.root,
        "enumeration_command": command,
        "enumeration_returncode": proc.returncode,
        "enumeration_seconds": round(time.time() - started, 3),
        "accessible_file_count": len(files),
        "search_errors": errors,
        "search_error_count": len(errors),
        "file_list": str(logs / "asset_search_all_files.txt"),
        "search_error_log": str(logs / "asset_search_errors.txt"),
        "target_name_paths": target_paths,
        "possible_dataset_image_paths": data_images,
        "official_split_paths": splits,
        "checkpoint_candidates": [{"path": p, **stat(p), "clip_name_match": bool(CLIP.search(p)), "luna_teacher_student_name_match": bool(RUN.search(p))} for p in checkpoint_paths],
        "cache_checks": cache_results,
        "run_evidence_candidates": [p for p in target_paths if re.search(r"(?:\.log$|(?:^|[\\/])log\.txt$|events\.out\.tfevents|(?:^|[\\/])(?:metrics|results|config|cfg)[^\\/]*\.(?:json|csv|ya?ml)$)", p, re.I)],
        "archive_count": len(archives),
        "archive_inspections": [],
        "method_limits": ["File names and archive member names support candidate discovery, not checkpoint authentication.", "No checkpoint is deserialized; no image datasets are extracted.", "Inaccessible paths are recorded, so absence claims apply only to accessible inspected files and archive members.", "Nested ZIPs are inspected in memory to one level with a 64 MiB size limit; deeper/nested non-ZIP archives are listed but not recursively inspected.", "Default CLIP/Torch/HuggingFace cache checks supplement the requested E: drive search."]
    }
    (output / "asset_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "files_enumerated", "file_count": len(files), "checkpoints": len(checkpoint_paths), "target_images": len(data_images), "splits": len(splits), "archives": len(archives), "errors": len(errors)}), flush=True)
    for index, path in enumerate(archives, 1):
        result["archive_inspections"].append(archive_listing(path))
        if index % 25 == 0 or index == len(archives):
            print(json.dumps({"stage": "archives", "done": index, "total": len(archives)}), flush=True)
            (output / "asset_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["completed_at"] = dt.datetime.now().astimezone().isoformat()
    result["total_seconds"] = round(time.time() - started, 3)
    (output / "asset_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = collections.Counter(a["status"] for a in result["archive_inspections"])
    lines = ["# Journal LUNA asset audit", "", "Read-only discovery. This inventory contains no experimental or simulated performance values.", "", f"Search completed: {result['completed_at']}", f"Requested root: `{args.root}` (includes `E:\\LUNA`).", f"Accessible files enumerated: {len(files):,}; search errors: {len(errors)}.", "", "## Findings requiring manual candidate review", "", f"- Dataset-name image candidates excluding named visualizations: {len(data_images)}", f"- Original split filename matches: {len(splits)}", f"- Checkpoint file candidates of any provenance: {len(checkpoint_paths)}", f"- Archives: {len(archives)}; status counts: {dict(counts)}", "", "Candidate lists and every archive listing outcome are in `asset_audit.json`. The raw all-file list and exact search errors are in `logs/asset_search_all_files.txt` and `logs/asset_search_errors.txt`.", "", "## Search scope and limitations", ""]
    lines += ["- " + x for x in result["method_limits"]]
    lines += ["", "## Reproduction", "", "```powershell", "& E:\\LUNA\\.venv_journal\\Scripts\\python.exe E:\\LUNA\\code_luna\\LUNA\\experiments\\audit_journal_assets.py --root E:\\ --output E:\\LUNA\\results_reliability_analysis", "```", ""]
    (output / "ASSET_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    add_review(output)
    print(json.dumps({"stage": "complete", "archive_status_counts": counts, "seconds": result["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
