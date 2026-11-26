"""Acquire official DTD images and the unmodified CoOp Zhou split.

Writes a provenance report, verifies the official archive checksum and every
split reference, and never generates or repairs a split.
"""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time

import requests

ROOT = Path(r"E:\LUNA\data")
LOG = Path(r"E:\LUNA\results_reliability_analysis\logs\asset_acquisition_dtd.json")
DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
SPLIT_URL = "https://drive.google.com/uc?id=1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x&export=download"
COOP_URL = "https://raw.githubusercontent.com/KaiyangZhou/CoOp/main/DATASETS.md"
ARCHIVE_MD5 = "fff73e5086ae6bdbea199a49dfb8a4c1"


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(data)
            md5.update(data)
    return {"sha256": sha.hexdigest(), "md5": md5.hexdigest(), "size_bytes": Path(path).stat().st_size}


def download_dataset():
    archive = ROOT / "downloads" / "dtd-r1.0.1.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    part = archive.with_suffix(archive.suffix + ".part")
    result = {"requested_url": DATASET_URL, "path": str(archive), "expected_md5": ARCHIVE_MD5,
              "expected_md5_evidence": r"E:\LUNA\.venv_journal\Lib\site-packages\torchvision\datasets\dtd.py"}
    if archive.exists() and hashes(archive)["md5"] == ARCHIVE_MD5:
        result.update(status="REUSED_VERIFIED_DOWNLOAD", **hashes(archive))
        return result
    errors = []
    for attempt in range(1, 4):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": "bytes=" + str(offset) + "-"} if offset else {}
            with requests.get(DATASET_URL, stream=True, headers=headers, timeout=(30, 60)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError("Server did not honor resume range; preserving partial download")
                result.update(final_url=response.url, http_status=response.status_code,
                              response_headers=dict(response.headers), resumed_from_bytes=offset)
                count, last = offset, time.monotonic()
                with part.open("ab" if offset else "wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            count += len(chunk)
                        if time.monotonic() - last >= 20:
                            print(json.dumps({"stage": "dataset_download", "bytes": count, "expected_total": 625239812}), flush=True)
                            last = time.monotonic()
            digest = hashes(part)
            if digest["md5"] != ARCHIVE_MD5:
                raise RuntimeError("Official DTD archive MD5 mismatch: " + digest["md5"])
            os.replace(part, archive)
            result.update(status="DOWNLOADED_AND_HASH_VERIFIED", **digest, prior_attempt_errors=errors)
            print(json.dumps({"stage": "dataset_download_complete", **digest}), flush=True)
            return result
        except Exception as exc:
            errors.append({"attempt": attempt, "error": type(exc).__name__ + ": " + str(exc)})
            print(json.dumps({"stage": "dataset_download_retry", **errors[-1]}), flush=True)
    result.update(status="FAILED", errors=errors)
    return result


def download_split():
    split = ROOT / "dtd" / "split_zhou_DescribableTextures.json"
    split.parent.mkdir(parents=True, exist_ok=True)
    result = {"requested_url": SPLIT_URL, "authoritative_link_source": COOP_URL,
              "path": str(split), "generated_or_repaired": False}
    try:
        response = requests.get(SPLIT_URL, timeout=(30, 60))
        response.raise_for_status()
        raw = response.content
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or set(parsed) != {"train", "val", "test"}:
            raise ValueError("Downloaded file is not the original train/val/test split JSON")
        if split.exists() and split.read_bytes() != raw:
            raise RuntimeError("Existing split differs from original download; no overwrite attempted")
        split.write_bytes(raw)
        result.update(status="DOWNLOADED_ORIGINAL_SPLIT", final_url=response.url,
                      response_headers=dict(response.headers), split_counts={key: len(value) for key, value in parsed.items()}, **hashes(split))
    except Exception as exc:
        result.update(status="FAILED", error=type(exc).__name__ + ": " + str(exc))
    print(json.dumps({"stage": "split_download", **{key: result[key] for key in ("status", "path", "split_counts", "sha256", "error") if key in result}}), flush=True)
    return result


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    result = {"started_at": dt.datetime.now().astimezone().isoformat(), "status": "ACQUIRING",
              "data_root": str(ROOT), "dataset_authority": "https://www.robots.ox.ac.uk/~vgg/data/dtd/",
              "split_authority": "https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#dtd"}
    LOG.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(download_dataset): "archive", pool.submit(download_split): "zhou_split"}
        for future in concurrent.futures.as_completed(futures):
            result[futures[future]] = future.result()
            LOG.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["archive"]["status"] == "FAILED" or result["zhou_split"]["status"] == "FAILED":
        result["status"] = "ACQUISITION_INCOMPLETE"
        LOG.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 1
    archive_path = Path(result["archive"]["path"])
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            rel = PurePosixPath(member.name)
            destination = (ROOT / member.name).resolve()
            if rel.is_absolute() or ".." in rel.parts or not destination.is_relative_to(ROOT.resolve()):
                raise ValueError("Unsafe archive member path: " + member.name)
            if not (member.isdir() or member.isfile()):
                raise ValueError("Unexpected linked/special archive member: " + member.name)
            if rel.parts and rel.parts[0] != "dtd":
                raise ValueError("Unexpected dataset archive root: " + member.name)
        archive.extractall(ROOT)
        result["extraction"] = {"destination": str(ROOT), "members": len(members), "method": "tarfile after path, type and dataset-root validation"}
    from journal_preflight import _inspect_dataset
    blockers = []
    result["dataset_preflight"] = _inspect_dataset("dtd", ROOT, blockers)
    result["dataset_preflight_blockers"] = blockers
    images = sorted((ROOT / "dtd" / "images").glob("*/*.jpg"))
    classes = sorted(p.name for p in (ROOT / "dtd" / "images").iterdir() if p.is_dir())
    result["image_count"] = len(images)
    result["class_count"] = len(classes)
    result["per_class_image_counts"] = {name: len(list((ROOT / "dtd" / "images" / name).glob("*.jpg"))) for name in classes}
    result["status"] = "READY_DATASET_AND_ORIGINAL_SPLIT" if not blockers and len(images) == 5640 and len(classes) == 47 else "VALIDATION_FAILED"
    result["finished_at"] = dt.datetime.now().astimezone().isoformat()
    documents = LOG.parent / "dtd_source_documents" / "manifest.json"
    if documents.exists():
        result["source_document_snapshots"] = json.loads(documents.read_text(encoding="utf-8"))
    result["preflight_source"] = {"path": str(Path(__file__).with_name("journal_preflight.py")), **hashes(Path(__file__).with_name("journal_preflight.py"))}
    LOG.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "complete", "status": result["status"], "images": len(images), "classes": len(classes), "blockers": blockers}), flush=True)
    return 0 if result["status"] == "READY_DATASET_AND_ORIGINAL_SPLIT" else 1


if __name__ == "__main__":
    sys.exit(main())
