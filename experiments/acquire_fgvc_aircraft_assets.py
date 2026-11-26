"""Download and verify official FGVC-Aircraft data without changing source files."""
from __future__ import annotations

import collections
from datetime import datetime, timezone
import hashlib
import gzip
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import time

import requests

ROOT = Path(r"E:\LUNA\data")
LOG_ROOT = Path(r"E:\LUNA\results_reliability_analysis\logs")
REPORT = LOG_ROOT / "asset_acquisition_fgvc_aircraft.json"
URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
PAGE = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/"
MD5_PAGE = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.html"
OFFICIAL_LISTED_MD5 = "32eca553f897e747706144f3bb444df0"
EXPECTED_MD5 = "d4acdd33327262359767eeaa97a4f732"
MAINTAINER_REFERENCE = "https://raw.githubusercontent.com/mlverse/torchvision/ef409505221fa210eaaef67b1fa3c014fce8a72c/R/dataset-fgvc.R"
EXPECTED_BYTES = 2753340328


def digest(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    return {"sha256": sha.hexdigest(), "md5": md5.hexdigest(), "size_bytes": Path(path).stat().st_size}


def save(report):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, REPORT)


def acquire_archive(report):
    path = ROOT / "downloads" / "fgvc-aircraft-2013b.tar.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    entry = report["archive"] = {"path": str(path), "source_url": URL, "expected_size_bytes": EXPECTED_BYTES,
                                  "expected_md5": EXPECTED_MD5, "expected_md5_source": MAINTAINER_REFERENCE,
                                  "official_listed_md5": OFFICIAL_LISTED_MD5, "official_listed_md5_source": MD5_PAGE,
                                  "official_listed_md5_matches": False,
                                  "checksum_discrepancy": "Current Oxford download matches maintained R torchvision reference; Oxford's linked checksum page lists a different digest. Both are retained explicitly."}
    if path.exists():
        value = digest(path)
        if value["md5"] != EXPECTED_MD5:
            raise ValueError("Existing archive has a different checksum; no overwrite attempted")
        entry.update(status="REUSED_HASH_VERIFIED", **value)
        return path
    failures = []
    for attempt in range(1, 5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == EXPECTED_BYTES:
                break
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(URL, stream=True, headers=headers, timeout=(30, 90)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError("Resume range not honored; partial file preserved")
                entry.update(http_status=response.status_code, final_url=response.url,
                             response_headers=dict(response.headers), resumed_from_bytes=offset)
                count, last = offset, time.monotonic()
                with partial.open("ab" if offset else "xb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
                            count += len(block)
                        if time.monotonic() - last >= 20:
                            entry.update(status="DOWNLOADING", downloaded_bytes=count)
                            save(report)
                            print(json.dumps({"stage": "download", "bytes": count, "total": EXPECTED_BYTES}), flush=True)
                            last = time.monotonic()
            break
        except Exception as exc:
            failures.append({"attempt": attempt, "error": type(exc).__name__ + ": " + str(exc)})
            entry["attempt_errors"] = failures
            save(report)
            print(json.dumps({"stage": "download_retry", **failures[-1]}), flush=True)
            if attempt == 4:
                raise
    value = digest(partial)
    if value["md5"] != EXPECTED_MD5 or value["size_bytes"] != EXPECTED_BYTES:
        raise ValueError("Downloaded archive failed published MD5 or expected byte-count verification")
    if path.exists():
        raise FileExistsError("Archive destination appeared during download; no overwrite attempted")
    partial.rename(path)
    entry.update(status="DOWNLOADED_AND_MAINTAINER_REFERENCE_MD5_VERIFIED", **value)
    save(report)
    print(json.dumps({"stage": "download_complete", **value}), flush=True)
    return path


def extract_data(archive, report):
    target = ROOT / "fgvc_aircraft"
    target.mkdir(parents=True, exist_ok=True)
    extracted, reused, skipped = 0, 0, 0
    report["status"] = "EXTRACTING"
    save(report)
    # Stream the tar once. Only the official data/ subtree is remapped, preserving
    # every image and metadata byte; differing existing user files are refused.
    with tarfile.open(archive, "r|gz") as handle:
        for member in handle:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != "fgvc-aircraft-2013b" or ".." in parts:
                raise ValueError("Unexpected/unsafe archive member: " + member.name)
            if len(parts) < 2 or parts[1] != "data":
                skipped += 1
                continue
            rel = Path(*parts[2:])
            destination = (target / rel).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise ValueError("Archive destination escaped dataset root")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with handle.extractfile(member) as stream:
                    if destination.exists():
                        incoming = hashlib.sha256()
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            incoming.update(block)
                        if digest(destination)["sha256"] != incoming.hexdigest():
                            raise ValueError("Existing dataset file differs; no overwrite: " + str(destination))
                        reused += 1
                    else:
                        with destination.open("xb") as output:
                            shutil.copyfileobj(stream, output, length=1024 * 1024)
                        extracted += 1
                if (extracted + reused) % 1000 == 0:
                    print(json.dumps({"stage": "extract", "new_files": extracted, "identical_reused": reused}), flush=True)
            else:
                raise ValueError("Refusing linked/special archive member: " + member.name)
    report["extraction"] = {"dataset_root": str(target), "source_subtree": "fgvc-aircraft-2013b/data",
                            "new_files": extracted, "identical_files_reused": reused,
                            "non_data_members_retained_in_archive": skipped,
                            "differing_existing_files_overwritten": 0}
    return target


def verify(target, report):
    from journal_preflight import _inspect_dataset
    blockers = []
    inspection = _inspect_dataset("fgvc_aircraft", ROOT, blockers)
    variants = (target / "variants.txt").read_text(encoding="utf-8").splitlines()
    classmap = {name: i for i, name in enumerate(variants)}
    rows = {}
    for split in ("train", "val", "test"):
        lines = (target / f"images_variant_{split}.txt").read_text(encoding="utf-8").splitlines()
        rows[split] = [line.split(maxsplit=1) for line in lines if line.strip()]
    reference_sets = {k: {x[0] for x in values} for k, values in rows.items()}
    overlaps = {a + "_" + b: len(reference_sets[a] & reference_sets[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    image_files = list((target / "images").glob("*.jpg"))
    metadata = [target / "variants.txt"] + [target / f"images_variant_{k}.txt" for k in rows]
    base_n = (len(variants) + 1) // 2
    report.update(dataset_preflight=inspection, dataset_preflight_blockers=blockers,
                  metadata_hashes={str(p): digest(p) for p in metadata},
                  image_count=len(image_files), class_count=len(variants), class_names=variants,
                  class_map=classmap, split_counts={k: len(v) for k, v in rows.items()},
                  per_class_split_counts={k: dict(collections.Counter(name for _, name in v)) for k, v in rows.items()},
                  pairwise_split_overlap_counts=overlaps,
                  within_split_duplicate_reference_counts={k: len(rows[k]) - len(s) for k, s in reference_sets.items()},
                  base_class_count=base_n, novel_class_count=len(variants) - base_n,
                  base_test_images=sum(classmap[name] < base_n for _, name in rows["test"]),
                  novel_test_images=sum(classmap[name] >= base_n for _, name in rows["test"]),
                  split_generation_or_repair=False)
    if blockers or any(overlaps.values()) or len(image_files) != inspection.get("unique_image_count"):
        raise ValueError("Downloaded dataset did not pass complete split/image verification: " + str(blockers))
    report["status"] = "READY_DATASET_AND_ORIGINAL_SPLITS"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["preflight_source"] = {"path": str(Path(__file__).with_name("journal_preflight.py")), **digest(Path(__file__).with_name("journal_preflight.py"))}
    save(report)
    print(json.dumps({k: report[k] for k in ("status", "image_count", "class_count", "split_counts", "pairwise_split_overlap_counts", "base_test_images", "novel_test_images")}), flush=True)


def main():
    report = {"dataset": "FGVCAircraft", "status": "ACQUIRING", "started_at": datetime.now(timezone.utc).isoformat(),
              "data_root": str(ROOT), "official_dataset_page": PAGE,
              "layout_source": "https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#fgvcaircraft"}
    try:
        docs = LOG_ROOT / "fgvc_aircraft_source_documents"
        docs.mkdir(parents=True, exist_ok=True)
        report["source_documents"] = []
        for name, url in (("Oxford_FGVC_Aircraft.html", PAGE), ("official_md5.html", MD5_PAGE),
                          ("mlverse_torchvision_dataset_fgvc.R", MAINTAINER_REFERENCE)):
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            path = docs / name
            path.write_bytes(response.content)
            report["source_documents"].append({"url": url, "path": str(path), **digest(path)})
            if url == MD5_PAGE and OFFICIAL_LISTED_MD5 not in response.text:
                raise ValueError("Published official archive checksum differs from inspected value")
            if url == MAINTAINER_REFERENCE and (EXPECTED_MD5 not in response.text or URL not in response.text):
                raise ValueError("Pinned maintainer reference does not authenticate expected archive URL/checksum")
        save(report)
        archive = acquire_archive(report)
        report["status"] = "VERIFYING_GZIP_CRC"
        save(report)
        unpacked_bytes = 0
        with gzip.open(archive, "rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                unpacked_bytes += len(chunk)
        report["gzip_integrity"] = {"status": "PASS_COMPLETE_STREAM_CRC_AND_SIZE", "uncompressed_bytes": unpacked_bytes}
        print(json.dumps({"stage": "gzip_integrity_verified", "uncompressed_bytes": unpacked_bytes}), flush=True)
        target = extract_data(archive, report)
        verify(target, report)
    except Exception as exc:
        report.update(status="FAILED_ACQUISITION_OR_VERIFICATION", error=type(exc).__name__ + ": " + str(exc),
                      last_updated_at=datetime.now(timezone.utc).isoformat())
        save(report)
        raise


if __name__ == "__main__":
    main()
