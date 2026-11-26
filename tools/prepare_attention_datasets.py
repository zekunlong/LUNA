#!/usr/bin/env python3
"""Export original dataset splits for controlled prompt-training comparisons.

No images or labels are sampled, reordered by display name, or reassigned.
The trainer selects its declared base-class / few-shot protocol afterwards.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


SPLITS = ("train", "val", "test")
EXPECTED_CLASSES = {"eurosat": 10, "fgvc_aircraft": 100, "stanford_cars": 196}
REPO_ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_zhou(path, image_root):
    result = {}
    source = json_load(path)
    for split in SPLITS:
        result[split] = [
            {"impath": str((image_root / str(impath).replace("\\", "/")).resolve()), "label": int(label), "classname": str(classname)}
            for impath, label, classname in source[split]
        ]
    labels = {}
    for split in SPLITS:
        for row in result[split]:
            old = labels.setdefault(row["label"], row["classname"])
            if old != row["classname"]:
                raise ValueError("Inconsistent class name for label {}".format(row["label"]))
    if sorted(labels) != list(range(len(labels))):
        raise ValueError("Dataset labels must be contiguous zero-based original labels")
    return [labels[label] for label in range(len(labels))], result


def read_aircraft(root):
    classnames = (root / "variants.txt").read_text(encoding="utf-8").splitlines()
    if len(set(classnames)) != len(classnames):
        raise ValueError("variants.txt contains duplicate class names")
    lookup = {name: index for index, name in enumerate(classnames)}
    result = {}
    sources = [root / "variants.txt"]
    for split in SPLITS:
        source = root / ("images_variant_" + split + ".txt")
        sources.append(source)
        rows = []
        for line in source.read_text(encoding="utf-8").splitlines():
            image_id, classname = line.split(" ", 1)
            rows.append({"impath": str((root / "images" / (image_id + ".jpg")).resolve()), "label": lookup[classname], "classname": classname})
        result[split] = rows
    return classnames, result, sources


def validate(dataset, classnames, splits):
    if len(classnames) != EXPECTED_CLASSES[dataset] or len(set(classnames)) != len(classnames):
        raise ValueError("Unexpected class count or duplicate class names for " + dataset)
    seen = set()
    counts = {}
    for split in SPLITS:
        counts[split] = len(splits[split])
        per_class = Counter()
        for row in splits[split]:
            if not 0 <= row["label"] < len(classnames) or classnames[row["label"]] != row["classname"]:
                raise ValueError("Class-order mismatch: " + repr(row))
            key = str(Path(row["impath"]).resolve()).casefold()
            if key in seen:
                raise ValueError("Duplicate image or overlapping dataset splits: " + row["impath"])
            if not Path(row["impath"]).is_file():
                raise FileNotFoundError("Dataset image missing: " + row["impath"])
            seen.add(key)
            per_class[row["label"]] += 1
        if set(per_class) != set(range(len(classnames))):
            raise ValueError("Every official class must appear in each original split: " + split)
    return counts


def prepare(dataset, data_root, output_dir):
    root = (data_root / dataset).resolve()
    if dataset == "fgvc_aircraft":
        classnames, splits, sources = read_aircraft(root)
        split_origin = "Original FGVC Aircraft images_variant_{train,val,test}.txt"
    elif dataset == "eurosat":
        sources = [root / "split_zhou_EuroSAT.json"]
        classnames, splits = read_zhou(sources[0], root / "2750")
        split_origin = "Existing prompt-learning split_zhou_EuroSAT.json; unchanged"
    else:
        sources = [root / "split_zhou_StanfordCars.json"]
        classnames, splits = read_zhou(sources[0], root)
        split_origin = "Stanford Cars source train/test and recorded train/validation split; see acquisition provenance"
    counts = validate(dataset, classnames, splits)
    acquisition = {}
    for candidate in (root / "acquisition_provenance.json", root / "provenance.json"):
        if candidate.is_file():
            acquisition[str(candidate)] = json_load(candidate)
    source_document = REPO_ROOT / "datasets" / (dataset + ".py")
    output = {
        "schema_version": 1,
        "dataset": dataset,
        "classnames": classnames,
        **splits,
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "data_root": str(data_root.resolve()),
            "split_origin": split_origin,
            "class_order": "Original integer label order from source annotations; no sorting of display names",
            "counts": counts,
            "source_files": [{"path": str(path), "sha256": digest(path)} for path in sources],
            "dataset_loader": {"path": str(source_document), "sha256": digest(source_document)},
            "acquisition_provenance": acquisition,
            "validation": "All image files exist; splits disjoint; every split covers all classes; exact label/name mapping verified",
            "few_shot_selection": "None here; separately recorded by the training command",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / (dataset + ".json")
    # A manifest is reproducible input; do not silently overwrite one referenced
    # by already running experiments with different split membership.
    if destination.exists():
        previous = json_load(destination)
        if any(previous.get(key) != output[key] for key in ("classnames", *SPLITS)):
            raise FileExistsError("Existing manifest has different samples: " + str(destination))
        print(json.dumps({"dataset": dataset, "status": "EXISTING_IDENTICAL", "path": str(destination), "counts": counts}))
    else:
        destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"dataset": dataset, "status": "READY", "path": str(destination), "counts": counts}))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(EXPECTED_CLASSES), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for dataset in args.datasets:
        prepare(dataset, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
