"""Validate downloaded datasets with LUNA's real dataset classes, without a GPU."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("E:/LUNA/data"))
    parser.add_argument("--output-root", type=Path, default=Path("E:/LUNA/results_reliability_analysis"))
    args = parser.parse_args()
    import outdated
    outdated.check_outdated = None
    import train
    from dassl.config import get_cfg_default
    from datasets.dtd import DescribableTextures
    from datasets.fgvc_aircraft import FGVCAircraft
    from datasets.eurosat import EuroSAT
    from experiments.journal_runtime import dataset_fingerprint, source_hashes
    from PIL import Image

    classes = {"DTD": DescribableTextures, "FGVCAircraft": FGVCAircraft, "EuroSAT": EuroSAT}
    prerequisites = {
        "DTD": ["dtd/images", "dtd/split_zhou_DescribableTextures.json"],
        "FGVCAircraft": ["fgvc_aircraft/images", "fgvc_aircraft/variants.txt", "fgvc_aircraft/images_variant_train.txt",
                         "fgvc_aircraft/images_variant_val.txt", "fgvc_aircraft/images_variant_test.txt"],
        "EuroSAT": ["eurosat/2750", "eurosat/split_zhou_EuroSAT.json"],
    }
    for dataset, required in prerequisites.items():
        missing = [relative for relative in required if not (args.data_root / relative).exists()]
        if missing:
            raise FileNotFoundError(f"{dataset}: required original files missing: {missing}")
    initial_sources = source_hashes()
    output = {"status": "PASS_ACTUAL_DATASET_LOADERS", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "data_root": str(args.data_root.resolve()), "core_sources_sha256": initial_sources,
              "check_scope": "Actual LUNA dataset classes and one decoded RGB image per class per train/Base/Novel loader. Archive/split authenticity is in separate acquisition receipts. No student/teacher model or optimizer is created.",
              "datasets": {}}
    for name, cls in classes.items():
        cfg = get_cfg_default()
        train.extend_cfg(cfg)
        cfg.merge_from_file(str(REPO / "configs/datasets" / {"DTD": "dtd.yaml", "FGVCAircraft": "fgvc_aircraft.yaml", "EuroSAT": "eurosat.yaml"}[name]))
        cfg.DATASET.ROOT = str(args.data_root.resolve())
        cfg.DATASET.NUM_SHOTS = 0
        cfg.DATASET.SUBSAMPLE_CLASSES = "all"
        cfg.DATASET.FEWSHOT_CACHE_ROOT = str(args.output_root / "logs/dataset_loader_check_cache")
        cfg.TRAINER.NAME = "LUNA"
        cfg.TRAINER.MODAL = "base2novel"
        cfg.freeze()
        dataset = cls(cfg)
        split_items = {"unlabeled_train": dataset.train_x, "base_test": dataset.val, "novel_test": dataset.test}
        sets, samples, split_summary = {}, {}, {}
        for part, items in split_items.items():
            paths = [str(Path(item.impath).resolve()) for item in items]
            if len(paths) != len(set(paths)):
                raise ValueError(f"{name}/{part} duplicate image references")
            absent = [path for path in paths if not Path(path).is_file()]
            if absent:
                raise FileNotFoundError(f"{name}/{part}: {len(absent)} referenced images missing")
            sets[part] = set(paths)
            label_counts = Counter(item.label for item in items)
            classnames = {str(item.label): item.classname for item in items}
            if sorted(label_counts) != list(range(len(label_counts))):
                raise ValueError(f"{name}/{part}: noncontiguous loader class labels")
            for item in items:
                samples.setdefault((part, item.label), item.impath)
            split_summary[part] = {"images": len(items), "classes": len(label_counts),
                                   "images_per_class": dict(sorted(label_counts.items())), "label_to_classname": classnames}
        overlaps = {"train_base": len(sets["unlabeled_train"] & sets["base_test"]),
                    "train_novel": len(sets["unlabeled_train"] & sets["novel_test"]),
                    "base_novel": len(sets["base_test"] & sets["novel_test"])}
        if any(overlaps.values()):
            raise ValueError(f"{name}: overlapping train/test image references: {overlaps}")

        def decode(path):
            with Image.open(path) as im:
                im.load()
                rgb = im.convert("RGB")
                if min(rgb.size) <= 0:
                    raise ValueError(f"Invalid image dimensions: {path}")
                return {"path": str(path), "width": rgb.width, "height": rgb.height, "mode": im.mode}

        with ThreadPoolExecutor(max_workers=4) as pool:
            decoded = list(pool.map(decode, samples.values()))
        row = {"status": "PASS", "dataset_class": cls.__name__, "splits": split_summary,
               "path_overlap_counts": overlaps, "dataset_fingerprint": dataset_fingerprint(dataset),
               "representative_images_fully_decoded": len(decoded), "decoded_images": decoded,
               "uses_unlabeled_novel_class_training_images": True, "generated_split": False}
        output["datasets"][name] = row
        print(json.dumps({"dataset": name, "status": "PASS", "counts": {k: v["images"] for k, v in split_summary.items()},
                          "decoded_samples": len(decoded), "overlaps": overlaps}), flush=True)
    if source_hashes() != initial_sources:
        raise RuntimeError("Core source changed during download/loader validation")
    destination = args.output_root / "logs/dataset_loader_validation.json"
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(destination), flush=True)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    main()
