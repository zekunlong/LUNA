#!/usr/bin/env python3
"""Recover original input filenames from the existing three-row figure panels.

Pixel matching is used ONLY to identify already chosen images, never to infer
heatmaps. Attribution must be recomputed from real model checkpoints.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def clip_crop(image, size=224):
    w, h = image.size
    new = (size, int(size*h/w)) if w < h else (int(size*w/h), size)
    image = image.resize(new, Image.Resampling.BICUBIC)
    x, y = round((image.width-size)/2), round((image.height-size)/2)
    return image.crop((x, y, x+size, y+size))


def match(panel_path, manifest_path, requested, output):
    source = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    panel = Image.open(panel_path).convert("RGB")
    if panel.size != (3295, 3612):
        raise ValueError("Matching coordinates are calibrated for original 3295x3612 source panels")
    samples = []
    for row, classname in enumerate(requested):
        target = np.asarray(panel.crop((166, 281+1135*row, 1176, 1291+1135*row)).resize(
            (32,32), Image.Resampling.BILINEAR), dtype=np.float32)/255
        ranked = []
        seen = set()
        for split in ("train", "val", "test"):
            for item in source[split]:
                if item["classname"] != classname or item["impath"] in seen:
                    continue
                seen.add(item["impath"])
                with Image.open(item["impath"]) as image:
                    small = np.asarray(clip_crop(image.convert("RGB")).resize((32,32),
                                       Image.Resampling.BILINEAR), dtype=np.float32)/255
                error = float(np.sqrt(np.mean((target-small)**2)))
                ranked.append((error, split, item))
        ranked.sort(key=lambda value: value[0])
        if not ranked:
            raise ValueError("No source images for "+classname)
        error, split, item = ranked[0]
        if error > .04:
            raise ValueError(f"No reliable pixel match for {classname}: RMSE {error}")
        record = {**item, "source_split": split, "pixel_match_rmse": error,
                  "next_best_rmse": ranked[1][0] if len(ranked)>1 else None,
                  "selection": "same input as user-supplied original panel", "panel_row": row+1}
        samples.append(record)
        print(json.dumps(record), flush=True)
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"source_panel": str(Path(panel_path).resolve()),
                                       "samples": samples}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--classes", nargs=3, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    match(a.panel, a.manifest, a.classes, a.output)
