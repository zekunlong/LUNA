#!/usr/bin/env python3
"""Measure native relevance inside dataset bounding boxes, after the actual crop.

Bounding boxes are annotation-based coarse foreground proxies, not segmentation
masks. Partial patch overlaps are weighted by area. No hand-drawn ROI is used.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image


def crop_box(box, image_size, resolution=224):
    """Map zero-based, half-open xyxy coordinates to CLIP's resized center crop."""
    w, h = image_size
    rw, rh = ((resolution, int(resolution*h/w)) if w < h
              else (int(resolution*w/h), resolution))
    ox, oy = round((rw-resolution)/2), round((rh-resolution)/2)
    x1, y1, x2, y2 = box
    return np.clip([x1*rw/w-ox, y1*rh/h-oy, x2*rw/w-ox, y2*rh/h-oy], 0, resolution)


def box_metrics(raw, box, resolution=224):
    raw = np.asarray(raw, dtype=np.float64)
    h, w = raw.shape
    x1, y1, x2, y2 = box
    xs, ys = np.linspace(0, resolution, w+1), np.linspace(0, resolution, h+1)
    horizontal = np.maximum(0, np.minimum(xs[1:], x2)-np.maximum(xs[:-1], x1))/(resolution/w)
    vertical = np.maximum(0, np.minimum(ys[1:], y2)-np.maximum(ys[:-1], y1))/(resolution/h)
    overlap = vertical[:,None]*horizontal[None,:]
    area = float(overlap.mean())
    invalid = ("nonfinite_map" if not np.isfinite(raw).all() else
               "negative_map" if (raw < 0).any() else
               "zero_map" if raw.sum() <= 0 else
               "empty_cropped_bbox" if area <= 0 else None)
    if invalid:
        return {"bbox_mass": None, "background_mass": None, "bbox_area_fraction": area,
                "bbox_enrichment": None, "pointing_inside_bbox": None,
                "invalid_reason": invalid, "pointing_status": "undefined_invalid_map"}
    mass = float(np.clip((raw/raw.sum()*overlap).sum(), 0, 1))
    peak_y, peak_x = np.unravel_index(raw.argmax(), raw.shape)
    px, py = (peak_x+.5)*resolution/w, (peak_y+.5)*resolution/h
    unique_peak = np.count_nonzero(raw == raw.max()) == 1
    return {"bbox_mass": mass, "background_mass": 1-mass, "bbox_area_fraction": area,
            "bbox_enrichment": mass/area,
            "pointing_inside_bbox": bool(x1<=px<x2 and y1<=py<y2) if unique_peak else None,
            "invalid_reason": None,
            "pointing_status": "defined" if unique_peak else "undefined_tied_maxima"}


def annotations(dataset, root):
    if dataset == "fgvc_aircraft":
        boxes = {}
        for line in (root/"fgvc_aircraft/images_box.txt").read_text().splitlines():
            key, *coords = line.split()
            x1, y1, x2, y2 = map(float, coords)
            boxes[key] = [x1-1, y1-1, x2, y2]
        return boxes, "Official images_box.txt: 1-based inclusive xyxy converted to zero-based half-open"
    if dataset == "stanford_cars":
        from scipy.io import loadmat
        boxes = {}
        for folder, file in (("cars_train", "devkit/cars_train_annos.mat"),
                             ("cars_test", "cars_test_annos_withlabels.mat")):
            rows = loadmat(root/"stanford_cars"/file, squeeze_me=True)["annotations"]
            for row in rows:
                name = str(row["fname"].item()) if isinstance(row["fname"], np.ndarray) else str(row["fname"])
                boxes[folder+"/"+name] = [float(row["bbox_x1"])-1, float(row["bbox_y1"])-1,
                                               float(row["bbox_x2"]), float(row["bbox_y2"])]
        return boxes, "Official Cars MAT bounding boxes: 1-based inclusive converted to zero-based half-open"
    raise ValueError("Bounding boxes are available only for Aircraft and Cars")


def analyze(bundle_path, data_root, output):
    with np.load(bundle_path, allow_pickle=False) as f:
        metadata = json.loads(str(f["metadata"].item()))
        maps, methods, labels = f["maps"], f["methods"].tolist(), f["classnames"].tolist()
        resolution = f["images"].shape[1]
    dataset = metadata["dataset"]
    boxes, provenance = annotations(dataset, Path(data_root))
    records = []
    for i, sample in enumerate(metadata["samples"]):
        path = Path(sample["impath"])
        key = path.stem if dataset == "fgvc_aircraft" else path.parent.name+"/"+path.name
        if key not in boxes:
            raise ValueError("No official bounding box for "+str(path))
        with Image.open(path) as image:
            box = crop_box(boxes[key], image.size, resolution)
        for j, method in enumerate(methods):
            records.append({"row": i+1, "classname": labels[i], "method": method,
                            **box_metrics(maps[i,j], box, resolution)})
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    with (out/"localization_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    values_by_method = {m: [r["bbox_mass"] for r in records if r["method"]==m and r["bbox_mass"] is not None] for m in methods}
    report = {"dataset": dataset, "annotation_source": provenance,
              "samples": len(labels), "sample_selection": metadata.get("sample_selection"),
              "limitations": "Bounding boxes include background inside the rectangle. Selected cases are not an unbiased test-set estimate.",
              "valid_samples": {m: len(values) for m, values in values_by_method.items()},
              "mean_bbox_mass": {m: float(np.mean(values)) if values else None for m, values in values_by_method.items()}}
    (out/"localization_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--data-root", default="E:/LUNA/data", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    analyze(a.bundle, a.data_root, a.output_dir)
