#!/usr/bin/env python3
"""Build class-aligned inference configs for the fully trained comparisons."""
import argparse
import json
from pathlib import Path

NAMES = {"eurosat": "EuroSAT", "fgvc_aircraft": "FGVC Aircraft", "stanford_cars": "Stanford Cars"}


def make(dataset, root):
    manifest = json.loads((root/"datasets"/(dataset+".json")).read_text(encoding="utf-8-sig"))
    classes = manifest["classnames"]
    specs = [{"name": "CLIP"}]
    for method in ("CoOp", "KgCoOp", "DeKg", "LUNA"):
        specs.append({"name": method, "checkpoint": str((root/"training"/dataset/method/"attention_checkpoint.pth").resolve()),
                      "checkpoint_classnames": classes,
                      "ctx_init": "" if method=="LUNA" else "a photo of a"})
    config = {"dataset": dataset, "title": NAMES[dataset], "classnames": classes,
              "clip_path": str(Path.home()/".cache/clip/ViT-B-16.pt"), "methods": specs,
              "figure_note": "Same target, crop, scale and opacity across methods",
              "variant_note": "LUNA: original repository legacy student; freshly trained final epoch 5"}
    out = root/"configs"; out.mkdir(parents=True, exist_ok=True)
    (out/(dataset+"_models.json")).write_text(json.dumps(config, indent=2), encoding="utf-8")
    matching = out/(dataset+"_original_inputs.json")
    if matching.is_file():
        original = json.loads(matching.read_text(encoding="utf-8"))
        samples = original["samples"]
        cutoff = (len(classes)+1)//2
        novel = all(classes.index(s["classname"]) >= cutoff for s in samples)
        base = all(classes.index(s["classname"]) < cutoff for s in samples)
        if not (novel or base):
            raise ValueError("Original rows span base/novel classes; separate evaluation sets required")
        config.update(classnames=classes[cutoff:] if novel else classes[:cutoff], samples=samples,
                      sample_selection="Matched original user-provided inputs; no reselection based on new method outputs")
        (out/(dataset+"_original.json")).write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Inference configs ready:", dataset)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=tuple(NAMES))
    p.add_argument("--output-root", type=Path, default=Path("E:/LUNA/attention_comparison"))
    a = p.parse_args(); make(a.dataset, a.output_root)
