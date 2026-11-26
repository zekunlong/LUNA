"""Export exactly the user's three original panels, preserving all nine inputs."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np


METHODS = ["CLIP", "CoOp", "KgCoOp", "DeKg", "LUNA"]
PANELS = (
    ("eurosat", "EuroSAT", ["Residential Buildings", "Sea or Lake", "Pasture Land"],
     ["Residential_1410.jpg", "SeaLake_2008.jpg", "Pasture_515.jpg"]),
    ("stanford_cars", "Stanford_Cars", ["2012 Ford F-450 Super Duty Crew Cab", "2012 Suzuki Kizashi Sedan", "2012 Mercedes-Benz C-Class Sedan"],
     ["01924.jpg", "01792.jpg", "01668.jpg"]),
    ("fgvc_aircraft", "FGVC_Aircraft", ["Challenger 600", "Tornado", "MD-11"],
     ["0780417.jpg", "1376307.jpg", "1048572.jpg"]),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(root):
    verified = []
    for dataset, title, classes, filenames in PANELS:
        source = root / "figures" / dataset / "original/contrastive_rollout"
        original = json.loads((root / "configs" / (dataset + "_original_inputs.json")).read_text(encoding="utf-8"))
        bundle = source / "attention_bundle.npz"
        with np.load(bundle, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if data["methods"].tolist() != METHODS or data["classnames"].tolist() != classes:
                raise ValueError("Wrong methods or original row order: " + dataset)
            if data["maps"].shape[:2] != (3, 5) or data["images"].shape[0] != 3:
                raise ValueError("Exactly three inputs and five measured maps are required")
            if not np.isfinite(data["maps"]).all() or (data["maps"] < 0).any():
                raise ValueError("Invalid measured maps")
        if len(metadata["samples"]) != 3 or len(original["samples"]) != 3:
            raise ValueError("Unexpected sample count")
        for index, (sample, expected) in enumerate(zip(metadata["samples"], original["samples"])):
            if Path(sample["impath"]) != Path(expected["impath"]) or Path(sample["impath"]).name != filenames[index]:
                raise ValueError("An input image was substituted: " + dataset)
            if sample["classname"] != classes[index] or expected["panel_row"] != index + 1:
                raise ValueError("Original panel row identity changed")
            if sha(Path(sample["impath"])) != sample["image_sha256"]:
                raise ValueError("Original image bytes changed")
        receipt = json.loads((source / "render_manifest.json").read_text(encoding="utf-8"))
        if receipt["status"] != "complete" or receipt["source_sha256"] != sha(bundle):
            raise ValueError("Figure and raw array receipt disagree")
        for extension in ("png", "pdf", "svg"):
            if not (source / ("attention_comparison." + extension)).is_file():
                raise FileNotFoundError(source)
        verified.append((dataset, title, classes, source, metadata))

    output = root / "requested_originals"
    output.mkdir(exist_ok=True)
    files, cards, evidence = [], [], []
    for dataset, title, classes, source, metadata in verified:
        for extension in ("png", "pdf", "svg"):
            origin = source / ("attention_comparison." + extension)
            target = output / (title + "." + extension)
            shutil.copyfile(origin, target)
            if sha(target) != sha(origin):
                raise ValueError("Figure copy differs")
            files.append(target)
        for source_name, suffix in (("attention_bundle.npz", ".npz"), ("render_manifest.json", "_render.json"), ("sample_metrics.csv", "_metrics.csv")):
            target = output / (title + suffix)
            shutil.copyfile(source / source_name, target)
            files.append(target)
        evidence.append({"dataset": dataset, "row_order": classes, "samples": metadata["samples"],
                         "source_bundle": str(source / "attention_bundle.npz"), "source_sha256": sha(source / "attention_bundle.npz")})
        cards.append('<section><h2>' + html.escape(title.replace("_", " ")) + '</h2><p>' +
                     ' · '.join(html.escape(x) for x in classes) + '</p><p>' +
                     ' · '.join('<a href="' + title + '.' + ext + '">' + ext.upper() + '</a>' for ext in ("png", "pdf", "svg")) +
                     '</p><img src="' + title + '.png" alt="' + html.escape(title) + ' six-column comparison"></section>')
    content = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>原始九张输入 · 完整六列对比</title>
<style>body{font:16px/1.7 system-ui,'Microsoft YaHei',sans-serif;margin:32px auto;max-width:1400px;padding:0 20px;color:#222}h1{font-size:27px}h2{font-size:22px}a{color:#245d83}section{margin:36px 0}img{width:100%;height:auto}p{margin:8px 0}</style>
<h1>你指定的三组原图：完整六列对比</h1><p>Input | CLIP | CoOp | KgCoOp | DeKg | LUNA</p>
<p>按本次附件顺序：EuroSAT、Stanford Cars、FGVC Aircraft。每组保留原来的三张输入及行序。</p>""" + "\n".join(cards) + "</html>"
    (output / "index.html").write_text(content, encoding="utf-8")
    (output / "README_zh.md").write_text("""# 仅包含你指定的三组原图

列顺序：Input | CLIP | CoOp | KgCoOp | DeKg | LUNA。

- EuroSAT：Residential Buildings / Sea or Lake / Pasture Land。
- Stanford Cars：2012 Ford F-450 / 2012 Suzuki Kizashi / 2012 Mercedes-Benz C-Class。
- FGVC Aircraft：Challenger 600 / Tornado / MD-11。

打开 index.html 查看三张完整面板；PNG、PDF、SVG 均可直接使用。所有输入和行序与原附件对应，五方法热图来自已经完成的真实模型计算。采用统一 Jet 配色、透明度与逐输入共同显示范围；本包是 contrastive gradient-weighted rollout 版本。热图为本轮重新计算结果，并非旧图已有 CLIP/LUNA 彩色像素的复刻。

此前推荐主图使用了另选成功案例，造成输入不对应。本包只保留以上九张原始输入，未加入成功案例替换图。原始数组、渲染参数和逐图指标一并提供。
""", encoding="utf-8")
    (output / "caption.tex").write_text(r"""\caption{Class-conditioned relevance for the original user-specified images from EuroSAT, Stanford Cars, and FGVC Aircraft. Columns show Input, CLIP, CoOp, KgCoOp, DeKg, and LUNA. Each dataset retains the original three inputs and row order. Maps show gradient-weighted rollout of the ground-truth cosine minus the mean of up to five highest-scoring incorrect classes through the final six visual blocks. All methods share the image crop, relative-mass normalization, per-input display range, Jet colormap, and opacity. The red border identifies LUNA.}
""", encoding="utf-8")
    files += [output / name for name in ("index.html", "README_zh.md", "caption.tex")]
    manifest = {"status": "COMPLETE_EXACT_ORIGINAL_INPUTS", "column_order": ["Input", *METHODS], "input_count": 9,
                "selection": "The user's three original panels only, without substitutions", "panels": evidence,
                "artifacts": [{"path": p.name, "sha256": sha(p)} for p in files]}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(manifest_path)
    archive_path = root / "requested_originals_only.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
    print(json.dumps({"status": manifest["status"], "output": str(output), "archive": str(archive_path), "input_count": 9}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/LUNA/attention_comparison"))
    export(parser.parse_args().root.resolve())
