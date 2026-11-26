"""Show weaker measured Aircraft responses using a shared monotonic color scale."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from attention_figure import METHODS, load_bundle, render_bundle, shared_row_maps


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(root):
    original = root / "requested_originals/FGVC_Aircraft.npz"
    before = sha(original)
    bundle = load_bundle(original)
    filenames = [Path(x["impath"]).name for x in bundle["metadata"]["samples"]]
    if filenames != ["0780417.jpg", "1376307.jpg", "1048572.jpg"]:
        raise ValueError("The three original Aircraft inputs are required")
    if bundle["methods"] != list(METHODS) or bundle["missing"]:
        raise ValueError("All five measured methods are required")
    # This preset applies everywhere, including background and every baseline.
    gamma = 0.6
    checks = []
    for row, image in enumerate(bundle["images"]):
        display, limits = shared_row_maps(bundle["maps"][row], image.shape[:2])
        for method, values in zip(METHODS, display):
            unique = np.unique(values)
            if len(unique) > 1 and not (np.diff(unique ** gamma) > 0).all():
                raise ValueError("The display transform must preserve response ordering")
            transformed = values ** gamma
            if not np.array_equal(values == values.max(), transformed == transformed.max()):
                raise ValueError("The maximum response locations changed")
        checks.append({"row": row + 1, "all_method_response_orders_preserved": True,
                       "all_method_maximum_locations_preserved": True,
                       "shared_range": list(limits)})
    output = root / "aircraft_response_detail"
    output.mkdir(exist_ok=True)
    copied = output / "attention_bundle.npz"
    shutil.copyfile(original, copied)
    receipt = render_bundle(copied, output, cmap="jet", alpha=0.48, dpi=600,
                            show_scores=True, display_gamma=gamma)
    if sha(copied) != before or sha(original) != before:
        raise ValueError("Measured source data changed")
    for extension in ("png", "pdf", "svg"):
        shutil.copyfile(output / ("attention_comparison." + extension),
                        output / ("FGVC_Aircraft_detail." + extension))
    caption = r"""\caption{Class-conditioned relevance on the original three FGVC Aircraft inputs. Columns show Input, CLIP, CoOp, KgCoOp, DeKg, and LUNA. Relevance uses the ground-truth cosine minus the mean of up to five highest-scoring incorrect classes, with gradient-weighted attention rollout through the last six visual blocks. The native maps are normalized to unit sum, bilinearly resized, and displayed using one pooled min--max range per input. A shared power color scale ($\gamma=0.6$) reveals weaker measured responses; the colorbar uses the same scale. Jet and opacity 0.48 are identical for all methods. The transform preserves response ordering and hotspot locations. The red border identifies LUNA.}
"""
    (output / "caption.tex").write_text(caption, encoding="utf-8")
    (output / "README_zh.md").write_text("""# Aircraft 弱响应显示细节版

三张输入与原附件一致：Challenger 600、Tornado、MD-11。
列顺序：Input | CLIP | CoOp | KgCoOp | DeKg | LUNA。

FGVC_Aircraft_detail.png / .pdf / .svg 是统一 gamma=0.6 色阶的版本，显露原始数组中的弱响应。所有方法、全部位置共用同一规则；背景弱响应也会显露。没有新增热点、移动热点或按飞机区域增亮。数据排序及极大值位置保持不变。色阶变换在图内和 caption.tex 中注明，定量指标依旧来自未经色阶变换的原始数组。

该显示版本不代表模型重新训练后改善了注意力。原来的线性色阶图仍保存在 requested_originals；本包的 attention_bundle.npz 与原始数组逐字节相同。

原始三例的官方飞机框内归因质量：
- Challenger 600：LUNA 69.53%，最高基线 KgCoOp 66.65%。
- Tornado：LUNA 75.54%，最高基线 CLIP 72.26%。
- MD-11：LUNA 53.22%，各基线为 59.82%–63.52%。
这些框是粗略前景代理，并非像素级分割；不能据此声称 LUNA 三张图都更关注飞机。前两张可分别讨论已有的局部响应，第三张需如实保留背景响应。

在有 NumPy、Pillow、Matplotlib 的环境中可复现本包图件：
python attention_figure.py attention_bundle.npz --output-dir rerender --display-gamma 0.6
""", encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("attention_figure.py"), output / "attention_figure.py")
    files = [output / ("FGVC_Aircraft_detail." + ext) for ext in ("png", "pdf", "svg")]
    files += [output / name for name in ("attention_bundle.npz", "render_manifest.json", "sample_metrics.csv",
                                         "caption.tex", "README_zh.md", "attention_figure.py")]
    verification = {"status": "VERIFIED_MEASURED_RESPONSE_DISPLAY", "source_sha256": before,
                    "source_unchanged": True, "copy_identical": True, "display_gamma": gamma,
                    "method_specific_transform": False, "spatial_mask": False,
                    "checks": checks, "artifacts": [{"path": p.name, "sha256": sha(p)} for p in files]}
    (output / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    files.append(output / "verification.json")
    archive = root / "Aircraft_response_detail.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in files:
            handle.write(path, path.name)
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise ValueError("ZIP verification failed")
    print(json.dumps({"status": verification["status"], "output": str(output), "archive": str(archive),
                      "source_unchanged": sha(original) == before, "gamma": receipt["display_gamma"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/LUNA/attention_comparison"))
    export(parser.parse_args().root.resolve())
