#!/usr/bin/env python3
"""Export the original contrastive attribution objective and the paper package.

Run after run_attention_queue.py completes. All five methods use the SAME
target-minus-up-to-five-hard-negatives objective, layers and display settings.
The target-score Rollout/Grad-CAM outputs remain available as diagnostics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("fgvc_aircraft", "eurosat", "stanford_cars")
TITLES = {"fgvc_aircraft": "Aircraft", "eurosat": "EuroSAT", "stanford_cars": "Cars"}


def run(command):
    subprocess.run([str(x) for x in command], cwd=REPO, check=True)


def delivery_notes(root):
    content = """# Attention 对比图交付入口

正式交付为你指定的三组原图、九张输入，列顺序为 **Input | CLIP | CoOp | KgCoOp | DeKg | LUNA**。打开 [原图专用浏览页](requested_originals/index.html) 查看，所有行序均与附件一致。热图来自本轮真实模型计算。

| 数据集 | 原始输入顺序 | 正式图件 |
|---|---|---|
| EuroSAT | Residential Buildings / Sea or Lake / Pasture Land | [PNG](requested_originals/EuroSAT.png) · [PDF](requested_originals/EuroSAT.pdf) |
| Stanford Cars | Ford F-450 / Suzuki Kizashi / Mercedes-Benz C-Class | [PNG](requested_originals/Stanford_Cars.png) · [PDF](requested_originals/Stanford_Cars.pdf) |
| FGVC Aircraft | Challenger 600 / Tornado / MD-11 | [PNG](requested_originals/FGVC_Aircraft.png) · [PDF](requested_originals/FGVC_Aircraft.pdf) |

`requested_originals` 同时提供 SVG、原始数组、渲染参数、逐图指标和图注。配色为统一 Jet，使用相同透明度和逐输入共同显示范围。此前推荐的另选成功案例没有对应你的原始输入；正式交付以此处三组为准。

## 原始输入的其他解释方法

| 数据集 | 参考配色 / contrastive rollout | Grad-CAM |
|---|---|---|
| EuroSAT | [PNG](figures/eurosat/original/contrastive_rollout/attention_comparison.png) · [PDF](figures/eurosat/original/contrastive_rollout/attention_comparison.pdf) | [PNG](figures/eurosat/original/gradcam/attention_comparison.png) |
| Aircraft | [PNG](figures/fgvc_aircraft/original/contrastive_rollout/attention_comparison.png) · [PDF](figures/fgvc_aircraft/original/contrastive_rollout/attention_comparison.pdf) | [PNG](figures/fgvc_aircraft/original/gradcam/attention_comparison.png) |
| Cars | [PNG](figures/stanford_cars/original/contrastive_rollout/attention_comparison.png) · [PDF](figures/stanford_cars/original/contrastive_rollout/attention_comparison.pdf) | [PNG](figures/stanford_cars/original/gradcam/attention_comparison.png) |

同目录还提供 SVG、原始 attention_bundle.npz、逐图指标和渲染参数。`figures/<dataset>/selected_success`、`random_control`、`luna_failures` 分别保留成功案例、随机对照及失败案例。

## 补充案例与完整分析

[分析报告](REPORT_zh.md) 和 [完整图件浏览页](review.html) 包含全部评估及案例。`paper/main_comparison.*` 和 `paper/alternatives` 中的跨数据集图使用另选成功案例，仅作为补充材料，不是本次指定的九张原图。

## 数值和复现

- `evaluation/<dataset>`：完整测试集 Base / Novel / HM、逐图预测及固定选样规则。
- `localization_audit`：预测无关的 100 张飞机 / 98 张汽车图、原始响应、官方框指标及配对区间。
- `training/<dataset>/<method>`：训练协议、最终 compact checkpoint，以及基线的逐轮指标。训练预算及数据使用差异在分析报告中列明；ZIP 不包含完整大模型或完整运行日志。
- `validation`：模型身份与评估一致性检查。
- 具体案例表述：[Aircraft 中英文分析](validation/fgvc_qualitative_notes.md) · [EuroSAT 中英文分析](validation/eurosat_qualitative_notes.md) · [Cars 中英文分析](validation/stanford_cars_qualitative_notes.md)。
- 本地代码：`E:/LUNA/code_luna/LUNA/tools`；解释器：`E:/LUNA/.venv_journal/Scripts/python.exe`。

从现有模型重新出图，可运行 `tools/finalize_attention_product.py`。重新组织完整顺序队列使用 `tools/prepare_attention_queue.py` 和 `tools/run_attention_queue.py --plan E:/LUNA/attention_comparison/queue_plan.json`；队列会识别已完成结果。正在运行的并行队列由各自计划管理，请勿在运行中另起同一训练。
"""
    (root/"README_zh.md").write_text(content, encoding="utf-8")
    (root/"paper/include_main.tex").write_text(r"""% Insert from the product root, or adjust the two paths to your paper project.
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{paper/main_comparison.pdf}
  \input{paper_caption_primary.tex}
  \label{fig:luna-attribution-comparison}
\end{figure*}
""", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("E:/LUNA/attention_comparison"))
    p.add_argument("--dataset", choices=DATASETS, help="Incrementally finalize one finished dataset")
    a = p.parse_args()
    root = a.root.resolve()
    selected_data = (a.dataset,) if a.dataset else DATASETS
    for dataset in selected_data:
        evaluation = root/"evaluation"/dataset
        report = json.loads((evaluation/"evaluation_summary.json").read_text(encoding="utf-8"))
        if report["status"] != "COMPLETE_HELDOUT_EVALUATION":
            raise ValueError("Complete held-out evaluation required: "+dataset)
        for group in ("original", "selected_success", "random_control", "luna_failures"):
            config = (root/"configs"/(dataset+"_original.json") if group=="original"
                      else evaluation/"selections"/(group+".json"))
            if not json.loads(config.read_text(encoding="utf-8"))["samples"]:
                print("No cases satisfy selection:", dataset, group); continue
            out = root/"figures"/dataset/group/"contrastive_rollout"
            if not (out/"attention_comparison.pdf").is_file():
                run([sys.executable, "tools/visualize_attention_comparison.py", "infer", "--config", config,
                     "--output-dir", out, "--attribution", "rollout", "--score", "target-margin", "--negative-k", "5"])
            if dataset != "eurosat":
                run([sys.executable, "tools/attention_localization.py", "--bundle", out/"attention_bundle.npz",
                     "--output-dir", out])
        if dataset != "eurosat":
            audit = root/"localization_audit"/dataset/"summary.json"
            if not audit.is_file() or json.loads(audit.read_text(encoding="utf-8")).get("status") != "COMPLETE_PREDICTION_INDEPENDENT_LOCALIZATION_AUDIT":
                run([sys.executable, "tools/audit_attention_localization.py", "--datasets", dataset,
                     "--comparison-root", root, "--output-root", root/"localization_audit"])
    if a.dataset:
        return
    paper = root/"paper"; paper.mkdir(exist_ok=True)
    combined = {key: [] for key in ("images", "maps", "classnames", "predictions", "target_probabilities", "correct")}
    sources = []
    for dataset in DATASETS:
        source = root/"figures"/dataset/"selected_success/contrastive_rollout"
        path = source/"attention_bundle.npz"
        with np.load(path, allow_pickle=False) as data:
            if data["methods"].tolist() != ["CLIP", "CoOp", "KgCoOp", "DeKg", "LUNA"]:
                raise ValueError("All five real methods are required in the paper figure")
            for key in combined:
                value = data[key][0]
                if key == "classnames":
                    value = TITLES[dataset]+" · "+str(value)
                combined[key].append(value)
            info = json.loads(str(data["metadata"].item()))
            sources.append({"dataset": dataset, "bundle": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "source_row": 0, "sample": info["samples"][0],
                            "class_order": info["class_order"], "effective_negative_k": min(5, len(info["class_order"])-1),
                            "sample_selection": info["sample_selection"]})
        for extension in ("png", "pdf", "svg"):
            shutil.copyfile(source/("attention_comparison."+extension), paper/(dataset+"."+extension))
    metadata = {"display_name": "Class-conditioned relevance across datasets",
                "dataset": "combined", "figure_note": "Selected LUNA success cases · one per dataset",
                "selection": "First ranked case from each dataset's declared success selection; no visual reranking",
                "sources": sources, "inference_settings": {"attribution": "class-contrastive gradient-weighted attention rollout",
                "rollout_last_layers": 6, "hard_negative_k": 5, "rollout_power": 1,
                "precision": "float32", "shared_image_encoder": "frozen ViT-B/16"}}
    bundle_path = paper/"main_bundle.npz"
    np.savez_compressed(bundle_path, **{k: np.stack(v) for k,v in combined.items()},
                        methods=np.asarray(["CLIP", "CoOp", "KgCoOp", "DeKg", "LUNA"]),
                        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    from attention_figure import render_bundle
    render_bundle(bundle_path, paper)
    for extension in ("png", "pdf", "svg"):
        shutil.copyfile(paper/("attention_comparison."+extension), paper/("main_comparison."+extension))
    run([sys.executable, "tools/export_attention_alternatives.py", "--root", root])
    run([sys.executable, "tools/build_attention_report.py", "--root", root, "--require-complete"])
    run([sys.executable, "tools/export_requested_originals.py", "--root", root])
    delivery_notes(root)
    run([sys.executable, "tools/package_attention_delivery.py", "--root", root])
    print("Final paper package:", paper)


if __name__ == "__main__":
    main()
