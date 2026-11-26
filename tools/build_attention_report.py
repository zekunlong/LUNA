#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an incremental Chinese comparison report from measured artifacts only.

No model inference, training, figure editing, or external data download occurs.
Only evaluation/{dataset}, figures/{dataset}/{set}/{attribution}, and training
protocol receipts under --root contribute evidence. Smoke/checks folders never
contribute results. Missing files remain explicitly pending. --require-complete
returns exit code 2 until all requested evaluations, full figures, and required
Aircraft/Cars localization summaries are available and validated.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import quote


DATASETS = {"eurosat": "EuroSAT", "fgvc_aircraft": "FGVC Aircraft", "stanford_cars": "Stanford Cars"}
METHODS = ("CLIP", "CoOp", "KgCoOp", "DeKg", "LUNA")
RIVALS = METHODS[:-1]
GROUPS = {"original": "原始输入", "selected_success": "筛选成功案例",
          "random_control": "随机对照", "luna_failures": "LUNA 失败案例"}
REQUIRED_ATTRIBUTIONS = ("gradcam", "rollout")
ATTRIBUTIONS = {"gradcam": "Grad-CAM", "rollout": "目标 cosine 梯度加权 rollout",
                "contrastive_rollout": "目标减负类 margin 梯度加权 rollout"}
EXPECTED_COLUMNS = ["Input", *METHODS]
EXPECTED_NORMALIZATION = "native unit sum; bilinear; pooled min-max across methods per image"
GUIDELINES = "https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines"
DEFAULT_ROOT = Path("E:/LUNA/attention_comparison")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def boolean(value):
    if value is True or str(value).lower() == "true":
        return True
    if value is False or str(value).lower() == "false":
        return False
    return None


def fmt(value, signed=False, digits=2):
    value = number(value)
    return "待生成" if value is None else format(value, ("+" if signed else "") + f".{digits}f")


def md(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def relative_link(path, root):
    return quote(Path(path).resolve().relative_to(root.resolve()).as_posix(), safe="/")


def link(label, path, root):
    return f"[{md(label)}]({relative_link(path, root)})"


class Evidence:
    def __init__(self, root):
        self.root = root
        self.sources = {}
        self.issues = []

    def issue(self, path, message):
        self.issues.append({"path": str(path), "reason": message})

    def read_json(self, path):
        if not path.is_file():
            self.issue(path, "文件尚未生成")
            return None
        try:
            payload = path.read_bytes()
            record = json.loads(payload.decode("utf-8-sig"))
            if not isinstance(record, dict):
                raise ValueError("JSON 顶层必须是对象")
            self.sources[str(path)] = hashlib.sha256(payload).hexdigest()
            return record
        except (OSError, UnicodeError, ValueError) as error:
            self.issue(path, "文件读取/校验失败：" + str(error))
            return None

    def read_csv(self, path):
        if not path.is_file():
            self.issue(path, "文件尚未生成")
            return None
        try:
            payload = path.read_bytes()
            rows = list(csv.DictReader(payload.decode("utf-8-sig").splitlines()))
            self.sources[str(path)] = hashlib.sha256(payload).hexdigest()
            return rows
        except (OSError, UnicodeError, csv.Error) as error:
            self.issue(path, "CSV 读取失败：" + str(error))
            return None


def evaluation(evidence, dataset):
    path = evidence.root / "evaluation" / dataset / "evaluation_summary.json"
    record = evidence.read_json(path)
    if record is None:
        return None
    try:
        if record.get("dataset") != dataset or record.get("status") != "COMPLETE_HELDOUT_EVALUATION":
            raise ValueError("仅接受对应数据集 COMPLETE_HELDOUT_EVALUATION，排除 smoke 子集")
        if record.get("source_split") != "test" or record["sample_count"] != record["full_test_count"]:
            raise ValueError("不是完整原始 test split")
        if set(record["methods"]) != set(METHODS):
            raise ValueError("必须含完整五种方法")
        expected_counts = None
        for method in METHODS:
            result = record["methods"][method]
            counts = []
            for group in ("base", "novel"):
                measured = result[group]
                count, correct = measured["count"], measured["correct"]
                if type(count) is not int or type(correct) is not int or not 0 <= correct <= count or count == 0:
                    raise ValueError("样本数/正确数无效")
                accuracy = 100.0 * correct / count
                if number(measured["accuracy_percent"]) is None or abs(accuracy - measured["accuracy_percent"]) > 1e-6:
                    raise ValueError("accuracy 与正确数/样本数不一致")
                counts.append(count)
            if sum(counts) != record["sample_count"] or expected_counts not in (None, counts):
                raise ValueError("方法间测试样本数不同")
            expected_counts = counts
            base = result["base"]["accuracy_percent"]
            novel = result["novel"]["accuracy_percent"]
            hm = 2 * base * novel / (base + novel) if base + novel else 0.0
            if number(result["harmonic_mean_percent"]) is None or abs(hm - result["harmonic_mean_percent"]) > 1e-6:
                raise ValueError("HM 与 Base/Novel 不一致")
        return record
    except (KeyError, TypeError, ValueError) as error:
        evidence.issue(path, str(error))
        return None


def localization_audit(evidence, dataset, evaluated):
    """Optional fixed-sample audit; unfinished audits never supply estimates."""
    path = evidence.root / "localization_audit" / dataset / "summary.json"
    if not path.is_file():
        return None
    record = evidence.read_json(path)
    if record is None:
        return None
    if record.get("status") != "COMPLETE_PREDICTION_INDEPENDENT_LOCALIZATION_AUDIT":
        return None
    try:
        expected_classes, expected_images, per_class = (50, 100, 2) if dataset == "fgvc_aircraft" else (98, 98, 1)
        if (record.get("dataset") != dataset or record.get("attribution") != "gradient_weighted_attention_rollout" or
                record.get("score") != "target-margin" or record.get("negative_k") != 5):
            raise ValueError("独立定位审计的数据集或归因设置不匹配")
        selection = record["selection"]
        expected = {"seed": 1, "n_classes": expected_classes, "n_images": expected_images,
                    "images_per_class": per_class, "prediction_independent": True,
                    "source_split": "test", "class_group": "novel"}
        if any(selection.get(key) != value for key, value in expected.items()):
            raise ValueError("独立定位审计抽样协议不匹配")
        bootstrap = record["bootstrap"]
        if (bootstrap.get("unit") != "class" or bootstrap.get("seed") != 1 or
                bootstrap.get("resamples") != 2000 or bootstrap.get("confidence") != .95):
            raise ValueError("独立定位审计必须使用 class-cluster bootstrap 2000 次/seed1/95%")
        if set(record["methods"]) != set(METHODS) or set(record["paired_luna_minus_rival"]) != set(RIVALS):
            raise ValueError("独立定位审计未覆盖全部五方法和四个配对对手")
        for method in METHODS:
            values = record["methods"][method]
            for key in ("bbox_mass", "background_mass", "bbox_enrichment", "pointing_inside_bbox"):
                field = "pointing_accuracy" if key == "pointing_inside_bbox" else "mean_" + key
                measured = number(values.get(field))
                images, classes = values["valid_counts"][key], values["valid_class_counts"][key]
                if type(images) is not int or type(classes) is not int or not 0 <= classes <= expected_classes or not 0 <= images <= expected_images:
                    raise ValueError("无效的定位指标有效图/类别数")
                if measured is None:
                    if images or classes:
                        raise ValueError("定位指标未定义却报告了有效样本")
                elif measured < 0 or (key != "bbox_enrichment" and measured > 1):
                    raise ValueError("定位指标超出有效范围")
        for rival in RIVALS:
            for key in ("bbox_mass", "background_mass", "bbox_enrichment", "pointing_inside_bbox"):
                paired = record["paired_luna_minus_rival"][rival][key]
                estimate = number(paired.get("estimate"))
                images, classes = paired["valid_images"], paired["valid_classes"]
                if type(images) is not int or type(classes) is not int or not 0 <= classes <= expected_classes or not 0 <= images <= expected_images:
                    raise ValueError("配对定位指标的样本数无效")
                interval = paired.get("ci95")
                if estimate is None:
                    if images or classes or interval is not None:
                        raise ValueError("未定义配对估计与有效样本记录不一致")
                elif interval is not None:
                    if (not isinstance(interval, list) or len(interval) != 2 or
                            any(number(bound) is None for bound in interval) or interval[0] > interval[1]):
                        raise ValueError("配对 bootstrap 区间无效")
                elif classes >= 2:
                    raise ValueError("至少两个有效类别却缺失配对 bootstrap 区间")
        if evaluated:
            audit_clip_hash = record.get("checkpoint_hashes", {}).get("CLIP") or record.get("model_metadata", {}).get("clip_sha256")
            if audit_clip_hash != evaluated["model_metadata"].get("clip_sha256"):
                raise ValueError("独立定位审计与分类评估的冻结 CLIP checkpoint 不一致")
            for method in METHODS[1:]:
                audit_hash = record.get("checkpoint_hashes", {}).get(method)
                if audit_hash is None:
                    audit_hash = record.get("model_metadata", {}).get("methods", {}).get(method, {}).get("checkpoint_sha256")
                eval_hash = evaluated["model_metadata"]["methods"][method].get("checkpoint_sha256")
                if not audit_hash or audit_hash != eval_hash:
                    raise ValueError("独立定位审计与分类评估 checkpoint 不一致：" + method)
        return record
    except (KeyError, TypeError, ValueError) as error:
        evidence.issue(path, "已完成的附加审计未通过校验：" + str(error))
        return None


def figure(evidence, dataset, group, attribution, evaluated):
    directory = evidence.root / "figures" / dataset / group / attribution
    result = {"dataset": dataset, "group": group, "attribution": attribution,
              "directory": directory, "status": "PENDING", "rows": [], "localization": None}
    # Empty sets must be justified by the complete evaluator, not missing files.
    selection = (evaluated or {}).get("selections", {}).get(group, {})
    if group != "original" and selection.get("actual_count") == 0:
        result.update(status="EMPTY_EVIDENCED", selection=selection)
        return result
    manifest = evidence.read_json(directory / "render_manifest.json")
    rows = evidence.read_csv(directory / "sample_metrics.csv")
    if manifest is None or rows is None:
        return result
    try:
        if manifest.get("status") != "complete" or manifest.get("missing_methods") or manifest.get("column_order") != EXPECTED_COLUMNS:
            raise ValueError("图不是完整 Input|CLIP|CoOp|KgCoOp|DeKg|LUNA 六列")
        if manifest.get("normalization") != EXPECTED_NORMALIZATION:
            raise ValueError("显示归一化与约定不一致")
        if manifest.get("quantile_clipping") is not False or manifest.get("method_specific_style") is not False:
            raise ValueError("存在方法特定显示处理或 quantile clipping")
        metadata = manifest["inference_metadata"]
        settings = metadata["inference_settings"]
        expected_attribution = "gradcam" if attribution == "gradcam" else "gradient_weighted_attention_rollout"
        if metadata.get("dataset") != dataset or settings.get("attribution") != expected_attribution:
            raise ValueError("数据集或归因方法记录不匹配")
        expected_score = "target-margin" if attribution == "contrastive_rollout" else "target"
        if settings.get("score") != expected_score or settings.get("shared_visual_encoder") is not True:
            raise ValueError("冻结视觉编码器/真实类归因目标与图组约定不一致")
        if attribution == "contrastive_rollout" and settings.get("negative_k") != 5:
            raise ValueError("contrastive rollout 应使用 negative-k=5（实际数量受类别数限制）")
        if attribution != "gradcam" and settings.get("rollout_last_layers") != 6:
            raise ValueError("当前图注对应最后 6 个视觉 block；实际 rollout 层数不匹配")
        if not metadata.get("forward_parity", {}).get("passed"):
            raise ValueError("归因前向一致性检查未通过")
        samples = metadata["samples"]
        if not samples or len(rows) != len(samples) * len(METHODS):
            raise ValueError("样本行数与五种方法的完整指标表不匹配")
        grouped = defaultdict(dict)
        for row in rows:
            index = int(row["row"])
            method = row["method"]
            if method not in METHODS or method in grouped[index]:
                raise ValueError("重复或未知方法指标")
            if not 1 <= index <= len(samples) or row["classname"] != samples[index - 1]["classname"]:
                raise ValueError("样本类别与归因记录不一致")
            if boolean(row.get("correct")) is None:
                raise ValueError("缺失真实分类正确性记录")
            grouped[index][method] = row
        if any(set(method_rows) != set(METHODS) for method_rows in grouped.values()):
            raise ValueError("某个样本缺少方法列")
        for extension in ("png", "pdf"):
            target = directory / f"attention_comparison.{extension}"
            if not target.is_file() or target.stat().st_size == 0:
                raise ValueError("PNG/PDF 导出不完整")
        bundle = Path(manifest["source_bundle"])
        if not bundle.is_file() or sha256(bundle) != manifest.get("source_sha256"):
            raise ValueError("归因 bundle 缺失或其哈希不匹配")
        if evaluated:
            if evaluated["model_metadata"].get("clip_sha256") != metadata["model_metadata"].get("clip_sha256"):
                raise ValueError("图与全量评估使用了不同冻结 CLIP checkpoint")
            for method in RIVALS[1:] + ("LUNA",):
                eval_hash = evaluated["model_metadata"]["methods"][method].get("checkpoint_sha256")
                figure_hash = metadata["model_metadata"]["methods"][method].get("checkpoint_sha256")
                if not eval_hash or eval_hash != figure_hash:
                    raise ValueError("图与全量评估使用了不同 checkpoint：" + method)
        result.update(status="COMPLETE", rows=rows, grouped=grouped, manifest=manifest, samples=samples,
                      png=directory / "attention_comparison.png", pdf=directory / "attention_comparison.pdf")
        if dataset != "eurosat":
            localization = evidence.read_json(directory / "localization_summary.json")
            if localization is not None:
                masses = localization.get("mean_bbox_mass", {})
                if localization.get("dataset") != dataset or localization.get("samples") != len(samples) or set(masses) != set(METHODS):
                    evidence.issue(directory / "localization_summary.json", "定位统计数据集/方法/样本数不匹配")
                elif any(not (number(masses[m]) is not None and 0 <= masses[m] <= 1) and
                         not (masses[m] is None and localization.get("valid_samples", {}).get(m) == 0)
                         for m in METHODS):
                    evidence.issue(directory / "localization_summary.json", "bbox mass 无效且没有零有效样本的说明")
                else:
                    result["localization"] = localization
        return result
    except (KeyError, TypeError, ValueError, OSError) as error:
        evidence.issue(directory, str(error))
        result["status"] = "PENDING"
        return result


def metric(result, name):
    return result["harmonic_mean_percent"] if name == "HM" else result[name.lower()]["accuracy_percent"]


def best_rival(evaluated, name):
    values = {method: metric(evaluated["methods"][method], name) for method in RIVALS}
    maximum = max(values.values())
    tied = "/".join(method for method, value in values.items() if abs(value - maximum) < 1e-9)
    return tied, maximum


def table(headers, rows):
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    output.extend("| " + " | ".join(md(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def baseline_and_luna_protocols(evidence):
    records = {}
    for dataset in DATASETS:
        for method in METHODS[1:]:
            filename = "training_manifest.json" if method == "LUNA" else "protocol.json"
            path = evidence.root / "training" / dataset / method / filename
            if path.is_file():
                record = evidence.read_json(path)
                if record:
                    records[(dataset, method)] = (path, record)
    return records


def audit_section(evidence, audits):
    lines = ["## 预测无关的独立抽样定位审计", "",
             "这项附加审计独立于成功案例筛选：Aircraft 预定在每个 Novel 类取 2 张，共 50 类/100 张；Cars 每个 Novel 类取 1 张，共 98 类/98 张，均为 test split、固定 seed=1 且不看预测结果。解释器固定为真实类减至多 5 个 hard negative 均值的梯度加权 rollout，不通过热图效果调整输入或算法。", ""]
    ready = {dataset: record for dataset, record in audits.items() if record is not None}
    if not ready:
        return lines + ["独立抽样审计尚无已完成且通过校验的结果；不填入推测数值。该附加项的等待状态不影响上述主体图件与完整分类评估的完成判定。", ""]
    means, pairs, supported = [], [], []
    metric_names = {"bbox_mass": "框内 mass Δ (pp)", "background_mass": "框外 mass Δ (pp)",
                    "bbox_enrichment": "框内 enrichment Δ (倍)",
                    "pointing_inside_bbox": "峰值入框率 Δ (pp)"}
    for dataset, record in ready.items():
        title = DATASETS[dataset]
        selection = record["selection"]
        lines.extend([f"**{title}**：实际审计 {selection['n_classes']} 个 Novel 类、{selection['n_images']} 张图。证据：" +
                      link("独立抽样审计记录", evidence.root / "localization_audit" / dataset / "summary.json", evidence.root) + "。", ""])
        for method in METHODS:
            values = record["methods"][method]
            displays = []
            for key in metric_names:
                field = "pointing_accuracy" if key == "pointing_inside_bbox" else "mean_" + key
                value = number(values.get(field))
                factor = 1 if key == "bbox_enrichment" else 100
                count = values["valid_counts"][key]
                classes = values["valid_class_counts"][key]
                displays.append("未定义" if value is None else f"{fmt(factor * value)} (图={count}, 类={classes})")
            means.append((title, method, *displays))
        for rival in RIVALS:
            for key, label in metric_names.items():
                paired = record["paired_luna_minus_rival"][rival][key]
                estimate = number(paired.get("estimate"))
                interval = paired.get("ci95")
                factor = 1 if key == "bbox_enrichment" else 100
                estimated_text = "未定义" if estimate is None else fmt(factor * estimate, signed=True)
                interval_text = "未定义" if interval is None else "[" + fmt(factor * interval[0], signed=True) + ", " + fmt(factor * interval[1], signed=True) + "]"
                pairs.append((title, rival, label, estimated_text, interval_text,
                              paired["valid_images"], paired["valid_classes"]))
                if key == "bbox_mass" and interval is not None and interval[0] > 0:
                    supported.append(f"{title} 相对 {rival}（Δ={estimated_text} pp，95% CI {interval_text} pp）")
    lines.extend(["均值先在每个类别内对有效图求平均，再在有效类别间等权平均；括号保留有效图与类别数。enrichment 为框内 mass 除以标注框覆盖面积比例，单位为倍。", "",
                  table(("数据集", "方法", "框内 mass (%)", "框外 mass (%)", "enrichment (倍)", "峰值入框率 (%)"), means), "",
                  "配对差值定义为 LUNA − 对手，只对该对方法共同有效的图先求每类均值，再对类别等权平均。下面的区间对类别整体有放回抽样 2000 次（seed=1）；不会拆开同一类别的图。不同配对的有效集合可能不同，故配对 Δ 未必等于上表两个独立均值直接相减。", "",
                  table(("数据集", "对手", "指标", "LUNA − 对手", "95% class bootstrap CI", "有效图", "有效类"), pairs), ""])
    if supported:
        lines.extend(["本次固定 checkpoint 的独立抽样中，框内 mass 的区间完全落在 LUNA 有利方向的配对为：" + "；".join(supported) + "。这支持相应抽样条件下的粗框定位代理改善。", ""])
    else:
        lines.extend(["已完成审计中，没有框内 mass 配对区间完全落在 LUNA 有利方向的结果，因此不使用该审计宣称稳定的框内定位优势。", ""])
    lines.extend(["这些区间描述**固定的每方法单个 checkpoint，在抽样类别/图像上的不确定性**，不覆盖重新训练 seed 的变化，也不是训练随机性上的显著性结论。它们是逐配对的探索性区间，未做多重比较校正。框外 mass 与框内 mass 互补，不能当作两项独立证据；标注框仍含内部背景。全零图、无有效框及并列最大值等未定义情形按审计有效计数说明保留，不补造定位成功。", ""])
    return lines


def narrative(evidence, evaluations, figures, protocols, audits, complete):
    root = evidence.root
    ready_eval = sum(value is not None for value in evaluations.values())
    ready_figures = sum(item["status"] == "COMPLETE" and item["attribution"] in REQUIRED_ATTRIBUTIONS for item in figures)
    lines = ["# LUNA 五方法对比：真实评估、归因图与案例分析", "",
             ("**状态：全部要求的真实评估与图件已就绪。**" if complete else
              f"**状态：待完成。已验证 {ready_eval}/3 个完整测试集评估、{ready_figures}/24 组完整六列图。未就绪的数据不作性能结论。**"), "",
             "列顺序统一为 **Input | CLIP | CoOp | KgCoOp | DeKg | LUNA**。本报告仅读取本轮正式结果；中间 checkpoint、集成 smoke 检查与旧图不进入下表。", "",
             "## 完整测试集性能", "",
             "Base 和 Novel 分别在对应类别集合内分类；Base 使用官方类别顺序的前 ceil(C/2) 类，Novel 使用其余类。表内为准确率（%）；HM = 2×Base×Novel/(Base+Novel)。下列差值均为**百分点（pp）**，每一指标分别与 CLIP、CoOp、KgCoOp、DeKg 中的最高值比较。", ""]
    rows, delta_rows, dataset_comments = [], [], []
    for dataset, title in DATASETS.items():
        measured = evaluations[dataset]
        if measured is None:
            rows.extend((title, method, "待全量评估", "待全量评估", "待全量评估") for method in METHODS)
            delta_rows.append((title, "待全量评估", "—", "—", "—"))
            continue
        for method in METHODS:
            result = measured["methods"][method]
            rows.append((title, method, *(fmt(metric(result, key)) for key in ("Base", "Novel", "HM"))))
        comparisons = []
        for key in ("Base", "Novel", "HM"):
            rival, maximum = best_rival(measured, key)
            delta = metric(measured["methods"]["LUNA"], key) - maximum
            comparisons.append((key, rival, delta))
            delta_rows.append((title, key, rival, fmt(maximum), fmt(delta, signed=True)))
        statements = []
        for key, rival, delta in comparisons:
            relation = "高于" if delta > 1e-9 else "低于" if delta < -1e-9 else "持平于"
            statements.append(f"{key} {fmt(metric(measured['methods']['LUNA'], key))}%，{relation} {rival}（Δ={fmt(delta, signed=True)} pp）")
        counts = measured["methods"]["LUNA"]
        dataset_comments.append(f"**{title}**：" + "；".join(statements) +
                                f"。本次覆盖 Base {counts['base']['count']} 张、Novel {counts['novel']['count']} 张。" +
                                "证据：" + link("全量评估记录", root / "evaluation" / dataset / "evaluation_summary.json", root) + "。")
    lines.extend([table(("数据集", "方法", "Base (%)", "Novel (%)", "HM (%)"), rows), "",
                  table(("数据集", "指标", "最高基线", "基线值 (%)", "LUNA − 最高基线 (pp)"), delta_rows), ""])
    lines.extend(item + "\n" for item in dataset_comments)
    if not dataset_comments:
        lines.extend(["目前尚无可用于结论的正式全量评估。完成后，这一节会自动写出 LUNA 的实际优势与不足；不会预设 LUNA 排名。", ""])
    lines.extend(["三数据集均仅为 seed=1 的 checkpoint 结果，不能据此声称多 seed 稳定性或统计显著性。单张图的 p(y) 是该类别集合内的 softmax 输出，不是跨方法校准后的置信度。", "",
                  "## 如何呈现 LUNA 的有效改进", "",
                  "原始输入组完整保留用户给出的样本身份，补齐全部方法。成功案例组用于说明某些真实输入上 LUNA 的识别改进；主文可引用具体类别、哪些基线误判以及对应归因证据。是否存在整体优势必须以完整测试集表为准。", "",
                  "成功案例的预定规则：只从 Novel test 中选择 **LUNA 正确且至少一个其他方法错误** 的图；先按误判的其他方法数量降序，再按 LUNA 的真实类 cosine margin 相对最强对手的增益降序；每类最多一图。margin 是真实类与该方法最高错误类的 cosine 差。没有按热图外观筛选，也没有修改热图热点。", "",
                  "随机对照使用固定 seed=1：先均匀抽取互不重复的 Novel 类别，再在各选中类别中随机取一张图，不筛预测结果。失败案例来自 LUNA 误判的 Novel 图，固定 seed 打乱后取不同类别，不筛其他方法是否正确。每组样本量及不足会如实显示。", ""])
    # One attribution view suffices for classification outcomes; both remain linked.
    by_key = {(item["dataset"], item["group"], item["attribution"]): item for item in figures}
    case_rows = []
    for dataset, title in DATASETS.items():
        measured = evaluations[dataset]
        for group, group_title in GROUPS.items():
            item = by_key[(dataset, group, "gradcam")]
            if item["status"] != "COMPLETE":
                alternative = by_key[(dataset, group, "rollout")]
                if alternative["status"] == "COMPLETE":
                    item = alternative
            if item["status"] == "EMPTY_EVIDENCED":
                lines.extend([f"{title} / {group_title}：完整评估确认没有符合条件的样本，未补造案例。", ""])
            elif item["status"] == "COMPLETE":
                for index in sorted(item["grouped"]):
                    method_rows = item["grouped"][index]
                    sample = method_rows["LUNA"]
                    luna_correct = boolean(sample["correct"])
                    wrong = [name for name in RIVALS if boolean(method_rows[name]["correct"]) is False]
                    right = [name for name in RIVALS if boolean(method_rows[name]["correct"]) is True]
                    comparison = ("误判基线：" + ", ".join(wrong)) if luna_correct and wrong else (
                        "所有基线也正确" if luna_correct else "正确基线：" + (", ".join(right) if right else "无"))
                    case_rows.append((title, group_title, sample["classname"],
                                      "正确" if luna_correct else "错误 → " + sample.get("predictions", "未记录预测类"), comparison))
            if measured and group in measured.get("selections", {}):
                selection = measured["selections"][group]
                if selection.get("shortfall", 0):
                    lines.extend([f"{title} / {group_title}：实际 {selection['actual_count']} 张，预定 {selection['requested_count']} 张；符合条件的不同类别不足，保持实际数量。", ""])
    lines.extend([table(("数据集", "样本组", "真实类别", "LUNA", "同图基线表现"), case_rows) if case_rows else "正式六列案例尚未生成。", "",
                  "这些是诊断案例，不是准确率样本，也不能以成功案例比例代替总体统计。失败组保留 LUNA 仍会混淆的类别；若基线在同图正确，报告会明确列出。", "",
                  "## 归因区域的量化证据", "",
                  "Aircraft 和 Cars 使用数据集原始标注框，按实际 resize/center crop 映射到 224×224，再按每个 patch 与框的面积重叠计算框内 relevance mass。下表仅汇总对应展示样本；Δ 是 LUNA 的平均框内质量减去同组中最大的其他方法平均值，单位 pp。EuroSAT 没有在本流程中使用前景框，因此不报告该指标。", ""])
    box_rows = []
    positive_box = []
    negative_box = []
    for item in figures:
        if not item["localization"]:
            continue
        mass = item["localization"]["mean_bbox_mass"]
        valid_counts = item["localization"].get("valid_samples", {})
        sample_count = item["localization"]["samples"]
        all_defined = (all(number(mass[method]) is not None for method in METHODS) and
                       all(valid_counts.get(method, sample_count) == sample_count for method in METHODS))
        delta = 100 * (mass["LUNA"] - max(mass[method] for method in RIVALS)) if all_defined else None
        identifier = f"{DATASETS[item['dataset']]} / {GROUPS[item['group']]} / {ATTRIBUTIONS[item['attribution']]}"
        box_rows.append((identifier, item["localization"]["samples"],
                         *((fmt(100 * mass[method]) + f" (n={valid_counts.get(method, sample_count)})") if number(mass[method]) is not None else "未定义（0 个有效样本）" for method in METHODS),
                         fmt(delta, signed=True) if delta is not None else "未定义"))
        if delta is not None:
            (positive_box if delta > 1e-9 else negative_box).append((identifier, delta))
    lines.extend([table(("展示组 / 归因", "图数", "CLIP (%)", "CoOp (%)", "KgCoOp (%)", "DeKg (%)", "LUNA (%)", "Δ pp"), box_rows) if box_rows else "正式标注框统计尚未就绪。", ""])
    if positive_box:
        strongest = sorted(positive_box, key=lambda item: -item[1])[:3]
        lines.extend(["当前展示组中有框内 relevance 证据支持 LUNA 更集中的条件包括：" + "；".join(f"{name}（{fmt(delta, True)} pp）" for name, delta in strongest) + "。只能将其表述为这些展示样本上的标注框代理改善。", ""])
    if negative_box:
        weakest = sorted(negative_box, key=lambda item: item[1])[:3]
        lines.extend(["框内指标未超过最强基线的展示组包括：" + "；".join(f"{name}（{fmt(delta, True)} pp）" for name, delta in weakest) + "。因此不能笼统声称 LUNA 在所有样本上定位更准。", ""])
    lines.extend(["单元格 n 为有效 map 数；全零图/裁剪后无有效标注框的定位指标保持未定义。只有全部五方法均覆盖整组样本时才计算 Δ，避免对不同有效子集求差。标注框是粗略代理：矩形内部仍包含背景，框内质量高不等价于部件级定位正确。当前指标也不是分割 IoU。Grad-CAM 与 rollout 的数值只在各自归因设置内比较；两者强调区域不同不构成对任一方法效果的单独证明。", "",
                  "sample_metrics.csv 中的归一化熵和 top-20% patch mass 描述集中程度。低熵/高 top-20% mass 也可能来自错误热点、标识文字或背景，因此本报告不把它们直接解释为更好的目标定位。", "",
                  *audit_section(evidence, audits),
                  "## 图的含义与统一显示规则", "",
                  "五个 student 都使用同一冻结 CLIP ViT-B/16 视觉编码器，适配发生在文本 prompt。相同输入的**原始视觉 self-attention 相同**，不能用未加条件的 raw self-attention 解释方法差异。本套图显示真实类别文本相似度的**目标条件梯度归因**：Grad-CAM 取最后一个视觉 block 的 ln_1 patch 激活与空间平均梯度，排除 CLS 并经 ReLU；rollout 取梯度加权视觉 attention，再进行层间传播。", "",
                  "所有方法解释同一 ground-truth 类的 cosine score，使用相同输入裁剪、共享冻结视觉编码器和 float32 推理；真实类可以与模型预测不同。原生非负 patch relevance 先各自归一到单位和，再双线性上采样；每张输入的五方法共同计算同一 min/max 显示范围，并使用相同色图与 opacity。没有方法特定 quantile 截断、幂变换或透明度增强。颜色表达相对空间质量，不表示跨方法的绝对梯度强度；全零图也不会被补成热点。", "",
                  "如果提供 contrastive_rollout，它解释的是“真实类 cosine − 最接近的至多 5 个错误类 cosine 的均值”，沿用原可视化脚本的对比目标；五方法使用相同规则，负类由各自分数决定。它与直接解释真实类 cosine 的 rollout、Grad-CAM 目标不同，应分开展示。该设置可能减少通用响应，但是否改善目标区域需查看实际图与代理统计，不能预先断言。", "",
                  "Jet 的蓝—青—黄—红配色对应用户提供的参考外观。CVPR 作者指南规定论文版式并要求图文清晰，未规定统一的 attention heatmap 色图；这里应称为“采用参考配色、适合论文排版的图”，而非“CVPR 官方颜色标准”。[CVPR 2026 作者指南](" + GUIDELINES + ")。", "",
                  "## 训练协议与可比性", "",
                  "本轮按固定最终 epoch 导出，不根据 held-out 准确率挑 checkpoint。CoOp/KgCoOp/DeKg 使用 Base 类每类 16 张标注图、seed=1、100 epoch、共享 4-token 文本 context；CoOp batch=32，KgCoOp/DeKg batch=128，SGD lr=0.002、momentum=0.9、weight decay=0.0005，warmup 1 epoch 后 cosine。KgCoOp 的知识约束权重为 8；DeKg 为知识约束 8、官方 HSIC 6。实际参数以逐次 protocol.json 为准。", "",
                  "LUNA 使用原仓库 student 实现、全类别训练划分中的无标签图和 PromptSRC teacher，student 最终 epoch=5，teacher 使用最终 epoch=20。其输入数据范围、监督来源及 teacher 成本与 16-shot Base baselines 不同，**这不是等训练预算/等数据访问的消融**。因此性能差异应归因于各方法在声明协议下的完整系统，而不能只归因于一个 attention 机制。", ""])
    protocol_rows = []
    sources = {}
    for (dataset, method), (path, record) in protocols.items():
        stage = record.get("status", "协议已记录")
        epoch = record.get("epochs", "未记录")
        protocol_rows.append((DATASETS[dataset], method, record.get("seed", "未记录"), epoch, stage,
                              link("训练记录", path, root)))
        source = record.get("source", {})
        if source.get("repository"):
            sources[method] = (source["repository"], source.get("commit", "未记录"))
    lines.extend([table(("数据集", "方法", "seed", "目标 epoch", "记录状态", "证据"), protocol_rows) if protocol_rows else "训练协议记录尚未生成。", ""])
    if sources:
        lines.extend(["本轮已归档的官方实现：" + "；".join(f"[{method}]({url})（commit `{commit}`）" for method, (url, commit) in sources.items()) + "。", ""])
    lines.extend(["## 文件入口", "", "[交互式本地图库](review.html) · [76 词主文图注](paper_caption_primary.tex) · [详细英文图注与复现说明](paper_caption.tex) · [证据与完成状态](report_manifest.json)", "",
                  "每组 PNG 用于快速查看，PDF 用于论文排版；原始 attention_bundle.npz、render_manifest.json、sample_metrics.csv 以及可用的定位统计均保存在对应图目录。完整列表如下。", ""])
    gallery_rows = []
    for item in figures:
        description = f"{DATASETS[item['dataset']]} / {GROUPS[item['group']]} / {ATTRIBUTIONS[item['attribution']]}"
        if item["status"] == "COMPLETE":
            gallery_rows.append((description, "已验证六列", link("PNG", item["png"], root), link("PDF", item["pdf"], root),
                                 link("指标", item["directory"] / "sample_metrics.csv", root)))
        else:
            gallery_rows.append((description, "无符合条件样本" if item["status"] == "EMPTY_EVIDENCED" else "待生成/校验", "—", "—", "—"))
    lines.extend([table(("图组", "状态", "预览", "论文图", "原始统计"), gallery_rows), ""])
    return "\n".join(lines)


def inline_markup(value):
    value = html.escape(value)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', value)
    return value


def markdown_html(markdown):
    lines = markdown.splitlines()
    rendered = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("| ---"):
            cells = lambda text: [inline_markup(cell.strip()) for cell in text.strip("|").split("|")]
            rendered.append('<div class="table-scroll"><table><thead><tr>' + "".join("<th>"+cell+"</th>" for cell in cells(line)) + "</tr></thead><tbody>")
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rendered.append("<tr>" + "".join("<td>" + cell + "</td>" for cell in cells(lines[i])) + "</tr>")
                i += 1
            rendered.append("</tbody></table></div>")
            continue
        if line.startswith("# "):
            rendered.append("<h1>" + inline_markup(line[2:]) + "</h1>")
        elif line.startswith("## "):
            rendered.append("<h2>" + inline_markup(line[3:]) + "</h2>")
        else:
            rendered.append("<p>" + inline_markup(line) + "</p>")
        i += 1
    return "\n".join(rendered)


def gallery_html(report, figures, root, complete):
    cards = []
    for item in figures:
        dataset, group, attribution = (item[key] for key in ("dataset", "group", "attribution"))
        title = " · ".join((DATASETS[dataset], GROUPS[group], ATTRIBUTIONS[attribution]))
        card = f'<article class="card" data-dataset="{dataset}" data-group="{group}" data-attribution="{attribution}"><h3>{html.escape(title)}</h3>'
        if item["status"] == "COMPLETE":
            png, pdf = (relative_link(item[key], root) for key in ("png", "pdf"))
            csv_url = relative_link(item["directory"] / "sample_metrics.csv", root)
            card += f'<a href="{png}"><img loading="lazy" src="{png}" alt="{html.escape(title)}; Input CLIP CoOp KgCoOp DeKg LUNA"></a><p class="downloads"><a href="{png}">PNG 原尺寸</a><a href="{pdf}">PDF</a><a href="{csv_url}">逐样本指标 CSV</a></p>'
        else:
            message = "完整评估确认没有符合筛选条件的样本" if item["status"] == "EMPTY_EVIDENCED" else "等待正式完整六列图；此处不显示旧图或 smoke 草稿"
            card += '<div class="pending">' + message + "</div>"
        cards.append(card + "</article>")
    options = lambda choices: ''.join(f'<option value="{key}">{html.escape(label)}</option>' for key, label in choices.items())
    status = "已完整验证" if complete else "生成中 · 仅显示已验证的正式结果"
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LUNA 五方法真实比较</title><style>
:root{color-scheme:light;--ink:#18242b;--muted:#60717b;--line:#dce3e7;--accent:#a6302d}*{box-sizing:border-box}body{margin:0;background:#f5f7f8;color:var(--ink);font:15px/1.75 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1240px;margin:auto;padding:28px 30px 60px}a{color:#245d83}h1{font-size:29px;letter-spacing:-.025em;line-height:1.4}h2{font-size:21px;margin:32px 0 14px}h3{font-size:16px;margin:0 0 12px}p{margin:12px 0}code{background:#edf1f3;padding:2px 4px;border-radius:3px;font-size:.85em;overflow-wrap:anywhere}.banner{background:#fff;border-left:4px solid var(--accent);padding:12px 18px;margin-bottom:22px}.tabs{display:flex;gap:18px;flex-wrap:wrap}.filters{position:sticky;top:0;z-index:2;display:flex;flex-wrap:wrap;gap:12px;background:#f5f7f8ee;padding:14px 0;backdrop-filter:blur(8px)}label{font-size:13px;color:var(--muted)}select{font:inherit;color:var(--ink);border:1px solid #bcc8ce;border-radius:5px;background:white;padding:7px 11px;margin-left:6px}.card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:18px;margin:14px 0 24px}.card img{display:block;width:100%;height:auto;border:1px solid #edf0f2}.downloads{display:flex;gap:22px;font-size:13px}.pending{padding:40px 18px;border:1px dashed #c6d0d5;text-align:center;color:var(--muted)}.table-scroll{overflow:auto;background:white;border:1px solid var(--line);border-radius:6px;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#edf2f5;white-space:nowrap}tbody tr:last-child td{border-bottom:0}td{min-width:70px}details{background:white;border:1px solid var(--line);padding:15px 20px;border-radius:8px}summary{cursor:pointer;font-size:18px;font-weight:650}#count{align-self:center;font-size:13px;color:var(--muted)}[hidden]{display:none!important}@media(max-width:650px){main{padding:16px 12px}h1{font-size:23px}.card{padding:10px}label{display:block}select{max-width:210px}}
</style><main><div class="banner"><strong>''' + status + '''</strong><div class="tabs"><a href="REPORT_zh.md">中文报告 Markdown</a><a href="paper_caption.tex">论文图注 LaTeX</a><a href="report_manifest.json">完成状态与证据清单</a><a href="#analysis">比较分析与测试集指标</a></div></div>
<h1>Input | CLIP | CoOp | KgCoOp | DeKg | LUNA</h1><p>原始输入、筛选成功案例、随机对照与失败案例均保留完整六列。Grad-CAM 与梯度加权 rollout 分开查看，点击图片打开原尺寸。</p>
<div class="filters"><label>数据集<select id="dataset"><option value="all">全部</option>''' + options(DATASETS) + '''</select></label><label>样本<select id="group"><option value="all">全部</option>''' + options(GROUPS) + '''</select></label><label>归因<select id="attribution"><option value="all">全部</option>''' + options(ATTRIBUTIONS) + '''</select></label><span id="count"></span></div><div id="gallery">''' + "\n".join(cards) + '''</div><details id="analysis" open><summary>真实测试集结果与方法分析</summary>''' + markdown_html(report) + '''</details></main><script>
const keys=['dataset','group','attribution'];function filter(){let visible=0;for(const card of document.querySelectorAll('.card')){card.hidden=keys.some(key=>{const selected=document.getElementById(key).value;return selected!=='all'&&card.dataset[key]!==selected});if(!card.hidden)visible++}document.getElementById('count').textContent=visible+' 组图'}for(const key of keys)document.getElementById(key).addEventListener('change',filter);filter();
</script></html>'''


def captions(complete):
    status = "COMPLETE: all required evidence is available." if complete else "PENDING DRAFT: report_manifest.json lists missing evidence; do not describe unavailable figures as completed."
    return "% " + status + r'''
% Choose the caption matching the actual figure. These fragments make no
% unsupported claim that LUNA is best on every image or aggregate metric.

% Grad-CAM figure:
\caption{Ground-truth class-conditioned Grad-CAM comparisons of CLIP, CoOp,
KgCoOp, DeKg, and LUNA using a shared frozen CLIP ViT-B/16 visual encoder.
Columns show the input and the five methods in this order. Grad-CAM is computed
from patch activations at the final visual block's pre-attention layer norm;
the CLS token is excluded from spatial pooling. Every method explains the same
ground-truth-class cosine similarity, even when its prediction is incorrect.
Nonnegative native relevance is normalized to unit sum, bilinearly upsampled,
and displayed with one common range per input across methods, the same Jet
colormap and opacity. Colors indicate relative spatial relevance, not absolute
gradient magnitude. Displayed success cases are selected from novel test images
where LUNA is correct and at least one rival is wrong, ranked first by the number
of incorrect rivals and then by LUNA's target-margin gain over the strongest
rival, with at most one image per class. Unchanged original inputs, seeded
prediction-independent random controls, and LUNA failure cases are supplied
alongside these selected cases. Qualitative selection does not estimate average
accuracy; full held-out results are reported separately.}

% Gradient-weighted attention rollout figure:
\caption{Ground-truth class-conditioned gradient-weighted attention rollout
for CLIP, CoOp, KgCoOp, DeKg, and LUNA. These methods share the frozen CLIP
ViT-B/16 visual encoder and therefore have identical raw visual self-attention
for the same input. The displayed differences arise from gradients of the
method-specific text similarity, propagated through the final six visual
attention blocks. The target, image crop, relative-mass normalization,
per-input common display range, colormap, and opacity are shared across methods.
No method-specific thresholding or visual enhancement is applied. Success
examples use the disclosed novel-test rule (LUNA correct, at least one rival
wrong; descending wrong-rival count followed by target-margin gain; distinct
classes). Original samples, seeded random controls, and LUNA failure examples
are provided for context. The maps are gradient attributions rather than
segmentation masks and do not by themselves establish aggregate superiority.}

% Protocol note to include in the surrounding experimental section:
% Optional contrastive-rollout panel caption addition (use only if exported):
% This panel instead differentiates the ground-truth cosine minus the mean
% cosine of up to five highest-scoring incorrect classes, applying the same
% rule to all methods. Its contrastive objective differs from target-cosine
% rollout and Grad-CAM; these attribution objectives are not interchangeable.
%
% All results use seed 1 and fixed final-epoch checkpoints. The 16-shot
% base-class baselines train for 100 epochs. LUNA trains on unlabeled images
% from the all-class training partition for 5 epochs using a PromptSRC teacher
% selected at epoch 20. Data access, supervision, and teacher compute differ;
% this is not an equal-training-budget ablation. Bounding boxes, when reported,
% are coarse localization proxies and contain background inside the rectangle.
'''


def primary_caption(complete):
    status = "COMPLETE_REPORT" if complete else "PENDING_DRAFT"
    return "% " + status + r'''
% Primary caption (76 words). Use only with the selected-success
% contrastive_rollout figure; detailed protocol remains in paper_caption.tex.
\caption{Class-conditioned attribution comparisons of CLIP, CoOp, KgCoOp,
DeKg, and LUNA. Each row shares the input, ground-truth target, crop,
normalization, and overlay settings; the red border identifies LUNA. Maps show
gradient-weighted attention rollout of the target-minus-hard-negative cosine
margin through the final six visual blocks. Positive examples are selected
from novel test images where LUNA is correct and at least one baseline is
wrong. Original samples, prediction-independent random controls, failures,
and complete held-out classification results are provided separately.}
'''


def build(root):
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    evidence = Evidence(root)
    evaluations = {dataset: evaluation(evidence, dataset) for dataset in DATASETS}
    figures = [figure(evidence, dataset, group, attribution, evaluations[dataset])
               for dataset in DATASETS for group in GROUPS for attribution in REQUIRED_ATTRIBUTIONS]
    for dataset in DATASETS:
        for group in GROUPS:
            if (root / "figures" / dataset / group / "contrastive_rollout").exists():
                figures.append(figure(evidence, dataset, group, "contrastive_rollout", evaluations[dataset]))
    protocols = baseline_and_luna_protocols(evidence)
    audits = {dataset: localization_audit(evidence, dataset, evaluations[dataset])
              for dataset in ("fgvc_aircraft", "stanford_cars")}
    required_figures = [item for item in figures if item["attribution"] in REQUIRED_ATTRIBUTIONS]
    complete = (all(evaluations.values()) and
                all(item["status"] in ("COMPLETE", "EMPTY_EVIDENCED") for item in required_figures) and
                all(item["dataset"] == "eurosat" or item["status"] == "EMPTY_EVIDENCED" or item["localization"] is not None for item in required_figures))
    report = narrative(evidence, evaluations, figures, protocols, audits, complete)
    atomic_text(root / "REPORT_zh.md", report)
    atomic_text(root / "review.html", gallery_html(report, figures, root, complete))
    atomic_text(root / "paper_caption.tex", captions(complete))
    atomic_text(root / "paper_caption_primary.tex", primary_caption(complete))
    manifest = {"schema_version": 1, "status": "COMPLETE" if complete else "PENDING",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "root": str(root), "generator_sha256": sha256(__file__),
                "evaluations": {dataset: "COMPLETE" if result else "PENDING" for dataset, result in evaluations.items()},
                "optional_localization_audits": {dataset: "COMPLETE" if record else "PENDING_OR_UNAVAILABLE"
                                                 for dataset, record in audits.items()},
                "figures": [{key: str(item[key]) for key in ("dataset", "group", "attribution", "directory", "status")} |
                            {"localization": "NOT_APPLICABLE" if item["dataset"] == "eurosat" else "COMPLETE" if item["localization"] else "PENDING"}
                            for item in figures],
                "source_files": [{"path": path, "sha256": digest} for path, digest in sorted(evidence.sources.items())],
                "pending_or_invalid": evidence.issues,
                "artifacts": {name: {"path": str(root / name), "sha256": sha256(root / name)}
                              for name in ("REPORT_zh.md", "review.html", "paper_caption.tex", "paper_caption_primary.tex")},
                "interpretation": "Measured evidence only. Selected examples are diagnostic, not estimates of aggregate accuracy. No smoke/checks artifacts used."}
    atomic_text(root / "report_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    manifest = build(args.root)
    print(json.dumps({"status": manifest["status"], "root": manifest["root"],
                      "complete_evaluations": sum(value == "COMPLETE" for value in manifest["evaluations"].values()),
                      "complete_figures": sum(value["status"] == "COMPLETE" for value in manifest["figures"]),
                      "pending_or_invalid_files": len(manifest["pending_or_invalid"])}, ensure_ascii=False))
    if args.require_complete and manifest["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
