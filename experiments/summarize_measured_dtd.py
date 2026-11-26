"""Publish current verified real measurements and an explicitly labeled comparison."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REAL = "REAL_EXPERIMENT"
SIM = "SIMULATED_POSITIVE_FOR_PLANNING_ONLY"
VARIANTS = ("AdaptiveLUNA", "StaticMean", "Equal")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquired_dataset_status(root):
    """Separate verified data availability from teacher/student measurement state."""
    result = {}
    for dataset, slug, ready_status in (("FGVCAircraft", "fgvc_aircraft", "READY_DATASET_AND_ORIGINAL_SPLITS"),
                                        ("EuroSAT", "eurosat", "READY_DATASET_AND_ORIGINAL_SPLIT")):
        receipt_path = root / "logs" / f"asset_acquisition_{slug}.json"
        item = {"status": "NOT_RUN", "data_status": "ACQUISITION_NOT_YET_VERIFIED", "teacher_status": "PENDING", "student_status": "PENDING", "result_source": None}
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
            item.update(acquisition_receipt=str(receipt_path), acquisition_receipt_sha256=digest(receipt_path), acquisition_status=receipt.get("status"))
            if receipt.get("status") == ready_status:
                if dataset == "FGVCAircraft":
                    preflight = receipt["dataset_preflight"]
                    metadata = receipt["metadata_hashes"]
                    valid = (Path(preflight["image_root"]).is_dir() and preflight["missing_image_count"] == 0
                             and preflight["duplicate_reference_count"] == 0 and not receipt["dataset_preflight_blockers"])
                    for path, expected in metadata.items():
                        value = expected.get("sha256") if isinstance(expected, dict) else expected
                        valid = valid and Path(path).is_file() and digest(Path(path)) == value
                    image_count, class_count, counts = receipt["image_count"], receipt["class_count"], receipt["split_counts"]
                else:
                    verification = receipt["verification"]
                    split_path = Path(receipt["split_path"])
                    valid = (Path(receipt["image_root"]).is_dir() and split_path.is_file()
                             and digest(split_path) == receipt["split"]["sha256"]
                             and verification["missing_image_count"] == 0 and verification["duplicate_reference_count"] == 0
                             and verification["all_images_decodable"] is True)
                    image_count, class_count, counts = verification["image_count"], verification["class_count"], verification["split_counts"]
                if not valid:
                    raise ValueError(f"{dataset} completed acquisition receipt no longer matches verified local assets")
                item.update(data_status="VERIFIED_DATASET_AND_ORIGINAL_SPLITS", image_count=image_count,
                            class_count=class_count, split_counts=counts, data_root=receipt["data_root"])
        teacher_path = root / "teacher_training" / f"{slug}_promptsrc_l14_seed1_final20" / "teacher_provenance.json"
        if teacher_path.is_file():
            teacher = json.loads(teacher_path.read_text(encoding="utf-8-sig"))
            item.update(teacher_status=teacher.get("status", "PENDING"), teacher_provenance=str(teacher_path))
        data_reason = (f"Verified dataset and original splits are available ({item['image_count']} images, {item['class_count']} classes). "
                       if item["data_status"].startswith("VERIFIED") else "Dataset acquisition has not yet supplied a completed verified receipt. ")
        item["reason"] = data_reason + f"Teacher status: {item['teacher_status']}; remaining student/evaluation conditions are tracked in the predeclared plan. Dataset acquisition supplies no measured model scores."
        result[dataset] = item
    return result


def verified_experiment_tables(root, validation):
    tables = {}
    for sheet, stem in (("Exp1_Weighting", "exp1_weighting"), ("Exp2_Reliability", "exp2_reliability"),
                        ("Exp3_TeacherQuality", "exp3_teacher_quality"), ("Exp4_RelationPreservation", "exp4_relation_preservation")):
        path = (root / "tables" / f"REAL_{stem}.csv").resolve()
        if validation["output_sha256"].get(str(path)) != digest(path):
            raise ValueError(f"{sheet} differs from its authenticated exporter receipt")
        tables[sheet] = read_csv(path)
        if any(row["ResultSource"] not in ("", REAL) for row in tables[sheet]):
            raise ValueError("Unexpected source in REAL experiment table")
    return tables


def experiment_progress(tables):
    lines = ["| Experiment | Completed measured rows | Partial measured rows | Pending rows |", "|---|---:|---:|---:|"]
    for sheet, rows in tables.items():
        count = lambda statuses: sum(row["Status"] in statuses for row in rows)
        lines.append(f"| {sheet} | {count(('REAL', 'UNDEFINED_REAL'))}/{len(rows)} | {count(('PARTIAL_REAL',))} | {count(('NOT_RUN',))} |")
    return lines


def exp234_findings(tables):
    lines = []
    from experiments.compare_measured_planning import reliability_interpretation
    correlations = [row for row in tables["Exp2_Reliability"] if row["Statistic"] == "Spearman" and row["ResultSource"] == REAL]
    if correlations:
        lines += ["## Measured reliability versus teacher correctness", "",
                  "| Dataset | Reliability | Spearman rho | Two-sided p | Full batches |", "|---|---|---:|---:|---:|"]
        for row in correlations:
            rho = f"{float(row['rho']):.7f}" if row["rho"] else "undefined"
            pvalue = f"{float(row['p_value']):.7g}" if row["p_value"] else "undefined"
            lines.append(f"| {row['Dataset']} | {row['Reliability']} | {rho} | {pvalue} | {row['N_batches']} |")
        lines += [*reliability_interpretation(tables), "", "P-values use SciPy's two-sided asymptotic Student-t approximation with average tied ranks. "
                  "They are unadjusted across three scores and depend on the fixed, class-correlated batch order. "
                  "They are descriptive evidence, not proof of causality or independent-batch/multi-seed robustness.", ""]
    return lines


def write_chinese_progress(root, tables, now):
    def cell(value, precision=6):
        return f"{float(value):.{precision}f}" if value not in (None, "") else "待完成"
    lines = ["# Journal LUNA：真实实验分阶段分析", "", f"更新时间 UTC：{now}", "",
             "所有数值直接读取经过验证的 REAL CSV，来源为 REAL_EXPERIMENT。空白项目没有用假设值填充；单个种子的结果不能代替多种子重复实验。", "",
             "## 当前完成度", "", *experiment_progress(tables), "",
             "## Exp2：可靠性与教师正确率", "",
             "相关性以固定原始测试顺序、每批 8 张图像为单位；尾部不足一批的图像单独记录。教师准确率在全部类别预测空间计算。", "",
             "| 数据集 | 可靠性 | Spearman rho | 双侧 p | 批次数 |", "|---|---|---:|---:|---:|"]
    for row in tables["Exp2_Reliability"]:
        if row["Statistic"] != "Spearman" or row["ResultSource"] != REAL:
            continue
        rho, p = cell(row["rho"], 7), (f"{float(row['p_value']):.7g}" if row["p_value"] else "未定义")
        lines.append(f"| {row['Dataset']} | {row['Reliability']} | {rho} | {p} | {row['N_batches']} |")
    lines += ["", "p 值采用 SciPy Spearman 双侧渐近 t 近似，并对并列值使用平均秩；没有进行多重比较校正。批次存在类别与顺序相关性，因此这些 p 值只支持对当前固定批次的描述，不能当作因果证据或跨随机种子的稳健性证据。", ""]
    from experiments.compare_measured_planning import reliability_interpretation
    lines += reliability_interpretation(tables, language="zh")
    lines += ["### Q1–Q5 教师准确率（%）", "",
              "每个可靠性分量分别按批次分数从低到高稳定排序，再分为五组；并列分数保持原始批次顺序。下表和曲线直接保留测量值，不按准确率重新排序，不平滑或强制单调。各组批次数和样本数见 REAL Exp2 CSV。", "",
              "| 数据集 | 可靠性 | Q1（低） | Q2 | Q3 | Q4 | Q5（高） |", "|---|---|---:|---:|---:|---:|---:|"]
    quintile_notes = []
    for dataset in ("DTD", "FGVCAircraft", "EuroSAT"):
        nonmonotonic = []
        for reliability in ("r_ins", "r_batch", "r_cls"):
            groups = {row["Quantile"]: row for row in tables["Exp2_Reliability"]
                      if row["Dataset"] == dataset and row["Reliability"] == reliability
                      and row["Statistic"] == "Reliability quantile" and row["ResultSource"] == REAL}
            if not groups:
                continue
            values = [groups.get(f"Q{q}", {}).get("teacher_batch_accuracy") for q in range(1, 6)]
            lines.append("| " + " | ".join([dataset, reliability, *[cell(value, 4) for value in values]]) + " |")
            if all(value not in (None, "") for value in values) and any(float(right) < float(left) for left, right in zip(values, values[1:])):
                nonmonotonic.append(reliability)
        if nonmonotonic:
            # Place notes after the complete table, so later datasets remain table rows.
            quintile_notes.append(f"{dataset} 的 {'、'.join(nonmonotonic)} 五分位准确率存在下降区间，不能写成严格单调上升。")
    lines += ["", *quintile_notes, "",
              "独立曲线图：[PNG](../figures/REAL_exp2_quintile_accuracy.png) / [PDF](../figures/REAL_exp2_quintile_accuracy.pdf)。尚未完成的数据集面板留空并标记 NOT_RUN。", ""]
    lines += ["## Exp3：教师训练阶段与学生性能", "",
              "Weak/Medium/Strong 按教师预定训练阶段 5/12/20 选择，名称不保证性能单调。教师准确率来自同一固定测试子集；学生 HM 来自各自完整 Base/Novel 测试评估，二者口径不同。", "",
              "教师 5/12 轮为同一条 20 轮训练轨迹的原始参数快照；只有最终 20 轮应用官方完整 GPA（mean=15，std=1）。因此阶段差异同时涉及训练进程与参数平均，不能完全归因于训练轮数，也不能把三个阶段都称为 GPA 教师。", "",
              "| 数据集 | 阶段 | 教师准确率 | Equal HM | Adaptive HM | Adaptive−Equal | 状态 |", "|---|---|---:|---:|---:|---:|---|"]
    for row in tables["Exp3_TeacherQuality"]:
        if row["ResultSource"] == REAL:
            lines.append("| " + " | ".join([row["Dataset"], row["TeacherQuality"], *[cell(row[key]) for key in ("TeacherAccuracy", "Equal_HM", "Adaptive_HM", "HM_Gain")], row["Status"]]) + " |")
    for dataset in ("DTD", "FGVCAircraft", "EuroSAT"):
        stages = {row["TeacherQuality"]: row for row in tables["Exp3_TeacherQuality"] if row["Dataset"] == dataset and row["ResultSource"] == REAL}
        if all(quality in stages and stages[quality]["HM_Gain"] not in (None, "") for quality in ("Weak", "Medium", "Strong")):
            gains = [float(stages[quality]["HM_Gain"]) for quality in ("Weak", "Medium", "Strong")]
            if gains[0] < gains[1] < gains[2]:
                lines += ["", f"**在当前单种子 {dataset} 实验中，Weak/Medium/Strong 的 Adaptive−Equal 增益依次为 "
                          + "、".join(f"{value:+.6f}" for value in gains)
                          + "，随阶段增强而增大。** 这不能支持“教师越弱，Adaptive 优势越大”的论断；与正向规划情景预设的弱教师收益最大方向相反。该结论限定于当前数据集、种子和 raw/GPA 阶段协议，不能推广到全部数据集。", ""]
            if all(value < 0 for value in gains):
                lines += ["", f"**{dataset} 三个阶段的 Adaptive−Equal HM 增益均为负："
                          + "、".join(f"{value:+.6f}" for value in gains)
                          + "（Weak、Medium、Strong）。** 当前这些真实运行没有显示 Adaptive 相对 Equal 的 HM 收益，不能由另一数据集的正向结果推断这里也有收益。", ""]
                drop_equal, drop_adaptive = stages["Weak"].get("Drop_equal"), stages["Weak"].get("Drop_adaptive")
                if drop_equal not in (None, "") and drop_adaptive not in (None, "") and float(drop_adaptive) > float(drop_equal):
                    lines += [f"{dataset} 的 Adaptive Drop={float(drop_adaptive):.10f} 大于 Equal Drop={float(drop_equal):.10f}；按 Strong−Weak HM Drop 指标，该数据集也未显示 Adaptive 有更强的弱教师鲁棒性。该描述仅限当前单种子和既定阶段协议。", ""]
    lines += ["", "Drop 定义为 Strong HM − Weak HM，保留正负号。负值表示该次 Weak 学生 HM 高于 Strong，不应改写为正向退化或强行解释成单调的教师质量收益。", ""]
    for row in tables["Exp3_TeacherQuality"]:
        if row["TeacherQuality"] == "Weak" and row["ResultSource"] == REAL and row["Drop_equal"] and row["Drop_adaptive"]:
            lines.append(f"{row['Dataset']}：Equal Drop={cell(row['Drop_equal'], 10)}；Adaptive Drop={cell(row['Drop_adaptive'], 10)}。")
    lines += ["", "## Exp4：关系距离与同一运行的 HM", "",
              "D_ins 为教师到学生 KL 的逐图像均值，单位为 nats；D_batch、D_cls 为按教师 Gram 范数归一化的 Frobenius 距离，按固定批次样本数加权。StaticHierarchical 对应固定等权 Equal，不是 Exp1 的训练均值 StaticMean。", "",
              "| 数据集 | 变体 | D_ins | D_batch | D_cls | 同运行 HM | 状态 |", "|---|---|---:|---:|---:|---:|---|"]
    for row in tables["Exp4_RelationPreservation"]:
        if row["ResultSource"] == REAL:
            lines.append("| " + " | ".join([row["Dataset"], row["Variant"], *[cell(row[key]) for key in ("D_ins", "D_batch", "D_cls", "HM")], row["Status"]]) + " |")
    from experiments.compare_measured_planning import relation_ranking_text
    lines += relation_ranking_text(tables, language="zh")
    lines += ["", "这些排序仅描述当前单种子的实际观察，不能独立证明关系保持导致泛化提升。", "",
              "所有结论以这里的实际排序为准。HYPOTHETICAL 文件中的正向规划情景保持独立，不能覆盖弱相关、未显著结果、负 Drop 或尚未完成的条件。", "",
              "来源：[REAL Exp2 CSV](../tables/REAL_exp2_reliability.csv)、[REAL Exp3 CSV](../tables/REAL_exp3_teacher_quality.csv)、[REAL Exp4 CSV](../tables/REAL_exp4_relation_preservation.csv)。每行包含实际运行或评估清单及 SHA256。", ""]
    if (root / "logs/dtd_posthoc_cpu_audit.json").is_file():
        lines += ["独立复核：[DTD 保存概率的 CPU 数值审计](../logs/dtd_posthoc_cpu_audit.json)。该审计复算前三个真实评估的可靠性、距离、正确率与相关/分位统计；同一 Strong 教师在 Equal/Adaptive 评估中的教师概率逐位一致。", ""]
    destination = root / "analysis/REAL_exp234_analysis_zh.md"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def verify_export(root, sanity, validation, rows):
    """Tie each published real row to the already verified exporter receipt."""
    if sanity["status"] != "PASS_REAL_DTD_SANITY" or validation.get("status") != "PASS":
        raise ValueError("Publish only verified actual sanity/measurement exports")
    if not validation.get("serialized_outputs_validated"):
        raise ValueError("Measured exporter must validate its serialized artifacts first")
    if validation.get("input_sha256", {}).get(str((root / "sanity_result.json").resolve())) != digest(root / "sanity_result.json"):
        raise ValueError("Sanity report changed since the measured export")
    real_csv = (root / "tables/REAL_exp1_weighting.csv").resolve()
    if validation.get("output_sha256", {}).get(str(real_csv)) != digest(real_csv):
        raise ValueError("REAL Exp1 CSV changed since the validated export")
    measured, manifests = [], {}
    for row in rows:
        if row["ResultSource"] not in ("", REAL):
            raise ValueError("Unexpected source in a REAL table")
        if row["ResultSource"] != REAL:
            if row["Status"] != "NOT_RUN" or any(row[key] for key in ("Base", "Novel", "HM")):
                raise ValueError("Unmeasured Exp1 rows must remain NOT_RUN with blank values")
            continue
        if row["Status"] != "REAL":
            raise ValueError("Measured row lacks REAL status")
        path = Path(row["RunManifest"]).resolve()
        if digest(path) != row["RunManifestSHA256"]:
            raise ValueError("Run manifest changed since the measured export")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETED_TRAINING":
            raise ValueError("An incomplete run cannot supply published accuracy")
        if any(float(row[key]) != manifest[key] for key in ("Base", "Novel", "HM")):
            raise ValueError("Published accuracy differs from its verified manifest")
        identity = row["Dataset"], row["Variant"]
        if identity in manifests:
            raise ValueError("Duplicate dataset/variant; summary cannot select by accuracy")
        manifests[identity] = manifest
        measured.append(row)
    if len(measured) != validation["measured_rows"]["Exp1_Weighting"]:
        raise ValueError("Measured row count differs from export validation")
    return measured, manifests


def queue_status(run_root, measured):
    exported = {row["Variant"] for row in measured if row["Dataset"] == "DTD"}
    queue = {}
    for variant in VARIANTS:
        path = run_root / variant / "journal_manifest.json"
        if path.is_file():
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                queue[variant] = {"status": "UPDATING_MANIFEST", "evidence": str(path)}
                continue
            status = manifest["status"]
            queue[variant] = {"status": "REAL" if variant in exported else ("COMPLETED_AWAITING_VERIFIED_EXPORT" if status == "COMPLETED_TRAINING" else status),
                              "evidence": str(path), "manifest": manifest}
        else:
            queue[variant] = {"status": "QUEUED", "evidence": str(run_root / variant)}
    return queue


def update_run_plan(root, queue, measured, now, assets, tables, validation):
    path = root / "run_plan.csv"
    if not path.is_file():
        return
    rows = read_csv(path)
    columns = list(dict.fromkeys([*rows[0], "HistoricalStatus", "HistoricalReason", "ResultSource", "ExecutionEvidence", "UpdatedUTC"]))
    measured_variants = {row["Variant"] for row in measured if row["Dataset"] == "DTD"}
    declared = {}
    plan_path = root / "logs/exp234_predeclared_plan_20260906.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        declared = {(item["dataset"], item["quality"], item["variant"]): item for item in plan["conditions"]}
        supplemental_path = root / "logs/exp1_static_mean_supplemental_predeclared_plan_20260906.json"
        if supplemental_path.is_file():
            supplemental = json.loads(supplemental_path.read_text(encoding="utf-8"))
            if (supplemental["original_plan_sha256"] != digest(plan_path)
                    or supplemental["core_source_sha256"] != plan["core_source_sha256"]
                    or supplemental["status"] != "SUPPLEMENTAL_PREDECLARED_NOT_EXECUTED"):
                raise ValueError("Supplemental run-plan provenance differs from the original declaration")
            declared.update({(item["dataset"], item["quality"], item["variant"]): item for item in supplemental["conditions"]})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot = root / "logs" / f"run_plan_before_measured_summary_{stamp}.csv"
    shutil.copy2(path, snapshot)
    for row in rows:
        row["HistoricalStatus"] = row.get("HistoricalStatus") or row["Status"]
        row["HistoricalReason"] = row.get("HistoricalReason") or row.get("CurrentReason", "")
        row.update(ResultSource="", ExecutionEvidence="", UpdatedUTC=now)
        dataset, experiment, variant = row["Dataset"], row["Experiment"], row["Variant"]
        candidates = [item for item in tables.get(experiment, []) if item["Dataset"] == dataset]
        if experiment == "Exp3_TeacherQuality":
            candidates = [item for item in candidates if item["TeacherQuality"] == row.get("TeacherQuality")]
            prefix = "Adaptive" if variant == "AdaptiveLUNA" else "Equal"
            available = [item for item in candidates if item.get(prefix + "_HM")]
        elif experiment == "Exp2_Reliability":
            available = [item for item in candidates if item["ResultSource"] == REAL]
        else:
            available = [item for item in candidates if item["Variant"] == variant and item["ResultSource"] == REAL]
        if available:
            done = (all(item["Status"] in ("REAL", "UNDEFINED_REAL") for item in available)
                    and (experiment != "Exp2_Reliability" or len(available) == len(candidates)))
            row.update(Status="REAL" if done else "PARTIAL_REAL", ResultSource=REAL,
                       ExecutionEvidence="; ".join(sorted({item["Evidence"] for item in available})),
                       CurrentReason="Authenticated measured cells are exported; blank cells remain pending." if not done else "Completed and authenticated actual result; retain source and final-checkpoint evidence.")
            continue
        reused = "Equal" if variant == "StaticHierarchical" else variant
        condition = declared.get((dataset, row.get("TeacherQuality") or "Strong", reused))
        if condition:
            condition_path = Path(condition["run_manifest"])
            condition_status = "QUEUED"
            if condition_path.is_file():
                try:
                    condition_status = json.loads(condition_path.read_text(encoding="utf-8"))["status"]
                except json.JSONDecodeError:
                    condition_status = "UPDATING_MANIFEST"
            if condition_status != "QUEUED":
                row.update(Status="COMPLETED_AWAITING_VERIFIED_EXPORT" if condition_status == "COMPLETED_TRAINING" else condition_status,
                           ExecutionEvidence=str(condition_path),
                           CurrentReason="Predeclared training condition status: " + condition_status + ". Required completed evaluation/export cells are not yet available; no measurement is assigned here.")
                continue
        if dataset != "DTD":
            row.update(Status="NOT_RUN", CurrentReason=assets[dataset]["reason"], ExecutionEvidence=assets[dataset].get("acquisition_receipt", ""))
            continue
        variant = row["Variant"]
        if row["Experiment"] == "Exp1_Weighting" and variant in queue:
            entry = queue[variant]
            row.update(Status=entry["status"], ExecutionEvidence=entry["evidence"],
                       ResultSource=REAL if variant in measured_variants else "",
                       CurrentReason=("Completed final-epoch metrics exported and verified." if variant in measured_variants else
                                      "Actual student training is running; no completed accuracy is exported." if entry["status"] == "RUNNING" else
                                      "Completed training awaits verified export; no accuracy is claimed in this plan." if entry["status"] == "COMPLETED_AWAITING_VERIFIED_EXPORT" else
                                      "Run requires recovery: " + str(entry.get("manifest", {}).get("error", "See execution log")) if entry["status"].startswith("FAILED") else
                                      "Scheduled in the fixed Adaptive -> StaticMean -> Equal queue; no completed accuracy is exported."))
        else:
            reused = "Equal" if variant == "StaticHierarchical" else variant
            available = reused in measured_variants
            row.update(Status="NOT_RUN", CurrentReason=(
                "Matching Exp1 checkpoint is available; required held-out analysis/teacher accuracy/relation distances have not been evaluated."
                if available and row.get("TeacherQuality") == "Strong" else
                "Required student comparison or checkpoint evaluation has not been run. DTD data and teacher checkpoints at epochs 5/12/20 are available."
            ))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("E:/LUNA/results_reliability_analysis"))
    parser.add_argument("--run-root", type=Path, help="Actual three-variant DTD cohort; default is the current first-real cohort.")
    args = parser.parse_args()
    root = args.output_root.resolve()
    sanity = json.loads((root / "sanity_result.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "logs/real_export_validation.json").read_text(encoding="utf-8"))
    rows = read_csv(root / "tables/REAL_exp1_weighting.csv")
    measured, manifests = verify_export(root, sanity, validation, rows)
    tables = verified_experiment_tables(root, validation)
    assets = acquired_dataset_status(root)
    for dataset, item in assets.items():
        exported = [row for records in tables.values() for row in records if row["Dataset"] == dataset and row["ResultSource"] == REAL]
        if exported:
            item.update(status="REAL_MEASUREMENTS_AVAILABLE", student_status="VERIFIED_MEASUREMENTS_AVAILABLE", result_source=REAL,
                        measured_rows=len(exported))
    progress_all = experiment_progress(tables)
    complete = {row["Variant"] for row in measured if row["Dataset"] == "DTD"} == set(VARIANTS)
    run_root = (args.run_root or root / "exp1_weighting/DTD/seed1_final20_20260906").resolve()
    queue = queue_status(run_root, measured)
    protocol_manifests = [entry["manifest"] for entry in queue.values() if "manifest" in entry]
    if not protocol_manifests:
        raise ValueError("At least one real run manifest is required to describe the student protocol")
    student_epochs = {entry["configuration"]["OPTIM"]["MAX_EPOCH"] for entry in protocol_manifests}
    student_batch_sizes = {entry["configuration"]["DATALOADER"]["TRAIN_X"]["BATCH_SIZE"] for entry in protocol_manifests}
    if len(student_epochs) != 1 or len(student_batch_sizes) != 1:
        raise ValueError("The live student cohort does not share one epoch/batch-size protocol")
    student_epochs = student_epochs.pop()
    student_batch_size = student_batch_sizes.pop()
    verified_runs = validation.get("verified_training_manifests", {})
    if verified_runs:
        completed_batches = 0
        for path, evidence in verified_runs.items():
            if digest(Path(path)) != evidence["sha256"]:
                raise ValueError("Verified training manifest changed since export")
            completed_batches += json.loads(Path(path).read_text(encoding="utf-8"))["statistics"]["batch_count"]
    else:
        completed_batches = sum(entry["statistics"]["batch_count"] for entry in manifests.values())
    now = datetime.now(timezone.utc).isoformat()
    from experiments.analyze_measured_exp234 import generate
    generated_analysis = generate(root, tables, now)
    findings = exp234_findings(tables)
    teacher_path = root / "teacher_training/dtd_promptsrc_l14_seed1_final20/teacher_provenance.json"
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    if teacher.get("status") != "COMPLETED_REAL_TEACHER_TRAINING":
        raise ValueError("Teacher training is not complete")
    teacher_batches = teacher["total_training_batches"]
    for entry in assets.values():
        if entry["teacher_status"] == "COMPLETED_REAL_TEACHER_TRAINING":
            teacher_batches += json.loads(Path(entry["teacher_provenance"]).read_text(encoding="utf-8"))["total_training_batches"]
    means, std = sanity["means"], sanity["lambda_std"]
    accuracy_lines = ["| Dataset | Variant | Base | Novel | HM | ResultSource |", "|---|---|---:|---:|---:|---|"]
    for row in measured:
        accuracy_lines.append("| " + " | ".join([row["Dataset"], row["Variant"],
                              *[f"{float(row[key]):.4f}" for key in ("Base", "Novel", "HM")], REAL]) + " |")
    if not measured:
        accuracy_lines = ["No completed, validated student Base/Novel/HM is claimed yet; current run states are listed below."]
    progress_lines = ["| DTD variant | Current execution/export status |", "|---|---|"]
    progress_lines.extend(f"| {variant} | {queue[variant]['status']} |" for variant in VARIANTS)
    recovery_lines = []
    recovery_records = []
    for (dataset, variant), manifest in manifests.items():
        recovery = manifest.get("final_evaluation_recovery")
        if recovery:
            recovery_lines.extend([
                f"{dataset} {variant}: final evaluation was recovered from the existing final checkpoint after "
                f"the original run recorded `{recovery['original_error']}`. The exporter verified zero additional "
                "optimizer steps and unchanged checkpoint/model state/training artifacts. The original failure "
                f"evidence remains under `{recovery['evidence_directory']}`; the recovery report is `{recovery['report_path']}`.", "",
            ])
            recovery_records.append({"dataset": dataset, "variant": variant, "report_path": recovery["report_path"],
                                     "training_restarted": recovery["training_restarted"], "additional_optimizer_steps": recovery["additional_optimizer_steps"]})
    by_variant = {row["Variant"]: row for row in measured if row["Dataset"] == "DTD"}
    gain_lines = []
    if complete:
        gain_equal = float(by_variant["AdaptiveLUNA"]["HM"]) - float(by_variant["Equal"]["HM"])
        gain_static = float(by_variant["AdaptiveLUNA"]["HM"]) - float(by_variant["StaticMean"]["HM"])
        gain_lines = [f"REAL DTD Adaptive HM gain vs Equal: {gain_equal:+.4f}; vs StaticMean: {gain_static:+.4f}.", "",
                      "These are one-seed measured differences, not a multi-seed robustness claim. "
                      "The observed ranking is retained regardless of whether it agrees with the planning scenario.", ""]
    lines = ["# Journal LUNA: verified real experiments", "", f"Updated UTC: {now}", "",
             "**DTD real sanity PASSED.** " + ("**DTD Exp1 is complete for all three variants.**" if complete else "**DTD Exp1 has pending variants; see their current statuses below.**"), "",
             "Public assets were acquired after the earlier local-only search. "
             "The earlier zero-result report has been preserved in logs/before_real_execution_20260906.", "",
             "## Real assets and teacher", "",
             "The official Oxford DTD archive and unmodified CoOp-linked Zhou split were downloaded and checked. "
             "All 5,640 referenced images exist; train/validation/test split counts are 2,820/1,128/1,692. "
             "The LUNA protocol uses all 2,820 training images without their labels for student optimization, "
             "including examples from all 47 classes (24 Base and 23 Novel). Thus Novel classes participate as "
             "unlabeled adaptation data. Final evaluation uses 864 Base and 828 Novel images from the original test split. "
             "The loader named `val` contains the Base test subset; its name does not mean that the original 1,128-image "
             "validation split supplied the reported Base accuracy.", "",
             "The exact OpenAI ViT-B/16 and ViT-L/14 weights passed the published URL SHA256 checks. "
             "A new PromptSRC ViT-L/14 teacher was trained from the official PromptKD checkout "
             "ec244f783c8b9b2503ad5a838a3fd97365686ddb: seed 1, 24 Base classes, 16 labeled shots per class, "
             "384 training samples, batch size 8, 20 epochs/960 batches. "
             "The final teacher uses official PromptSRC GPA at epoch 20; epochs 5 and 12 are also retained. "
             "No accuracy evaluation selected the teacher. This is a newly trained teacher, not an authenticated copy of the authors' released checkpoint.", "",
             "Teacher loader workers were set to 0 for Windows; the method, seed, optimizer, learning rate, "
             "batch size and GPA configuration were preserved. Journal student settings and fixed reliability formulas were unchanged.", "",
             "## REAL DTD sanity", "",
             f"Executed {sanity['batch_count']} distinct batches / {sum(row['n_samples'] for row in sanity['records'])} images; all {len(sanity['checks'])} required checks passed.", "",
             "| Weight | REAL mean | REAL population std | ResultSource |", "|---|---:|---:|---|"]
    for name in ("lambda_ins", "lambda_batch", "lambda_cls"):
        lines.append(f"| {name} | {means[name]:.8f} | {std[name]:.8f} | {REAL} |")
    lines += ["", "Student prompt gradients were present, finite and nonzero; teacher parameters were frozen with no gradients. "
              "Weights were bounded and normalized, changed across batches, and the objective passed the nonproportional "
              "legacy-contribution check. These are genuine training-batch diagnostics, not synthetic fixture values.", "",
              "## REAL Exp1: final-epoch Base/Novel evaluation", "",
              "The initial DTD execution order was Adaptive → complete training-only mean lambdas → StaticMean → Equal. "
              "For FGVCAircraft and EuroSAT, the original Exp2–4 conditions run first; two separately predeclared StaticMean controls are added afterward, "
              "using each completed Strong Adaptive training mean. Their already completed Strong Equal runs are reused without retraining. "
              f"Each student uses seed 1, batch size {student_batch_size}, {student_epochs} epochs, "
              "and the same final teacher. Epoch count comes from the actual run configuration, independently of the teacher's 20 epochs. "
              "Final Base/Novel evaluation is explicit; no test accuracy selects checkpoints.", "",
              "The effective common distillation multiplier is **1** in the existing implementation. The command-line "
              "`TRAINER.LUNA.KD_WEIGHT=1000.0` value is ignored by that implementation; it is not the multiplier used in these "
              "measured runs. This applies identically to all three variants and is reported as an implementation fact, "
              "without changing the active training code.", "",
              "The seed is fixed across variants, but cuDNN benchmark mode is enabled. These results document one controlled "
              "seed and its actual artifacts; they do not establish bitwise deterministic reproduction or variation across seeds.", "",
              *progress_lines, "", *accuracy_lines, "", *recovery_lines, *gain_lines, *findings,
              "## What remains", "",
              *[f"{dataset}: {item['reason']}" for dataset, item in assets.items()], "",
              *progress_all, "",
              "The progress table counts authenticated exported rows, including explicitly partial comparisons. "
              "Exp2 correlations/quintiles require held-out evaluation; Exp3 teacher accuracy and each student arm have separate evidence; "
              "Exp4 distances use the same student checkpoint as their reported HM. Blank cells remain pending or carry an explicit undefined-statistic reason. "
              "Strong student metrics are reused from matching completed Exp1 runs. Strong-minus-Weak drops are signed and may be negative. "
              "The earlier 71 software tests were offline validation, separate from the real measurements above.", "",
              "Held-out batch correlations describe association under the declared fixed order and batching. "
              "They do not establish causality, batch-order robustness, or multi-seed significance; raw batches, tie handling and p-value method are retained in the evaluation receipts.", "",
              "The hypothetical workbook remains unchanged and clearly labeled. It is not empirical evidence. "
              "A conditional PR assessment based on that positive scenario does not override the actual measured ranking.", "",
              "## Files", "",
              "- [REAL workbook](Journal_LUNA_REAL_Results.xlsx)",
              "- [Actual sanity report](SANITY_CHECK.md) and [raw batch records](sanity_result.json)",
              "- [Real Exp1 CSV](tables/REAL_exp1_weighting.csv)",
              "- [Real Exp2 CSV](tables/REAL_exp2_reliability.csv), [Exp3 CSV](tables/REAL_exp3_teacher_quality.csv), [Exp4 CSV](tables/REAL_exp4_relation_preservation.csv)",
              "- [Measured Exp2-4 analysis in Chinese](analysis/REAL_exp234_analysis_zh.md), [PNG overview](figures/REAL_exp234_overview.png), [PDF overview](figures/REAL_exp234_overview.pdf)",
              "- [Measured Exp2 Q1-Q5 curves: PNG](figures/REAL_exp2_quintile_accuracy.png), [PDF](figures/REAL_exp2_quintile_accuracy.pdf)",
              "- [Real vs hypothetical comparison](REAL_VS_HYPOTHETICAL_SUMMARY.md)",
              "- [Failure/availability log](FAILURE_ANALYSIS.md)",
              "- Supplemental Exp1 declaration (when present): logs/exp1_static_mean_supplemental_predeclared_plan_20260906.json; its immutable declaration status is separate from actual run status.",
              "- [Teacher provenance](teacher_training/dtd_promptsrc_l14_seed1_final20/teacher_provenance.json)",
              "- Acquisition evidence: logs/asset_acquisition_dtd.json, logs/asset_acquisition_clip.json, logs/asset_acquisition_fgvc_aircraft.json and logs/asset_acquisition_eurosat.json", ""]
    if (root / "analysis/REAL_DTD_exp1_analysis_zh.md").is_file():
        lines += ["- [Measured DTD analysis in Chinese](analysis/REAL_DTD_exp1_analysis_zh.md)",
                  "- [Measured DTD figure](figures/REAL_DTD_exp1.png)", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    hypothetical = {(row["Dataset"], row["Variant"]): row for row in read_csv(root / "tables/HYPOTHETICAL_exp1_weighting.csv")}
    if any(row.get("ResultSource") != SIM for row in hypothetical.values()):
        raise ValueError("Planning table contains an invalid source label")
    comparison = ["# REAL vs HYPOTHETICAL comparison", "", f"Updated UTC: {now}", "",
                  "REAL: actual DTD sanity is passed. " + ("All three DTD Exp1 variants are measured." if complete else "DTD Exp1 has pending variants; only completed and validated rows are shown."), "",
                  f"REAL values carry `{REAL}`. Planning values carry `{SIM}` and were designed before these real runs. "
                  "NOT_RUN means no verified measurement is available. Gap = REAL value minus the corresponding SIMULATED value; blank gaps are unavailable, not zero.", "",
                  "## REAL vs HYPOTHETICAL / SIMULATED POSITIVE RESULTS FOR PLANNING ONLY — Exp1", "",
                  "| Dataset | Variant | REAL status | REAL Base | REAL Novel | REAL HM | REAL source | SIMULATED HM | Planning ResultSource | Gap |",
                  "|---|---|---|---:|---:|---:|---|---:|---|---:|"]
    for row in rows:
        plan = hypothetical[row["Dataset"], row["Variant"]]
        available = row["ResultSource"] == REAL
        metrics = [f"{float(row[key]):.4f}" for key in ("Base", "Novel", "HM")] if available else ["—"] * 3
        gap = f"{float(row['HM']) - float(plan['HM']):+.4f}" if available else "—"
        comparison.append("| " + " | ".join([row["Dataset"], row["Variant"], "REAL" if available else "NOT_RUN",
                          *metrics, REAL if available else "—", f"{float(plan['HM']):.2f}", SIM, gap]) + " |")
    from experiments.compare_measured_planning import build_comparison_sections
    detailed_comparison, comparison_files = build_comparison_sections(root, tables)
    comparison += ["", *gain_lines, *detailed_comparison,
                   "## Remaining experiments", "",
                   *progress_all, "",
                   "The REAL tables contain only authenticated observed values, including partial comparisons. Planned correlations, robustness gains "
                   "and relation distances remain SIMULATED and cannot fill any missing real cell. "
                   "The full planning scenarios and source labels remain in the separate "
                   "[hypothetical workbook](HYPOTHETICAL_POSITIVE_RESULTS_FOR_PLANNING.xlsx) and "
                   "[hypothetical analysis](HYPOTHETICAL_POSITIVE_ANALYSIS.md).", "",
                   "The measured DTD teacher is newly trained under the official PromptSRC configuration. "
                   "All Exp1 variants share that teacher and the same declared protocol. Absolute differences from the "
                   "planning targets are not estimates of reproducibility error, because planning values came from no measured run.", "",
                   "Current conclusions should follow the actual ranking and one-seed limitation. The hypothetical "
                   "PR_REASONABLE_FIRST_SUBMISSION label remains conditional and does not establish empirical readiness.", ""]
    (root / "REAL_VS_HYPOTHETICAL_SUMMARY.md").write_text("\n".join(comparison), encoding="utf-8")
    state = {"updated_utc": now, "first_real_dtd_exp1_complete": complete, "original_full_project_complete": False,
             "real_sanity_batches": sanity["batch_count"], "real_batch_count": sanity["batch_count"] + completed_batches,
             "completed_student_training_batches": completed_batches, "real_teacher_training_batches": teacher_batches,
             "student_epochs": student_epochs, "student_batch_size": student_batch_size,
             "measured_result_count": sum(sum(row["ResultSource"] == REAL for row in items) for items in tables.values()), "measured_sanity_row_count": 1,
             "measured_rows": validation["measured_rows"], "partial_measured_rows": validation.get("partial_measured_rows", {}),
             "completed_evaluations": validation.get("completed_evaluations", 0),
             "completed_original_training_conditions": validation.get("completed_original_training_conditions"),
             "completed_supplemental_training_conditions": validation.get("completed_supplemental_training_conditions", 0),
             "count_scope": "Counts include verified completed-evaluation runs only; other executed batches, including recovery-pending runs, remain in streamed journal_batches.csv.",
             "result_source": REAL, "offline_test_count": 71,
             "datasets": {"DTD": {"status": "REAL_EXP1_COMPLETED" if complete else "REAL_SANITY_PASS_EXP1_PENDING",
                                     "data_root": "E:/LUNA/data", "teacher_provenance": str(teacher_path)},
                          **assets},
             "measured_exp1": measured, "teacher_status": teacher["status"],
             "recovered_final_evaluation_runs": recovery_records,
             "student_queue": {variant: {key: value for key, value in entry.items() if key != "manifest"} for variant, entry in queue.items()},
             "source_separation": "Hypothetical artifacts were not changed or imported into REAL tables."}
    (root / "logs/real_execution_status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    update_run_plan(root, queue, measured, now, assets, tables, validation)
    files = [root / "REPORT.md", root / "REAL_VS_HYPOTHETICAL_SUMMARY.md", root / "sanity_result.json",
             root / "Journal_LUNA_REAL_Results.xlsx", root / "FAILURE_ANALYSIS.md", root / "run_plan.csv", *sorted((root / "tables").glob("REAL_*"))]
    files.extend(sorted((root / "analysis").glob("REAL_DTD_exp1*")))
    files.extend(sorted((root / "figures").glob("REAL_DTD_exp1*")))
    files.extend(generated_analysis)
    files.extend(comparison_files)
    files.extend(sorted((root / "logs").glob("asset_acquisition_*.json")))
    manifest = {"status": "PASS", "updated_utc": now, "real_sanity_batches": sanity["batch_count"],
                "real_exp1_rows": len(measured), "first_real_dtd_exp1_complete": complete,
                "real_export_validation": str(root / "logs/real_export_validation.json"),
                "artifacts": [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)} for path in files]}
    (root / "logs/final_deliverable_validation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
