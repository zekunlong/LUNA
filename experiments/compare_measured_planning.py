"""Align measured and planning metrics without changing either source table."""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

REAL = "REAL_EXPERIMENT"
SIM = "SIMULATED_POSITIVE_FOR_PLANNING_ONLY"


def read_planning(path, keys):
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(row.get("ResultSource") != SIM for row in rows):
        raise ValueError("Comparison planning input has an invalid source label")
    result = {tuple(row[key] for key in keys): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate planning alignment key")
    return result, before


def number(value):
    if value in (None, ""):
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite real/planning comparison value")
    return value


def reliability_interpretation(tables, language="en"):
    """Interpret measured correlation direction separately from nominal p-values."""
    correlations = {(row["Dataset"], row["Reliability"]): row for row in tables["Exp2_Reliability"]
                    if row["Statistic"] == "Spearman" and row["ResultSource"] == REAL
                    and row.get("rho") not in (None, "")}
    datasets = ("DTD", "FGVCAircraft", "EuroSAT")
    lines = []
    if all((dataset, key) in correlations and float(correlations[dataset, key]["rho"]) > 0
           for dataset in datasets for key in ("r_ins", "r_cls")):
        lines += ["", ("三个已测数据集的 r_ins 和 r_cls 的 Spearman 系数均为正；这不能推广为三个可靠性分量都满足“分数越高、教师越正确”。"
                       if language == "zh" else
                       "r_ins and r_cls have positive Spearman coefficients on all three measured datasets. This does not generalize to a higher-score/higher-correctness claim for every reliability component."), ""]
    unsupported = False
    for dataset in datasets:
        row = correlations.get((dataset, "r_batch"))
        if row is None:
            continue
        rho, p = float(row["rho"]), number(row.get("p_value"))
        p_text = f"{p:.7g}" if p is not None else ("未定义" if language == "zh" else "undefined")
        if rho < 0:
            unsupported = True
            if language == "zh":
                lines += [f"**{dataset} 的 r_batch 呈负向关联：rho={rho:.7f}，名义双侧 p={p_text}。** 在该固定测试顺序和批次协议下，更高 r_batch 没有对应更高教师正确率；即使名义 p 很小，负方向也不能被解释为正向可靠性证据。", ""]
            else:
                lines += [f"{dataset} r_batch has a negative association: rho={rho:.7f}, nominal two-sided p={p_text}. "
                          "Under this fixed test order and batching, higher r_batch does not track higher teacher correctness; a small nominal p-value does not turn a negative coefficient into positive reliability evidence.", ""]
        if p is not None and p >= .05:
            unsupported = True
            if language == "zh":
                lines += [f"**{dataset} 的 r_batch：rho={rho:.7f}，名义 p={p_text}，未达到未校正 0.05 阈值。** 该结果不支持正向正确率关联的笼统论断；不显著也不等于证明完全没有关联。", ""]
            else:
                lines += [f"{dataset} r_batch does not reach the conventional unadjusted 0.05 threshold (rho={rho:.7f}, nominal p={p_text}). "
                          "It does not support a blanket positive-correctness association; non-significance does not establish absence of association.", ""]
    if unsupported:
        lines += [("因此，r_batch 的实测结果不支持“该分数越高，教师越正确”的统一结论。所有 p 值仍是未校正的渐近名义值，受固定、类别相关批次影响，不能据此作因果或独立批次稳健性推断。"
                   if language == "zh" else
                   "The measured r_batch results therefore do not support a uniform higher-score/higher-correctness conclusion. All p-values remain unadjusted asymptotic nominal values under fixed, class-correlated batches, without causal or independent-batch robustness inference."), ""]
    return lines


def relation_ranking_text(tables, language="en"):
    """Describe complete measured Exp4 rankings, with explicit pending cases."""
    variants = ("InstanceOnly", "StaticHierarchical", "AdaptiveLUNA")
    metrics = ("D_ins", "D_batch", "D_cls", "HM")
    lines = []
    for dataset in ("DTD", "FGVCAircraft", "EuroSAT"):
        rows = {row["Variant"]: row for row in tables["Exp4_RelationPreservation"]
                if row["Dataset"] == dataset and row["ResultSource"] == REAL and row["Status"] == "REAL"
                and all(row.get(metric) not in (None, "") for metric in metrics)}
        missing = [variant for variant in variants if variant not in rows]
        if missing:
            lines += ["", (f"{dataset} 的 Exp4 尚待完成：{'、'.join(missing)}；当前不作完整三个方法的排名结论。"
                           if language == "zh" else
                           f"{dataset} Exp4 remains pending for {', '.join(missing)}; no complete three-method ranking is claimed."), ""]
            continue
        winners, optimum = {}, {}
        for metric in metrics:
            values = {variant: float(rows[variant][metric]) for variant in variants}
            optimum[metric] = (max if metric == "HM" else min)(values.values())
            winners[metric] = [variant for variant, value in values.items()
                               if math.isclose(value, optimum[metric], rel_tol=0, abs_tol=1e-12)]
        adaptive_best = [metric for metric in metrics[:3] if "AdaptiveLUNA" in winners[metric]]
        if language == "zh":
            details = [f"{metric} 最{'高' if metric == 'HM' else '低'}为 {' / '.join(winners[metric])}（{optimum[metric]:.10f}）" for metric in metrics]
            lines += ["", f"**{dataset} 三个方法的真实排序：**" + "；".join(details) + "。", ""]
            if len(adaptive_best) < 3:
                best = "、".join(adaptive_best) if adaptive_best else "没有任何一个距离指标"
                lines += [f"在这次 {dataset} 测量中，AdaptiveLUNA 仅在 {best} 取得最小值，因此不支持它在全部关系距离上均优于全部对照的说法。"
                          "距离排名与 HM 排名应分别报告，不补充未经验证的因果解释。", ""]
        else:
            details = [f"{'maximum' if metric == 'HM' else 'minimum'} {metric}: {' / '.join(winners[metric])} ({optimum[metric]:.10f})" for metric in metrics]
            lines += ["", f"Measured {dataset} three-method ranking: " + "; ".join(details) + ".", ""]
            if len(adaptive_best) < 3:
                best = ", ".join(adaptive_best) if adaptive_best else "none of the three distances"
                lines += [f"AdaptiveLUNA minimizes only {best} in this {dataset} run. "
                          "This does not support improvement in every relation distance against all controls; distance and HM rankings remain separate observations, without a causal explanation.", ""]
    return lines


def build_comparison_sections(root, tables):
    planning, input_hashes = {}, {}
    specs = {"Exp1": ("exp1_weighting", ("Dataset", "Variant")),
             "Exp2": ("exp2_reliability", ("Dataset", "Reliability")),
             "Exp2Q": ("exp2_reliability_quantiles", ("Dataset", "Reliability", "Quantile")),
             "Exp3": ("exp3_teacher_quality", ("Dataset", "TeacherQuality")),
             "Exp3Drop": ("exp3_teacher_quality_drops", ("Dataset",)),
             "Exp4": ("exp4_relation_preservation", ("Dataset", "Variant"))}
    for name, (stem, keys) in specs.items():
        path = root / "tables" / f"HYPOTHETICAL_{stem}.csv"
        planning[name], input_hashes[path] = read_planning(path, keys)
    comparisons = {name: [] for name in ("Exp1", "Exp2", "Exp3", "Exp4")}

    def add(experiment, actual, assumed, metric, real_key, plan_key, plan_name):
        real_value, plan_value = number(actual.get(real_key)), number(assumed.get(plan_key))
        if real_value is not None and actual.get("ResultSource") != REAL:
            raise ValueError("Unverified real value cannot enter a comparison")
        undefined = actual.get("Status") == "UNDEFINED_REAL" and real_key in ("rho", "p_value")
        status = "REAL" if real_value is not None else "UNDEFINED_REAL" if undefined else "NOT_RUN"
        real_source = REAL if real_value is not None or undefined else None
        gap = real_value - plan_value if real_value is not None and plan_value is not None else None
        comparisons[experiment].append({"Experiment": experiment, "Dataset": actual["Dataset"], "Variant": actual.get("Variant"),
            "TeacherQuality": actual.get("TeacherQuality"), "Reliability": actual.get("Reliability"),
            "Statistic": actual.get("Statistic"), "Quantile": actual.get("Quantile"), "Metric": metric,
            "REAL_Value": real_value, "REAL_Status": status, "REAL_RowStatus": actual["Status"], "REAL_ResultSource": real_source,
            "Planning_Value": plan_value, "Planning_Status": "SIMULATED", "Planning_ResultSource": SIM,
            "Gap_REAL_Minus_Planning": gap, "REAL_Evidence": actual.get("Evidence"),
            "Planning_Evidence": str(root / "tables" / f"HYPOTHETICAL_{specs[plan_name][0]}.csv"),
            "Interpretation": "Arithmetic difference from an assumed scenario; not a replication error, causal effect, or significance test."})
    for actual in tables["Exp1_Weighting"]:
        assumed = planning["Exp1"][actual["Dataset"], actual["Variant"]]
        for metric in ("Base", "Novel", "HM"):
            add("Exp1", actual, assumed, metric, metric, metric, "Exp1")
    for actual in tables["Exp2_Reliability"]:
        if actual["Statistic"] == "Spearman":
            assumed = planning["Exp2"][actual["Dataset"], actual["Reliability"]]
            for metric, target in (("rho", "Rho"), ("p_value", "PValue"), ("N_batches", "NBatches")):
                add("Exp2", actual, assumed, metric, metric, target, "Exp2")
        else:
            assumed = planning["Exp2Q"][actual["Dataset"], actual["Reliability"], actual["Quantile"]]
            for metric, target in (("teacher_batch_accuracy", "TeacherBatchAccuracy"), ("N_batches", "NBatches")):
                add("Exp2", actual, assumed, metric, metric, target, "Exp2Q")
    mapping = {"TeacherAccuracy": "TeacherAccuracy", "Equal_Base": "EqualBase", "Equal_Novel": "EqualNovel", "Equal_HM": "EqualHM",
               "Adaptive_Base": "AdaptiveBase", "Adaptive_Novel": "AdaptiveNovel", "Adaptive_HM": "AdaptiveHM", "HM_Gain": "HMGain"}
    for actual in tables["Exp3_TeacherQuality"]:
        assumed = planning["Exp3"][actual["Dataset"], actual["TeacherQuality"]]
        for metric, target in mapping.items():
            add("Exp3", actual, assumed, metric, metric, target, "Exp3")
        drops = planning["Exp3Drop"][actual["Dataset"],]
        for metric, target in (("Drop_equal", "DropEqual"), ("Drop_adaptive", "DropAdaptive")):
            add("Exp3", actual, drops, metric, metric, target, "Exp3Drop")
    for actual in tables["Exp4_RelationPreservation"]:
        assumed = planning["Exp4"][actual["Dataset"], actual["Variant"]]
        for metric in ("D_ins", "D_batch", "D_cls", "HM"):
            add("Exp4", actual, assumed, metric, metric, metric, "Exp4")
    destination = root / "comparison"
    destination.mkdir(exist_ok=True)
    outputs = []
    for experiment, records in comparisons.items():
        path = destination / f"REAL_VS_HYPOTHETICAL_{experiment.lower()}_gaps.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        outputs.append(path)

    def find(experiment, dataset, metric, **identity):
        rows = [row for row in comparisons[experiment] if row["Dataset"] == dataset and row["Metric"] == metric
                and all(row.get(key) == value for key, value in identity.items())]
        if len(rows) != 1:
            raise ValueError(f"Ambiguous comparison identity: {experiment} {dataset} {metric} {identity}")
        return rows[0]

    def triplet(row, precision=4):
        if row["Metric"] == "rho":
            precision = 6
        def fmt(value, signed=False):
            if value is None:
                return "—"
            if row["Metric"] == "p_value":
                return f"{value:+.3g}" if signed else f"{value:.3g}"
            return f"{value:+.{precision}f}" if signed else f"{value:.{precision}f}"
        return " / ".join((fmt(row["REAL_Value"]), fmt(row["Planning_Value"]), fmt(row["Gap_REAL_Minus_Planning"], True)))

    def status_and_sources(rows):
        states = {row["REAL_Status"] for row in rows}
        status = "REAL" if states == {"REAL"} else "NOT_RUN" if states == {"NOT_RUN"} else "PARTIAL_REAL"
        sources = "R / S" if any(row["REAL_ResultSource"] == REAL for row in rows) else "— / S"
        return status + " / SIMULATED", sources

    lines = ["## Exp2–4: aligned REAL / PLANNING / GAP values", "",
             "Each metric cell below is **REAL / SIMULATED planning / (REAL − planning)**. A dash means unavailable or explicitly undefined; no gap is computed for that cell. "
             "Accuracy/HM/gain/drop differences are percentage points; rho and normalized distances are dimensionless. "
             f"Source abbreviations: **R = `{REAL}`**, **S = `{SIM}`**. Planning status is always SIMULATED; per-metric status, full source labels and evidence paths are retained in the gap CSVs.", "",
             "### Exp2: correlations and quintile accuracy", "",
             "| Dataset | Reliability | REAL / planning status | rho R / S / gap | p R / S / gap | Q1 accuracy R / S / gap | Q5 accuracy R / S / gap | Sources |",
             "|---|---|---|---:|---:|---:|---:|---|"]
    for dataset in ("DTD", "FGVCAircraft", "EuroSAT"):
        for reliability in ("r_ins", "r_batch", "r_cls"):
            selected = [find("Exp2", dataset, metric, Reliability=reliability, Statistic="Spearman") for metric in ("rho", "p_value")]
            selected += [find("Exp2", dataset, "teacher_batch_accuracy", Reliability=reliability, Quantile=q) for q in ("Q1", "Q5")]
            status, sources = status_and_sources(selected)
            lines.append("| " + " | ".join([dataset, reliability, status, *map(triplet, selected), sources]) + " |")
    lines += ["", "[All Exp2 correlation, Q1–Q5 accuracy and batch-count gaps](comparison/REAL_VS_HYPOTHETICAL_exp2_gaps.csv). "
              "[Measured Q1–Q5 table and curves](analysis/REAL_exp234_analysis_zh.md) retain all reversals. "
              "The planning scenario assumed batch size 64 and assumed N; the measured evaluation uses fixed batches of 8. "
              "Its p-values and the illustrative planning p-values do not share an empirical sampling design, so their numerical gaps are not a significance comparison.", ""]
    lines += reliability_interpretation(tables)
    lines += ["### Exp3: teacher accuracy, student HM and gain", "",
              "| Dataset | Stage | REAL / planning status | Teacher accuracy R / S / gap | Equal HM R / S / gap | Adaptive HM R / S / gap | HM gain R / S / gap | Sources |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for dataset in ("DTD", "FGVCAircraft", "EuroSAT"):
        for quality in ("Weak", "Medium", "Strong"):
            selected = [find("Exp3", dataset, metric, TeacherQuality=quality) for metric in ("TeacherAccuracy", "Equal_HM", "Adaptive_HM", "HM_Gain")]
            status, sources = status_and_sources(selected)
            lines.append("| " + " | ".join([dataset, quality, status, *map(triplet, selected), sources]) + " |")
    for dataset in ("DTD", "FGVCAircraft", "EuroSAT"):
        gains = [find("Exp3", dataset, "HM_Gain", TeacherQuality=quality) for quality in ("Weak", "Medium", "Strong")]
        if all(row["REAL_Value"] is not None for row in gains):
            observed = [row["REAL_Value"] for row in gains]
            assumed = [row["Planning_Value"] for row in gains]
            if observed[0] < observed[1] < observed[2] and assumed[0] > assumed[2]:
                lines += ["", f"In these one-seed {dataset} runs, Adaptive-minus-Equal gains rise from Weak to Medium to Strong: "
                          + ", ".join(f"{value:+.6f}" for value in observed)
                          + ". This is opposite to the planning scenario's larger Weak-teacher advantage. "
                          "These measured stages therefore do not support a claim that Adaptive benefits grow as the teacher weakens; this limitation is specific to the observed dataset/protocol.", ""]
            if all(value < 0 for value in observed):
                lines += ["", f"All three measured {dataset} Adaptive-minus-Equal HM gains are negative (Weak, Medium, Strong): "
                          + ", ".join(f"{value:+.6f}" for value in observed)
                          + ". These runs show no Adaptive HM benefit over Equal on this dataset; a positive result on another dataset cannot be generalized to it.", ""]
                equal_drop = find("Exp3", dataset, "Drop_equal", TeacherQuality="Weak")["REAL_Value"]
                adaptive_drop = find("Exp3", dataset, "Drop_adaptive", TeacherQuality="Weak")["REAL_Value"]
                if equal_drop is not None and adaptive_drop is not None and adaptive_drop > equal_drop:
                    lines += [f"For {dataset}, Adaptive Drop={adaptive_drop:.10f} exceeds Equal Drop={equal_drop:.10f}. "
                              "This also fails to support stronger Weak-teacher robustness under the signed Strong-minus-Weak HM-drop metric in the measured one-seed protocol.", ""]
    lines += ["", "Signed Strong-minus-Weak drops (REAL / planning / gap):", ""]
    for dataset in ("DTD", "FGVCAircraft", "EuroSAT"):
        eq, ad = [find("Exp3", dataset, metric, TeacherQuality="Weak") for metric in ("Drop_equal", "Drop_adaptive")]
        status, sources = status_and_sources([eq, ad])
        lines.append(f"- {dataset}: Equal **{triplet(eq)}**; Adaptive **{triplet(ad)}**; status {status}; sources {sources}.")
    lines += ["", "[All Exp3 Base/Novel/HM, teacher-accuracy, gain and signed-drop gaps](comparison/REAL_VS_HYPOTHETICAL_exp3_gaps.csv). "
              "Negative measured drops are retained: they indicate a higher Weak student HM in that run. "
              "Actual teacher accuracy covers the fixed all-class held-out subset; planning accuracy was assumed. "
              "Actual epochs 5/12 are raw snapshots and epoch 20 is final GPA, so the stage comparison includes both training progress and averaging.", "",
              "### Exp4: relation distances and same-run HM", "",
              "| Dataset | Variant | REAL / planning status | D_ins R / S / gap | D_batch R / S / gap | D_cls R / S / gap | HM R / S / gap | Sources |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for dataset in ("DTD", "FGVCAircraft", "EuroSAT"):
        for variant in ("InstanceOnly", "StaticHierarchical", "AdaptiveLUNA"):
            selected = [find("Exp4", dataset, metric, Variant=variant) for metric in ("D_ins", "D_batch", "D_cls", "HM")]
            status, sources = status_and_sources(selected)
            lines.append("| " + " | ".join([dataset, variant, status, *map(triplet, selected), sources]) + " |")
    lines += relation_ranking_text(tables)
    lines += ["", "[All Exp4 distance/HM gaps](comparison/REAL_VS_HYPOTHETICAL_exp4_gaps.csv). "
              "D_ins is KL in nats per image; D_batch/D_cls are normalized Frobenius distances under the declared batching. "
              "StaticHierarchical reuses Equal, and each REAL HM comes from the same student checkpoint as its distances. "
              "Planning distances are designed values, so an arithmetic gap is not a reproduction error or evidence that one protocol is causally better.", "",
              "[Complete Exp1 Base/Novel/HM gaps](comparison/REAL_VS_HYPOTHETICAL_exp1_gaps.csv) supplement the Exp1 summary above.", ""]
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != expected for path, expected in input_hashes.items()):
        raise ValueError("A planning source file changed during comparison generation")
    return lines, outputs
