"""Generate a wholly simulated manuscript-planning scenario, never real results.

This module does not read experiment tables, models, checkpoints, or datasets.
Run --validate-only while real work is ongoing. Writing requires the explicit
--real-execution-complete acknowledgment, after all possible real execution.
All statistical quantities are planning assumptions, not significance evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


SOURCE = "SIMULATED_POSITIVE_FOR_PLANNING_ONLY"
TITLE = "HYPOTHETICAL / SIMULATED POSITIVE RESULTS FOR PLANNING ONLY"
WARNING = (
    "These values are hypothetical planning values and are NOT experimental "
    "measurements. Do not submit them as real results."
)
TEX_WARNING = (
    "% WARNING:\n"
    "% These values are hypothetical planning values and are NOT\n"
    "% experimental measurements. Do not submit them as real results.\n"
)
DATASETS = ("FGVCAircraft", "DTD", "EuroSAT")
SHEETS = (
    "Exp1_Weighting", "Exp2_Reliability", "Exp3_TeacherQuality",
    "Exp4_RelationPreservation", "PR_Assessment",
)
SHEET_TABLES = {
    "Exp1_Weighting": ("exp1_weighting",),
    "Exp2_Reliability": ("exp2_reliability", "exp2_reliability_quantiles"),
    "Exp3_TeacherQuality": ("exp3_teacher_quality", "exp3_teacher_quality_drops"),
    "Exp4_RelationPreservation": ("exp4_relation_preservation",),
    "PR_Assessment": ("pr_assessment",),
}
ASSESSMENT = "PR_REASONABLE_FIRST_SUBMISSION"
SCOPE_URL = "https://shop.elsevier.com/journals/pattern-recognition/0031-3203"
SCOPE_CHECKED = "2026-09-05"
P_METHOD = "ILLUSTRATIVE_TWO_SIDED_T_APPROXIMATION_FROM_ASSUMED_RHO_AND_N"
AGGREGATE_NOTE = (
    "Designed summary targets; no synthetic batch sample was fitted or tested. "
    "Rho, N, quintiles and teacher accuracies are assumptions."
)

Rows = List[Dict[str, Any]]


def harmonic_mean(base: float, novel: float) -> float:
    return 2.0 * base * novel / (base + novel)


def accuracy_pair(base: float, hm: float) -> Dict[str, float]:
    """Retain the target HM; solve Novel without rounding its stored value."""
    novel = hm * base / (2.0 * base - hm)
    return {"Base": base, "Novel": novel, "HM": hm}


def tagged(section: str, **values: Any) -> Dict[str, Any]:
    return dict(values, ResultSource=SOURCE, TableTitle=f"{TITLE} - {section}")


def build_scenario() -> Dict[str, Rows]:
    # scipy is used only to map assumed rho/N to an illustrative p-value.
    from scipy.stats import t

    scenario: Dict[str, Rows] = {}
    weighting_spec = {
        "FGVCAircraft": ((44.10, 40.36), (44.85, 40.82), (45.95, 42.01)),
        "DTD": ((77.70, 73.50), (78.20, 74.02), (79.35, 75.21)),
        "EuroSAT": ((92.00, 86.89), (92.45, 87.32), (93.40, 88.36)),
    }
    weighting: Rows = []
    for dataset, pairs in weighting_spec.items():
        for variant, (base, hm) in zip(("Equal", "StaticMean", "AdaptiveLUNA"), pairs):
            weighting.append(tagged(
                "Exp1 Weighting", Dataset=dataset, Variant=variant,
                **accuracy_pair(base, hm),
                Assumption="Designed single-scenario percentage accuracies; no measured run or seed.",
            ))
    scenario["exp1_weighting"] = weighting
    by_run = {(r["Dataset"], r["Variant"]): r for r in weighting}

    correlations: Rows = []
    quantiles: Rows = []
    rho_spec = {
        "FGVCAircraft": (0.57, 0.43, 0.48),
        "DTD": (0.61, 0.46, 0.52),
        "EuroSAT": (0.49, 0.41, 0.45),
    }
    n_spec = {"FGVCAircraft": 50, "DTD": 30, "EuroSAT": 100}
    quintile_accuracy = {
        "FGVCAircraft": {
            "r_ins": (38.8, 44.6, 49.6, 54.5, 59.0),
            "r_batch": (40.2, 45.3, 49.4, 53.6, 58.0),
            "r_cls": (39.5, 44.9, 49.5, 54.0, 58.6),
        },
        "DTD": {
            "r_ins": (67.1, 72.0, 77.1, 81.7, 86.5),
            "r_batch": (68.4, 72.7, 76.9, 80.8, 85.6),
            "r_cls": (68.0, 72.4, 76.8, 81.2, 86.0),
        },
        "EuroSAT": {
            "r_ins": (86.5, 88.9, 91.2, 93.0, 95.4),
            "r_batch": (87.4, 89.3, 91.1, 92.4, 94.8),
            "r_cls": (87.0, 89.1, 91.0, 92.7, 95.2),
        },
    }
    reliability_means = {
        "FGVCAircraft": {
            "r_ins": (0.23, 0.30, 0.36, 0.42, 0.48),
            "r_batch": (0.80, 0.84, 0.87, 0.90, 0.93),
            "r_cls": (0.74, 0.79, 0.83, 0.87, 0.92),
        },
        "DTD": {
            "r_ins": (0.37, 0.45, 0.52, 0.59, 0.65),
            "r_batch": (0.83, 0.86, 0.89, 0.92, 0.95),
            "r_cls": (0.77, 0.81, 0.85, 0.89, 0.94),
        },
        "EuroSAT": {
            "r_ins": (0.57, 0.64, 0.70, 0.76, 0.81),
            "r_batch": (0.86, 0.89, 0.92, 0.94, 0.97),
            "r_cls": (0.81, 0.85, 0.88, 0.92, 0.96),
        },
    }
    for dataset in DATASETS:
        n = n_spec[dataset]
        for reliability, rho in zip(("r_ins", "r_batch", "r_cls"), rho_spec[dataset]):
            statistic = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
            pvalue = float(2.0 * t.sf(abs(statistic), df=n - 2))
            correlations.append(tagged(
                "Exp2 Reliability Correlations", Dataset=dataset,
                Reliability=reliability, Target="teacher_batch_accuracy",
                Rho=rho, PValue=pvalue, NBatches=n, TStatistic=statistic,
                DegreesOfFreedom=n - 2, PValueMethod=P_METHOD,
                EvaluationProtocol=(
                    "ASSUMED: fixed held-out split/order; shuffle=False; seed=1; "
                    "batch_size=64. N is a planning count, not an observed split count."
                ),
                Assumption=AGGREGATE_NOTE,
            ))
            for index, (reliability_mean, accuracy) in enumerate(zip(
                reliability_means[dataset][reliability],
                quintile_accuracy[dataset][reliability],
            ), start=1):
                quantiles.append(tagged(
                    "Exp2 Reliability Quintiles", Dataset=dataset,
                    Reliability=reliability, Quantile=f"Q{index}",
                    MeanReliability=reliability_mean,
                    TeacherBatchAccuracy=accuracy, NBatches=n // 5,
                    Assumption=AGGREGATE_NOTE,
                ))
    scenario["exp2_reliability"] = correlations
    scenario["exp2_reliability_quantiles"] = quantiles

    # Epochs are deliberately assumed. They are not claims about discovered
    # PromptSRC checkpoints or their availability. Round-half-up is irrelevant
    # for 20 epochs: 25% -> 5, 60% -> 12, final -> 20 exactly.
    quality_spec = {
        "FGVCAircraft": {
            "Weak": (39.00, 34.70, 41.00, 37.80, 40.10),
            "Medium": (42.10, 38.40, 44.00, 40.55, 46.20),
            "Strong": (44.10, 40.36, 45.95, 42.01, 49.30),
        },
        "DTD": {
            "Weak": (72.80, 68.00, 75.40, 71.00, 69.80),
            "Medium": (76.20, 71.80, 78.50, 74.00, 74.20),
            "Strong": (77.70, 73.50, 79.35, 75.21, 76.88),
        },
        "EuroSAT": {
            "Weak": (87.30, 81.55, 89.75, 84.33, 84.60),
            "Medium": (90.60, 85.13, 92.60, 87.11, 88.60),
            "Strong": (92.00, 86.89, 93.40, 88.36, 91.00),
        },
    }
    epoch_spec = {"Weak": (5, 0.25), "Medium": (12, 0.60), "Strong": (20, 1.0)}
    quality_rows: Rows = []
    drop_rows: Rows = []
    for dataset in DATASETS:
        for quality in ("Weak", "Medium", "Strong"):
            ebase, ehm, abase, ahm, teacher = quality_spec[dataset][quality]
            equal = accuracy_pair(ebase, ehm)
            adaptive = accuracy_pair(abase, ahm)
            if quality == "Strong":
                equal = {k: by_run[(dataset, "Equal")][k] for k in ("Base", "Novel", "HM")}
                adaptive = {k: by_run[(dataset, "AdaptiveLUNA")][k] for k in ("Base", "Novel", "HM")}
            epoch, fraction = epoch_spec[quality]
            quality_rows.append(tagged(
                "Exp3 Teacher Quality", Dataset=dataset, TeacherQuality=quality,
                TeacherEpoch=epoch, TeacherTotalEpochs=20,
                TargetEpochFraction=fraction, TeacherAccuracy=teacher,
                TeacherAccuracyDefinition="Mean teacher batch accuracy on the assumed fixed evaluation split",
                EqualBase=equal["Base"], EqualNovel=equal["Novel"], EqualHM=equal["HM"],
                AdaptiveBase=adaptive["Base"], AdaptiveNovel=adaptive["Novel"], AdaptiveHM=adaptive["HM"],
                HMGain=adaptive["HM"] - equal["HM"],
                Assumption=(
                    "20 teacher epochs assumed; epochs 5/12/20 selected by schedule only. "
                    "No checkpoint or test-performance selection occurred. Strong values share Exp1 identity."
                ),
            ))
        weak = next(r for r in quality_rows if r["Dataset"] == dataset and r["TeacherQuality"] == "Weak")
        strong = next(r for r in quality_rows if r["Dataset"] == dataset and r["TeacherQuality"] == "Strong")
        equal_drop = strong["EqualHM"] - weak["EqualHM"]
        adaptive_drop = strong["AdaptiveHM"] - weak["AdaptiveHM"]
        drop_rows.append(tagged(
            "Exp3 Strong-to-Weak Drops", Dataset=dataset,
            DropEqual=equal_drop, DropAdaptive=adaptive_drop,
            DropReduction=equal_drop - adaptive_drop,
            Assumption="Arithmetic differences between assumed Strong and Weak HMs; percentage points.",
        ))
    scenario["exp3_teacher_quality"] = quality_rows
    scenario["exp3_teacher_quality_drops"] = drop_rows

    relation_spec = {
        "FGVCAircraft": ((0.168, 0.248, 0.284), (0.150, 0.200, 0.225), (0.132, 0.173, 0.193)),
        "DTD": ((0.144, 0.218, 0.253), (0.129, 0.176, 0.199), (0.113, 0.153, 0.171)),
        "EuroSAT": ((0.117, 0.182, 0.216), (0.105, 0.146, 0.167), (0.091, 0.127, 0.143)),
    }
    instance_hm = {"FGVCAircraft": 39.75, "DTD": 72.78, "EuroSAT": 86.14}
    relation_rows: Rows = []
    for dataset in DATASETS:
        for variant, distances in zip(("InstanceOnly", "StaticHierarchical", "AdaptiveLUNA"), relation_spec[dataset]):
            exp1_variant = "Equal" if variant == "StaticHierarchical" else variant
            hm = instance_hm[dataset] if variant == "InstanceOnly" else by_run[(dataset, exp1_variant)]["HM"]
            relation_rows.append(tagged(
                "Exp4 Relation Preservation", Dataset=dataset, Variant=variant,
                D_ins=distances[0], D_batch=distances[1], D_cls=distances[2], HM=hm,
                SameScenarioRun=f"{dataset}:{exp1_variant}",
                MetricDefinition=(
                    "D_ins=mean KL(P_teacher||P_student); "
                    "D_batch=Frobenius(Mb_teacher-Mb_student)/(Frobenius(Mb_teacher)+1e-8); "
                    "D_cls=Frobenius(Mc_teacher-Mc_student)/(Frobenius(Mc_teacher)+1e-8)."
                ),
                Assumption="Designed relation distances; shared HM identifies the same assumed run, not an actual checkpoint.",
            ))
    scenario["exp4_relation_preservation"] = relation_rows

    assessment_rows = [
        ("Conclusion", ASSESSMENT, "Conditional assessment based solely on this HYPOTHETICAL positive result scenario; current empirical submission readiness is NOT_READY."),
        ("Exp1", "Adaptive advantage beyond one static control", "Mean gains: 1.61 HM vs Equal and 1.14 HM vs StaticMean. Real paired runs and uncertainty would be needed; StaticMean alone is not exhaustive coefficient tuning."),
        ("Exp2", "Moderate association with correctness", "Assumed rho spans 0.41-0.61. The p-values are illustrative arithmetic from assumptions; they cannot establish significance, calibration, or causality."),
        ("Exp3", "Greater benefit with weaker teachers", "Assumed mean gains are 2.96 / 2.11 / 1.61 HM for Weak / Medium / Strong, with a smaller Strong-to-Weak drop for Adaptive."),
        ("Exp4", "Better preservation of hierarchical relations", "Designed average D_ins/D_batch/D_cls: Adaptive 0.112/0.151/0.169; Static 0.128/0.174/0.197; InstanceOnly 0.143/0.216/0.251."),
        ("Existing qualitative material", "Context only until independently verified", "Attention heatmaps and class-wise t-SNE can illustrate mechanisms; their existence is not validation of the journal reliability formulation or of unseen results."),
        ("Existing quantitative material", "Verify protocol and numerical provenance", "Base-to-novel, cross-dataset and domain-generalization evidence would strengthen the case if actual artifacts show fair comparisons and match the fixed journal formulation."),
        ("Remaining empirical evidence", "Repeat, quantify uncertainty, contextualize novelty", "Obtain measured Exp1-4, multiple seeds with uncertainty, fixed teacher schedules, suitable current baselines, and cost/overhead reporting before submission."),
        ("Journal scope", "Plausible methodological fit, not acceptance prediction", f"Official publisher scope includes original pattern-recognition methodology and applications grounded in its literature. Source: {SCOPE_URL} (checked {SCOPE_CHECKED})."),
    ]
    scenario["pr_assessment"] = [tagged("PR Assessment", Topic=topic, Assessment=assessment, Rationale=rationale) for topic, assessment, rationale in assessment_rows]
    return scenario


def validate_scenario(scenario: Dict[str, Rows]) -> List[str]:
    """Fail closed on missing provenance, broken identities, or invalid targets."""
    from scipy.stats import t

    checks: List[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)
        checks.append(message)

    expected_counts = {
        "exp1_weighting": 9, "exp2_reliability": 9,
        "exp2_reliability_quantiles": 45, "exp3_teacher_quality": 9,
        "exp3_teacher_quality_drops": 3, "exp4_relation_preservation": 9,
        "pr_assessment": 9,
    }
    require(set(scenario) == set(expected_counts), "All requested scenario tables are present")
    for name, rows in scenario.items():
        require(len(rows) == expected_counts[name], f"{name}: expected row count")
        require(all(row["ResultSource"] == SOURCE and TITLE in row["TableTitle"] for row in rows), f"{name}: every row has exact simulated source and warning title")
        require(all(math.isfinite(value) for row in rows for value in row.values() if isinstance(value, (int, float))), f"{name}: all numeric values are finite")
        if name != "pr_assessment":
            require({r["Dataset"] for r in rows} == set(DATASETS), f"{name}: all three datasets are covered")

    first = {(r["Dataset"], r["Variant"]): r for r in scenario["exp1_weighting"]}
    quality = {(r["Dataset"], r["TeacherQuality"]): r for r in scenario["exp3_teacher_quality"]}
    fourth = {(r["Dataset"], r["Variant"]): r for r in scenario["exp4_relation_preservation"]}
    rho_rows = {(r["Dataset"], r["Reliability"]): r for r in scenario["exp2_reliability"]}
    for row in scenario["exp1_weighting"]:
        require(all(0 < row[k] <= 100 for k in ("Base", "Novel", "HM")), f"Exp1 {row['Dataset']} {row['Variant']}: valid percentage ranges")
        require(math.isclose(harmonic_mean(row["Base"], row["Novel"]), row["HM"], abs_tol=1e-12, rel_tol=0), f"Exp1 {row['Dataset']} {row['Variant']}: HM matches full-precision Base/Novel")
    for dataset in DATASETS:
        for metric in ("Base", "Novel", "HM"):
            require(first[(dataset, "Equal")][metric] < first[(dataset, "StaticMean")][metric] < first[(dataset, "AdaptiveLUNA")][metric], f"Exp1 {dataset}: Adaptive > StaticMean > Equal for {metric}")
        gain = first[(dataset, "AdaptiveLUNA")]["HM"] - first[(dataset, "Equal")]["HM"]
        static_gain = first[(dataset, "AdaptiveLUNA")]["HM"] - first[(dataset, "StaticMean")]["HM"]
        require(1.45 <= gain <= 1.8 and 0.8 <= static_gain <= 1.3, f"Exp1 {dataset}: moderate requested gains")
        for variant, prefix in (("Equal", "Equal"), ("AdaptiveLUNA", "Adaptive")):
            require(all(quality[(dataset, "Strong")][prefix + metric] == first[(dataset, variant)][metric] for metric in ("Base", "Novel", "HM")), f"Exp3 {dataset} Strong {variant}: exact Exp1 identity for Base/Novel/HM")
            require(quality[(dataset, "Weak")][prefix + "HM"] < quality[(dataset, "Medium")][prefix + "HM"] < quality[(dataset, "Strong")][prefix + "HM"], f"Exp3 {dataset} {variant}: HM increases with teacher stage")
        gains = [quality[(dataset, stage)]["HMGain"] for stage in ("Weak", "Medium", "Strong")]
        require(gains[0] > gains[1] > gains[2] > 0, f"Exp3 {dataset}: positive gains decline from Weak to Strong")
        require(quality[(dataset, "Weak")]["TeacherAccuracy"] < quality[(dataset, "Medium")]["TeacherAccuracy"] < quality[(dataset, "Strong")]["TeacherAccuracy"], f"Exp3 {dataset}: teacher accuracy increases with stage")
        drop = next(r for r in scenario["exp3_teacher_quality_drops"] if r["Dataset"] == dataset)
        require(drop["DropEqual"] > drop["DropAdaptive"] > 0 and math.isclose(drop["DropReduction"], gains[0] - gains[2], abs_tol=1e-12), f"Exp3 {dataset}: positive smaller adaptive drop and consistent drop reduction")
        for variant, exp1_variant in (("StaticHierarchical", "Equal"), ("AdaptiveLUNA", "AdaptiveLUNA")):
            require(fourth[(dataset, variant)]["HM"] == first[(dataset, exp1_variant)]["HM"], f"Exp4 {dataset} {variant}: exact shared-run Exp1 HM")
        require(fourth[(dataset, "InstanceOnly")]["HM"] < fourth[(dataset, "StaticHierarchical")]["HM"] < fourth[(dataset, "AdaptiveLUNA")]["HM"], f"Exp4 {dataset}: requested HM ordering")
        for metric in ("D_ins", "D_batch", "D_cls"):
            values = [fourth[(dataset, variant)][metric] for variant in ("AdaptiveLUNA", "StaticHierarchical", "InstanceOnly")]
            require(0 < values[0] < values[1] < values[2], f"Exp4 {dataset}: positive distances with Adaptive lowest for {metric}")
        for reliability in ("r_ins", "r_batch", "r_cls"):
            corr = rho_rows[(dataset, reliability)]
            require(0.3 < corr["Rho"] < 0.7 and 0 < corr["PValue"] < 1 and corr["NBatches"] > 2, f"Exp2 {dataset} {reliability}: moderate rho and valid illustrative p/N")
            require(corr["PValueMethod"] == P_METHOD and corr["DegreesOfFreedom"] == corr["NBatches"] - 2, f"Exp2 {dataset} {reliability}: illustrative p-value assumption is explicit")
            expected_t = corr["Rho"] * math.sqrt((corr["NBatches"] - 2) / (1 - corr["Rho"] ** 2))
            expected_p = float(2 * t.sf(abs(expected_t), df=corr["NBatches"] - 2))
            require(math.isclose(corr["TStatistic"], expected_t, abs_tol=1e-12) and math.isclose(corr["PValue"], expected_p, rel_tol=1e-12, abs_tol=0), f"Exp2 {dataset} {reliability}: p and t reproduce the declared approximation")
            qrows = [r for r in scenario["exp2_reliability_quantiles"] if r["Dataset"] == dataset and r["Reliability"] == reliability]
            require([r["Quantile"] for r in qrows] == [f"Q{i}" for i in range(1, 6)] and sum(r["NBatches"] for r in qrows) == corr["NBatches"], f"Exp2 {dataset} {reliability}: Q1-Q5 and batch counts match N")
            for metric in ("MeanReliability", "TeacherBatchAccuracy"):
                values = [r[metric] for r in qrows]
                require(all(a < b for a, b in zip(values, values[1:])), f"Exp2 {dataset} {reliability}: monotonic {metric}")
            require(all(0 <= r["MeanReliability"] <= 1 and 0 <= r["TeacherBatchAccuracy"] <= 100 for r in qrows), f"Exp2 {dataset} {reliability}: valid quintile metric ranges")
            batch_average = sum(r["TeacherBatchAccuracy"] * r["NBatches"] for r in qrows) / corr["NBatches"]
            require(math.isclose(batch_average, quality[(dataset, "Strong")]["TeacherAccuracy"], abs_tol=1e-12, rel_tol=0), f"Exp2/3 {dataset} {reliability}: quintile mean equals Strong teacher accuracy")
    for row in scenario["exp3_teacher_quality"]:
        for prefix in ("Equal", "Adaptive"):
            require(math.isclose(harmonic_mean(row[prefix + "Base"], row[prefix + "Novel"]), row[prefix + "HM"], abs_tol=1e-12, rel_tol=0), f"Exp3 {row['Dataset']} {row['TeacherQuality']} {prefix}: HM matches Base/Novel")
        require(row["TeacherEpoch"] == int(row["TeacherTotalEpochs"] * row["TargetEpochFraction"]), f"Exp3 {row['Dataset']} {row['TeacherQuality']}: epoch follows assumed schedule")
        require(math.isclose(row["HMGain"], row["AdaptiveHM"] - row["EqualHM"], abs_tol=1e-12), f"Exp3 {row['Dataset']} {row['TeacherQuality']}: HM gain arithmetic")
    target_means = {
        "InstanceOnly": (0.143, 0.216, 0.251),
        "StaticHierarchical": (0.128, 0.174, 0.197),
        "AdaptiveLUNA": (0.112, 0.151, 0.169),
    }
    for variant, targets in target_means.items():
        for metric, target in zip(("D_ins", "D_batch", "D_cls"), targets):
            require(math.isclose(sum(fourth[(dataset, variant)][metric] for dataset in DATASETS) / 3, target, abs_tol=1e-12), f"Exp4 {variant}: exact requested mean {metric}")
    return checks


def tex_escape(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(c, c) for c in str(value))


TEX_COLUMNS = {
    "exp1_weighting": ("Dataset", "Variant", "Base", "Novel", "HM"),
    "exp2_reliability": ("Dataset", "Reliability", "Rho", "PValue", "NBatches"),
    "exp2_reliability_quantiles": ("Dataset", "Reliability", "Quantile", "MeanReliability", "TeacherBatchAccuracy", "NBatches"),
    "exp3_teacher_quality": ("Dataset", "TeacherQuality", "TeacherEpoch", "TeacherAccuracy", "EqualBase", "EqualNovel", "EqualHM", "AdaptiveBase", "AdaptiveNovel", "AdaptiveHM", "HMGain"),
    "exp3_teacher_quality_drops": ("Dataset", "DropEqual", "DropAdaptive", "DropReduction"),
    "exp4_relation_preservation": ("Dataset", "Variant", "D_ins", "D_batch", "D_cls", "HM"),
}


def write_csv_tex(tables_dir: Path, name: str, rows: Rows) -> List[Path]:
    csv_path = tables_dir / f"HYPOTHETICAL_{name}.csv"
    tex_path = tables_dir / f"HYPOTHETICAL_{name}.tex"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = TEX_COLUMNS[name] + ("ResultSource",)
    lines = [TEX_WARNING.rstrip(), "% Requires \\usepackage{graphicx} for width-constrained export.", "% Full precision and all assumptions are retained in the companion CSV."]
    if name.startswith("exp2"):
        lines.append("% Assumed rho/N and designed quintiles; p-values use a two-sided t approximation, not observed significance.")
    if name.startswith("exp3"):
        lines.append("% Teacher total epochs=20 and Weak/Medium/Strong epochs=5/12/20 are assumptions, not available checkpoints.")
    lines += [r"\begin{table}[htbp]", r"\centering", r"\tiny", r"\caption{" + tex_escape(rows[0]["TableTitle"]) + "}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline", " & ".join(tex_escape(c) for c in columns) + r" \\", r"\hline"]
    lines.extend(" & ".join(tex_escape(row[c]) for c in columns) + r" \\" for row in rows)
    lines += [r"\hline", r"\end{tabular}", "}", r"\end{table}", ""]
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return [csv_path, tex_path]


def write_workbook(path: Path, scenario: Dict[str, Rows]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    workbook.remove(workbook.active)
    workbook.properties.title = TITLE
    workbook.properties.subject = WARNING
    workbook.properties.description = SOURCE
    orange = PatternFill("solid", fgColor="FFF2CC")
    dark = PatternFill("solid", fgColor="1F4E78")
    for sheet_name, table_names in SHEET_TABLES.items():
        ws = workbook.create_sheet(sheet_name)
        for table_index, table_name in enumerate(table_names):
            if table_index:
                ws.append([None])
                ws.append([None])
            rows = scenario[table_name]
            columns = list(rows[0])
            title_row = ws.max_row + 1 if ws.max_row > 1 or ws.cell(1, 1).value else 1
            ws.cell(title_row, 1, rows[0]["TableTitle"])
            ws.merge_cells(start_row=title_row, start_column=1, end_row=title_row, end_column=len(columns))
            ws.cell(title_row, 1).font = Font(bold=True, color="9C0006", size=12)
            ws.cell(title_row, 1).fill = orange
            ws.cell(title_row, 1).alignment = Alignment(wrap_text=True, vertical="center")
            ws.row_dimensions[title_row].height = 34
            ws.cell(title_row + 1, 1, WARNING)
            ws.merge_cells(start_row=title_row + 1, start_column=1, end_row=title_row + 1, end_column=len(columns))
            ws.cell(title_row + 1, 1).font = Font(bold=True, color="9C0006")
            ws.cell(title_row + 1, 1).alignment = Alignment(wrap_text=True)
            ws.row_dimensions[title_row + 1].height = 30
            ws.append(columns)
            header_row = ws.max_row
            for cell in ws[header_row]:
                cell.fill = dark
                cell.font = Font(bold=True, color="FFFFFF")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            ws.row_dimensions[header_row].height = 34
            for row in rows:
                ws.append([row[col] for col in columns])
                for col_index, col in enumerate(columns, start=1):
                    cell = ws.cell(ws.max_row, col_index)
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
                    if isinstance(row[col], float):
                        cell.number_format = "0.0000E+00" if col == "PValue" else "0.000000"
                    if col == "ResultSource":
                        cell.fill = orange
                        cell.font = Font(bold=True, color="9C0006")
                ws.row_dimensions[ws.max_row].height = 60 if sheet_name == "PR_Assessment" else 46
            for col_index, col in enumerate(columns, start=1):
                width = 22
                if col in ("Assumption", "Rationale", "EvaluationProtocol", "MetricDefinition"):
                    width = 72
                elif col in ("ResultSource", "TableTitle", "PValueMethod", "TeacherAccuracyDefinition"):
                    width = 48
                ws.column_dimensions[get_column_letter(col_index)].width = max(ws.column_dimensions[get_column_letter(col_index)].width or 0, width)
        ws.freeze_panes = "C4"
        ws.sheet_view.showGridLines = False
        ws.print_options.horizontalCentered = True
        ws.page_setup.orientation = "landscape"
        ws.page_setup.paperSize = ws.PAPERSIZE_A3
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.print_title_rows = "1:3"
        ws.oddHeader.center.text = "SIMULATED PLANNING ONLY"
        ws.oddFooter.center.text = "NOT EXPERIMENTAL MEASUREMENTS | Page &P of &N"
    workbook.save(path)


def analysis_markdown(scenario: Dict[str, Rows]) -> str:
    exp1 = {(r["Dataset"], r["Variant"]): r for r in scenario["exp1_weighting"]}
    rows = []
    for dataset in DATASETS:
        e, s, a = (exp1[(dataset, variant)]["HM"] for variant in ("Equal", "StaticMean", "AdaptiveLUNA"))
        rows.append(f"| {dataset} | {e:.2f} | {s:.2f} | {a:.2f} | {a-e:.2f} | {a-s:.2f} | {SOURCE} |")
    original_hm = harmonic_mean(44.10, 37.20)
    adjusted_novel = exp1[("FGVCAircraft", "Equal")]["Novel"]
    return f"""# {TITLE}

**ResultSource: {SOURCE}**

**{WARNING}** This is a complete, intentionally positive planning scenario for
all three datasets and Exp1-4. No value below is a measured result; no real table
was read to construct it. Real execution status and evidence belong in the
separate REAL workbook, FAILURE_ANALYSIS.md and REAL_VS_HYPOTHETICAL_SUMMARY.md.

## Conditional journal assessment

**{ASSESSMENT}, based explicitly on a HYPOTHETICAL positive result scenario.**
If real experiments reproduce these trends with credible uncertainty estimates,
fair comparisons and a clearly established methodological contribution, LUNA
would be a reasonable first submission to Pattern Recognition. This planning
exercise cannot establish empirical readiness: a submission relying on these
simulated values remains **NOT_READY**. The category does not predict acceptance.

The publisher's scope includes original contributions to pattern-recognition
methods and applications grounded in the field's literature. Reliability-aware
vision-language adaptation is plausibly within that scope; the proposed fit is
an inference, not an editorial judgment. [Official Pattern Recognition scope]({SCOPE_URL})
(checked {SCOPE_CHECKED}).

## Exp1: value beyond a static mean

Table: {TITLE} - Exp1 HM and gains

| Dataset | Equal HM | StaticMean HM | Adaptive HM | Gain vs Equal | Gain vs StaticMean | ResultSource |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

The designed average gain is +1.61 HM against Equal and +1.14 against StaticMean.
If measured, this would support the usefulness of dynamic reliability weighting
beyond the specific fixed training-mean control. It would not establish that
every alternative static coefficient choice is inferior. Adaptive and StaticMean
must share the same full protocol, and StaticMean coefficients must come only
from the completed adaptive training trace. Keep the reliability formulation
fixed; quantify variation across matched seeds without selecting by test results.

The supplied Aircraft Base=44.10 and Novel=37.20 yield HM={original_hm:.12f}, which
already rounds correctly to 40.36. To make the full-precision stored HM target
exactly 40.36, only stored Novel is adjusted to {adjusted_novel:.15f}; its displayed
value remains 37.20. All other Novel values are solved by N=H*B/(2*B-H), with
full precision retained in CSV/XLSX. Rounded table numbers need not recompute
to the final displayed HM without their hidden precision.

## Exp2: association with actual teacher correctness

The assumed Spearman correlations range from 0.41 to 0.61. Positive moderate
associations and increasing Q1-Q5 teacher batch accuracies would be encouraging
evidence that the fixed reliabilities track correctness. These are batch-level
relationships; they would not alone establish per-sample confidence calibration
or causality. No entropy comparison is used as independent correctness evidence.

For this scenario only, N is assumed to be 50 Aircraft, 30 DTD and 100 EuroSAT
batches. These counts are placeholders, not verified dataset/split counts.
The hypothetical evaluation protocol assumes shuffle=False, fixed order and
split, batch_size=64 and seed=1. Actual evaluation must derive N from the genuine
loader and state how a final partial batch is handled. The illustrative p-value
is 2*sf_t(abs(rho*sqrt((N-2)/(1-rho^2))), df=N-2. It is an arithmetic consequence
of assumed inputs, **not an observed significance test**. No synthetic batch
vectors were fitted; correlations and quintiles are separately designed summary
targets, not statistics claimed to originate from one reconstructed dataset.

Every reliability has five equal-count quintiles with increasing mean teacher
accuracy. DTD r_batch follows 68.4, 72.7, 76.9, 80.8, 85.6 percent. The mean
teacher batch accuracy from each reliability's quintiles matches the corresponding
Strong teacher accuracy in Exp3 (49.30, 76.88 and 91.00 percent). These means are
batch averages, not separately asserted per-image pooled accuracies. Real
inference should address tied ranks, dependent batches and the nine correlations.
Fixed ordering can cluster classes or difficulty, so shared batch composition can
confound both reliability and correctness; examine class composition and confirm
the interpretation under a prespecified batching sensitivity analysis. Also
report uncertainty or permutation tests suited to the actual data rather than
copying these illustrative p-values.

## Exp3: imperfect teachers

Teacher training is assumed to last 20 epochs. Weak/Medium/Strong use epochs
5/12/20, chosen only by 25%/60%/100% of that schedule. These are not claims that
such checkpoints exist. In real execution, use the nearest available checkpoint
under a prespecified tie rule, with the final teacher epoch verified from its
configuration or log, never selected by test accuracy.

The designed mean Equal/Adaptive HMs are 61.42/64.38 (Weak), 65.11/67.22 (Medium)
and 66.92/68.53 (Strong). Thus mean gains decline from +2.96 to +2.11 to +1.61 HM.
The mean Strong-to-Weak decline is 5.50 HM for Equal and 4.15 for Adaptive, a
1.35-point reduction. If reproduced, that pattern would support the interpretation
that reliability weighting becomes more useful with imperfect teachers. Shared
Strong Base, Novel and HM values are copied exactly from the corresponding Exp1
scenario rows. TeacherAccuracy uses the same assumed fixed evaluation split
across quality levels. Other teacher architectures or corruption sources could
test the breadth of the interpretation after these required runs are measured.

## Exp4: hierarchical relation preservation

The designed mean D_ins/D_batch/D_cls is 0.143/0.216/0.251 for InstanceOnly,
0.128/0.174/0.197 for StaticHierarchical, and 0.112/0.151/0.169 for AdaptiveLUNA.
Adaptive has the lowest distances and highest HM in all three assumed datasets.
StaticHierarchical and Adaptive use exactly the corresponding Equal and Adaptive
Exp1 HMs; there is no selection of a different accuracy run. Distances use the
requested mean teacher-to-student KL and normalized Frobenius definitions with
epsilon 1e-8. All distances here are designed values, not measurements from a
teacher/student forward pass.

If real, simultaneous relation preservation and HM improvements would connect
the mechanism to the behavioral gain. These metrics are closely related to the
training objectives, so they are supportive mechanism evidence, not independent
proof of generalization. Compute them from matched actual checkpoints on the
same deterministic held-out batches and report their units/normalization.

## Combining with existing material

Attention heatmaps could illustrate changes in attended regions if examples,
preprocessing and comparison models are matched. Class-wise t-SNE could illustrate
embedding structure if the sampling, seed and projection settings are disclosed.
Neither visualization validates a numerical claim on its own. The asset inventory
located attention PNGs and class-wise t-SNE panels under
`E:/LUNA/LUNA/tsne/split_panels`, and legacy quantitative claims in
`E:/LUNA/LUNA/IEEE Conference Template - ICME 2026/comb.tex`. Presence establishes
that material exists; it does not authenticate the runs, numerical claims or
visual interpretations. This analysis does not claim to have validated them.

Base-to-novel evaluation supplies the central adaptation outcome. Cross-dataset
and domain-generalization results could establish broader usefulness if their
actual artifacts demonstrate fair training data, class splits, backbone and
checkpoint selection. Existing legacy results must be labeled with their actual
method and protocol: they cannot automatically validate the fixed journal
reliability formulation. Reconcile their numeric provenance before combining
them with new experiments. No missing existing accuracy is invented here.

## What would make the conditional case convincing

Measured Exp1-4 with valid assets and paired seeds would be the foundation. Report
dispersion or confidence intervals, inspect effect consistency by dataset, and
explain how the fixed reliability formulation advances appropriate current
baselines. Document teacher/student compute and the overhead of reliability and
relational losses. The three requested datasets offer distinct fine-grained,
texture and remote-sensing settings, but the claimed breadth should match the
full verified benchmark suite. Qualitative examples should accompany quantitative
evidence and include difficult cases.

This is a **{ASSESSMENT}** assessment **only under the HYPOTHETICAL positive
result scenario**. Real execution failures, absent provenance or irreproducible
gains remain open issues regardless of the attractive planning numbers. These
files must stay separate from final manuscript experimental tables.
"""


def validate_written_outputs(output_root: Path, scenario: Dict[str, Rows], paths: List[Path]) -> List[str]:
    from openpyxl import load_workbook

    checks: List[str] = []
    for path in paths:
        if path.name.startswith("REAL_") or path.name == "Journal_LUNA_REAL_Results.xlsx":
            raise ValueError(f"Refusing a real-results output name: {path}")
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            if not rows or any(row.get("ResultSource") != SOURCE or TITLE not in row.get("TableTitle", "") for row in rows):
                raise ValueError(f"Missing simulated provenance/title: {path}")
            expected_rows = scenario[path.stem.removeprefix("HYPOTHETICAL_")]
            if len(rows) != len(expected_rows) or any(
                {key: str(value) for key, value in expected.items()} != actual
                for expected, actual in zip(expected_rows, rows)
            ):
                raise ValueError(f"CSV full-precision round-trip mismatch: {path}")
            checks.append(f"{path.name}: all serialized CSV rows retain source/title")
            checks.append(f"{path.name}: all fields round-trip without precision loss")
        elif path.suffix == ".tex":
            content = path.read_text(encoding="utf-8")
            if not content.startswith(TEX_WARNING) or TITLE not in content:
                raise ValueError(f"Missing mandatory TeX warning/title: {path}")
            name = path.stem.removeprefix("HYPOTHETICAL_")
            if content.count(tex_escape(SOURCE)) != len(scenario[name]):
                raise ValueError(f"TeX data-row source count mismatch: {path}")
            checks.append(f"{path.name}: warning comment, title, and every data-row source validated")
    wb_path = output_root / "HYPOTHETICAL_POSITIVE_RESULTS_FOR_PLANNING.xlsx"
    workbook = load_workbook(wb_path, data_only=False)
    if tuple(workbook.sheetnames) != SHEETS:
        raise ValueError(f"Workbook sheets differ from requested schema: {workbook.sheetnames}")
    source_count = 0
    for ws in workbook.worksheets:
        source_column = None
        headers = None
        expected_rows = [row for name in SHEET_TABLES[ws.title] for row in scenario[name]]
        row_index = 0
        for cells in ws.iter_rows(values_only=True):
            if not any(value is not None for value in cells):
                source_column = None
                continue
            if "ResultSource" in cells:
                source_column = cells.index("ResultSource")
                headers = cells
                continue
            if source_column is not None:
                if cells[source_column] != SOURCE:
                    raise ValueError(f"Workbook row without simulated source in {ws.title}: {cells}")
                expected = expected_rows[row_index]
                for column, key in enumerate(headers):
                    if key is None:
                        continue
                    actual = cells[column]
                    target = expected[key]
                    if isinstance(target, float):
                        matches = isinstance(actual, (int, float)) and math.isclose(actual, target, rel_tol=1e-14, abs_tol=1e-12)
                    else:
                        matches = actual == target
                    if not matches:
                        raise ValueError(f"Workbook value mismatch in {ws.title}, row {row_index}, {key}: {actual!r} != {target!r}")
                row_index += 1
                source_count += 1
        if row_index != len(expected_rows):
            raise ValueError(f"Missing workbook rows in {ws.title}")
    if source_count != sum(len(rows) for rows in scenario.values()):
        raise ValueError("Workbook does not contain all scenario data rows")
    checks.append(f"Workbook: exact five sheet names and all {source_count} data-row sources validated")
    checks.append("Workbook: all fields and numeric precision validated after reopening")
    workbook.close()
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("E:/LUNA/results_reliability_analysis"))
    parser.add_argument("--validate-only", action="store_true", help="Validate the in-memory scenario; create no files.")
    parser.add_argument("--real-execution-complete", action="store_true", help="Acknowledge all possible real execution is complete before writing planning artifacts.")
    args = parser.parse_args()
    if not args.validate_only and not args.real_execution_complete:
        parser.error("Writing is allowed only after real work completes: pass --real-execution-complete, or use --validate-only.")
    scenario = build_scenario()
    checks = validate_scenario(scenario)
    if args.validate_only:
        print(json.dumps({"Status": "PASS", "Mode": "IN_MEMORY_VALIDATION_ONLY", "OutputFilesCreated": 0, "InvariantChecks": len(checks), "RowCounts": {key: len(rows) for key, rows in scenario.items()}, "ResultSource": SOURCE}, indent=2))
        return 0
    output_root = args.output_root.resolve()
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for name, rows in scenario.items():
        if name == "pr_assessment":
            continue
        paths.extend(write_csv_tex(tables_dir, name, rows))
    workbook_path = output_root / "HYPOTHETICAL_POSITIVE_RESULTS_FOR_PLANNING.xlsx"
    write_workbook(workbook_path, scenario)
    paths.append(workbook_path)
    analysis_path = output_root / "HYPOTHETICAL_POSITIVE_ANALYSIS.md"
    analysis_path.write_text(analysis_markdown(scenario), encoding="utf-8")
    paths.append(analysis_path)
    checks.extend(validate_written_outputs(output_root, scenario, paths))
    report = {
        "Title": TITLE, "ResultSource": SOURCE, "Warning": WARNING,
        "Status": "PASS", "RealExecutionCompleteAcknowledgedByOperator": True,
        "GeneratedUTC": datetime.now(timezone.utc).isoformat(),
        "GeneratorSHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ScenarioSHA256": hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
        "InputMeasuredResultsRead": 0,
        "InvariantChecksPassed": len(checks), "Checks": checks,
        "RowCounts": {name: len(rows) for name, rows in scenario.items()},
        "WorkbookSheets": list(SHEETS),
        "Assumptions": {
            "Source": SOURCE,
            "TeacherEpochs": "20 total epochs, stages 5/12/20; schedule assumed, not measured or discovered.",
            "EvaluationCounts": "FGVCAircraft=50, DTD=30, EuroSAT=100 assumed batches; not verified split sizes.",
            "StatisticalMethod": P_METHOD,
            "StatisticalCaveat": AGGREGATE_NOTE,
            "AircraftRounding": "44.10/37.20 already rounds to HM 40.36; stored Novel=37.204765886287625 enforces the exact full-precision HM target.",
            "Precision": "HM verified from stored Base/Novel to absolute tolerance 1e-12; no pre-export rounding.",
            "Assessment": f"{ASSESSMENT} conditional on the HYPOTHETICAL positive scenario only.",
        },
        "Artifacts": [{"Path": str(path), "SHA256": hashlib.sha256(path.read_bytes()).hexdigest(), "ResultSource": SOURCE} for path in paths],
    }
    report_path = output_root / "HYPOTHETICAL_VALIDATION_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"Status": "PASS", "InvariantChecks": len(checks), "Workbook": str(workbook_path), "Analysis": str(analysis_path), "ValidationReport": str(report_path), "ResultSource": SOURCE}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
