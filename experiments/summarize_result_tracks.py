"""Join availability and planning reports without copying planning into REAL files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from openpyxl import load_workbook

SOURCE = "SIMULATED_POSITIVE_FOR_PLANNING_ONLY"
TITLE = "HYPOTHETICAL / SIMULATED POSITIVE RESULTS FOR PLANNING ONLY"
SPECS = [
    ("Exp1", "exp1_weighting", ["Dataset", "Variant", "Base", "Novel", "HM"]),
    ("Exp2 correlations", "exp2_reliability", ["Dataset", "Reliability", "Rho", "PValue", "NBatches"]),
    ("Exp2 quantiles", "exp2_reliability_quantiles", ["Dataset", "Reliability", "Quantile", "TeacherBatchAccuracy", "NBatches"]),
    ("Exp3", "exp3_teacher_quality", ["Dataset", "TeacherQuality", "TeacherEpoch", "TeacherAccuracy", "EqualHM", "AdaptiveHM", "HMGain"]),
    ("Exp3 drops", "exp3_teacher_quality_drops", ["Dataset", "DropEqual", "DropAdaptive", "DropReduction"]),
    ("Exp4", "exp4_relation_preservation", ["Dataset", "Variant", "D_ins", "D_batch", "D_cls", "HM"]),
]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def display(key, value):
    if value in (None, ""):
        return "—"
    try:
        number = float(value)
    except ValueError:
        return value.replace("|", "/")
    if key == "PValue":
        return f"{number:.3g}"
    if key in ("NBatches", "TeacherEpoch"):
        return str(int(number))
    if key.startswith("D_"):
        return f"{number:.3f}"
    return f"{number:.2f}"


def build(root):
    state = json.loads((root / "logs" / "real_execution_status.json").read_text(encoding="utf-8"))
    if state["real_batch_count"] != 0 or state["measured_result_count"] != 0:
        raise ValueError("This status comparison needs extension before reporting measured results")
    records = {}
    for _, stem, _ in SPECS:
        rows = read_csv(root / "tables" / f"HYPOTHETICAL_{stem}.csv")
        if not rows or any(row["ResultSource"] != SOURCE or TITLE not in row["TableTitle"] for row in rows):
            raise AssertionError("Every hypothetical row must carry its explicit source and warning title")
        records[stem] = rows
    for file in (root / "tables").glob("REAL_*.csv"):
        if SOURCE in file.read_text(encoding="utf-8-sig"):
            raise AssertionError(f"Planning values leaked into {file}")
        metric_keys = {"Base", "Novel", "HM", "r_ins", "r_batch", "r_cls", "rho", "p_value", "N_batches",
                       "teacher_batch_accuracy", "TeacherEpoch", "TeacherAccuracy", "Equal_Base", "Equal_Novel",
                       "Equal_HM", "Adaptive_Base", "Adaptive_Novel", "Adaptive_HM", "HM_Gain", "Drop_equal",
                       "Drop_adaptive", "D_ins", "D_batch", "D_cls", "mean_lambda_ins", "std_lambda_ins",
                       "mean_lambda_batch", "std_lambda_batch", "mean_lambda_cls", "std_lambda_cls"}
        for row in read_csv(file):
            if any(row[key] not in (None, "") for key in metric_keys.intersection(row)):
                raise AssertionError(f"Unexpected numeric experiment result in unavailable table: {file}")
    for file in (root / "tables").glob("HYPOTHETICAL_*.tex"):
        if not file.read_text(encoding="utf-8").startswith("% WARNING:"):
            raise AssertionError(f"Missing LaTeX warning: {file}")
    real = load_workbook(root / "Journal_LUNA_REAL_Results.xlsx", read_only=True, data_only=True)
    for sheet in real:
        for row in sheet.iter_rows(values_only=True):
            if any(SOURCE in str(cell) for cell in row if cell is not None):
                raise AssertionError("Planning source appeared in real workbook")
    real.close()
    lines = ["# REAL vs HYPOTHETICAL comparison", "",
             "**REAL:** zero measured dataset batches and zero measured Exp1–4 results. "
             "**NOT_RUN:** every required experimental cell. "
             f"**SIMULATED:** every planning number below has source `{SOURCE}`.", "",
             "Software tests and asset preflight are actual executed checks; they are not paper experiment results. "
             "No real-to-planning numerical gap can be computed while the real value is absent. "
             "The dash in the gap column means unavailable, never zero.", "",
             "## REAL sanity", "",
             "NOT_RUN: 0/10 batches. No real r_ins/r_batch/r_cls, lambda means/stds, "
             "gradient values or loss-comparison values were obtained. "
             "No hypothetical sanity measurements were requested or invented.", "",
             "## REAL failure reasons", "",
             "| Dataset | Real status | Exact reason | Evidence |", "|---|---|---|---|"]
    for dataset, entry in state["datasets"].items():
        lines.append(f"| {dataset} | NOT_RUN | {entry['exact_reason'].replace('|', '/')} | "
                     "[Asset audit](ASSET_AUDIT.md), [cell-level failure analysis](FAILURE_ANALYSIS.md) |")
    lines += ["", "Exp1 additionally lacks complete adaptive training means for StaticMean. "
              "Exp2 lacks adaptive checkpoints and deterministic evaluation batches. "
              "Exp3 lacks the original teacher epoch series and verified total epochs; Strong cannot reuse absent Exp1 results. "
              "Exp4 lacks compatible final student checkpoints and paired teacher/student probabilities. "
              "The work still needed to run checkpoint analyses is recorded in OFFLINE_VALIDATION.md.", ""]
    for name, stem, columns in SPECS:
        lines += [f"## {name}: {TITLE}", "",
                  "All numeric metric columns in this table are SIMULATED planning values. "
                  "REAL values are unavailable; the real status is NOT_RUN for every row.", "",
                  "| " + " | ".join(columns + ["REAL status", "Planning label", "ResultSource", "Gap"]) + " |",
                  "|" + "---|" * (len(columns) + 4)]
        for row in records[stem]:
            lines.append("| " + " | ".join([display(key, row[key]) for key in columns] +
                                               ["NOT_RUN", "SIMULATED", SOURCE, "—"]) + " |")
        lines.append("")
    lines += ["## Interpretation and provenance", "",
              "The hypothetical scenario is a conditional planning exercise, not an inference from available model outputs. "
              "P-values are illustrative t approximations from assumed rho/N, and quantiles are constructed targets; "
              "no underlying real or simulated batch predictions are asserted. "
              "The spreadsheets retain the exact source label on every planning row.", "",
              "Existing manuscript numbers and rendered attention/t-SNE plots were retained as prior materials without "
              "importing them as results of the fixed journal implementation. Their run provenance still needs validation.", "",
              "Conditional scenario assessment: **PR_REASONABLE_FIRST_SUBMISSION**, based exclusively on a "
              "HYPOTHETICAL positive result scenario and subject to reproducing the findings and authenticating broader experiments. "
              "Current empirical submission readiness: **NOT_READY**.", "",
              "Files: [REAL workbook](Journal_LUNA_REAL_Results.xlsx), "
              "[planning workbook](HYPOTHETICAL_POSITIVE_RESULTS_FOR_PLANNING.xlsx), "
              "[hypothetical analysis](HYPOTHETICAL_POSITIVE_ANALYSIS.md), "
              "[failure analysis](FAILURE_ANALYSIS.md).", ""]
    (root / "REAL_VS_HYPOTHETICAL_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    exp1 = {(row["Dataset"], row["Variant"]): float(row["HM"]) for row in records["exp1_weighting"]}
    datasets = state["datasets"]
    gains_equal = [exp1[(d, "AdaptiveLUNA")] - exp1[(d, "Equal")] for d in datasets]
    gains_static = [exp1[(d, "AdaptiveLUNA")] - exp1[(d, "StaticMean")] for d in datasets]
    metrics = {"status": "PASS", "source_separation_validated": True,
               "real_measured_result_count": 0, "real_batch_count": 0,
               "hypothetical_source": SOURCE,
               "planning_mean_HM_gain_vs_Equal": sum(gains_equal) / len(gains_equal),
               "planning_mean_HM_gain_vs_StaticMean": sum(gains_static) / len(gains_static),
               "scenario_assessment": "PR_REASONABLE_FIRST_SUBMISSION", "empirical_readiness": "NOT_READY"}
    report_path = root / "REPORT.md"
    archive = root / "logs" / "REPORT_before_current_continuation.md"
    if report_path.exists() and not archive.exists():
        archive.write_bytes(report_path.read_bytes())
    report = ["# Journal LUNA continuation: final available-work status", "",
              "The requested two-track deliverables are complete for the assets currently available. "
              "The real DTD sanity and Exp1–4 remain **NOT_RUN**; no measured paper result was produced.", "",
              "## REAL execution", "",
              "The audit enumerated 601,580 accessible files across E: and listed 71 actual loose archives. "
              "No usable DTD/FGVCAircraft/EuroSAT images, original splits, required OpenAI CLIP weights, "
              "dataset-specific PromptSRC teachers, teacher epoch provenance, or reusable journal students were found. "
              "Eleven inaccessible directories and nested-archive inspection limits are recorded explicitly in ASSET_AUDIT.md.", "",
              "The real DTD entry was invoked. Its final preflight completed with CUDA available and stopped for missing "
              "external assets before the first batch. An initial 45-second import timeout was diagnosed as cold "
              "scikit-learn import and resolved on retry; it is preserved as historical evidence, not a current blocker.", "",
              "All 71 software tests passed with zero failures/errors/skips. Two technical fixes were made: intermediate "
              "teacher checkpoints now pass the correct epoch preflight, and the sanity comparison rejects scalar-rescaled "
              "legacy contributions. The fixed reliability formulation was not changed or tuned. "
              "These tests and a synthetic source diagnostic do not substitute for real model batches.", "",
              "The REAL workbook has all six requested sheets. Experimental metric/source cells are blank for unavailable "
              "results; 82 entries in FAILURE_ANALYSIS.md identify every missing cell group and how to obtain it. "
              "BatchCount=0 is the observed execution count. No performance cell is filled with a zero or planning value.", "",
              "## HYPOTHETICAL planning", "",
              f"Every planning table and row is explicitly labeled `{SOURCE}`. The complete Exp1–4 scenario exists only "
              "in the separate hypothetical workbook/tables. Its generator checks full-precision HM, shared Strong/Equal/Adaptive "
              "results across experiments, teacher-quality declines, constructed correlations/quantiles, and source provenance. "
              "The workbook and CSV/TeX files passed 234 scenario/export checks.", "",
              "SIMULATED average Exp1 gains: +1.61 HM vs Equal, +1.14 vs StaticMean. SIMULATED Exp2 rho: 0.41–0.61. "
              "SIMULATED Exp3 gains: +2.96/+2.11/+1.61 for Weak/Medium/Strong, with smaller Adaptive drops. "
              "SIMULATED Exp4 Adaptive distances: 0.112/0.151/0.169 on average. These are designed planning values.", "",
              "The conditional assessment is **PR_REASONABLE_FIRST_SUBMISSION**, based only on the HYPOTHETICAL positive "
              "scenario and conditional on real replication and authenticated broader experiments. Actual empirical readiness is "
              "**NOT_READY**. Existing legacy figures and manuscript tables were not promoted into new measured journal results.", "",
              "## Deliverables", "",
              "- [REAL results workbook](Journal_LUNA_REAL_Results.xlsx)",
              "- [HYPOTHETICAL planning workbook](HYPOTHETICAL_POSITIVE_RESULTS_FOR_PLANNING.xlsx)",
              "- [Failure analysis](FAILURE_ANALYSIS.md)",
              "- [Hypothetical paper analysis](HYPOTHETICAL_POSITIVE_ANALYSIS.md)",
              "- [REAL vs HYPOTHETICAL comparison](REAL_VS_HYPOTHETICAL_SUMMARY.md)",
              "- [Asset search](ASSET_AUDIT.md) and [offline validation](OFFLINE_VALIDATION.md)",
              "- CSV/TeX tables in `tables/`; checksums and source-separation checks in `logs/final_deliverable_validation.json`.", "",
              "## Reproduction and remaining execution", "",
              "From `E:/LUNA/code_luna/LUNA`, use `E:/LUNA/.venv_journal/Scripts/python.exe` to run, in order: "
              "`experiments/audit_journal_assets.py`, `experiments/run_journal_sanity.py`, "
              "`experiments/record_real_availability.py`, `experiments/export_unavailable_real_results.py`, "
              "`experiments/generate_hypothetical_planning.py --real-execution-complete`, and "
              "`experiments/summarize_result_tracks.py`. The unavailable-result exporters deliberately refuse ready datasets "
              "or measured batches; genuine recovered assets must enter actual execution/reuse, not this zero-measurement path.", "",
              "With real assets, execute Adaptive → full training mean → StaticMean → Equal. Complete the deterministic "
              "checkpoint evaluation entry for Exp2/4, select Weak/Medium teachers by original epoch distance, and reuse "
              "protocol-identical Exp1 checkpoints. These missing execution entry points are documented in OFFLINE_VALIDATION.md; "
              "this report does not claim that untested full experiment orchestration has been completed. "
              "No independently runnable dataset was withheld because DTD was absent.", ""]
    report_path.write_text("\n".join(report), encoding="utf-8")
    deliverables = [root / "Journal_LUNA_REAL_Results.xlsx", root / "HYPOTHETICAL_POSITIVE_RESULTS_FOR_PLANNING.xlsx",
                    root / "FAILURE_ANALYSIS.md", root / "HYPOTHETICAL_POSITIVE_ANALYSIS.md",
                    root / "REAL_VS_HYPOTHETICAL_SUMMARY.md", report_path, *sorted((root / "tables").glob("REAL_*")),
                    *sorted((root / "tables").glob("HYPOTHETICAL_*"))]
    metrics["artifacts"] = [{"path": str(path), "bytes": path.stat().st_size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in deliverables]
    (root / "logs" / "final_deliverable_validation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "artifacts"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("E:/LUNA/results_reliability_analysis"))
    build(parser.parse_args().output_root)
