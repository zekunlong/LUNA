"""Export audited unavailable experimental cells without inventing measurements.

This exporter is intentionally for the zero-measurement case. It refuses to run
if the execution audit contains a ready dataset or any measured batch/results.
It never imports the hypothetical scenario. Missing metric and source cells stay
blank; NOT_RUN is a status, never an experimental measurement.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

DATASETS = ("FGVCAircraft", "DTD", "EuroSAT")
RELIABILITIES = ("r_ins", "r_batch", "r_cls")
SCHEMAS = {
    "Sanity": ["Dataset", "Variant", "BatchCount", "r_ins", "r_batch", "r_cls",
               "mean_lambda_ins", "std_lambda_ins", "mean_lambda_batch", "std_lambda_batch",
               "mean_lambda_cls", "std_lambda_cls", "lambda_sum_approximately_one",
               "student_prompt_gradients_positive", "teacher_gradients_zero",
               "adaptive_not_scalar_rescaled_legacy"],
    "Exp1_Weighting": ["Dataset", "Variant", "Base", "Novel", "HM"],
    "Exp2_Reliability": ["Dataset", "Variant", "Statistic", "Reliability", "Quantile",
                         "rho", "p_value", "N_batches", "teacher_batch_accuracy"],
    "Exp3_TeacherQuality": ["Dataset", "Variant", "TeacherQuality", "TeacherEpoch", "TeacherAccuracy",
                           "Equal_Base", "Equal_Novel", "Equal_HM", "Adaptive_Base", "Adaptive_Novel",
                           "Adaptive_HM", "HM_Gain", "Drop_equal", "Drop_adaptive"],
    "Exp4_RelationPreservation": ["Dataset", "Variant", "D_ins", "D_batch", "D_cls", "HM"],
}
NAMES = {
    "Exp1_Weighting": "exp1_weighting", "Exp2_Reliability": "exp2_reliability",
    "Exp3_TeacherQuality": "exp3_teacher_quality", "Exp4_RelationPreservation": "exp4_relation_preservation",
}
META = ["Status", "ResultSource", "FailureID", "ReasonCategories", "ExactReason", "Evidence"]


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def write_tex(path, sheet, rows):
    metrics = {
        "Exp1_Weighting": ["Base", "Novel", "HM"],
        "Exp2_Reliability": ["rho", "p_value", "N_batches", "teacher_batch_accuracy"],
        "Exp3_TeacherQuality": ["TeacherEpoch", "TeacherAccuracy", "Equal_Base", "Equal_Novel", "Equal_HM",
                               "Adaptive_Base", "Adaptive_Novel", "Adaptive_HM", "HM_Gain", "Drop_equal", "Drop_adaptive"],
        "Exp4_RelationPreservation": ["D_ins", "D_batch", "D_cls", "HM"],
    }[sheet]
    columns = ["Dataset", "Variant"] + metrics + ["Status", "FailureID"]
    # All missing cells are --, never zeros or hypothetical placeholders.
    lines = ["% Audited real-result availability. -- means no measurement was produced.",
             r"% Requires \usepackage{graphicx} for width-constrained export.",
             r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
             r"\caption{REAL " + tex_escape(sheet) + r": no measured results available.}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline",
             " & ".join(map(tex_escape, columns)) + r" \\", r"\hline"]
    for row in rows:
        lines.append(" & ".join(tex_escape(row.get(key, "")) if row.get(key) not in (None, "") else "--"
                                for key in columns) + r" \\")
    lines += [r"\hline", r"\end{tabular}", "}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def worksheet(book, name, title, rows, columns):
    sheet = book.create_sheet(name)
    sheet.append([title])
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(columns))
    sheet.cell(1, 1).font = Font(bold=True, color="FFFFFF", size=12)
    sheet.cell(1, 1).fill = PatternFill("solid", fgColor="244062")
    sheet.append(["Blank metric/source cells mean NOT_RUN; values are not imputed. See Failure_Log."])
    sheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(columns))
    sheet.append(columns)
    for cell in sheet[3]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DCE6F1")
    for row in rows:
        sheet.append([row.get(key) if row.get(key) != "" else None for key in columns])
    sheet.freeze_panes = "C4"
    sheet.auto_filter.ref = f"A3:{get_column_letter(len(columns))}{max(3, sheet.max_row)}"
    for index, key in enumerate(columns, 1):
        sheet.column_dimensions[get_column_letter(index)].width = min(52, max(15, len(key) + 3))
    for row in sheet.iter_rows(min_row=4):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    return sheet


def build(root):
    state = json.loads((root / "logs" / "real_execution_status.json").read_text(encoding="utf-8"))
    if state["real_execution_complete"] is not True:
        raise ValueError("Complete all possible real execution before exporting its final status")
    if state["measured_result_count"] or state["real_batch_count"]:
        raise ValueError("This availability-only exporter cannot handle measured results")
    if any(item["status"] == "READY" for item in state["datasets"].values()):
        raise ValueError("A runnable dataset must be executed before exporting unavailable cells")
    audit = root / "asset_audit.json"
    if not audit.is_file():
        raise FileNotFoundError("Asset search evidence is required")
    tables = {key: [] for key in SCHEMAS}
    failures = []
    paths = "; ".join(state["paths_searched"])

    def add(sheet, dataset, variant, **identity):
        entry = state["datasets"][dataset]
        categories = list(entry["reason_categories"])
        reason = entry["exact_reason"]
        requirement = entry["required_assets"]
        if sheet == "Exp1_Weighting" and variant == "StaticMean":
            reason += " No complete AdaptiveLUNA training trace exists from which to obtain training-only mean lambdas."
            requirement += " Complete AdaptiveLUNA first, then reuse its verified training_mean_weights.json."
        if sheet == "Exp2_Reliability":
            reason += " No valid AdaptiveLUNA checkpoint or deterministic evaluation-batch predictions are available. No correlation or quantile estimate was calculated."
            requirement += " Complete AdaptiveLUNA and implement/run the separate fixed-order, fixed-seed, fixed-batch checkpoint evaluation entry on the declared split without retraining. The existing entry rejects eval-only; see OFFLINE_VALIDATION.md."
        if sheet == "Exp3_TeacherQuality" and identity.get("TeacherQuality") in ("Weak", "Medium"):
            categories.append("MISSING_INTERMEDIATE_TEACHER_CHECKPOINT")
            reason += " No teacher epoch series or verified total-epoch metadata exists to select the nearest 25%/60% checkpoint."
            requirement += " Recover the original teacher epoch series and total epochs; select by epoch distance only."
        if sheet == "Exp4_RelationPreservation":
            reason += " No compatible final student checkpoint or teacher/student evaluation probability tensors exist for this variant."
            requirement += " Reuse Exp1 Equal/Adaptive or train InstanceOnly; implement/run the separate checkpoint evaluation entry on the same run and fixed split for KL and normalized Gram distances. See OFFLINE_VALIDATION.md for the current entry-point gap."
        failure_id = f"F{len(failures) + 1:03d}"
        evidence = "; ".join([str(audit), *entry["evidence"]])
        row = {key: None for key in SCHEMAS[sheet] + META}
        row.update(Dataset=dataset, Variant=variant, Status="NOT_RUN", FailureID=failure_id,
                   ReasonCategories="; ".join(dict.fromkeys(categories)), ExactReason=reason, Evidence=evidence,
                   **identity)
        if sheet == "Sanity":
            row["BatchCount"] = state["real_batch_count"]  # observed execution count, not a model metric
        tables[sheet].append(row)
        missing = [key for key in SCHEMAS[sheet] if row[key] is None]
        failures.append({"FailureID": failure_id, "Experiment": sheet, "Dataset": dataset,
                         "Variant": variant, "Status": "NOT_RUN", "ReasonCategory": row["ReasonCategories"],
                         "ExactReason": reason, "Evidence": evidence, "PathsSearched": paths,
                         "RequiredForRealResult": requirement, "MissingFields": ", ".join(missing)})

    add("Sanity", "DTD", "10-batch sanity")
    for dataset in DATASETS:
        for variant in ("AdaptiveLUNA", "StaticMean", "Equal"):
            add("Exp1_Weighting", dataset, variant)
        for reliability in RELIABILITIES:
            add("Exp2_Reliability", dataset, f"AdaptiveLUNA/{reliability}/Spearman",
                Statistic="Spearman", Reliability=reliability)
            for quantile in range(1, 6):
                add("Exp2_Reliability", dataset, f"AdaptiveLUNA/{reliability}/Q{quantile}",
                    Statistic="Reliability quantile", Reliability=reliability, Quantile=f"Q{quantile}")
        for quality in ("Weak", "Medium", "Strong"):
            add("Exp3_TeacherQuality", dataset, f"Equal vs AdaptiveLUNA/{quality}", TeacherQuality=quality)
        for variant in ("InstanceOnly", "StaticHierarchical", "AdaptiveLUNA"):
            add("Exp4_RelationPreservation", dataset, variant)

    (root / "tables").mkdir(exist_ok=True)
    for sheet, stem in NAMES.items():
        write_csv(root / "tables" / f"REAL_{stem}.csv", tables[sheet], SCHEMAS[sheet] + META)
        write_tex(root / "tables" / f"REAL_{stem}.tex", sheet, tables[sheet])
    write_csv(root / "tables" / "REAL_sanity.csv", tables["Sanity"], SCHEMAS["Sanity"] + META)
    write_csv(root / "tables" / "REAL_failure_log.csv", failures, list(failures[0]))

    lines = ["# Failure analysis for REAL experimental results", "",
             "All entries below are **NOT_RUN**. Preflight ran; no real model batch was executed. "
             "These are unavailable results, not observed numerical or training failures. "
             "Blank measurements have not been replaced with zeros.", "",
             "The search covers accessible paths and inspected archive listings documented in ASSET_AUDIT.md. "
             "Uninspected or inaccessible locations are explicitly retained there; absence is limited to that scope.", "",
             "The fixed reliability formulation was preserved. Offline fixture tests are recorded in "
             "OFFLINE_VALIDATION.md and do not supply experimental cells below.", ""]
    for fail in failures:
        lines += [f"## {fail['FailureID']}", "",
                  f"Experiment: {fail['Experiment']}  ", f"Dataset: {fail['Dataset']}  ",
                  f"Variant: {fail['Variant']}  ", "Status: NOT_RUN", "",
                  f"Reason category: {fail['ReasonCategory']}", "", f"Exact reason: {fail['ExactReason']}", "",
                  f"Evidence: {fail['Evidence']}", "", f"Paths searched: {fail['PathsSearched']}", "",
                  f"What would be required to obtain the real result: {fail['RequiredForRealResult']}", "",
                  f"Missing result fields covered by this entry: {fail['MissingFields']}", ""]
    (root / "FAILURE_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    book = Workbook()
    book.remove(book.active)
    for sheet, rows in tables.items():
        worksheet(book, sheet, f"REAL {sheet} - experimental availability", rows, SCHEMAS[sheet] + META)
    worksheet(book, "Failure_Log", "REAL experimental execution failure/availability log", failures, list(failures[0]))
    path = root / "Journal_LUNA_REAL_Results.xlsx"
    book.save(path)
    verified = load_workbook(path, read_only=True, data_only=True)
    expected = [*SCHEMAS, "Failure_Log"]
    if verified.sheetnames != expected:
        raise AssertionError("Unexpected workbook sheets")
    for sheet, columns in SCHEMAS.items():
        for row in verified[sheet].iter_rows(min_row=4, values_only=True):
            mapping = dict(zip(columns + META, row))
            if mapping["Status"] != "NOT_RUN" or mapping["ResultSource"] is not None:
                raise AssertionError("Unavailable results must not claim a measured source")
    verified.close()
    result = {"status": "PASS", "real_metric_count": 0, "failure_entries": len(failures),
              "table_row_counts": {key: len(rows) for key, rows in tables.items()},
              "workbook_sheets": expected,
              "checks": ["missing experimental metrics are blank", "no measured source claimed for NOT_RUN",
                         "separate workbook", "all missing cells mapped to failure entries"]}
    (root / "logs" / "real_export_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("E:/LUNA/results_reliability_analysis"))
    build(parser.parse_args().output_root)
