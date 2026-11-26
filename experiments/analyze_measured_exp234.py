"""Rebuild Chinese analysis and standalone plots from authenticated REAL CSVs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def generate_quintile_figure(root, tables):
    """Show each score's accuracy ranking in its own panel, using observed bins."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    datasets = (("DTD", "DTD"), ("FGVCAircraft", "FGVC Aircraft"), ("EuroSAT", "EuroSAT"))
    scores = (
        ("r_ins", "Prediction certainty", r"$r_{\mathrm{ins}}$", "#28699B", "o"),
        ("r_batch", "Sample relation stability", r"$r_{\mathrm{batch}}$", "#BF642C", "s"),
        ("r_cls", "Class relation stability", r"$r_{\mathrm{cls}}$", "#21816D", "^"),
    )
    destination = root / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9,
        "text.color": "#202A34", "axes.labelcolor": "#202A34",
        "xtick.color": "#53606C", "ytick.color": "#53606C",
        "axes.edgecolor": "#ABB3BB", "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }):
        fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.8), sharex=True, sharey=True)
        fig.subplots_adjust(left=.105, right=.979, bottom=.158, top=.783,
                            hspace=.55, wspace=.19)
        fig.text(.5, .967, "Do higher scores identify more accurate teacher predictions?",
                 ha="center", va="top", fontsize=12, weight="semibold")
        for column, (_, title, symbol, color, _) in enumerate(scores):
            bounds = axes[0, column].get_position()
            center = (bounds.x0 + bounds.x1) / 2
            fig.text(center, .898, title, ha="center", va="center", fontsize=9.5,
                     weight="semibold")
            fig.text(center, .867, symbol, ha="center", va="center", color=color, fontsize=10)
            fig.add_artist(Line2D([bounds.x0, bounds.x1], [.844, .844],
                                 transform=fig.transFigure, color=color, linewidth=1.7))

        for row_index, (dataset, display_name) in enumerate(datasets):
            rows = [row for row in tables["Exp2_Reliability"]
                    if row["Dataset"] == dataset and row["Statistic"] == "Reliability quantile"
                    and row["ResultSource"] == "REAL_EXPERIMENT"]
            row_top = axes[row_index, 0].get_position().y1
            fig.text(.105, row_top + .014, display_name, ha="left", va="bottom",
                     fontsize=10, weight="semibold")
            # Every score ranks the same set of batches; count that set once.
            count_rows = [row for row in rows if row["Reliability"] == "r_ins"]
            if len(count_rows) == 5:
                n_batches = sum(int(row["N_batches"]) for row in count_rows)
                n_samples = sum(int(row["N_samples"]) for row in count_rows)
                fig.text(.979, row_top + .014,
                         f"{n_batches:,} test batches · {n_samples:,} images",
                         ha="right", va="bottom", fontsize=8, color="#63707C")
            for column, (key, _, _, color, marker) in enumerate(scores):
                ax = axes[row_index, column]
                groups = {row["Quantile"]: row for row in rows if row["Reliability"] == key}
                values = [float(groups[f"Q{q}"]["teacher_batch_accuracy"])
                          if f"Q{q}" in groups and groups[f"Q{q}"]["teacher_batch_accuracy"] not in (None, "")
                          else np.nan for q in range(1, 6)]
                if any(np.isfinite(value) for value in values):
                    ax.plot(range(1, 6), values, marker=marker, color=color,
                            linewidth=1.65, markersize=4.4, markeredgecolor="white",
                            markeredgewidth=.5, zorder=3)
                    for x, value in ((1, values[0]), (5, values[-1])):
                        if np.isfinite(value):
                            below = value >= 86
                            ax.annotate(f"{value:.1f}", (x, value), textcoords="offset points",
                                        xytext=(0, -9 if below else 6), ha="center",
                                        va="top" if below else "bottom", fontsize=8.5,
                                        color=color, weight="semibold")
                else:
                    ax.text(.5, .5, "No measured data", transform=ax.transAxes,
                            ha="center", va="center", color="#63707C", fontsize=8)
                ax.set_xticks(range(1, 6),
                              ("Lowest\n20%", "", "Middle\n20%", "", "Highest\n20%"))
                ax.set_yticks((0, 25, 50, 75, 100))
                ax.set_xlim(.6, 5.4)
                ax.set_ylim(0, 100)
                ax.tick_params(axis="x", labelsize=8, length=2.5, pad=5)
                ax.tick_params(axis="y", labelsize=8, length=0, pad=5)
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.grid(axis="y", color="#E3E7EB", linewidth=.6)
                ax.set_axisbelow(True)

        fig.text(.024, .474, "Teacher accuracy (%)", rotation=90, ha="center",
                 va="center", fontsize=10)
        fig.text(.542, .079, "Test batches ranked by score: lowest to highest",
                 ha="center", va="center", fontsize=9)
        fig.text(.542, .036,
                 "Five groups of ≈20% of batches, ranked separately for each score.",
                 ha="center", va="center", fontsize=7.5, color="#63707C")
        for suffix in ("png", "pdf", "svg"):
            path = destination / f"REAL_exp2_quintile_accuracy.{suffix}"
            fig.savefig(path, dpi=300, facecolor="white")
            outputs.append(path)
        plt.close(fig)

    caption = destination / "REAL_exp2_quintile_accuracy_caption.md"
    caption.write_text(
        "**Suggested manuscript caption**\n\n"
        "Teacher accuracy across five groups of test batches ranked by three reliability scores. "
        "Rows correspond to datasets; columns correspond to prediction certainty "
        "($r_{\\mathrm{ins}}$), sample relation stability ($r_{\\mathrm{batch}}$), and class relation "
        "stability ($r_{\\mathrm{cls}}$). Within each dataset, batches are ranked independently "
        "by each score from lowest to highest, with approximately 20% of batches per group. "
        "The vertical axis shows the fraction of images classified correctly by the teacher "
        "in each group. An upward trend indicates that higher scores identify batches with "
        "more accurate teacher predictions. Numbers mark the lowest and highest groups. "
        "Teacher accuracy increases across prediction certainty groups in all three datasets; "
        "the relation stability scores do not show a consistent increase across datasets.\n\n"
        "**Score definitions and evaluation protocol**\n\n"
        "Prediction certainty is one minus the normalized mean prediction entropy across "
        "the original and augmented views. Sample and class relation stability measure the "
        "cosine similarity of the corresponding teacher probability Gram matrices across "
        "the two views. All three scores are batch scalars computed without labels. Labels "
        "are used only to evaluate accuracy in the full class prediction space.\n\n"
        "Results use seed 1 and the original test order, with eight images per batch and "
        "incomplete final batches excluded. Ties retain the original batch order before "
        "splitting into five groups. The lines connect the five observed values; no values "
        "are smoothed, reordered by accuracy, or imputed. This is a descriptive comparison "
        "under fixed, class-correlated batches; it does not establish a student training "
        "benefit or robustness across seeds and batch orders.\n\n"
        "**中文说明**\n\n"
        "这张图检查：被可靠性分数判为更可靠的测试批次，教师是否预测得更准确。"
        "每行是一个数据集，每列分别表示预测确定性、样本间关系稳定性、类别间关系稳定性。"
        "各指标独立排序，分为五组，每组约占批次数的 20%；三个指标的同序分组成员可以不同。"
        "纵轴为组内教师分类准确率，端点标注最低分组和最高分组的数值。"
        "按预测确定性从低到高分组后，教师准确率在三个数据集上均逐组上升；"
        "按关系稳定性分组后的准确率趋势则随数据集而变化。\n\n"
        "Source: `../tables/REAL_exp2_reliability.csv`, authenticated against "
        "`../logs/real_export_validation.json`.\n\n"
        "Regenerate only this figure and caption:\n\n"
        "```powershell\n"
        "python E:/LUNA/code_luna/LUNA/experiments/analyze_measured_exp234.py "
        "--output-root E:/LUNA/results_reliability_analysis --quintiles-only\n"
        "```\n",
        encoding="utf-8",
    )
    outputs.append(caption)
    return outputs


def generate(root, tables, now):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from experiments.summarize_measured_dtd import REAL, write_chinese_progress

    outputs = [write_chinese_progress(root, tables, now)]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    datasets = ("DTD", "FGVCAircraft", "EuroSAT")
    reliabilities = ("r_ins", "r_batch", "r_cls")
    colors = ("#3975AA", "#D88035", "#389474")
    for index, dataset in enumerate(datasets):
        a, b, c = axes[index]
        a.set_title(f"{dataset}: Exp2 reliability/correctness", fontsize=11)
        b.set_title(f"{dataset}: Exp3 teacher stage", fontsize=11)
        c.set_title(f"{dataset}: Exp4 relation distances", fontsize=11)
        correlations = {row["Reliability"]: row for row in tables["Exp2_Reliability"] if row["Dataset"] == dataset and row["Statistic"] == "Spearman" and row["ResultSource"] == REAL}
        if correlations:
            for i, key in enumerate(reliabilities):
                row = correlations.get(key)
                if row and row["rho"]:
                    value = float(row["rho"])
                    a.bar(i, value, color=colors[i], width=.6)
                    a.annotate(f"{value:.3f}\np={float(row['p_value']):.3g}", (i, value),
                               xytext=(0, 7 if value >= 0 else -8), textcoords="offset points",
                               ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
                else:
                    a.text(i, 0, "undefined", ha="center", fontsize=8)
            a.set_xticks(range(3), reliabilities)
            a.set_ylim(-1, 1)
            a.set_ylabel("Spearman rho")
            a.axhline(0, color="#555555", linewidth=.8)
        else:
            a.text(.5, .5, "NOT_RUN\nNo measured correlation", transform=a.transAxes, ha="center", va="center", color="#666666")
            a.set_xticks([]); a.set_yticks([])
        stage_rows = {row["TeacherQuality"]: row for row in tables["Exp3_TeacherQuality"] if row["Dataset"] == dataset and row["ResultSource"] == REAL}
        if stage_rows:
            for variant, prefix, color in (("Equal", "Equal", colors[0]), ("Adaptive", "Adaptive", colors[2])):
                values = [float(stage_rows[q][prefix + "_HM"]) if q in stage_rows and stage_rows[q][prefix + "_HM"] else np.nan for q in ("Weak", "Medium", "Strong")]
                b.plot(range(3), values, "o-", color=color, label=variant)
                for i, value in enumerate(values):
                    if np.isfinite(value):
                        other_key = "Adaptive_HM" if prefix == "Equal" else "Equal_HM"
                        other_raw = stage_rows[("Weak", "Medium", "Strong")[i]].get(other_key)
                        other = float(other_raw) if other_raw not in (None, "") else np.nan
                        above = not np.isfinite(other) or value > other or (value == other and variant == "Equal")
                        b.annotate(f"{value:.3f}", (i, value), textcoords="offset points", xytext=(0, 8 if above else -14), ha="center", fontsize=8)
            b.set_xticks(range(3), ("Weak\nraw e5", "Medium\nraw e12", "Strong\nGPA e20"))
            b.set_ylabel("Student HM (%)")
            b.margins(y=.3)
            b.legend(fontsize=8, loc="upper center", ncol=2)
        else:
            b.text(.5, .5, "NOT_RUN\nNo measured stage comparison", transform=b.transAxes, ha="center", va="center", color="#666666")
            b.set_xticks([]); b.set_yticks([])
        relation_rows = {row["Variant"]: row for row in tables["Exp4_RelationPreservation"] if row["Dataset"] == dataset and row["ResultSource"] == REAL}
        if relation_rows:
            x = np.arange(3)
            for i, variant in enumerate(("InstanceOnly", "StaticHierarchical", "AdaptiveLUNA")):
                row = relation_rows.get(variant)
                if row:
                    values = [float(row[key]) for key in ("D_ins", "D_batch", "D_cls")]
                    c.bar(x + (i - 1) * .24, values, .24, color=colors[i], label=f"{variant} (HM {float(row['HM']):.3f})")
            c.set_xticks(x, ("KL / image\n(nats)", "Batch Gram\n(normalized)", "Class Gram\n(normalized)"))
            c.set_ylabel("Distance (lower is closer)")
            c.legend(fontsize=7)
            pending = [key for key in ("InstanceOnly", "StaticHierarchical", "AdaptiveLUNA") if key not in relation_rows]
            if pending:
                c.text(.01, .98, "Pending: " + ", ".join(pending), transform=c.transAxes, fontsize=7, va="top")
            c.margins(y=.2)
        else:
            c.text(.5, .5, "NOT_RUN\nNo measured relation distances", transform=c.transAxes, ha="center", va="center", color="#666666")
            c.set_xticks([]); c.set_yticks([])
        for ax in (a, b, c):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
    fig.suptitle("REAL_EXPERIMENT | seed 1 | measured stages only; missing cells are not imputed\nExp2: unadjusted asymptotic p-values; class-correlated fixed batches. Exp3: raw snapshots vs final GPA.", fontsize=12)
    destination = root / "figures"
    destination.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        path = destination / f"REAL_exp234_overview.{suffix}"
        fig.savefig(path, dpi=200)
        outputs.append(path)
    plt.close(fig)
    outputs.extend(generate_quintile_figure(root, tables))
    # Exp1's historical results are unchanged; its forward pointer follows current evidence.
    previous = root / "analysis/REAL_DTD_exp1_analysis_zh.md"
    if previous.is_file():
        text = previous.read_text(encoding="utf-8")
        old = "可靠性是否反映教师正确率仍需 Exp2；关系项是否带来泛化收益仍需 Exp4。"
        new = "可靠性与正确率、关系距离的最新真实证据和未完成条件见 [Exp2–4 分阶段分析](REAL_exp234_analysis_zh.md)；本文件仅分析 Exp1，不能代替后续实验。"
        if old in text:
            previous.write_text(text.replace(old, new), encoding="utf-8")
    return outputs


def main():
    from experiments.summarize_measured_dtd import verified_experiment_tables
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("E:/LUNA/results_reliability_analysis"))
    parser.add_argument("--quintiles-only", action="store_true",
                        help="Rebuild only the Exp2 quintile figure and its manuscript caption")
    args = parser.parse_args()
    root = args.output_root.resolve()
    validation = json.loads((root / "logs/real_export_validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or not validation.get("serialized_outputs_validated"):
        raise ValueError("Run the measured exporter before rebuilding analysis")
    tables = verified_experiment_tables(root, validation)
    outputs = (generate_quintile_figure(root, tables) if args.quintiles_only else
               generate(root, tables, datetime.now(timezone.utc).isoformat()))
    print(json.dumps({"status": "PASS", "source": "REAL_EXPERIMENT", "outputs": [str(path) for path in outputs]}))


if __name__ == "__main__":
    main()
