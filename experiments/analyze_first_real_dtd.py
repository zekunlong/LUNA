"""Describe and plot the completed, verified first real DTD weighting experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("E:/LUNA/results_reliability_analysis"))
    args = parser.parse_args()
    root = args.output_root
    table = root / "tables/REAL_exp1_weighting.csv"
    validation = json.loads((root / "logs/real_export_validation.json").read_text(encoding="utf-8"))
    if validation["status"] != "PASS":
        raise ValueError("Require a successful measured export first")
    measured = {r["Variant"]: r for r in records(table) if r["Dataset"] == "DTD" and r["ResultSource"] == "REAL_EXPERIMENT"}
    variants = ("Equal", "StaticMean", "AdaptiveLUNA")
    if set(measured) != set(variants):
        raise ValueError("Require all three verified real DTD variants")
    batches, manifests = {}, {}
    for variant in variants:
        row = measured[variant]
        manifest_path = Path(row["RunManifest"])
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != row["RunManifestSHA256"]:
            raise ValueError("Exported manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "COMPLETED_TRAINING":
            raise ValueError("Training did not complete")
        batches[variant] = records(manifest_path.parent / "journal_batches.csv")
        manifests[variant] = manifest
    traces = batches["AdaptiveLUNA"]
    weights = np.array([[float(row[f"lambda_{term}"]) for term in ("ins", "batch", "cls")] for row in traces])
    losses = np.array([[float(row[f"L_{term}"]) for term in ("ins", "batch", "cls")] for row in traces])
    contributions = weights * losses
    # Shares of the sum of logged weighted scalar losses, not gradient influence.
    shares = contributions.sum(axis=0) / contributions.sum()
    epochs = sorted({int(row["epoch"]) for row in traces})
    adaptive = measured["AdaptiveLUNA"]
    deltas = {v: {k: float(adaptive[k]) - float(measured[v][k]) for k in ("Base", "Novel", "HM")} for v in variants[:2]}
    if all(deltas[v]["HM"] > 0 for v in variants[:2]):
        verdict = "本轮 Adaptive 的 HM 高于两个对照，支持在当前 DTD / seed 1 配置下存在动态加权收益；仍需多 seed 和其他数据集复核。"
    elif deltas["Equal"]["HM"] > 0:
        verdict = "本轮 Adaptive 的 HM 高于 Equal，但没有超过 StaticMean；因此本轮不能证明动态变化优于固定训练均值。"
    else:
        verdict = "本轮 Adaptive 的 HM 没有超过 Equal，不支持当前配置下动态加权优于等权的结论。应保留这个真实结果，不能用规划值替代。"
    paired_audit = {}
    sample_fields = ("epoch", "batch_index", "sample_index", "impath", "ground_truth")
    adaptive_samples = records(Path(manifests["AdaptiveLUNA"]["raw_reliability_directory"]) / "samples.csv")
    for variant in variants[:2]:
        other_batches = batches[variant]
        if len(other_batches) != len(traces) or any(
            any(a[key] != b[key] for key in ("epoch", "batch_index", "n_samples"))
            for a, b in zip(traces, other_batches)
        ):
            raise ValueError(f"Unmatched batch counts or identities for {variant}")
        other_samples = records(Path(manifests[variant]["raw_reliability_directory"]) / "samples.csv")
        identity_matches = len(other_samples) == len(adaptive_samples) and all(
            all(a[key] == b[key] for key in sample_fields) for a, b in zip(adaptive_samples, other_samples))
        paired_audit[variant] = {
            "initialization_teacher_dataset_protocol_match": all(manifests[variant][key] == manifests["AdaptiveLUNA"][key]
                for key in ("initialization_sha256", "teacher_checkpoint_sha256", "dataset_fingerprint", "protocol_fingerprint", "source_hashes")),
            "training_sample_order_identical": identity_matches,
            "sample_observations_compared": len(other_samples),
            "teacher_reliability_exact_mismatch_batches": sum(any(a[key] != b[key] for key in ("r_ins", "r_batch", "r_cls"))
                for a, b in zip(traces, batches[variant])),
        }
        if not paired_audit[variant]["initialization_teacher_dataset_protocol_match"]:
            raise ValueError(f"Cohort provenance mismatch for {variant}")
    result = {
        "ResultSource": "REAL_EXPERIMENT", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "DTD", "seed": int(adaptive["Seed"]),
        "metrics_percent": {v: {k: float(measured[v][k]) for k in ("Base", "Novel", "HM")} for v in variants},
        "adaptive_delta_percentage_points": deltas,
        "interpretation_zh": verdict,
        "adaptive_training_batches": len(traces),
        "adaptive_lambda_mean": weights.mean(axis=0).tolist(),
        "adaptive_lambda_population_std": weights.std(axis=0).tolist(),
        "term_order": ["ins", "batch", "cls"],
        "weighted_scalar_loss_share": shares.tolist(),
        "paired_training_audit": paired_audit,
        "share_interpretation": "Sum of lambda_k * L_k divided by sum over all terms and batches; this is not a gradient contribution or causal attribution.",
        "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "limitations": ["One seed only; no significance or robustness claim", "Newly trained official-configuration teacher, not authors' authenticated released weights", "Exp2 held-out reliability validity and Exp4 mechanism tests remain unevaluated"],
    }
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    (analysis_dir / "REAL_DTD_exp1_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# DTD 首个真实 Exp1 结果分析", "", verdict, "", "所有下列数值来源：`REAL_EXPERIMENT`。", "",
             "| 方法 | Base (%) | Novel (%) | HM (%) |", "|---|---:|---:|---:|"]
    for variant in variants:
        lines.append("| " + " | ".join([variant, *[f"{float(measured[variant][k]):.4f}" for k in ("Base", "Novel", "HM")]]) + " |")
    lines += ["", f"Adaptive 相对 Equal：Base {deltas['Equal']['Base']:+.4f}、Novel {deltas['Equal']['Novel']:+.4f}、HM {deltas['Equal']['HM']:+.4f} 个百分点。",
              f"Adaptive 相对 StaticMean：Base {deltas['StaticMean']['Base']:+.4f}、Novel {deltas['StaticMean']['Novel']:+.4f}、HM {deltas['StaticMean']['HM']:+.4f} 个百分点。", "",
              "本轮仅一个 seed，差值不能直接解释为统计显著性；不能据此证明跨数据集稳定提升。", "",
              "## 权重是否真的在变化", "",
              f"以下统计覆盖 Adaptive 全部 {len(traces)} 个训练 batch，StaticMean 使用同一完整训练过程的均值。标准差表示 batch 间变化，不是均值置信区间。", "",
              "| 项 | lambda 均值 | lambda 总体标准差 | 加权标量损失占比 |", "|---|---:|---:|---:|"]
    for i, term in enumerate(("ins", "batch", "cls")):
        lines.append(f"| {term} | {weights[:, i].mean():.6f} | {weights[:, i].std():.6f} | {shares[i] * 100:.2f}% |")
    lines += ["", "加权损失占比按所有 batch 的 lambda_k × L_k 求和后归一化。它不是梯度贡献，不能用来证明某项的因果作用。",
              "lambda 大小接近不表示三个损失项贡献接近，还取决于原始损失尺度。可靠性公式与损失系数保持不变。", "",
              "DTD 10-batch sanity 检查通过，说明梯度、归一化及实现行为符合检查要求。可靠性与正确率、关系距离的最新真实证据和未完成条件见 [Exp2–4 分阶段分析](REAL_exp234_analysis_zh.md)；本文件仅分析 Exp1，不能代替后续实验。", "",
              "教师为基于官方 PromptKD / PromptSRC 配置新训练的 ViT-L/14，seed 1、20 epochs、最终 GPA 权重；学生为已有配置的 ViT-B/16、5 epochs。三组共享教师、初始化和训练协议，使用最终 epoch 评估，没有按测试准确率挑选 checkpoint。", "",
              "[真实结果图](../figures/REAL_DTD_exp1.png) · [原始结果 CSV](../tables/REAL_exp1_weighting.csv)", ""]
    lines += ["## 三组配对核查", "", "以下核查比较真实训练日志，未改变训练或结果。", ""]
    for variant, audit in paired_audit.items():
        lines.append(f"- {variant} 对 Adaptive：初始化/教师/数据/协议一致={audit['initialization_teacher_dataset_protocol_match']}；"
                     f"全部 {audit['sample_observations_compared']} 条图像记录顺序一致={audit['training_sample_order_identical']}；"
                     f"教师可靠性数值不完全相同的 batch 数={audit['teacher_reliability_exact_mismatch_batches']}。")
    lines += [""]
    (analysis_dir / "REAL_DTD_exp1_analysis_zh.md").write_text("\n".join(lines), encoding="utf-8")
    colors = ("#3975AA", "#D88035", "#389474")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = np.arange(3)
    for idx, key in enumerate(("Base", "Novel", "HM")):
        values = [float(measured[v][key]) for v in variants]
        bars = axes[0].bar(x + (idx - 1) * .25, values, .25, color=colors[idx], label=key)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axes[0].set_xticks(x, ("Equal", "StaticMean", "Adaptive"))
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy / HM (%)")
    axes[0].set_title("Final-epoch held-out evaluation")
    axes[0].legend(loc="upper left", ncol=3, fontsize=8)
    for idx, term in enumerate(("ins", "batch", "cls")):
        stats = [weights[[int(row["epoch"]) == epoch for row in traces], idx] for epoch in epochs]
        means, deviations = np.array([v.mean() for v in stats]), np.array([v.std() for v in stats])
        axes[1].plot(epochs, means, "o-", label=term, color=colors[idx])
        axes[1].fill_between(epochs, means - deviations, means + deviations, color=colors[idx], alpha=.13)
    axes[1].set_xticks(epochs)
    axes[1].set_xlabel("Training epoch")
    axes[1].set_ylabel("Adaptive lambda")
    axes[1].set_title("Mean weights; shading = batch std")
    axes[1].legend(ncol=3, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("DTD | REAL_EXPERIMENT | seed 1 | 5 student epochs", fontsize=12)
    figure_dir = root / "figures"
    figure_dir.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"REAL_DTD_exp1.{suffix}", dpi=200)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
