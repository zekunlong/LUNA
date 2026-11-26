"""Run the requested real DTD ten-batch gate before any journal experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.journal_preflight import inspect_assets
from experiments.journal_runtime import (
    DATASETS, DEFAULT_ANALYSIS_ROOT, JournalRecorder, METRICS, WEIGHTS,
    require_sanity_pass, source_hashes, write_json,
)


def assess_sanity(records):
    """Assess real trainer records; synthetic tests must never publish a pass."""
    values = {key: [float(row[key]) for row in records] for key in METRICS}
    finite = bool(records) and all(math.isfinite(value) for column in values.values() for value in column)
    deviations = {key: statistics.pstdev(values[key]) if values[key] and finite else None for key in WEIGHTS}
    independent_weighting = bool(records)
    for row in records:
        expected = [row[key] * row[weight] for key, weight in zip(("L_ins", "L_batch", "L_cls"), WEIGHTS)]
        independent_weighting &= math.isclose(sum(expected), row["L_adapt"], rel_tol=2e-6, abs_tol=1e-6)
    probe_rows = [row for row in records if "alternate_weight_diagnostic" in row]
    relative_change = False
    for row in probe_rows:
        probe = row["alternate_weight_diagnostic"]
        if probe["used_for_optimization"]:
            raise ValueError("Diagnostic alternate weights must never be used for optimization")
        active = [key for key in ("L_ins", "L_batch", "L_cls") if abs(row["weighted_terms"][key]) > 1e-12]
        ratios = [probe["weighted_terms"][key] / row["weighted_terms"][key] for key in active]
        relative_change |= len(ratios) >= 2 and max(ratios) - min(ratios) > 1e-6
    # Unequal totals alone do not exclude a scalar-rescaled legacy objective.
    # Compare the recorded contributions of the three independent components:
    # proportional vectors would still be explained by one scalar multiplier.
    legacy_different = False
    for row in records:
        legacy = row.get("legacy_components", {})
        adaptive = row.get("weighted_terms", {})
        names = ("L_ins", "L_batch", "L_cls")
        if not all(key in legacy and key in adaptive for key in names):
            continue
        original = [legacy[key] for key in names]
        current = [adaptive[key] for key in names]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (*original, *current)):
            continue
        original_norm = math.sqrt(sum(value * value for value in original))
        current_norm = math.sqrt(sum(value * value for value in current))
        if min(original_norm, current_norm) <= 1e-12:
            continue
        distance = math.sqrt(sum((old / original_norm - new / current_norm) ** 2
                                 for old, new in zip(original, current)))
        legacy_different |= distance > 1e-6
    identities = {(row.get("epoch"), row.get("batch_index")) for row in records}
    checks = {
        "ten_real_batches": len(records) >= 10 and len(identities) == len(records)
        and all(row["run_kind"] == "sanity_training_batches" for row in records),
        "no_nan_inf": finite and all(row.get("legacy_numeric_status") == "finite" for row in records),
        "reliability_in_unit_interval": finite and all(0 <= value <= 1 for key in ("r_ins", "r_batch", "r_cls") for value in values[key]),
        "weights_in_unit_interval": finite and all(0 <= value <= 1 for key in WEIGHTS for value in values[key]),
        "weights_sum_to_one": finite and all(abs(sum(row[key] for key in WEIGHTS) - 1) < 1e-5 for row in records),
        "weights_change_across_batches": finite and any(value is not None and value > 0 for value in deviations.values()),
        "student_prompt_gradients": bool(records) and all(row["student_prompt_grad_present"] and row["student_prompt_grad_finite"] for row in records)
        and any(row["student_prompt_grad_nonzero"] for row in records),
        "teacher_frozen_without_gradients": bool(records) and all(row["teacher_requires_grad_false"] and row["teacher_grad_none"] and row["teacher_weights_detached"] for row in records),
        "independent_weighted_scalars": independent_weighting,
        "relative_contributions_change": relative_change,
        "legacy_is_mathematically_different": legacy_different,
        "same_sample_two_views": bool(records) and all(row["paired_same_sample"] and row["paired_view_count"] == 2 for row in records),
        "no_test_checkpoint_selection": bool(records) and all(row["selection_protocol"] == "final_epoch" and row["Base"] is None and row["Novel"] is None for row in records),
    }
    return checks, deviations, {key: statistics.mean(column) if column and finite else None for key, column in values.items()}


def write_sanity_report(root, result):
    root = Path(root)
    write_json(root / "sanity_result.json", result)
    lines = ["# Journal LUNA DTD sanity check", "", f"状态：**{result['status']}**。", "",
             "真实 DTD 十批次检查是完整实验的前置条件。合成单元测试不替代此检查。", "",
             "## 已实现的公式与模式", "",
             "`H(p)=-sum(p*log(p+1e-8))`；`r_ins=clamp(1-(mean H(Pt)+mean H(Pt_aug))/(2 log C),0,1)`。",
             "`r_batch`、`r_cls` 分别为两视图 `P Pᵀ`、`Pᵀ P` 的展平余弦：`dot/(norm_product+1e-8)`，裁剪到 [0,1]。",
             "`lambda_k=(r_k+1e-8)/(r_ins+r_batch+r_cls+3e-8)`；三个独立标量 loss 分别乘各自 lambda 后求和。",
             "可靠性概率使用原基础温度 1；三项损失保留温度 [1,3,5,2,6] 及原 KL 的 T²/Gram 平方和定义。",
             "journal 统一使用同图的原 weak/strong 增强与两视图损失；legacy 保留原行为。", "",
             "## 修改和新增文件", "",
             "- `train.py`：显式新模式配置和入口；默认 legacy。",
             "- `trainers/luna.py`：只提取教师路径函数作为覆盖接口，legacy 数值路径不变。",
             "- `trainers/luna_journal.py`：冻结教师、配对视图、独立加权、最终 epoch 生命周期与真实 sanity。",
             "- `experiments/journal_losses.py`、`journal_runtime.py`、`journal_preflight.py`、`run_journal_sanity.py`：数学、日志、预检与运行控制。",
             "- `experiments/tests/`：公式、legacy 兼容、预检和训练结构测试。", ""]
    software_path = root / "logs" / "software_tests.json"
    if software_path.is_file():
        software = json.loads(software_path.read_text(encoding="utf-8"))
        lines.extend(["## 软件验证（不等于真实 DTD sanity）", "",
                      f"软件测试状态：{software.get('status')}；运行 {software.get('tests_run')} 项。",
                      f"解释器：`{software.get('interpreter')}`。详细记录见 `logs/software_tests.json` 和 `logs/software_tests.txt`。", ""])
    if result.get("blockers"):
        lines.extend(["## 当前未满足的条件", ""])
        lines.extend("- " + message for message in result["blockers"])
    if result.get("checks"):
        lines.extend(["", "## 检查结果", "", "| Check | Result |", "|---|---|"])
        lines.extend(f"| {key} | {'PASS' if passed else 'FAIL'} |" for key, passed in result["checks"].items())
        lines.extend(["", "## 十批次 lambda 统计", "", "| Weight | Mean | Population std |", "|---|---:|---:|"])
        for key in WEIGHTS:
            lines.append(f"| {key} | {result['means'].get(key)} | {result['lambda_std'].get(key)} |")
    else:
        count = result.get("batch_count", 0)
        if count:
            lines.extend(["", f"已记录真实 DTD batch 数：{count}；运行中断，完整十批检查为 **NOT_COMPLETED**。",
                          "已完成 batch 的数值与梯度状态保留在下方和 sanity_result.json；它们不能替代完整通过报告。"])
        else:
            lines.extend(["", "真实 DTD batch 数：0；十批次均值/标准差、prompt 梯度、teacher 梯度和真实 legacy/adaptive 对比均为 **NOT_RUN**。"])
    if result.get("records"):
        lines.extend(["", "## 一个真实 batch", "", "```json", json.dumps(result["records"][0], ensure_ascii=False, indent=2), "```",
                      "", "所有 batch 的梯度状态、相对贡献探针、legacy 数值和原始可靠性见 sanity_result.json 与 run_dir 日志。",
                      "legacy 对比保留原 weak-only mask 公式；journal 同时修正双视图配对/汇总。因此该对比用于验证实现差异，不用于隔离 λ 的科学归因。"])
    lines.extend(["", "## Checkpoint 与执行顺序", "",
                  "sanity 不调用测试集评估、不保存学生实验 checkpoint；journal 完整训练只能显式保存/加载最终 epoch。",
                  "当前命令：`" + result.get("command", "not recorded") + "`。",
                  "完整 Exp1–4 未在此 sanity 命令中启动。只有 PASS_REAL_DTD_SANITY 且代码哈希一致的报告可通过完整训练入口。"])
    (root / "SANITY_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _blocked(root, preflight, command):
    result = {"status": "BLOCKED_BEFORE_REAL_DTD_SANITY", "dataset": "DTD", "batch_count": 0,
              "checks": {}, "records": [], "blockers": preflight["blockers"], "preflight": preflight,
              "source_hashes": source_hashes(), "command": command,
              "command_argv": list(sys.argv),
              "created_utc": datetime.now(timezone.utc).isoformat(), "full_experiments_started": False}
    write_sanity_report(root, result)
    print("Real DTD sanity check is BLOCKED; no training was started.")
    for reason in preflight["blockers"]:
        print("-", reason)
    return 2


def run_journal_mode(args, cfg):
    """Entry from train.py; full training is explicitly gated on real sanity."""
    root = Path(cfg.JOURNAL.OUTPUT_ROOT)
    sanity = cfg.JOURNAL.SANITY_BATCHES > 0
    if getattr(args, "eval_only", False) or getattr(args, "no_train", False):
        raise ValueError("Journal mode uses its explicit sanity/final-epoch lifecycle; legacy eval flags are not supported here")
    if sanity and (cfg.DATASET.NAME != "DescribableTextures" or cfg.JOURNAL.MODE != "reliability_adaptive"
                   or cfg.JOURNAL.TEACHER_QUALITY != "strong"):
        raise ValueError("The required sanity gate is DTD with reliability_adaptive weighting and the strong teacher")
    if not sanity:
        require_sanity_pass(cfg.JOURNAL.SANITY_REPORT)
    preflight = inspect_assets(
        DATASETS[cfg.DATASET.NAME], cfg.DATASET.ROOT, cfg.TRAINER.LUNA.TEACHER_CLIP_PATH,
        cfg.JOURNAL.TEACHER_CHECKPOINT, cfg.JOURNAL.TEACHER_TOTAL_EPOCHS,
        teacher_quality=cfg.JOURNAL.TEACHER_QUALITY,
    )
    command = " ".join(sys.argv)
    if preflight["status"] != "READY":
        if sanity:
            return _blocked(root, preflight, command)
        raise ValueError("Journal preflight failed: " + "; ".join(preflight["blockers"]))

    import torch
    from dassl.utils import set_random_seed, setup_logger
    from trainers.luna_journal import LUNAJournal
    set_random_seed(cfg.SEED)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    setup_logger(cfg.OUTPUT_DIR)
    trainer = None
    recorder = None
    records = []
    try:
        trainer = LUNAJournal(cfg)
        recorder = JournalRecorder(trainer, cfg)
        if not sanity:
            trainer.train()
            metrics = trainer.journal_final_result
            if not metrics or not all(math.isfinite(metrics[key]) for key in ("Base", "Novel", "HM")):
                raise ValueError("The final student checkpoint did not produce finite Base/Novel/HM metrics")
            recorder.finish("COMPLETED_TRAINING", metrics=metrics)
            return 0
        records = trainer.run_sanity(max_batches=cfg.JOURNAL.SANITY_BATCHES)
        checks, deviations, means = assess_sanity(records)
        passed = all(checks.values())
        result = {
            "status": "PASS_REAL_DTD_SANITY" if passed else "FAIL_REAL_DTD_SANITY",
            "dataset": "DTD", "batch_count": len(records), "checks": checks,
            "records": records, "lambda_std": deviations, "means": means,
            "source_hashes": source_hashes(), "preflight": preflight,
            "configuration": recorder.manifest["configuration"],
            "initialization_sha256": recorder.manifest["initialization_sha256"],
            "run_dir": cfg.OUTPUT_DIR, "command": command,
            "command_argv": list(sys.argv),
            "created_utc": datetime.now(timezone.utc).isoformat(), "full_experiments_started": False,
        }
        recorder.finish("COMPLETED_SANITY" if passed else "FAILED_SANITY")
        write_sanity_report(root, result)
        for record in records:
            print("DTD batch", record["batch_index"], json.dumps({key: record[key] for key in METRICS}))
        print("Lambda population standard deviations:", json.dumps(deviations))
        print("Sanity checks:", json.dumps(checks, sort_keys=True))
        return 0 if passed else 1
    except Exception as exc:
        if recorder is not None:
            recorder.finish("FAILED_SANITY" if sanity else "FAILED_TRAINING", error=f"{type(exc).__name__}: {exc}")
        if sanity:
            result = {"status": "FAIL_REAL_DTD_SANITY", "dataset": "DTD",
                      "batch_count": len(getattr(trainer, "journal_records", [])), "checks": {},
                      "records": getattr(trainer, "journal_records", []),
                      "blockers": [f"{type(exc).__name__}: {exc}"], "preflight": preflight,
                      "source_hashes": source_hashes(), "command": command, "full_experiments_started": False}
            result["command_argv"] = list(sys.argv)
            write_sanity_report(root, result)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("DTD", "dtd"), default="DTD")
    parser.add_argument("--root", default=None)
    parser.add_argument("--teacher-clip-path", default=None)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--teacher-total-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1, help="Existing queue seed; keep identical for all controls")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_ROOT, help="Analysis root, including SANITY_CHECK.md")
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.batches < 10:
        parser.error("at least 10 actual DTD batches are required")
    if arguments.seed < 0:
        parser.error("a fixed nonnegative seed is required")
    preflight = inspect_assets(arguments.dataset, arguments.root, arguments.teacher_clip_path,
                               arguments.teacher_checkpoint, arguments.teacher_total_epochs)
    root = arguments.output_dir.expanduser().resolve()
    command = " ".join(sys.argv)
    if preflight["status"] != "READY":
        return _blocked(root, preflight, command)
    if arguments.preflight_only:
        write_json(root / "logs" / "journal_asset_preflight.json", preflight)
        print("Asset/environment preflight READY; real DTD sanity has not run.")
        return 0
    assets = preflight["assets"]
    run_dir = root / "sanity_dtd" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    train_args = argparse.Namespace(
        root=assets["data_root"]["path"], output_dir=str(run_dir), resume="", seed=arguments.seed,
        source_domains=None, target_domains=None, transforms=None, trainer="LUNA", backbone=None, head=None,
        config_file=str(REPO / "configs/trainers/LUNA/vit_b16_c2_ep20_batch8_4+4ctx.yaml"),
        dataset_config_file=str(REPO / "configs/datasets/dtd.yaml"), eval_only=False, model_dir="",
        load_epoch=None, no_train=False, distill_weighting="reliability_adaptive", selection_protocol="final_epoch",
        teacher_checkpoint=assets["teacher_checkpoint"]["path"],
        teacher_total_epochs=assets["teacher_total_epochs"]["value"], teacher_quality="strong",
        static_weights=None, analysis_output_dir=str(root), sanity_batches=arguments.batches, sanity_report=None,
        opts=["DATASET.NUM_SHOTS", "0", "TRAINER.MODAL", "base2novel", "TRAINER.LUNA.TEMPERATURE", "1.0",
              "TRAINER.LUNA.KD_WEIGHT", "1000.0", "TRAINER.LUNA.TEACHER_CLIP_PATH", assets["teacher_clip"]["path"],
              "TEST.SPLIT", "val"],
    )
    from train import setup_cfg
    cfg = setup_cfg(train_args)
    return run_journal_mode(train_args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
