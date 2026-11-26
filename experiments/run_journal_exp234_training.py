"""Run a predeclared Exp3/Exp4 student condition using unchanged journal code."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ROOT = Path("E:/LUNA/results_reliability_analysis")
DATA_ROOT = Path("E:/LUNA/data")
DATASETS = {"DTD": "dtd", "FGVCAircraft": "fgvc_aircraft", "EuroSAT": "eurosat"}
STAGES = {"Weak": 5, "Medium": 12, "Strong": 20}
MODES = {"Equal": "equal", "AdaptiveLUNA": "reliability_adaptive", "InstanceOnly": "instance_only"}
COHORT = "seed1_final20_20260906"


def teacher_directory(dataset):
    return ROOT / "teacher_training" / f"{DATASETS[dataset]}_promptsrc_l14_seed1_final20"


def destination(dataset, quality, variant):
    if variant == "InstanceOnly":
        return ROOT / "exp4_relation_preservation" / dataset / COHORT / variant
    if quality == "Strong":
        return ROOT / "exp1_weighting" / dataset / COHORT / variant
    return ROOT / "exp3_teacher_quality" / dataset / COHORT / quality / variant


def train_condition(dataset, quality, variant):
    from experiments.journal_runtime import require_sanity_pass, sha256_file, source_hashes
    from experiments.run_first_real_dtd import run
    require_sanity_pass(ROOT / "sanity_result.json")
    if variant == "InstanceOnly" and quality != "Strong":
        raise ValueError("Exp4 InstanceOnly uses the same final teacher as Exp1")
    teacher_dir = teacher_directory(dataset)
    receipt = json.loads((teacher_dir / "teacher_provenance.json").read_text(encoding="utf-8"))
    if receipt["status"] != "COMPLETED_REAL_TEACHER_TRAINING":
        raise ValueError("Teacher's predeclared training must finish first")
    checkpoint = teacher_dir / "VLPromptLearner" / f"model.pth.tar-{STAGES[quality]}"
    if sha256_file(checkpoint) != receipt["checkpoint_sha256"][str(checkpoint)]:
        raise ValueError("Teacher checkpoint differs from actual completed training receipt")
    output = destination(dataset, quality, variant)
    manifest = output / "journal_manifest.json"
    if manifest.exists():
        measured = json.loads(manifest.read_text(encoding="utf-8"))
        required = {"status": "COMPLETED_TRAINING", "dataset": dataset, "seed": 1,
                    "variant": MODES[variant], "teacher_quality": quality.lower(),
                    "teacher_checkpoint_sha256": sha256_file(checkpoint), "source_hashes": source_hashes(),
                    "selection_protocol": "final_epoch", "test_used_for_selection": False}
        if any(measured.get(key) != value for key, value in required.items()):
            raise ValueError(f"Existing condition is incomplete or incompatible; preserve and inspect: {output}")
        if measured.get("epoch") != 5 or not Path(measured["student_checkpoint"]).is_file():
            raise ValueError("Reusable result lacks the final student epoch checkpoint")
        print("REUSED_REAL_RESULT " + json.dumps({key: measured[key] for key in ("dataset", "variant", "Base", "Novel", "HM")}), flush=True)
        return measured
    clip = Path.home() / ".cache/clip/ViT-L-14.pt"
    root_arg = DATA_ROOT.as_posix()
    paired_manifest = destination(dataset, quality, "AdaptiveLUNA") / "journal_manifest.json"
    if paired_manifest.is_file():
        root_arg = json.loads(paired_manifest.read_text(encoding="utf-8"))["configuration"]["DATASET"]["ROOT"]
        if Path(root_arg).resolve() != DATA_ROOT.resolve():
            raise ValueError("Paired condition uses a different dataset root")
    command = [sys.executable, "-X", "utf8", "-u", "train.py", "--root", root_arg, "--seed", "1",
               "--trainer", "LUNA", "--dataset-config-file", f"configs/datasets/{DATASETS[dataset]}.yaml",
               "--config-file", "configs/trainers/LUNA/vit_b16_c2_ep20_batch8_4+4ctx.yaml",
               "--output-dir", str(output), "--distill-weighting", MODES[variant],
               "--selection-protocol", "final_epoch", "--teacher-checkpoint", str(checkpoint),
               "--teacher-total-epochs", "20", "--teacher-quality", quality.lower(),
               "--analysis-output-dir", str(ROOT), "--sanity-report", str(ROOT / "sanity_result.json"),
               "DATASET.NUM_SHOTS", "0", "TRAINER.MODAL", "base2novel", "TRAINER.LUNA.TEMPERATURE", "1.0",
               "TRAINER.LUNA.KD_WEIGHT", "1000.0", "TRAINER.LUNA.TEACHER_CLIP_PATH", str(clip), "TEST.SPLIT", "val"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    log = ROOT / "logs/exp234_execution" / dataset / f"{quality}_{variant}_{stamp}.stdout.txt"
    run(command, log)
    measured = json.loads(manifest.read_text(encoding="utf-8"))
    if measured["status"] != "COMPLETED_TRAINING":
        raise RuntimeError("Condition did not finish its predeclared training/evaluation")
    print("REAL_RESULT " + json.dumps({"ResultSource": "REAL_EXPERIMENT", "quality": quality,
          **{key: measured[key] for key in ("dataset", "variant", "Base", "Novel", "HM")}}), flush=True)
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--quality", choices=STAGES, required=True)
    parser.add_argument("--variant", choices=MODES, required=True)
    args = parser.parse_args()
    train_condition(args.dataset, args.quality, args.variant)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    main()
