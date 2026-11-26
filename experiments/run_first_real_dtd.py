"""Run real DTD sanity and the fixed Adaptive -> StaticMean -> Equal protocol.

This runner never changes reliability equations or experimental coefficients.
Every subprocess is streamed to an append-only execution log. Existing outputs
are not overwritten; choose a fresh --run-id after a failed attempt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("E:/LUNA/results_reliability_analysis")
TEACHER_DIR = ROOT / "teacher_training/dtd_promptsrc_l14_seed1_final20"
MODES = (("AdaptiveLUNA", "reliability_adaptive"), ("StaticMean", "static_mean"), ("Equal", "equal"))


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    print("COMMAND", json.dumps(command), flush=True)
    with log.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"command": command, "cwd": str(REPO), "created_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        handle.flush()
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"Process exit {code}; see {log}")


def main():
    # Windows shells may inherit cp1252; tqdm emits Unicode block characters.
    # Keep the parent stream as UTF-8 just like the child and archived log.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "sanity", "adaptive", "static_mean", "equal", "controls"), default="all")
    parser.add_argument("--run-id", default="seed1_final20_20260906")
    parser.add_argument("--teacher-dir", type=Path, default=TEACHER_DIR)
    args = parser.parse_args()
    provenance = json.loads((args.teacher_dir / "teacher_provenance.json").read_text(encoding="utf-8"))
    if provenance["status"] != "COMPLETED_REAL_TEACHER_TRAINING":
        raise ValueError("The actual teacher must finish its declared training before distillation")
    teacher = args.teacher_dir / "VLPromptLearner/model.pth.tar-20"
    teacher_clip = Path.home() / ".cache/clip/ViT-L-14.pt"
    common = ["--root", "E:/LUNA/data", "--teacher-clip-path", str(teacher_clip),
              "--teacher-checkpoint", str(teacher), "--teacher-total-epochs", "20", "--seed", "1"]
    logs = ROOT / "logs" / "real_dtd_execution" / args.run_id
    if args.phase in ("all", "sanity"):
        command = [sys.executable, "-u", "experiments/run_journal_sanity.py", *common,
                   "--batches", "10", "--output-dir", str(ROOT)]
        run(command, logs / "sanity.stdout.txt")
        sanity = json.loads((ROOT / "sanity_result.json").read_text(encoding="utf-8"))
        if sanity["status"] != "PASS_REAL_DTD_SANITY":
            raise ValueError("Actual DTD sanity did not pass; inspect and fix before training")
        if args.phase == "sanity":
            return 0
    run_root = ROOT / "exp1_weighting/DTD" / args.run_id
    for variant, mode in MODES:
        phase_mode = "reliability_adaptive" if args.phase == "adaptive" else args.phase
        if args.phase == "controls" and mode == "reliability_adaptive":
            continue
        if args.phase not in ("all", "controls") and mode != phase_mode:
            continue
        destination = run_root / variant
        command = [sys.executable, "-u", "train.py", "--root", "E:/LUNA/data", "--seed", "1",
                   "--trainer", "LUNA", "--dataset-config-file", "configs/datasets/dtd.yaml",
                   "--config-file", "configs/trainers/LUNA/vit_b16_c2_ep20_batch8_4+4ctx.yaml",
                   "--output-dir", str(destination), "--distill-weighting", mode,
                   "--selection-protocol", "final_epoch", "--teacher-checkpoint", str(teacher),
                   "--teacher-total-epochs", "20", "--teacher-quality", "strong",
                   "--analysis-output-dir", str(ROOT), "--sanity-report", str(ROOT / "sanity_result.json")]
        if mode == "static_mean":
            command += ["--static-weights", str(run_root / "AdaptiveLUNA/training_mean_weights.json")]
        command += ["DATASET.NUM_SHOTS", "0", "TRAINER.MODAL", "base2novel",
                    "TRAINER.LUNA.TEMPERATURE", "1.0", "TRAINER.LUNA.KD_WEIGHT", "1000.0",
                    "TRAINER.LUNA.TEACHER_CLIP_PATH", str(teacher_clip), "TEST.SPLIT", "val"]
        run(command, logs / f"{variant}.stdout.txt")
        record = json.loads((destination / "journal_manifest.json").read_text(encoding="utf-8"))
        if record["status"] != "COMPLETED_TRAINING":
            raise ValueError(f"Incomplete real {variant} run")
        print("REAL_RESULT", json.dumps({"ResultSource": "REAL_EXPERIMENT", "Variant": variant,
                                          **{key: record[key] for key in ("Base", "Novel", "HM")}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
