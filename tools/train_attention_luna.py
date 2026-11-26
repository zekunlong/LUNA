#!/usr/bin/env python3
"""Train the repository's original LUNA for the attention comparison.

Only teacher path resolution and final-epoch export are adapted. The original
LUNA model, losses, two loaders, optimizer and forward_backward run unchanged.
The dataset's held-out partitions are not used for checkpoint selection.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("eurosat", "fgvc_aircraft", "stanford_cars"))
    parser.add_argument("--root", type=Path, default=Path("E:/LUNA/data"))
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-clip", type=Path, default=Path.home()/".cache/clip/ViT-L-14.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    for path in (args.teacher_checkpoint, args.teacher_clip):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a fresh output directory; original trainer is not resumed implicitly")
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO/"Dassl.pytorch"))
    import torch
    import train
    from dassl.config import get_cfg_default
    from dassl.utils import set_random_seed, setup_logger
    from trainers.luna import LUNA
    torch.set_num_threads(4)
    cfg = get_cfg_default()
    train.extend_cfg(cfg)
    cfg.merge_from_file(str(REPO/"configs/datasets"/(args.dataset+".yaml")))
    cfg.merge_from_file(str(REPO/"configs/trainers/LUNA/vit_b16_c2_ep20_batch8_4+4ctx.yaml"))
    cfg.DATASET.ROOT = str(args.root.resolve())
    cfg.DATASET.NUM_SHOTS = 0
    cfg.DATASET.FEWSHOT_CACHE_ROOT = str(args.output_dir/"fewshot_cache")
    cfg.SEED = args.seed
    cfg.TRAINER.NAME = "LUNA"
    cfg.TRAINER.MODAL = "base2novel"
    cfg.TRAINER.LUNA.TEMPERATURE = 1.0
    cfg.TRAINER.LUNA.KD_WEIGHT = 1000.0
    cfg.TRAINER.LUNA.TEACHER_CLIP_PATH = str(args.teacher_clip.resolve())
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = str(args.output_dir.resolve())
    cfg.TEST.NO_TEST = True
    cfg.TEST.FINAL_MODEL = "last_step"
    cfg.freeze()
    set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    # An explicit checkpoint path replaces only the legacy hard-coded directory
    # convention; the original trainer still performs the actual teacher load.
    class ExplicitTeacherLUNA(LUNA):
        def _resolve_teacher_checkpoint(self):
            return str(args.teacher_checkpoint.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir/"training_manifest.json"
    record = {"status": "RUNNING", "method": "LUNA", "variant": "original_repository_legacy",
              "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": args.dataset,
              "seed": args.seed, "config_yaml": cfg.dump(), "checkpoint_selection": "final_epoch",
              "test_used_for_selection": False, "epochs": cfg.OPTIM.MAX_EPOCH,
              "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
              "teacher_checkpoint_sha256": digest(args.teacher_checkpoint),
              "teacher_clip_sha256": digest(args.teacher_clip),
              "trainer_source_sha256": digest(REPO/"trainers/luna.py"),
              "adaptations": ["explicit teacher path", "Windows worker count",
                              "fixed final epoch; held-out tests disabled during training"]}
    def save():
        manifest_path.write_text(json.dumps(record, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    save()
    began = time.monotonic()
    try:
        trainer = ExplicitTeacherLUNA(cfg)
        record["classnames"] = list(trainer.dm.dataset.classnames)
        record["training_samples"] = len(trainer.dm.dataset.train_x)
        save()
        trainer.train()
        checkpoint = args.output_dir/"VLPromptLearner"/("model.pth.tar-"+str(cfg.OPTIM.MAX_EPOCH))
        if not checkpoint.is_file():
            raise FileNotFoundError("Final trained checkpoint was not exported: "+str(checkpoint))
        # Compact genuine inference checkpoint, preserving class identity.
        trained_ctx = trainer.model.prompt_learner.ctx.detach().cpu()
        if not torch.isfinite(trained_ctx).all():
            raise ValueError("LUNA learned context contains nonfinite values")
        compact = args.output_dir/"attention_checkpoint.pth"
        torch.save({"state_dict": {"prompt_learner.ctx": trained_ctx}, "method": "LUNA",
                    "classnames": record["classnames"], "epoch": cfg.OPTIM.MAX_EPOCH,
                    "variant": record["variant"], "source_checkpoint_sha256": digest(checkpoint)}, compact)
        record.update(status="COMPLETED", checkpoint=str(checkpoint.resolve()),
                      checkpoint_sha256=digest(checkpoint), attention_checkpoint=str(compact.resolve()),
                      elapsed_seconds=time.monotonic()-began)
        save()
        print(json.dumps({k: record[k] for k in ("status", "dataset", "attention_checkpoint", "elapsed_seconds")}), flush=True)
    except Exception as exc:
        record.update(status="FAILED", error=repr(exc), elapsed_seconds=time.monotonic()-began)
        save()
        raise


if __name__ == "__main__":
    main()
