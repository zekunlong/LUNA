"""Run or reuse a fixed checkpoint-only evaluation for the Exp2/3/4 cohort."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.run_journal_exp234_training import ROOT, COHORT, DATASETS, STAGES, MODES, destination


def evaluate_condition(dataset, quality, variant):
    from experiments.journal_runtime import sha256_file
    from experiments.run_first_real_dtd import run
    source_dir = destination(dataset, quality, variant)
    if quality == "Strong":
        output = ROOT / "exp4_relation_preservation" / dataset / COHORT / variant / "evaluation"
    else:
        output = source_dir / "evaluation"
    manifest_path = output / "evaluation_manifest.json"
    if manifest_path.is_file():
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        if record["status"] != "COMPLETED_REAL_EVALUATION":
            raise ValueError(f"Preserved incomplete evaluation requires inspection: {output}")
        for path, digest in ((record["source_run_manifest"], record["source_run_manifest_sha256"]),
                             (record["student_checkpoint"], record["checkpoint_sha256_after"]),
                             (record["helper_path"], record["helper_sha256"])):
            if sha256_file(path) != digest:
                raise ValueError(f"Reused evaluation source changed: {path}")
        print("REUSED_REAL_EVALUATION " + str(manifest_path), flush=True)
        return record
    command = [sys.executable, "-X", "utf8", "-u", "experiments/evaluate_journal_reliability_relations.py",
               "--run-dir", str(source_dir), "--output-dir", str(output), "--evaluation-seed", "1"]
    if quality == "Strong" and variant == "AdaptiveLUNA":
        command += ["--exp2-output-dir", str(ROOT / "exp2_reliability_validation" / dataset / COHORT)]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    log = ROOT / "logs/exp234_execution" / dataset / f"evaluate_{quality}_{variant}_{stamp}.stdout.txt"
    run(command, log)
    result = json.loads(manifest_path.read_text(encoding="utf-8"))
    if result["status"] != "COMPLETED_REAL_EVALUATION":
        raise ValueError("Checkpoint-only evaluation did not complete")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--quality", choices=STAGES, required=True)
    parser.add_argument("--variant", choices=MODES, required=True)
    args = parser.parse_args()
    evaluate_condition(args.dataset, args.quality, args.variant)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    main()
