#!/usr/bin/env python3
"""Write the full three-dataset training/evaluation/figure execution plan."""
import json
from pathlib import Path

ROOT = Path("E:/LUNA/attention_comparison")
REPO = Path(__file__).resolve().parents[1]
PYTHON = "E:/LUNA/.venv_journal/Scripts/python.exe"
stages = []


def add(name, command, outputs, completion=None, **kwargs):
    row = {"id": name, "command": [str(x) for x in command], "outputs": [str(x) for x in outputs], **kwargs}
    if completion:
        row["completion_json"] = {"path": str(completion[0]), "value": completion[1]}
    stages.append(row)


def main():
    for dataset in ("fgvc_aircraft", "eurosat", "stanford_cars"):
        train = ROOT/"training"/dataset
        manifest = ROOT/"datasets"/(dataset+".json")
        if dataset == "stanford_cars":
            add("wait_cars_data", [], [manifest], wait_only=True)
            teacher_dir = ROOT/"training/stanford_cars/PromptSRC_teacher"
            add("cars_teacher", [PYTHON, "-u", "E:/LUNA/external/PromptKD/run_stanford_cars_promptsrc_final.py",
                                  "--output-dir", teacher_dir, "--num-workers", "0"],
                [teacher_dir/"VLPromptLearner/model.pth.tar-20"],
                (teacher_dir/"teacher_provenance.json", "COMPLETED_REAL_TEACHER_TRAINING"),
                cwd="E:/LUNA/external/PromptKD")
        else:
            teacher_dir = Path("E:/LUNA/results_reliability_analysis/teacher_training")/(dataset+"_promptsrc_l14_seed1_final20")
        luna = train/"LUNA"
        command = [PYTHON, "-u", "tools/train_attention_luna.py", dataset,
                   "--teacher-checkpoint", teacher_dir/"VLPromptLearner/model.pth.tar-20",
                   "--output-dir", luna, "--num-workers", "4"]
        add(dataset+"_LUNA", command, [luna/"attention_checkpoint.pth"],
            (luna/"training_manifest.json", "COMPLETED"))
        for method in ("CoOp", "KgCoOp", "DeKg"):
            destination = train/method
            add(dataset+"_"+method, [PYTHON, "-u", "tools/train_attention_baselines.py", "--manifest", manifest,
                                     "--method", method, "--clip-path", Path.home()/".cache/clip/ViT-B-16.pt",
                                     "--output-dir", destination, "--cache-images", "--skip-eval"],
                [destination/"attention_checkpoint.pth"], (destination/"status.json", "COMPLETED_REAL_TRAINING"))
        if dataset == "stanford_cars":
            add("cars_match_originals", [PYTHON, "tools/match_attention_inputs.py", "--panel", "E:/LUNA/attention/attention_cars.png",
                                         "--manifest", manifest, "--classes", "2012 Ford F-450 Super Duty Crew Cab",
                                         "2012 Suzuki Kizashi Sedan", "2012 Mercedes-Benz C-Class Sedan",
                                         "--output", ROOT/"configs/stanford_cars_original_inputs.json"],
                [ROOT/"configs/stanford_cars_original_inputs.json"])
        add(dataset+"_configs", [PYTHON, "tools/make_attention_configs.py", dataset],
            [ROOT/"configs"/(dataset+"_models.json"), ROOT/"configs"/(dataset+"_original.json")])
        evaluation = ROOT/"evaluation"/dataset
        add(dataset+"_evaluate", [PYTHON, "-u", "tools/evaluate_attention_comparison.py", "--dataset-manifest", manifest,
                                 "--models-config", ROOT/"configs"/(dataset+"_models.json"), "--output-dir", evaluation],
            [evaluation/"evaluation_arrays.npz", evaluation/"evaluation_summary.json"],
            (evaluation/"evaluation_summary.json", "COMPLETE_HELDOUT_EVALUATION"))
        for group in ("original", "selected_success", "random_control", "luna_failures"):
            config = (ROOT/"configs"/(dataset+"_original.json") if group=="original"
                      else evaluation/"selections"/(group+".json"))
            for attribution in ("rollout", "gradcam"):
                destination = ROOT/"figures"/dataset/group/attribution
                name = dataset+"_"+group+"_"+attribution
                add(name, [PYTHON, "-u", "tools/visualize_attention_comparison.py", "infer", "--config", config,
                           "--output-dir", destination, "--attribution", attribution],
                    [destination/"attention_bundle.npz", destination/"attention_comparison.png", destination/"attention_comparison.pdf"],
                    skip_if_empty_samples=str(config))
                if dataset != "eurosat":
                    add(name+"_localization", [PYTHON, "tools/attention_localization.py", "--bundle", destination/"attention_bundle.npz",
                                               "--output-dir", destination], [destination/"localization_summary.json"],
                        skip_if_empty_samples=str(config))
    plan = {"cwd": str(REPO), "log_dir": str(ROOT/"logs/queue"), "stages": stages}
    path = ROOT/"queue_plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Prepared {len(stages)} stages: {path}")


if __name__ == "__main__":
    main()
