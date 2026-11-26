"""Predeclared, checkpoint-only held-out evaluation for journal Exp2/3/4.

No training method is called. The original ordered test split is evaluated in
full batches of eight; discarded tail identities are explicit. Metrics are
computed in float32 from native model logits at temperature one.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BATCH_SIZE = 8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def object_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def stable_transform_repr(transform):
    # Dassl RandAugment inherits object.__repr__, whose address varies by
    # process. Its fixed N=2,M=10 semantics are declared separately above.
    return re.sub(r" at 0x[0-9a-fA-F]+(?=>)", "", repr(transform))


def write_csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relation_distances(log_teacher, log_student):
    """KL per image and normalized Frobenius of the unmodified Gram matrices."""
    import torch
    from experiments.journal_losses import EPS
    teacher, student = log_teacher.exp(), log_student.exp()
    require(teacher.shape == student.shape and teacher.ndim == 2, "Probability shapes differ")
    instance = (teacher * (log_teacher - log_student)).sum(dim=1).mean()
    batch_t, batch_s = teacher @ teacher.T, student @ student.T
    class_t, class_s = teacher.T @ teacher, student.T @ student
    batch = torch.linalg.vector_norm(batch_t - batch_s) / (torch.linalg.vector_norm(batch_t) + EPS)
    classes = torch.linalg.vector_norm(class_t - class_s) / (torch.linalg.vector_norm(class_t) + EPS)
    return {"D_ins": float(instance), "D_batch": float(batch), "D_cls": float(classes)}


def verify_distance_formulas():
    """Software-only numerical check; never included among real observations."""
    import numpy as np
    import torch
    teacher = torch.tensor([[.6, .3, .1], [.1, .2, .7]], dtype=torch.float64)
    student = torch.tensor([[.4, .4, .2], [.2, .3, .5]], dtype=torch.float64)
    zero = relation_distances(teacher.log(), teacher.log())
    require(all(value == 0 for value in zero.values()), "Equal distributions must have zero distances")
    actual = relation_distances(teacher.log(), student.log())
    pt, ps = teacher.numpy(), student.numpy()
    expected = {
        "D_ins": float(np.mean(np.sum(pt * np.log(pt / ps), axis=1))),
        "D_batch": float(np.linalg.norm(pt @ pt.T - ps @ ps.T) / (np.linalg.norm(pt @ pt.T) + 1e-8)),
        "D_cls": float(np.linalg.norm(pt.T @ pt - ps.T @ ps) / (np.linalg.norm(pt.T @ pt) + 1e-8)),
    }
    require(all(math.isclose(actual[k], expected[k], rel_tol=1e-12, abs_tol=1e-14) for k in expected),
            "Distance formulas differ from independent NumPy reference")
    return {"status": "PASSED_SOFTWARE_VALIDATION", "real_experiment": False,
            "checks": ["identical distributions give exactly zero", "perturbed distributions match independent NumPy reference"]}


def original_test_records(config):
    root = Path(config["DATASET"]["ROOT"]).resolve()
    name = config["DATASET"]["NAME"]
    sources = []
    records = []
    if name in ("DescribableTextures", "EuroSAT"):
        folder, images, split = (("dtd", "images", "split_zhou_DescribableTextures.json")
                                 if name == "DescribableTextures" else ("eurosat", "2750", "split_zhou_EuroSAT.json"))
        source = root / folder / split
        sources.append(source)
        data = json.loads(source.read_text(encoding="utf-8"))
        for index, (relative, label, classname) in enumerate(data["test"]):
            path = (root / folder / images / relative).resolve()
            require(path.is_relative_to(root / folder / images), "Split path escapes image directory")
            records.append({"sample_index": index, "sample_id": f"{folder}/{images}/{Path(relative).as_posix()}",
                            "impath": str(path), "ground_truth": int(label), "classname": classname})
    elif name == "FGVCAircraft":
        folder = root / "fgvc_aircraft"
        sources = [folder / "variants.txt", folder / "images_variant_test.txt"]
        classnames = sources[0].read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(sources[1].read_text(encoding="utf-8").splitlines()):
            image_id, classname = line.split(" ", 1)
            records.append({"sample_index": index, "sample_id": f"fgvc_aircraft/images/{image_id}.jpg",
                            "impath": str(folder / "images" / f"{image_id}.jpg"),
                            "ground_truth": classnames.index(classname), "classname": classname})
    else:
        raise ValueError("Unsupported original dataset")
    require(records and len({r["sample_id"] for r in records}) == len(records), "Empty or duplicated test records")
    require(all(Path(r["impath"]).is_file() for r in records), "An original test image is missing")
    return records, sources


def analysis_tables(batches):
    import numpy as np
    from scipy.stats import spearmanr
    correlations, quintiles = [], []
    accuracy = np.asarray([row["teacher_batch_accuracy_percent"] for row in batches])
    for key in ("r_ins", "r_batch", "r_cls"):
        values = np.asarray([row[key] for row in batches])
        reason = "constant_reliability" if np.ptp(values) == 0 else "constant_accuracy" if np.ptp(accuracy) == 0 else ""
        if reason:
            rho, p_value = None, None
        else:
            result = spearmanr(values, accuracy, alternative="two-sided")
            rho, p_value = float(result.statistic), float(result.pvalue)
            require(math.isfinite(rho) and math.isfinite(p_value), "Unexpected nonfinite Spearman result")
        correlations.append({"reliability": key, "rho": rho, "p_value": p_value,
                             "N_batches": len(batches), "status": "undefined" if reason else "defined", "reason": reason})
        order = np.argsort(values, kind="stable")
        for quintile, indices in enumerate(np.array_split(order, 5), start=1):
            require(len(indices) > 0, "Too few full batches for five bins")
            selected = [batches[int(index)] for index in indices]
            count = sum(row["n_samples"] for row in selected)
            quintiles.append({"reliability": key, "quintile": quintile, "N_batches": len(selected),
                              "N_samples": count, "reliability_min": float(values[indices].min()),
                              "reliability_max": float(values[indices].max()), "reliability_mean": float(values[indices].mean()),
                              "teacher_accuracy_percent": 100 * sum(row["teacher_correct_count"] for row in selected) / count,
                              "batch_indices": json.dumps([int(index) for index in indices])})
    return correlations, quintiles


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--exp2-output-dir", type=Path)
    parser.add_argument("--evaluation-seed", type=int, default=1)
    parser.add_argument("--verify-formulas-only", action="store_true")
    options = parser.parse_args()
    if options.verify_formulas_only:
        print(json.dumps(verify_distance_formulas()))
        return 0
    require(options.run_dir and options.output_dir, "--run-dir and --output-dir are required")
    require(options.evaluation_seed >= 0, "Evaluation seed must be nonnegative")
    from experiments.journal_runtime import (dataset_fingerprint, model_state_hash, protocol_fingerprint,
                                             sha256_file, source_hashes, write_json)
    run_dir, output = options.run_dir.resolve(), options.output_dir.resolve()
    source_path = run_dir / "journal_manifest.json"
    original = json.loads(source_path.read_text(encoding="utf-8"))
    config = original["configuration"]
    require(original["status"] == "COMPLETED_TRAINING", "A completed real training manifest is required")
    require(original["source_hashes"] == source_hashes(), "Core source hashes differ from the trained run")
    require(original["protocol_fingerprint"] == protocol_fingerprint(config), "Source configuration fingerprint mismatch")
    require(original["selection_protocol"] == "final_epoch" and original["test_used_for_selection"] is False,
            "Final-epoch checkpoint without test selection is required")
    require(config["JOURNAL"]["SANITY_BATCHES"] == 0 and config["DATASET"]["NUM_SHOTS"] == 0,
            "Full real unlabeled training is required")
    require(config["INPUT"]["SIZE"] == [224, 224] and config["INPUT"]["INTERPOLATION"] == "bicubic"
            and config["INPUT"]["TRANSFORMS"] == ["random_resized_crop", "random_flip", "normalize"]
            and config["INPUT"]["RRCROP_SCALE"] == [0.08, 1.0], "Input transforms differ from predeclared paired evaluation")
    require(Path(config["OUTPUT_DIR"]).resolve() == run_dir and original["seed"] == config["SEED"], "Source run identity mismatch")
    epochs = int(config["OPTIM"]["MAX_EPOCH"])
    checkpoint = run_dir / "VLPromptLearner" / f"model.pth.tar-{epochs}"
    teacher = Path(original["teacher_checkpoint"])
    require(checkpoint.is_file() and teacher.is_file(), "A final student/teacher checkpoint is missing")
    require(sha256_file(teacher) == original["teacher_checkpoint_sha256"], "Teacher checkpoint hash changed")
    require(sha256_file(config["TRAINER"]["LUNA"]["TEACHER_CLIP_PATH"]) == original["teacher_clip_sha256"], "Teacher foundation hash changed")
    trace_path = run_dir / "journal_batches.csv"
    with trace_path.open(encoding="utf-8", newline="") as stream:
        trace = list(csv.DictReader(stream))
    groups = {}
    for row in trace:
        epoch, index = int(row["epoch"]), int(row["batch_index"])
        require(index not in groups.setdefault(epoch, set()), "Duplicated training batch")
        groups[epoch].add(index)
        require(row["dataset"] == original["dataset"] and row["variant"] == original["variant"]
                and int(row["seed"]) == original["seed"] and int(row["n_samples"]) == BATCH_SIZE,
                "Training trace identity/batch size mismatch")
    require(set(groups) == set(range(1, epochs + 1)), "Incomplete training epochs")
    expected_batches = len(groups[1])
    require(all(indices == set(range(expected_batches)) for indices in groups.values()), "Incomplete training batch coverage")
    require(len(trace) == original["statistics"]["batch_count"] and BATCH_SIZE * len(trace) == original["statistics"]["n_observations"],
            "Training trace/summary counts differ")
    records, split_sources = original_test_records(config)
    analyzed = records[:len(records) // BATCH_SIZE * BATCH_SIZE]
    discarded = records[len(analyzed):]
    eligible = original["teacher_quality"] == "strong" and original["variant"] == "reliability_adaptive"
    require(options.exp2_output_dir is None or eligible, "Exp2 mirror is only for Strong AdaptiveLUNA")
    if options.exp2_output_dir:
        mirror = options.exp2_output_dir.resolve()
        require(mirror != output and (not mirror.exists() or not any(mirror.iterdir())), "Exp2 mirror must be new or empty")
    require(not output.exists() or not any(output.iterdir()), "Evaluation output must be new or empty; existing evidence is preserved")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol_id": "original_test_order_full8_paired_seeded_T1_v1", "evaluation_seed": options.evaluation_seed,
        "sample_order": "unchanged original held-out test split file order", "split": "original_test", "prediction_space": "all_classes",
        "batch_size": BATCH_SIZE, "shuffle": False, "drop_last": True, "num_workers": 0,
        "N_original_test": len(records), "N_analyzed": len(analyzed), "N_discarded": len(discarded),
        "analyzed_samples": [{k: r[k] for k in ("sample_index", "sample_id", "ground_truth", "classname")} for r in analyzed],
        "discarded_samples": discarded, "split_sources": [{"path": str(p), "sha256": sha256_file(p)} for p in split_sources],
        "weak_view": "existing build_transform(cfg,is_train=False): resize shorter edge224 bicubic,center crop224,ToTensor,CLIP normalization",
        "augmented_view": "existing build_transform(cfg,is_train=True,strong=True): random resized crop224 scale(.08,1),horizontal flip,RandAugment(2,10),ToTensor,CLIP normalization",
        "augmentation_seed_rule": "int.from_bytes(SHA256(f'{evaluation_seed}:{dataset}:{original_test_index}').digest()[:4],'big'); reset/restore Python,NumPy,CPU Torch RNG per sample",
        "reliability_function": "experiments.journal_losses.compute_reliability, unchanged", "temperature": 1.0,
        "probability_dtype": "torch.float32", "probability_computation": "native logits -> float32 log_softmax -> exp",
        "distance_formulas": {"D_ins": "mean_i sum_c Pt[i,c]*(logPt[i,c]-logPs[i,c])",
                              "D_batch": "||Pt@Pt.T-Ps@Ps.T||F/(||Pt@Pt.T||F+1e-8)",
                              "D_cls": "||Pt.T@Pt-Ps.T@Ps||F/(||Pt.T@Pt||F+1e-8)"},
        "distance_units": {"D_ins": "nats per image", "D_batch": "dimensionless", "D_cls": "dimensionless"},
        "distance_aggregation": "sample-weighted mean of full-batch distances; all batches contain8",
        "teacher_accuracy_scope": "original_test_full_batch_subset_all_class_micro",
        "spearman": {"implementation": "scipy.stats.spearmanr", "alternative": "two-sided", "p_method": "asymptotic Student-t approximation, no permutation",
                     "tie_method": "average ranks", "undefined": "constant scores or accuracy yield null rho/p plus reason", "unit": "full batch"},
        "quintiles": {"method": "stable ascending reliability sort, ties retain original batch index; numpy.array_split into5", "direction": "1 lowest,5 highest"},
        "inferential_limitations": "single-seed descriptive post-training analysis; ordered class-correlated batches and discrete batch accuracy limit independent-batch inference; p-values unadjusted across3 reliabilities",
        "training_restarted": False, "additional_optimizer_steps": 0,
    }
    protocol["analyzed_sample_hash"] = object_hash(protocol["analyzed_samples"])
    manifest = {
        "schema_version": 1, "status": "PREPARED_REAL_EVALUATION", "ResultSource": "REAL_EXPERIMENT",
        "dataset": original["dataset"], "variant": original["variant"], "seed": original["seed"],
        "teacher_quality": original["teacher_quality"], "student_epoch": epochs, "evaluation_seed": options.evaluation_seed,
        "source_run_manifest": str(source_path), "source_run_manifest_sha256": sha256_file(source_path),
        "source_hashes": original["source_hashes"], "source_protocol_fingerprint": original["protocol_fingerprint"],
        "training_trace": str(trace_path), "training_trace_sha256": sha256_file(trace_path),
        "student_checkpoint": str(checkpoint), "checkpoint_sha256_before": sha256_file(checkpoint),
        "teacher_checkpoint": str(teacher), "teacher_checkpoint_sha256": original["teacher_checkpoint_sha256"],
        "helper_path": str(Path(__file__).resolve()), "helper_sha256": sha256_file(__file__),
        "command_argv": list(sys.argv), "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol, "analyzed_sample_hash": protocol["analyzed_sample_hash"],
        "N_batches": len(analyzed) // BATCH_SIZE, "N_samples": len(analyzed), "exp2_eligible": eligible,
        "teacher_accuracy_scope": protocol["teacher_accuracy_scope"], "training_restarted": False, "additional_optimizer_steps": 0,
    }
    manifest_path = output / "evaluation_manifest.json"
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    try:
        # No call into the train entry point or its orchestration; suppress only
        # WILDS' optional remote version notification during module registration.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import outdated
        outdated.check_outdated = None
        import train
        import numpy as np
        import scipy
        import torch
        from PIL import Image
        from yacs.config import CfgNode
        from dassl.config import get_cfg_default
        from dassl.data.transforms import build_transform
        from dassl.utils import set_random_seed
        from trainers.luna_journal import LUNAJournal
        from experiments.journal_losses import compute_reliability
        cfg = get_cfg_default()
        train.extend_cfg(cfg)
        cfg.merge_from_other_cfg(CfgNode(config))
        cfg.freeze()
        require(protocol_fingerprint(cfg) == original["protocol_fingerprint"], "Reconstructed configuration differs")
        # Our own source/hash-validated checkpoints contain live Dassl scheduler metadata.
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        require(payload.get("epoch") == epochs and payload.get("val_result") is None, "Checkpoint is not the configured final unselected epoch")
        require(payload.get("state_dict") and all(not isinstance(value, torch.Tensor) or bool(torch.isfinite(value).all())
                                                for value in payload["state_dict"].values()), "Nonfinite/missing checkpoint state")
        del payload
        set_random_seed(cfg.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        trainer = LUNAJournal(cfg)
        require(model_state_hash(trainer.model) == original["initialization_sha256"], "Student initialization changed")
        require(dataset_fingerprint(trainer.dm.dataset) == original["dataset_fingerprint"], "Dataset/split fingerprint changed")
        require(len(trainer.train_loader_x) == expected_batches, "Real training-loader length differs from full trace")
        classnames = list(trainer.dm.dataset.classnames)
        require(all(0 <= r["ground_truth"] < len(classnames) and classnames[r["ground_truth"]] == r["classname"] for r in records),
                "Original test labels/classnames disagree with the model all-class mapping")
        require(trainer.temperature == 1.0, "Existing reliability temperature must remain one")
        trainer.load_model(str(run_dir), epoch=epochs)
        trainer.model.eval()
        trainer.model_teacher.eval()
        model_before = model_state_hash(trainer.model)
        teacher_before = model_state_hash(trainer.model_teacher)
        weak_transform = build_transform(cfg, is_train=False)
        augmented_transform = build_transform(cfg, is_train=True, strong=True)
        protocol.update(classnames=classnames, weak_transform_repr=stable_transform_repr(weak_transform),
                        augmented_transform_repr=stable_transform_repr(augmented_transform),
                        cudnn_benchmark=False, cudnn_deterministic=True,
                        torch_deterministic_algorithms=True, cublas_workspace_config=":4096:8", allow_tf32=False,
                        model_dtype=str(next(trainer.model.parameters()).dtype), teacher_model_dtype=str(next(trainer.model_teacher.parameters()).dtype))
        manifest.update(teacher_epoch=trainer.journal_teacher_epoch, model_state_sha256_before=model_before,
                        teacher_model_state_sha256_before=teacher_before, protocol_fingerprint=object_hash(protocol),
                        environment={"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "cuda": torch.version.cuda,
                                     "device": str(trainer.device), "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        # Persist the complete protocol before the first outcome is computed.
        write_json(manifest_path, manifest)

        class PairedTestDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(records)

            def __getitem__(self, index):
                record = records[index]
                with Image.open(record["impath"]) as opened:
                    source = opened.convert("RGB")
                weak = weak_transform(source)
                seed = int.from_bytes(hashlib.sha256(f"{options.evaluation_seed}:{original['dataset']}:{index}".encode()).digest()[:4], "big")
                python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
                try:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.random.default_generator.manual_seed(seed)
                    augmented = augmented_transform(source)
                finally:
                    random.setstate(python_state)
                    np.random.set_state(numpy_state)
                    torch.set_rng_state(torch_state)
                return weak, augmented, index, seed, sha256_file(record["impath"])

        loader = torch.utils.data.DataLoader(PairedTestDataset(), batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=0)
        batches, samples, probability_arrays = [], [], {key: [] for key in ("teacher_log_probability", "student_log_probability", "teacher_aug_log_probability")}
        with torch.no_grad():
            for batch_index, (weak, augmented, indices, augmentation_seeds, image_hashes) in enumerate(loader):
                require(indices.tolist() == list(range(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE)), "Test order changed")
                weak, augmented = weak.to(trainer.device), augmented.to(trainer.device)
                teacher_logits = trainer.model_teacher(weak)
                augmented_logits = trainer.model_teacher(augmented)
                log_teacher = teacher_logits.float().log_softmax(dim=1)
                log_augmented = augmented_logits.float().log_softmax(dim=1)
                scale, image_features, text_features = trainer.model(weak, None)
                student_logits = scale * image_features @ text_features.T
                log_student = student_logits.float().log_softmax(dim=1)
                if batch_index == 0:
                    manifest["observed_logit_dtypes"] = {"teacher": str(teacher_logits.dtype), "student": str(student_logits.dtype)}
                require(all(value.shape == (BATCH_SIZE, len(classnames)) and bool(torch.isfinite(value).all())
                            for value in (log_teacher, log_augmented, log_student)), "Invalid all-class logits")
                reliability = {key: float(value) for key, value in compute_reliability(log_teacher.exp(), log_augmented.exp()).items()}
                distances = relation_distances(log_teacher, log_student)
                teacher_pred = log_teacher.argmax(dim=1).cpu().tolist()
                student_pred = log_student.argmax(dim=1).cpu().tolist()
                correct = 0
                for offset, index in enumerate(indices.tolist()):
                    record = records[index]
                    is_correct = int(teacher_pred[offset] == record["ground_truth"])
                    correct += is_correct
                    samples.append({**record, "batch_index": batch_index, "index_in_batch": offset,
                                    "augmentation_seed": int(augmentation_seeds[offset]), "image_sha256": image_hashes[offset],
                                    "teacher_prediction": teacher_pred[offset], "student_prediction": student_pred[offset], "teacher_correct": is_correct})
                batches.append({"batch_index": batch_index, "n_samples": BATCH_SIZE,
                                "sample_indices": json.dumps(indices.tolist()), "sample_ids": json.dumps([records[i]["sample_id"] for i in indices.tolist()]),
                                "teacher_correct_count": correct, "teacher_batch_accuracy_percent": 100 * correct / BATCH_SIZE,
                                **reliability, **distances})
                for key, tensor in zip(probability_arrays, (log_teacher, log_student, log_augmented)):
                    probability_arrays[key].append(tensor.cpu().numpy())
                if batch_index % 25 == 0 or batch_index + 1 == len(loader):
                    print(f"Real held-out evaluation: {batch_index + 1}/{len(loader)} full batches", flush=True)
        require(len(batches) == manifest["N_batches"] and len(samples) == manifest["N_samples"], "Incomplete held-out evaluation")
        metrics = {key: sum(row[key] * row["n_samples"] for row in batches) / len(samples) for key in ("D_ins", "D_batch", "D_cls")}
        require(all(math.isfinite(value) for value in metrics.values()), "Nonfinite relation metrics")
        correlations, quintiles = analysis_tables(batches) if eligible else ([], [])
        write_csv(output / "batches.csv", batches, list(batches[0]))
        write_csv(output / "samples.csv", samples, list(samples[0]))
        write_csv(output / "exp2_correlations.csv", correlations, ("reliability", "rho", "p_value", "N_batches", "status", "reason"))
        write_csv(output / "exp2_quintiles.csv", quintiles, ("reliability", "quintile", "N_batches", "N_samples", "reliability_min", "reliability_max", "reliability_mean", "teacher_accuracy_percent", "batch_indices"))
        write_json(output / "exp4_metrics.json", {"ResultSource": "REAL_EXPERIMENT", "N_batches": len(batches), "N_samples": len(samples),
                                                "relation_metrics": metrics, "units": protocol["distance_units"], "aggregation": protocol["distance_aggregation"]})
        np.savez_compressed(output / "probabilities.npz", **{key: np.concatenate(value) for key, value in probability_arrays.items()})
        model_after, teacher_after = model_state_hash(trainer.model), model_state_hash(trainer.model_teacher)
        require(model_before == model_after and teacher_before == teacher_after and trainer._journal_optimizer_steps == 0,
                "Evaluation changed model state or executed optimizer steps")
        require(sha256_file(checkpoint) == manifest["checkpoint_sha256_before"] and sha256_file(teacher) == original["teacher_checkpoint_sha256"], "A checkpoint changed during evaluation")
        require(sha256_file(source_path) == manifest["source_run_manifest_sha256"] and sha256_file(trace_path) == manifest["training_trace_sha256"], "Source training evidence changed")
        require(source_hashes() == manifest["source_hashes"], "Core sources changed during evaluation")
        artifacts = {path.name: {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                     for path in output.iterdir() if path.is_file() and path != manifest_path}
        manifest.update(status="COMPLETED_REAL_EVALUATION", completed_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started, checkpoint_sha256_after=sha256_file(checkpoint),
                        model_state_sha256_after=model_after, teacher_model_state_sha256_after=teacher_after,
                        teacher_accuracy_micro_percent=100 * sum(row["teacher_correct"] for row in samples) / len(samples),
                        teacher_correct_count=sum(row["teacher_correct"] for row in samples), relation_metrics=metrics,
                        artifacts=artifacts, source_validation={"complete_training_trace": True, "final_epoch_checkpoint": True,
                                                              "initialization": True, "dataset_fingerprint": True, "all_class_mapping": True,
                                                              "core_source_hashes": True, "unchanged_model_and_checkpoint": True})
        write_json(manifest_path, manifest)
        if options.exp2_output_dir:
            mirror = options.exp2_output_dir.resolve()
            require(mirror != output and (not mirror.exists() or not any(mirror.iterdir())), "Exp2 mirror must be new or empty")
            mirror.mkdir(parents=True, exist_ok=True)
            for name in ("evaluation_manifest.json", "exp2_correlations.csv", "exp2_quintiles.csv"):
                shutil.copy2(output / name, mirror / name)
        print("COMPLETED_REAL_EVALUATION " + json.dumps({"manifest": str(manifest_path), "teacher_accuracy_micro_percent": manifest["teacher_accuracy_micro_percent"], **metrics}), flush=True)
    except Exception as exc:
        manifest.update(status="FAILED_REAL_EVALUATION", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - started)
        write_json(manifest_path, manifest)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
