"""Separate journal training with paired views and explicit final-epoch selection."""

import hashlib
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from dassl.data.data_manager import build_data_loader
from dassl.data.transforms import build_transform
from dassl.engine import TrainerX
from dassl.utils import load_checkpoint

from experiments.journal_losses import EPS, aggregate_losses, compute_reliability
from .luna import LUNA, bc_loss, cc_loss, kd_loss


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trainable_sha256(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            value = parameter.detach().cpu().contiguous()
            digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class LUNAJournal(LUNA):
    """Instantiate directly with TRAINER.NAME='LUNA' to retain dataset semantics.

    The parent retains model construction, prompt architecture, optimizers,
    scalar loss helpers, and Base/Novel inference. Only this class opts into the
    new paired-view objective and final-epoch selection protocol.
    """

    def __init__(self, cfg):
        self.journal_batch_callback = None
        self.journal_epoch_callback = None
        self.journal_records = []
        self.journal_last_record = None
        self.journal_final_result = None
        self._journal_sanity = False
        self._journal_optimizer_steps = 0
        self._journal_epoch_sums = {}
        self._journal_epoch_samples = 0
        super().__init__(cfg)

    def check_cfg(self, cfg):
        super().check_cfg(cfg)
        if cfg.TRAINER.NAME != "LUNA" or cfg.TRAINER.MODAL != "base2novel":
            raise ValueError("Journal mode requires the existing LUNA base2novel dataset path")
        if cfg.DATASET.NAME not in {"FGVCAircraft", "DescribableTextures", "EuroSAT"}:
            raise ValueError("Journal mechanism comparisons are restricted to the three requested datasets")
        if cfg.JOURNAL.MODE not in {"equal", "static_mean", "reliability_adaptive", "instance_only"}:
            raise ValueError("Unknown journal weighting mode; legacy uses the original trainer")
        if cfg.JOURNAL.SELECTION_PROTOCOL != "final_epoch" or cfg.TEST.FINAL_MODEL != "last_step":
            raise ValueError("Journal mode requires final_epoch / TEST.FINAL_MODEL=last_step")
        if not cfg.TEST.NO_TEST:
            raise ValueError("Journal mode requires TEST.NO_TEST=True; final Base/Novel evaluation is explicit")
        if cfg.RESUME:
            raise ValueError("Journal comparisons require a fresh, controlled initialization; automatic resume is disabled")
        if cfg.SEED < 0:
            raise ValueError("A fixed nonnegative main-experiment seed is required")
        if cfg.DATASET.NUM_SHOTS != 0:
            raise ValueError("The inspected student protocol uses NUM_SHOTS=0 and the full unlabeled training set")
        if cfg.DATALOADER.K_TRANSFORMS != 1:
            raise ValueError("Paired journal views require the original K_TRANSFORMS=1 setting")
        if cfg.TRAINER.IVLP.PREC == "amp":
            raise ValueError("AMP is not part of the verified original training path")
        if int(cfg.JOURNAL.TEACHER_TOTAL_EPOCHS) <= 0:
            raise ValueError("The original total teacher-training epoch count must be supplied")
        if cfg.JOURNAL.TEACHER_QUALITY not in {"weak", "medium", "strong"}:
            raise ValueError("Teacher quality must be a preselected weak, medium, or strong stage")
        output_root = Path(cfg.JOURNAL.OUTPUT_ROOT).expanduser()
        if not output_root.is_absolute():
            raise ValueError("JOURNAL.OUTPUT_ROOT must be an absolute separate output directory")
        output = Path(cfg.OUTPUT_DIR).expanduser().resolve()
        if not output.is_relative_to(output_root.resolve()):
            raise ValueError("Journal OUTPUT_DIR must stay inside JOURNAL.OUTPUT_ROOT")
        if not str(cfg.DATASET.FEWSHOT_CACHE_ROOT).strip():
            raise ValueError("Set a separate FEWSHOT_CACHE_ROOT so dataset construction cannot write into raw data")
        cache = Path(cfg.DATASET.FEWSHOT_CACHE_ROOT).expanduser().resolve()
        if not cache.is_relative_to(output_root.resolve()):
            raise ValueError("Journal few-shot cache must stay inside its separate output root")
        if cfg.JOURNAL.MODE == "static_mean" and len(cfg.JOURNAL.STATIC_WEIGHTS) != 3:
            raise ValueError("static_mean requires three fixed weights from a compatible training-only run")

    def build_data_loader(self):
        root = Path(self.cfg.DATASET.ROOT).expanduser().resolve()
        required = {
            "DescribableTextures": ["dtd/split_zhou_DescribableTextures.json"],
            "EuroSAT": ["eurosat/split_zhou_EuroSAT.json"],
            "FGVCAircraft": [
                "fgvc_aircraft/variants.txt", "fgvc_aircraft/images_variant_train.txt",
                "fgvc_aircraft/images_variant_val.txt", "fgvc_aircraft/images_variant_test.txt",
            ],
        }[self.cfg.DATASET.NAME]
        missing = [str(root / relative) for relative in required if not (root / relative).is_file()]
        if missing:
            raise FileNotFoundError("Existing splits are required; no new split will be generated: " + ", ".join(missing))
        super().build_data_loader()
        source = self.dm.dataset.train_x
        other = self.dm.dataset.train_u
        if len(source) != len(other) or any(
            (left.impath, left.label) != (right.impath, right.label)
            for left, right in zip(source, other)
        ):
            raise ValueError("The original x/u training sources differ; paired journal protocol cannot be inferred")
        weak_transform = build_transform(self.cfg, is_train=True)
        augmented_transform = build_transform(self.cfg, is_train=True, strong=True)
        if weak_transform is None or augmented_transform is None:
            raise ValueError("Journal pairing requires the original weak and strong image transforms")
        loader = build_data_loader(
            self.cfg,
            sampler_type=self.cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=source,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=[weak_transform, augmented_transform],
            is_train=True,
        )
        # DatasetWrapper applies both transforms to the same opened image and
        # emits img/img2 with one shared index/impath. There is one sampler.
        self.train_loader_x = self.train_loader = loader
        self.train_loader_u = None
        self.dm.train_loader_x = loader
        self.dm.train_loader_u = None
        self.journal_pairing = {
            "policy": "one_sample_two_existing_transforms_one_sampler",
            "weak_transform": repr(weak_transform),
            "augmented_transform": repr(augmented_transform),
            "sample_count": len(source),
            "batch_size": loader.batch_size,
            "drop_last": loader.drop_last,
        }

    def _resolve_teacher_checkpoint(self):
        value = os.path.expandvars(str(self.cfg.JOURNAL.TEACHER_CHECKPOINT).strip())
        if not value:
            raise ValueError("JOURNAL.TEACHER_CHECKPOINT must identify an explicit, preselected teacher file")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit journal teacher checkpoint not found: {path}")
        if "best" in path.name.lower():
            raise ValueError("Journal teachers require a named training-stage checkpoint, not a best-selected file")
        self.journal_teacher_checkpoint = str(path)
        return str(path)

    def build_model(self):
        super().build_model()
        checkpoint = load_checkpoint(self.journal_teacher_checkpoint)
        epoch = checkpoint.get("epoch")
        total = int(self.cfg.JOURNAL.TEACHER_TOTAL_EPOCHS)
        if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or not 0 < epoch <= total:
            raise ValueError("Teacher checkpoint requires an integer epoch in (0, TEACHER_TOTAL_EPOCHS]")
        if self.cfg.JOURNAL.TEACHER_QUALITY == "strong" and epoch != total:
            raise ValueError("Strong teacher must be the final teacher-training epoch, not a selected earlier checkpoint")
        if self.cfg.JOURNAL.TEACHER_QUALITY != "strong" and epoch >= total:
            raise ValueError("Weak/medium teacher checkpoints must precede the final teacher epoch")
        state = checkpoint["state_dict"]
        prompt_keys = [
            name for name, _ in self.model_teacher.named_parameters()
            if "prompt_learner" in name or "VPT" in name
        ]
        missing = [name for name in prompt_keys if name not in state]
        if missing:
            raise ValueError("Teacher checkpoint is missing prompted parameters: " + ", ".join(missing))
        parameters = dict(self.model_teacher.named_parameters())
        for name in prompt_keys:
            actual = parameters[name].detach()
            expected = state[name].to(device=actual.device, dtype=actual.dtype)
            if not torch.equal(actual, expected):
                raise ValueError(f"The loaded teacher prompt differs from the selected checkpoint: {name}")
        self.model_teacher.requires_grad_(False)
        self.model_teacher.eval()
        for parameter in self.model_teacher.parameters():
            parameter.grad = None
        self.journal_teacher_epoch = int(epoch)
        self.journal_teacher_sha256 = _file_sha256(self.journal_teacher_checkpoint)
        self.journal_initial_trainable_sha256 = _trainable_sha256(self.model)
        self.journal_trainable_names = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        if not self.journal_trainable_names or any("prompt_learner" not in name for name in self.journal_trainable_names):
            raise ValueError("The journal student must retain the original text-prompt-only trainable parameters")
        self.journal_temperatures = (self.temperature, 3.0, 5.0, 2.0, 6.0)

    def parse_batch_train(self, batch):
        if "img2" not in batch or "index" not in batch or "impath" not in batch:
            raise ValueError("Journal forward requires one paired batch from its dedicated loader")
        weak, augmented = batch["img"], batch["img2"]
        if not isinstance(weak, torch.Tensor) or not isinstance(augmented, torch.Tensor) or weak.shape != augmented.shape:
            raise ValueError("The two views must be aligned image tensors with identical shapes")
        if len(batch["index"]) != weak.shape[0] or len(batch["impath"]) != weak.shape[0]:
            raise ValueError("Paired sample IDs must match the image batch")
        return weak.to(self.device), augmented.to(self.device)

    def _gradient_record(self):
        prompt_gradients = [
            parameter.grad for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "prompt_learner" in name and parameter.grad is not None
        ]
        finite = bool(prompt_gradients) and all(bool(torch.isfinite(value).all()) for value in prompt_gradients)
        norm = math.sqrt(sum(value.detach().double().square().sum().item() for value in prompt_gradients))
        result = {
            "teacher_requires_grad_false": all(not parameter.requires_grad for parameter in self.model_teacher.parameters()),
            "teacher_grad_none": all(parameter.grad is None for parameter in self.model_teacher.parameters()),
            "student_prompt_grad_present": bool(prompt_gradients),
            "student_prompt_grad_finite": finite,
            "student_prompt_grad_nonzero": finite and norm > 0,
            "student_prompt_grad_norm": norm,
            "student_frozen_parameter_grad_none": all(
                parameter.grad is None for parameter in self.model.parameters() if not parameter.requires_grad
            ),
        }
        if not result["teacher_requires_grad_false"] or not result["teacher_grad_none"]:
            raise RuntimeError("The teacher must remain frozen with no gradients")
        if self.cfg.JOURNAL.VALIDATE_GRADIENTS and (
            not finite or not result["student_frozen_parameter_grad_none"]
        ):
            raise RuntimeError("Student prompt gradients are missing/nonfinite or a frozen parameter received gradients")
        return result

    @torch.no_grad()
    def _legacy_comparison(self, teacher_logits, weak_terms):
        probabilities = F.softmax(teacher_logits.detach(), dim=1)
        confidence = probabilities.max(dim=1)[0].detach()
        mask = confidence.le(np.percentile(confidence.cpu().numpy().flatten(), 50)).bool()
        class_mass = probabilities.sum(dim=0).detach()
        class_mask = class_mass.le(np.percentile(class_mass.cpu().numpy().flatten(), 50)).bool()
        masks = (mask, mask, class_mask)
        names = ("L_ins", "L_batch", "L_cls")
        values = {}
        for name, terms, selected in zip(names, weak_terms, masks):
            # Preserve the original scalar -> broadcast mask -> mean order.
            value = sum(self.kd_loss_weight * (term.detach() * selected).mean() for term in terms)
            values[name] = value
        total = sum(values[name].mean() for name in ("L_ins", "L_cls", "L_batch"))
        finite = all(bool(torch.isfinite(value)) for value in (*values.values(), total))
        return {
            "legacy_loss": total.item() if bool(torch.isfinite(total)) else None,
            "legacy_components": {name: value.item() if bool(torch.isfinite(value)) else None for name, value in values.items()},
            "legacy_sample_selected_fraction": mask.float().mean().item(),
            "legacy_class_selected_fraction": class_mask.float().mean().item(),
            "legacy_numeric_status": "finite" if finite else "nonfinite",
            "legacy_comparison_scope": "same weak logits; original weak-only scalar-mask formula; no legacy optimization",
        }

    @torch.no_grad()
    def _sample_records(self, batch, prob_teacher, prob_augmented):
        if not self.cfg.JOURNAL.LOG_BATCH_SAMPLES:
            return []
        teacher = prob_teacher.detach().float()
        augmented = prob_augmented.detach().float()
        entropy = -(teacher * (teacher + EPS).log()).sum(dim=1)
        entropy_aug = -(augmented * (augmented + EPS).log()).sum(dim=1)
        confidence, prediction = teacher.max(dim=1)
        confidence_aug, prediction_aug = augmented.max(dim=1)
        labels = batch["label"].detach().cpu().tolist()
        indices = batch["index"].detach().cpu().tolist()
        return [
            {
                "index": int(index), "sample_index": int(index),
                "impath": str(path), "label": int(label), "ground_truth": int(label),
                "teacher_entropy": float(ent), "teacher_max_prob": float(conf),
                "teacher_max_probability": float(conf),
                "teacher_pred": int(pred), "teacher_prediction": int(pred),
                "teacher_correct": int(pred) == int(label),
                "teacher_aug_entropy": float(ent_aug), "teacher_entropy_aug": float(ent_aug),
                "teacher_aug_max_prob": float(conf_aug),
                "teacher_aug_pred": int(pred_aug), "teacher_aug_correct": int(pred_aug) == int(label),
                "candidate_space": "all_training_classes", "reliability_granularity": "batch",
            }
            for index, path, label, ent, conf, pred, ent_aug, conf_aug, pred_aug in zip(
                indices, batch["impath"], labels, entropy.cpu().tolist(), confidence.cpu().tolist(),
                prediction.cpu().tolist(), entropy_aug.cpu().tolist(), confidence_aug.cpu().tolist(),
                prediction_aug.cpu().tolist(),
            )
        ]

    def forward_backward(self, batch):
        weak, augmented = self.parse_batch_train(batch)
        self.model_teacher.eval()
        scale, image, text = self.model(weak, None)
        student_weak = scale * image @ text.t()
        scale_aug, image_aug, text_aug = self.model(augmented, None)
        student_aug = scale_aug * image_aug @ text_aug.t()
        with torch.no_grad():
            teacher_weak = self.model_teacher(weak)
            teacher_aug = self.model_teacher(augmented)
            prob_teacher = F.softmax(teacher_weak.detach() / self.temperature, dim=1)
            prob_aug = F.softmax(teacher_aug.detach() / self.temperature, dim=1)
            reliabilities = compute_reliability(prob_teacher, prob_aug)

        functions = (kd_loss, bc_loss, cc_loss)
        weak_terms = [
            [function(student_weak, teacher_weak, temperature) for temperature in self.journal_temperatures]
            for function in functions
        ]
        augmented_terms = [
            [function(student_aug, teacher_aug, temperature) for temperature in self.journal_temperatures]
            for function in functions
        ]
        components = [
            sum(self.kd_loss_weight * term for term in original)
            + sum(self.kd_loss_weight * term for term in other)
            for original, other in zip(weak_terms, augmented_terms)
        ]
        total, weights = aggregate_losses(
            *components,
            mode=self.cfg.JOURNAL.MODE,
            reliabilities=reliabilities,
            static_weights=self.cfg.JOURNAL.STATIC_WEIGHTS,
        )
        if any(value.requires_grad for value in (*reliabilities.values(), *weights.values())):
            raise RuntimeError("Teacher reliabilities and hierarchical weights must be detached")
        applied_lr = self.get_current_lr()
        comparison = self._legacy_comparison(teacher_weak, weak_terms)
        samples = self._sample_records(batch, prob_teacher, prob_aug)
        self.optim.zero_grad()
        total.backward()
        gradient_record = self._gradient_record()
        self.optim.step()
        self._journal_optimizer_steps += 1

        component_names = ("L_ins", "L_batch", "L_cls")
        weight_names = ("lambda_ins", "lambda_batch", "lambda_cls")
        record = {
            "dataset": self.cfg.DATASET.NAME, "seed": int(self.cfg.SEED),
            "epoch": int(self.epoch + 1), "batch_index": int(self.batch_idx),
            "n_samples": int(weak.shape[0]), "lr": float(applied_lr),
            "variant": self.cfg.JOURNAL.MODE, "selection_protocol": "final_epoch",
            "run_kind": "sanity_training_batches" if self._journal_sanity else "journal_training",
            "teacher_checkpoint": self.journal_teacher_checkpoint,
            "teacher_checkpoint_sha256": self.journal_teacher_sha256,
            "teacher_epoch": self.journal_teacher_epoch,
            "teacher_quality": self.cfg.JOURNAL.TEACHER_QUALITY,
            "student_initial_trainable_sha256": self.journal_initial_trainable_sha256,
            "Base": None, "Novel": None, "HM": None,
            "accuracy_status": "not_evaluated_during_training",
            "loss": total.detach().item(), "L_adapt": total.detach().item(),
            **{name: value.detach().item() for name, value in zip(component_names, components)},
            **{name: value.detach().item() for name, value in reliabilities.items()},
            **{name: value.detach().item() for name, value in weights.items()},
            "weighted_terms": {
                name: (value.detach().to(weights[weight_name].dtype) * weights[weight_name]).item()
                for name, value, weight_name in zip(component_names, components, weight_names)
            },
            "sample_indices": batch["index"].detach().cpu().tolist(),
            "sample_paths": [str(path) for path in batch["impath"]],
            "paired_same_sample": True, "paired_view_count": 2,
            "reliability_granularity": "batch",
            "reliability_temperature": float(self.temperature),
            "loss_temperatures": list(self.journal_temperatures),
            "probability_precision": str(prob_teacher.dtype),
            "loss_component_precision": [str(value.dtype) for value in components],
            "reliability_precision": str(reliabilities["r_ins"].dtype),
            "teacher_weights_detached": True,
            **gradient_record, **comparison,
        }
        if self._journal_sanity and self.batch_idx == 7:
            diagnostic_r = {"r_ins": 0.1, "r_batch": 0.2, "r_cls": 0.7}
            diagnostic_total, diagnostic_weights = aggregate_losses(
                *(value.detach() for value in components),
                mode="reliability_adaptive", reliabilities=diagnostic_r,
            )
            record["alternate_weight_diagnostic"] = {
                "used_for_optimization": False, "input_reliabilities": diagnostic_r,
                "weights": {name: value.item() for name, value in diagnostic_weights.items()},
                "weighted_terms": {
                    name: (value.detach().to(diagnostic_weights[weight_name].dtype) * diagnostic_weights[weight_name]).item()
                    for name, value, weight_name in zip(component_names, components, weight_names)
                },
                "total": diagnostic_total.item(),
            }
        self.journal_last_record = record
        if self._journal_sanity:
            self.journal_records.append(record)
        for name in (*component_names, *weight_names, "r_ins", "r_batch", "r_cls", "loss", "L_adapt"):
            self._journal_epoch_sums[name] = self._journal_epoch_sums.get(name, 0.0) + record[name] * record["n_samples"]
        self._journal_epoch_samples += record["n_samples"]
        if self.journal_batch_callback is not None:
            self.journal_batch_callback(record, samples)
        if not self._journal_sanity and (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return {name: record[name] for name in ("loss", *component_names, *weight_names, "r_ins", "r_batch", "r_cls")}

    def run_epoch(self):
        self.model_teacher.eval()
        TrainerX.run_epoch(self)

    def before_epoch(self):
        self._journal_epoch_sums = {}
        self._journal_epoch_samples = 0

    def before_train(self):
        if self._journal_optimizer_steps:
            raise RuntimeError("Full journal training needs a fresh instance; sanity-updated weights cannot be reused as initialization")
        if any(Path(self.output_dir).glob("*/model*.pth.tar*")):
            raise FileExistsError("Journal output already contains checkpoints; use a new controlled run directory")
        self.start_epoch = 0
        self.init_writer(str(Path(self.output_dir) / "tensorboard"))
        self.time_start = time.time()

    def train(self):
        if int(self.cfg.JOURNAL.SANITY_BATCHES) > 0:
            raise ValueError("Use run_sanity() for the configured finite batch check; full training is disabled")
        from experiments.journal_runtime import require_sanity_pass
        require_sanity_pass(self.cfg.JOURNAL.SANITY_REPORT)
        super().train()

    def after_epoch(self):
        last = self.epoch + 1 == self.max_epoch
        frequency = self.cfg.TRAIN.CHECKPOINT_FREQ
        if last or (frequency > 0 and (self.epoch + 1) % frequency == 0):
            self.save_model(self.epoch, self.output_dir)
        if self.journal_epoch_callback is not None:
            record = {
                "epoch": self.epoch + 1, "n_samples": self._journal_epoch_samples,
                "selection_protocol": "final_epoch", "accuracy_status": "not_evaluated_during_training",
                **{name: value / self._journal_epoch_samples for name, value in self._journal_epoch_sums.items()},
            }
            self.journal_epoch_callback(record)

    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
        if self._journal_sanity:
            raise RuntimeError("Sanity batches must not produce a student experiment checkpoint")
        if is_best or val_result is not None or "best" in model_name.lower():
            raise ValueError("Journal checkpoints cannot be selected using evaluation performance")
        return super().save_model(epoch, directory, is_best=False, val_result=None, model_name=model_name)

    def load_model(self, directory, epoch=None):
        if epoch is None or int(epoch) != self.max_epoch:
            raise ValueError("Journal evaluation requires the explicit final student epoch")
        for name in self.get_model_names():
            path = Path(directory) / name / f"model.pth.tar-{epoch}"
            checkpoint = load_checkpoint(str(path))
            if checkpoint.get("epoch") != epoch:
                raise ValueError(f"Final student filename and stored epoch disagree: {path}")
        return super().load_model(directory, epoch=epoch)

    def after_train(self):
        self.load_model(self.output_dir, epoch=self.max_epoch)
        base = float(self.test(split="val"))
        novel = float(self.test(split="test"))
        hm = 0.0 if base + novel == 0 else 2 * base * novel / (base + novel)
        self.journal_final_result = {
            "dataset": self.cfg.DATASET.NAME, "seed": int(self.cfg.SEED),
            "epoch": self.max_epoch, "variant": self.cfg.JOURNAL.MODE,
            "Base": base, "Novel": novel, "HM": hm,
            "selection_protocol": "final_epoch",
            "student_checkpoint": str(Path(self.output_dir) / "VLPromptLearner" / f"model.pth.tar-{self.max_epoch}"),
            "teacher_checkpoint": self.journal_teacher_checkpoint,
        }
        print(f"Journal final-epoch evaluation: {self.journal_final_result}")
        self.close_writer()

    def run_sanity(self, max_batches=10):
        """Run real paired training batches without saving or evaluating a model."""
        if self.cfg.DATASET.NAME != "DescribableTextures" or self.cfg.JOURNAL.MODE != "reliability_adaptive":
            raise ValueError("The required journal sanity run is DTD with reliability_adaptive weighting")
        count = int(max_batches)
        if count < 10:
            raise ValueError("The requested journal sanity check requires at least ten actual batches")
        if self._journal_optimizer_steps:
            raise RuntimeError("Sanity must start from a fresh student initialization")
        if len(self.train_loader_x) < count:
            raise ValueError(f"Paired training loader has fewer than the requested {count} batches")
        self._journal_sanity = True
        self.journal_records = []
        self.num_batches = count
        self.epoch = 0
        self.before_epoch()
        self.set_model_mode("train")
        self.model_teacher.eval()
        iterator = iter(self.train_loader_x)
        try:
            for self.batch_idx in range(count):
                self.forward_backward(next(iterator))
            if not any(record["student_prompt_grad_nonzero"] for record in self.journal_records):
                raise RuntimeError("No nonzero student prompt gradient was observed during the actual sanity batches")
            return self.journal_records
        finally:
            self._journal_sanity = False
