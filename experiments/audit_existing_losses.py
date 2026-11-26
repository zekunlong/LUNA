"""Audit source loss semantics with synthetic inputs; never run an experiment."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F


DEFAULT_OUTPUT = Path("E:/LUNA/results_reliability_analysis/logs")
SOURCE = Path(__file__).resolve().parents[1] / "trainers" / "luna.py"
FUNCTIONS = ("kd_loss", "bc_loss", "cc_loss")
COMPONENTS = {
    "loss_kd_weak": ("kd_loss", "mask"),
    "loss_bc_weak": ("bc_loss", "mask"),
    "loss_cc_weak": ("cc_loss", "class_conf_mask"),
}
MASK_TARGETS = {
    "pred_teacher_weak", "confidence", "pseudo_labels", "conf_thresh",
    "mask", "class_confidence", "class_confidence_thresh", "class_conf_mask",
}


def targets(node):
    if not isinstance(node, ast.Assign):
        return set()
    return {
        child.id
        for target in node.targets
        for child in ast.walk(target)
        if isinstance(child, ast.Name)
    }


def assignment(nodes, name):
    matches = [node for node in nodes if name in targets(node)]
    if len(matches) != 1:
        raise ValueError(f"Expected one source assignment for {name}, found {len(matches)}")
    return matches[0]


def execute_nodes(nodes, namespace):
    # Only the explicitly inspected functions/assignments are executed. Importing
    # the trainer would otherwise load Dassl, CLIP, and training infrastructure.
    module = ast.Module(body=copy.deepcopy(nodes), type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)


def extract_source():
    source_bytes = SOURCE.read_bytes()
    source_text = source_bytes.decode("utf-8-sig")
    tree = ast.parse(source_text, filename=str(SOURCE))
    definitions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    if {node.name for node in definitions} != set(FUNCTIONS):
        raise ValueError("The expected three original loss functions were not found")
    trainer = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LUNA"
    )
    build = next(node for node in trainer.body if getattr(node, "name", None) == "build_model")
    forward = next(node for node in trainer.body if getattr(node, "name", None) == "forward_backward")
    branches = [
        node for node in forward.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "prec == 'amp'"
    ]
    if len(branches) != 1:
        raise ValueError("Cannot identify the original non-AMP training branch")
    branch = branches[0].orelse
    start = next(i for i, node in enumerate(branch) if "pred_teacher_weak" in targets(node))
    end = next(i for i, node in enumerate(branch) if "class_conf_mask" in targets(node))
    mask_nodes = branch[start:end + 1]
    if any(not targets(node) or not targets(node) <= MASK_TARGETS for node in mask_nodes):
        raise ValueError("Unexpected code in the confidence/mask assignment block")

    constants = {}
    for name in ("temperature", "kd_loss_weight"):
        matches = [
            node for node in build.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Attribute)
                    and ast.unparse(target) == f"self.{name}" for target in node.targets)
        ]
        if len(matches) != 1:
            raise ValueError(f"Cannot resolve the original {name} constant")
        constants[name] = ast.literal_eval(matches[0].value)

    component_nodes = [assignment(branch, name) for name in COMPONENTS]
    active_dict = assignment(branch, "losses_dict")
    if not isinstance(active_dict.value, ast.Dict):
        raise ValueError("The active loss dictionary is no longer a literal dictionary")
    active_names = [ast.unparse(value) for value in active_dict.value.values]
    if set(active_names) != set(COMPONENTS) or len(active_names) != len(COMPONENTS):
        raise ValueError("Active components differ from the inspected weak-only source")
    total_node = assignment(branch, "loss")

    temperatures = {}
    for node in component_nodes:
        name = next(iter(targets(node)))
        function_name, _ = COMPONENTS[name]
        calls = sorted(
            (call for call in ast.walk(node) if isinstance(call, ast.Call)
             and isinstance(call.func, ast.Name) and call.func.id == function_name),
            key=lambda call: (call.lineno, call.col_offset),
        )
        if not calls:
            raise ValueError(f"No source loss calls found for {name}")
        values = []
        for call in calls:
            if len(call.args) != 3 or call.keywords:
                raise ValueError("Loss call reduction/options changed; review the diagnostic")
            if [ast.unparse(arg) for arg in call.args[:2]] != [
                "logits_student_weak", "logits_teacher_weak"
            ]:
                raise ValueError("The active loss no longer uses only weak logits")
            temperature = call.args[2]
            if isinstance(temperature, ast.Attribute) and ast.unparse(temperature) == "self.temperature":
                values.append(constants["temperature"])
            else:
                values.append(ast.literal_eval(temperature))
        temperatures[name] = values

    source_nodes = definitions + mask_nodes + component_nodes + [active_dict, total_node]
    metadata = {
        "path": str(SOURCE),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "extraction": "AST-selected original functions and assignments; trainer not imported",
        "constants": constants,
        "active_temperatures_by_component": temperatures,
        "active_component_names": active_names,
        "active_weak_only": True,
        "source_fragments": [
            {"line_start": node.lineno, "line_end": node.end_lineno,
             "text": ast.get_source_segment(source_text, node)}
            for node in source_nodes
        ],
    }
    return definitions, mask_nodes, component_nodes, active_dict, total_node, metadata


def run_case(name, class_count, tied_teacher, generator, extracted):
    definitions, mask_nodes, component_nodes, active_dict, total_node, source = extracted
    batch_size = 8
    student = torch.randn(batch_size, class_count, generator=generator, dtype=torch.float64)
    teacher = torch.randn(batch_size, class_count, generator=generator, dtype=torch.float64)
    if tied_teacher:
        teacher = torch.zeros_like(teacher)
    namespace = {
        "torch": torch, "F": F, "np": np,
        "self": SimpleNamespace(**source["constants"]),
        "logits_student_weak": student, "logits_teacher_weak": teacher,
    }
    execute_nodes(definitions + mask_nodes + component_nodes + [active_dict, total_node], namespace)
    components = {}
    for component_name, (function_name, mask_name) in COMPONENTS.items():
        mask = namespace[mask_name]
        fraction = mask.to(torch.float64).mean()
        original_function = namespace[function_name]
        scalar_sum = torch.zeros((), dtype=torch.float64)
        per_temperature = []
        for temperature in source["active_temperatures_by_component"][component_name]:
            scalar = original_function(student, teacher, temperature)
            if scalar.ndim != 0:
                raise AssertionError(f"{function_name} no longer returns a reduced scalar")
            masked = (scalar * mask).mean()
            equivalent = scalar * fraction
            torch.testing.assert_close(masked, equivalent, rtol=1e-12, atol=1e-12)
            scalar_sum += scalar
            per_temperature.append({
                "temperature": temperature,
                "original_function_result_shape": list(scalar.shape),
                "broadcast_product_shape": list((scalar * mask).shape),
                "original_scalar_loss": scalar.item(),
                "source_masked_scalar_mean": masked.item(),
                "scalar_times_selected_fraction": equivalent.item(),
                "absolute_difference": abs((masked - equivalent).item()),
                "identity_assertion": "PASS",
            })
        equivalent_component = source["constants"]["kd_loss_weight"] * scalar_sum * fraction
        original_component = namespace[component_name]
        torch.testing.assert_close(original_component, equivalent_component, rtol=1e-12, atol=1e-12)
        components[component_name] = {
            "mask_source_name": mask_name,
            "selected_fraction": fraction.item(),
            "actual_source_component": original_component.item(),
            "equivalent_scalar_sum_times_fraction": equivalent_component.item(),
            "component_assertion": "PASS",
            "per_temperature": per_temperature,
        }
    equivalent_total = sum(value["equivalent_scalar_sum_times_fraction"] for value in components.values())
    actual_total = namespace["loss"].item()
    if not np.isclose(actual_total, equivalent_total, rtol=1e-12, atol=1e-12):
        raise AssertionError("Original source total differs from the scalar-fraction identity")
    return {
        "synthetic_case": name,
        "batch_size": batch_size,
        "class_count": class_count,
        "constant_teacher_logits": tied_teacher,
        "raw_teacher_confidence": namespace["confidence"].tolist(),
        "raw_class_probability_mass": namespace["class_confidence"].tolist(),
        "confidence_percentile_threshold": float(namespace["conf_thresh"]),
        "class_mass_percentile_threshold": float(namespace["class_confidence_thresh"]),
        "sample_mask": namespace["mask"].tolist(),
        "class_mask": namespace["class_conf_mask"].tolist(),
        "sample_selected_fraction": namespace["mask"].to(torch.float64).mean().item(),
        "class_selected_fraction": namespace["class_conf_mask"].to(torch.float64).mean().item(),
        "components": components,
        "actual_source_total_loss": actual_total,
        "equivalent_total_loss": equivalent_total,
        "total_assertion": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    extracted = extract_source()
    global_rng_before = torch.random.get_rng_state().clone()
    synthetic_seed = 20260905
    generator = torch.Generator(device="cpu").manual_seed(synthetic_seed)
    cases = [
        run_case("random_logits_B8_C10", 10, False, generator, extracted),
        run_case("random_logits_B8_C47", 47, False, generator, extracted),
        run_case("tied_teacher_logits_B8_C10", 10, True, generator, extracted),
    ]
    if not torch.equal(global_rng_before, torch.random.get_rng_state()):
        raise AssertionError("Diagnostic changed the global CPU Torch RNG state")
    cautions = [
        "Synthetic source diagnostic only: no real dataset, model, checkpoint, or training run was used.",
        "The synthetic RNG seed is not the main experiment seed.",
        "The scalar-broadcast identity verifies actual source behavior; it does not supply a missing reliability formulation.",
        "Selected fractions depend on ranks, ties, batch/class sizes, and numerical precision; do not assume they are exactly fixed.",
        "CPU float64 verifies the identity only, not production fp16 numerics or full trainer executability.",
        "Do not insert these synthetic numbers into Exp1-Exp4 result tables or claim experimental validation.",
    ]
    report = {
        "status": "PASS_SYNTHETIC_SOURCE_DIAGNOSTIC_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": extracted[-1],
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__,
            "device": "cpu", "dtype": "float64",
        },
        "synthetic_rng": {"seed": synthetic_seed, "generator": "local CPU torch.Generator",
                          "global_cpu_rng_unchanged": True},
        "cases": cases,
        "cautions": cautions,
    }
    lines = [
        report["status"],
        f"Source: {SOURCE}",
        f"Source SHA256: {report['source']['sha256']}",
        "AST-derived active loss components: " + ", ".join(report["source"]["active_component_names"]),
        "AST-derived temperature lists: " + json.dumps(report["source"]["active_temperatures_by_component"]),
        "Identity: mean(scalar_loss * mask) = scalar_loss * mean(mask).",
        "Global CPU Torch RNG unchanged: PASS",
    ]
    for case in cases:
        lines.append(
            f"{case['synthetic_case']}: PASS; sample selected fraction="
            f"{case['sample_selected_fraction']:.12g}; class selected fraction="
            f"{case['class_selected_fraction']:.12g}; source total="
            f"{case['actual_source_total_loss']:.12g}"
        )
    lines.extend(["", *cautions, "", "Raw source fragments, scores, masks, scalar shapes, and identity checks are in the JSON file."])
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "source_semantics_diagnostic.json"
    txt_path = output_dir / "source_semantics_diagnostic.txt"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"JSON: {json_path}\nTXT: {txt_path}")


if __name__ == "__main__":
    main()
