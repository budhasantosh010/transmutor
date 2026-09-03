from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated, v837_capacity_criterion_sha256
from experiments.v837_primitive_invention.common.metrics import binary_summary, bootstrap_mean_ci, continuous_summary
from experiments.v837_primitive_invention.common.reference_models import build_reference_model
from experiments.v837_primitive_invention.common.reference_training import matched_budget_signature, train_sequence_model
from experiments.v837_primitive_invention.common.seeds import deterministic_int, gate_sha256
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
EXPECTED_GATE_HASH = "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867"
MODEL_NAMES = ["neutral_high_capacity", "gru_reference", "residual_rnn_reference", "vanilla_rnn_reference"]
FAMILIES = [task.name for task in all_tasks()]


def _configure_torch() -> None:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _neutral_factory(replicate: int):
    graph = high_capacity_generic_graph(replicate)
    return graph, lambda: NeutralGraphModel(graph, obs_dim=6, state_dim=4, message_dim=4)


def _reference_factory(model_name: str):
    selection = CONFIG["reference_hidden_size_selection"][model_name]
    hidden_size = int(selection["hidden_size"])
    return lambda: build_reference_model(model_name, hidden_size, 6)


def _compatibility_worker(family: str, restart: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    graph = high_capacity_generic_graph(restart)
    run_seed = deterministic_int("blocker-reference-train", family, restart)
    initialization_seed = deterministic_int("train", graph.graph_id, task.name, run_seed)
    result = train_sequence_model(
        model_factory=lambda: NeutralGraphModel(graph, obs_dim=6, state_dim=4, message_dim=4),
        task=task,
        train_seeds=list(range(30000, 30128)),
        validation_seeds=list(range(30200, 30328)),
        initialization_seed=initialization_seed,
        steps=192,
        learning_rate=0.005,
        weight_decay=0.0001,
        gradient_clip=5.0,
        curve_steps=(0, 192),
    )
    return {
        "family": family,
        "restart": int(restart),
        "development_success": result.development.success_rate,
        "validation_success": result.validation.success_rate,
        "development_loss": result.development.loss,
        "validation_loss": result.validation.loss,
        "initialization_seed": int(initialization_seed),
        "resources": result.resources.to_dict(),
    }


def _primary_worker(model_name: str, family: str, replicate: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    base_seed = deterministic_int("v837j-primary-init", family, replicate)
    if model_name == "neutral_high_capacity":
        graph, factory = _neutral_factory(replicate)
        graph_id = graph.graph_id
    else:
        factory = _reference_factory(model_name)
        graph_id = None
    training = CONFIG["primary_training"]
    train_seeds = list(range(training["development_seed_range"][0], training["development_seed_range"][1] + 1))
    validation_seeds = list(range(training["validation_seed_range"][0], training["validation_seed_range"][1] + 1))
    result = train_sequence_model(
        model_factory=factory,
        task=task,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        initialization_seed=base_seed,
        steps=int(training["steps"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        curve_steps=tuple(training["curve_steps"]),
    )
    return {
        "model": model_name,
        "family": family,
        "replicate": int(replicate),
        "initialization_seed": int(base_seed),
        "graph_id": graph_id,
        "development_success": result.development.success_rate,
        "validation_success": result.validation.success_rate,
        "development_loss": result.development.loss,
        "validation_loss": result.validation.loss,
        "capacity_demonstrated": capacity_demonstrated(result.development.success_rate, result.validation.success_rate),
        "learning_curve": result.learning_curve,
        "resources": result.resources.to_dict(),
        "train_seed_first": train_seeds[0],
        "train_seed_last": train_seeds[-1],
        "validation_seed_first": validation_seeds[0],
        "validation_seed_last": validation_seeds[-1],
        "task_family_label_in_model_input": False,
        "gpu_seconds": 0.0,
    }


def _run_compatibility_probe() -> dict:
    preserved = json.loads((ROOT / "experiments/v837_primitive_invention/failures/blocker_diagnostic_results.json").read_text(encoding="utf-8"))
    preserved_map = {(row["family"], int(row["restart"])): row for row in preserved["rows"]}
    rows = []
    jobs = [(family, restart) for family in FAMILIES for restart in range(3)]
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_compatibility_worker, family, restart): (family, restart) for family, restart in jobs}
        for future in as_completed(futures):
            row = future.result()
            old = preserved_map[(row["family"], row["restart"])]
            row["historical_validation_success"] = float(old["validation_success"])
            row["absolute_validation_drift"] = abs(float(row["validation_success"]) - float(old["validation_success"]))
            rows.append(row)
            print(f"compat {row['family']} r{row['restart']}: val={row['validation_success']:.3f} drift={row['absolute_validation_drift']:.6f}", flush=True)
    rows.sort(key=lambda row: (row["family"], row["restart"]))
    per_family = {}
    families_over = 0
    threshold = float(CONFIG["baseline_compatibility_probe"]["drift_threshold"])
    for family in FAMILIES:
        fr = [row for row in rows if row["family"] == family]
        current = max(float(row["validation_success"]) for row in fr)
        historical = max(float(row["historical_validation_success"]) for row in fr)
        drift = abs(current - historical)
        families_over += int(drift > threshold)
        per_family[family] = {"current_best": current, "historical_best": historical, "absolute_drift": drift}
    compatible = families_over <= int(CONFIG["baseline_compatibility_probe"]["max_families_over_threshold"])
    payload = {
        "status": "PASS" if compatible else "IMPLEMENTATION_FAILURE",
        "compatible": compatible,
        "drift_threshold": threshold,
        "families_over_threshold": families_over,
        "per_family": per_family,
        "rows": rows,
        "fresh_audit_consumed": False,
    }
    write_json(HERE / "diagnostics" / "baseline_compatibility.json", payload)
    return payload


def _summarize_model(model_name: str, rows: list[dict]) -> dict:
    model_rows = [row for row in rows if row["model"] == model_name]
    per_family = {}
    families_passing = 0
    for family in FAMILIES:
        fr = sorted((row for row in model_rows if row["family"] == family), key=lambda row: row["replicate"])
        dev = np.asarray([row["development_success"] for row in fr], dtype=float)
        val = np.asarray([row["validation_success"] for row in fr], dtype=float)
        passes = np.asarray([row["capacity_demonstrated"] for row in fr], dtype=bool)
        aggregate_pass = capacity_demonstrated(float(np.median(dev)), float(np.median(val)))
        families_passing += int(aggregate_pass)
        if float(np.median(dev)) >= 0.90 and float(np.median(val)) < 0.85:
            behavior = "GENERALIZATION_FAILURE"
        elif float(np.median(dev)) < 0.90:
            behavior = "OPTIMIZATION_OR_CAPACITY_FAILURE"
        else:
            behavior = "ADEQUATE"
        per_family[family] = {
            "development": continuous_summary(dev),
            "validation": continuous_summary(val),
            "validation_bootstrap": bootstrap_mean_ci(val, seed=deterministic_int("v837j-bootstrap", model_name, family)),
            "replicate_capacity_rate": binary_summary(passes),
            "aggregate_capacity_pass": aggregate_pass,
            "training_validation_gap_median": float(np.median(dev - val)),
            "behavior": behavior,
        }
    resources = {
        "model_fits": len(model_rows),
        "optimizer_steps": int(sum(row["resources"]["optimizer_steps"] for row in model_rows)),
        "examples_processed": int(sum(row["resources"]["examples_processed"] for row in model_rows)),
        "environment_interactions": int(sum(row["resources"]["environment_steps"] for row in model_rows)),
        "forward_calls": int(sum(row["resources"]["forward_calls"] for row in model_rows)),
        "wall_seconds_sum_workers": float(sum(row["resources"]["wall_seconds"] for row in model_rows)),
        "cpu_seconds_sum_workers": float(sum(row["resources"]["cpu_seconds"] for row in model_rows)),
        "gpu_seconds": 0.0,
        "parameter_count": int(model_rows[0]["resources"]["parameter_count"]),
        "parameter_bytes": int(model_rows[0]["resources"]["model_parameter_bytes"]),
    }
    return {
        "parameter_count": resources["parameter_count"],
        "parameter_bytes": resources["parameter_bytes"],
        "families_passing": int(families_passing),
        "family_results": per_family,
        "resource_accounting": resources,
    }


def _learning_curve_summary(rows: list[dict]) -> dict:
    output = {}
    for model_name in MODEL_NAMES:
        output[model_name] = {}
        for family in FAMILIES:
            fr = [row for row in rows if row["model"] == model_name and row["family"] == family]
            steps = sorted({int(point["step"]) for row in fr for point in row["learning_curve"]})
            points = []
            for step in steps:
                at = [next(point for point in row["learning_curve"] if int(point["step"]) == step) for row in fr]
                points.append({
                    "step": step,
                    "training_loss_median": float(np.median([point["training_loss"] for point in at])),
                    "training_success_median": float(np.median([point["training_success"] for point in at])),
                    "validation_loss_median": float(np.median([point["validation_loss"] for point in at])),
                    "validation_success_median": float(np.median([point["validation_success"] for point in at])),
                    "gradient_norm_median": float(np.median([point["gradient_norm"] for point in at])),
                    "state_norm_median": float(np.median([point["state_norm"] for point in at])),
                    "activation_saturation_median": float(np.median([point["activation_saturation"] for point in at])),
                })
            output[model_name][family] = points
    return output


def _diagnose(models: dict) -> tuple[str, bool, list[str], str]:
    neutral = int(models["neutral_high_capacity"]["families_passing"])
    gru = int(models["gru_reference"]["families_passing"])
    residual = int(models["residual_rnn_reference"]["families_passing"])
    vanilla = int(models["vanilla_rnn_reference"]["families_passing"])
    if neutral >= 4:
        return ("NEUTRAL_BASELINE_RECOVERY_CONFOUND", False, ["BENCHMARK_LEARNABILITY_UNRESOLVED"], "Stop: the matched neutral baseline itself now reaches >=4/5; reconcile this with the historical lineage before further calibration.")
    if gru >= 4 and residual >= 4:
        return ("REPRESENTATION_FAMILY_FAILURE_STRENGTHENED", True, [], "Matched GRU and residual recurrent references both reach >=4/5; V837m is justified as the next isolated Transmutor cell-law diagnostic.")
    if gru >= 4 or residual >= 4:
        return ("REPRESENTATION_FAMILY_FAILURE_STRENGTHENED", True, [], "At least one parameter-matched conventional recurrent reference reaches >=4/5; benchmark learnability is established, with architecture-specific differences retained for the next mechanism choice.")
    if vanilla >= 4:
        return ("REPRESENTATION_FAMILY_FAILURE_STRENGTHENED", True, [], "The optional dense vanilla tanh RNN reaches >=4/5 while the neutral graph does not; dense recurrent organization is sufficient under the matched regime.")
    return ("BENCHMARK_LEARNABILITY_UNRESOLVED", False, ["REFERENCE_MODEL_FAILURE", "BENCHMARK_LEARNABILITY_UNRESOLVED"], "GRU and residual references both remain below 4/5 at matched budget; run V837k with optimizer-step budget as the only changed variable.")


def main() -> int:
    if gate_sha256() != EXPECTED_GATE_HASH:
        raise SystemExit("frozen V837 gate hash mismatch")
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise SystemExit("capacity criterion fingerprint mismatch")
    compatibility = _run_compatibility_probe()
    if not compatibility["compatible"]:
        raise SystemExit("IMPLEMENTATION_FAILURE: neutral baseline compatibility drift exceeded frozen tolerance")

    jobs = [(model, family, replicate) for model in MODEL_NAMES for family in FAMILIES for replicate in range(int(CONFIG["primary_training"]["replicates"]))]
    rows = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_primary_worker, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{row['model']} {row['family']} r{row['replicate']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}", flush=True)
    rows.sort(key=lambda row: (row["model"], row["family"], row["replicate"]))
    write_json(HERE / "diagnostics" / "raw_runs.json", {"rows": rows, "fresh_audit_consumed": False})

    models = {name: _summarize_model(name, rows) for name in MODEL_NAMES}
    curves = _learning_curve_summary(rows)
    write_json(HERE / "diagnostics" / "learning_curve_summary.json", curves)
    diagnosis, passed, failure_classes, next_experiment = _diagnose(models)
    training = CONFIG["primary_training"]
    budget = matched_budget_signature(
        optimizer=training["optimizer"], optimizer_steps=training["steps"], train_episodes=training["train_episodes"],
        validation_episodes=training["validation_episodes"], learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"], gradient_clip=training["gradient_clip"],
    )
    compute = {
        "primary": {name: models[name]["resource_accounting"] for name in MODEL_NAMES},
        "totals": {
            "model_fits": int(sum(models[name]["resource_accounting"]["model_fits"] for name in MODEL_NAMES)),
            "optimizer_steps": int(sum(models[name]["resource_accounting"]["optimizer_steps"] for name in MODEL_NAMES)),
            "examples_processed": int(sum(models[name]["resource_accounting"]["examples_processed"] for name in MODEL_NAMES)),
            "environment_interactions": int(sum(models[name]["resource_accounting"]["environment_interactions"] for name in MODEL_NAMES)),
            "forward_calls": int(sum(models[name]["resource_accounting"]["forward_calls"] for name in MODEL_NAMES)),
            "wall_seconds_sum_workers": float(sum(models[name]["resource_accounting"]["wall_seconds_sum_workers"] for name in MODEL_NAMES)),
            "cpu_seconds_sum_workers": float(sum(models[name]["resource_accounting"]["cpu_seconds_sum_workers"] for name in MODEL_NAMES)),
            "gpu_seconds": 0.0,
        },
        "matched_budget_signature": budget,
    }
    payload = {
        "version": "V837j",
        "parent": "V837h",
        "single_change": "replace neutral cell representation with diagnostic learned reference models while keeping task/data/training criteria matched",
        "historical_gate_hash": EXPECTED_GATE_HASH,
        "capacity_criterion_hash": CONFIG["capacity_criterion_hash"],
        "fresh_audit_consumed": False,
        "primitive_mining_allowed": False,
        "task_family_label_allowed": False,
        "baseline_compatibility": compatibility,
        "matching_check": {
            "same_task_generators": True,
            "same_primary_training_seeds": True,
            "same_primary_validation_episodes": True,
            "same_optimizer_steps": True,
            "same_examples_processed_per_fit": True,
            "same_optimizer": True,
            "same_learning_rate": True,
            "same_weight_decay": True,
            "same_gradient_clip": True,
            "parameter_target": CONFIG["parameter_target"],
            "parameter_selections": CONFIG["reference_hidden_size_selection"],
        },
        "models": models,
        "learning_curve_summary": curves,
        "compute_accounting": compute,
        "diagnosis": diagnosis,
        "pass": bool(passed),
        "failure_classification": failure_classes,
        "next_experiment": next_experiment,
    }
    write_json(HERE / "results.json", payload)
    doc = HERE / ("PASS.md" if passed else "FAILURE.md")
    lines = [
        f"# V837j {'DIAGNOSTIC PASS' if passed else 'FAILURE'}",
        "",
        "This is a learned-reference calibration result, not a Transmutor primitive-invention PASS.",
        "",
        f"Diagnosis: **{diagnosis}**.",
        "",
        "Families passing aggregate capacity criterion:",
    ]
    for name in MODEL_NAMES:
        lines.append(f"- {name}: {models[name]['families_passing']}/5 ({models[name]['parameter_count']} parameters)")
    lines += ["", next_experiment, "", "Fresh-audit episodes consumed: 0. Primitives promoted: 0. Primitive mining remains blocked."]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
