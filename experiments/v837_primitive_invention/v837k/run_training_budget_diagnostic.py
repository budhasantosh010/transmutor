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
from experiments.v837_primitive_invention.common.reference_training import train_sequence_model
from experiments.v837_primitive_invention.common.seeds import deterministic_int, gate_sha256
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
J_RESULTS = ROOT / "experiments/v837_primitive_invention/v837j/results.json"
EXPECTED_GATE_HASH = "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867"
MODELS = ["neutral_high_capacity", "gru_reference", "residual_rnn_reference"]
LEARNED_MODELS = ["gru_reference", "residual_rnn_reference"]
FAMILIES = [task.name for task in all_tasks()]


def _configure_torch() -> None:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _factory(model_name: str, replicate: int):
    if model_name == "neutral_high_capacity":
        graph = high_capacity_generic_graph(replicate)
        return lambda: NeutralGraphModel(graph, obs_dim=6, state_dim=4, message_dim=4)
    selection = CONFIG["hidden_size_selection"][model_name]
    return lambda: build_reference_model(model_name, int(selection["hidden_size"]), 6)


def _curve_steps(total_steps: int) -> tuple[int, ...]:
    base = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192]
    for value in (256, 320, 384, 512, 640, 768):
        if value <= total_steps:
            base.append(value)
    base.append(total_steps)
    return tuple(sorted(set(base)))


def _worker(model_name: str, family: str, replicate: int, multiplier: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    training = CONFIG["base_training"]
    train_seeds = list(range(training["development_seed_range"][0], training["development_seed_range"][1] + 1))
    validation_seeds = list(range(training["validation_seed_range"][0], training["validation_seed_range"][1] + 1))
    steps = int(training["steps"]) * int(multiplier)
    initialization_seed = deterministic_int("v837j-primary-init", family, replicate)
    result = train_sequence_model(
        model_factory=_factory(model_name, replicate),
        task=task,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        initialization_seed=initialization_seed,
        steps=steps,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        curve_steps=_curve_steps(steps),
    )
    return {
        "model": model_name,
        "family": family,
        "replicate": int(replicate),
        "step_multiplier": int(multiplier),
        "optimizer_steps_per_fit": int(steps),
        "initialization_seed": int(initialization_seed),
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
        "fresh_audit_consumed": False,
        "gpu_seconds": 0.0,
    }


def _summarize_rows(rows: list[dict]) -> dict:
    output: dict[str, dict] = {}
    for model_name in MODELS:
        mr = [row for row in rows if row["model"] == model_name]
        per_family = {}
        families_passing = 0
        for family in FAMILIES:
            fr = sorted((row for row in mr if row["family"] == family), key=lambda row: row["replicate"])
            dev = np.asarray([row["development_success"] for row in fr], dtype=float)
            val = np.asarray([row["validation_success"] for row in fr], dtype=float)
            passes = np.asarray([row["capacity_demonstrated"] for row in fr], dtype=bool)
            aggregate_pass = capacity_demonstrated(float(np.median(dev)), float(np.median(val)))
            families_passing += int(aggregate_pass)
            behavior = "ADEQUATE"
            if float(np.median(dev)) >= 0.90 and float(np.median(val)) < 0.85:
                behavior = "GENERALIZATION_FAILURE"
            elif float(np.median(dev)) < 0.90:
                behavior = "OPTIMIZATION_OR_CAPACITY_FAILURE"
            per_family[family] = {
                "development": continuous_summary(dev),
                "validation": continuous_summary(val),
                "validation_bootstrap": bootstrap_mean_ci(val, seed=deterministic_int("v837k-bootstrap", model_name, family, rows[0]["step_multiplier"])),
                "replicate_capacity_rate": binary_summary(passes),
                "aggregate_capacity_pass": aggregate_pass,
                "training_validation_gap_median": float(np.median(dev - val)),
                "behavior": behavior,
            }
        resources = {
            "model_fits": len(mr),
            "optimizer_steps": int(sum(row["resources"]["optimizer_steps"] for row in mr)),
            "examples_processed": int(sum(row["resources"]["examples_processed"] for row in mr)),
            "environment_interactions": int(sum(row["resources"]["environment_steps"] for row in mr)),
            "forward_calls": int(sum(row["resources"]["forward_calls"] for row in mr)),
            "wall_seconds_sum_workers": float(sum(row["resources"]["wall_seconds"] for row in mr)),
            "cpu_seconds_sum_workers": float(sum(row["resources"]["cpu_seconds"] for row in mr)),
            "gpu_seconds": 0.0,
            "parameter_count": int(mr[0]["resources"]["parameter_count"]),
            "parameter_bytes": int(mr[0]["resources"]["model_parameter_bytes"]),
        }
        output[model_name] = {
            "parameter_count": resources["parameter_count"],
            "parameter_bytes": resources["parameter_bytes"],
            "families_passing": int(families_passing),
            "family_results": per_family,
            "resource_accounting": resources,
        }
    return output


def _learning_curve_summary(rows: list[dict]) -> dict:
    output = {}
    for model_name in MODELS:
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


def _run_multiplier(multiplier: int) -> tuple[list[dict], dict]:
    jobs = [
        (model, family, replicate, multiplier)
        for model in MODELS
        for family in FAMILIES
        for replicate in range(int(CONFIG["base_training"]["replicates"]))
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{multiplier}x {row['model']} {row['family']} r{row['replicate']}: "
                f"dev={row['development_success']:.3f} val={row['validation_success']:.3f}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["model"], row["family"], row["replicate"]))
    summary = _summarize_rows(rows)
    return rows, summary


def _learned_pass(summary: dict) -> bool:
    return any(int(summary[name]["families_passing"]) >= 4 for name in LEARNED_MODELS)


def main() -> int:
    if gate_sha256() != EXPECTED_GATE_HASH:
        raise SystemExit("frozen V837 gate hash mismatch")
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise SystemExit("capacity criterion fingerprint mismatch")
    j = json.loads(J_RESULTS.read_text(encoding="utf-8"))
    if j.get("diagnosis") != "BENCHMARK_LEARNABILITY_UNRESOLVED":
        raise SystemExit("V837k is only justified after unresolved V837j learned-reference calibration")

    conditions: dict[str, dict] = {
        "1x": {
            "source": "preserved V837j matched-budget result",
            "optimizer_steps_per_fit": int(CONFIG["base_training"]["steps"]),
            "models": {name: j["models"][name] for name in MODELS},
        }
    }
    all_new_rows: list[dict] = []
    executed = []
    for multiplier in (2, 4):
        rows, summary = _run_multiplier(multiplier)
        all_new_rows.extend(rows)
        executed.append(multiplier)
        conditions[f"{multiplier}x"] = {
            "source": "V837k executed diagnostic",
            "optimizer_steps_per_fit": int(CONFIG["base_training"]["steps"]) * multiplier,
            "models": summary,
            "learning_curve_summary": _learning_curve_summary(rows),
        }
        write_json(
            HERE / "diagnostics" / f"raw_runs_{multiplier}x.json",
            {"rows": rows, "fresh_audit_consumed": False, "single_change": "optimizer_steps"},
        )
        print(
            f"{multiplier}x families passing: "
            + ", ".join(f"{name}={summary[name]['families_passing']}/5" for name in MODELS),
            flush=True,
        )
        if _learned_pass(summary):
            break

    winning_multiplier = None
    winning_model = None
    for multiplier in (2, 4):
        condition = conditions.get(f"{multiplier}x")
        if condition is None:
            continue
        for model_name in LEARNED_MODELS:
            if int(condition["models"][model_name]["families_passing"]) >= 4:
                winning_multiplier = multiplier
                winning_model = model_name
                break
        if winning_model:
            break

    if winning_model is not None:
        diagnosis = "OPTIMIZATION_BUDGET_FAILURE"
        passed = True
        failure_classes: list[str] = []
        next_experiment = (
            f"{winning_model} reaches >=4/5 at {winning_multiplier}x optimizer steps. "
            "Benchmark learnability is established only with increased optimization; V837m may be considered with this training-efficiency confound documented."
        )
    else:
        diagnosis = "BENCHMARK_LEARNABILITY_UNRESOLVED"
        passed = False
        failure_classes = ["REFERENCE_MODEL_FAILURE", "BENCHMARK_LEARNABILITY_UNRESOLVED"]
        next_experiment = (
            "Neither GRU nor residual RNN reaches >=4/5 through the frozen 4x optimizer-step budget. "
            "Proceed to V837l, changing unique development data as the next single variable."
        )

    totals = {
        "model_fits": int(sum(row["resources"]["model_fits"] for row in all_new_rows)),
        "optimizer_steps": int(sum(row["resources"]["optimizer_steps"] for row in all_new_rows)),
        "examples_processed": int(sum(row["resources"]["examples_processed"] for row in all_new_rows)),
        "environment_interactions": int(sum(row["resources"]["environment_steps"] for row in all_new_rows)),
        "forward_calls": int(sum(row["resources"]["forward_calls"] for row in all_new_rows)),
        "wall_seconds_sum_workers": float(sum(row["resources"]["wall_seconds"] for row in all_new_rows)),
        "cpu_seconds_sum_workers": float(sum(row["resources"]["cpu_seconds"] for row in all_new_rows)),
        "gpu_seconds": 0.0,
    }
    payload = {
        "version": "V837k",
        "parent": "V837j",
        "single_change": CONFIG["single_change"],
        "historical_gate_hash": EXPECTED_GATE_HASH,
        "capacity_criterion_hash": CONFIG["capacity_criterion_hash"],
        "fresh_audit_consumed": False,
        "primitive_mining_allowed": False,
        "task_family_label_allowed": False,
        "matching_check": {
            "same_architectures_as_v837j": True,
            "same_hidden_sizes": True,
            "same_parameter_counts": True,
            "same_task_generators": True,
            "same_training_episode_ids": True,
            "same_validation_episode_ids": True,
            "same_unique_data_amount": True,
            "same_optimizer": True,
            "same_learning_rate": True,
            "same_weight_decay": True,
            "same_gradient_clip": True,
            "same_initialization_seeds": True,
            "only_changed_variable": "optimizer_steps",
        },
        "conditions": conditions,
        "executed_multipliers": executed,
        "compute_accounting": {"new_v837k_runs_only": totals},
        "diagnosis": diagnosis,
        "pass": bool(passed),
        "failure_classification": failure_classes,
        "next_experiment": next_experiment,
    }
    write_json(HERE / "results.json", payload)
    write_json(
        HERE / "diagnostics" / "analysis_summary.json",
        {
            "diagnosis": diagnosis,
            "winning_multiplier": winning_multiplier,
            "winning_model": winning_model,
            "executed_multipliers": executed,
            "families_passing": {
                name: {condition: int(data["models"][name]["families_passing"]) for condition, data in conditions.items()}
                for name in MODELS
            },
            "fresh_audit_consumed": False,
        },
    )
    doc = HERE / ("PASS.md" if passed else "FAILURE.md")
    doc.write_text(
        "# V837k " + ("DIAGNOSTIC PASS" if passed else "FAILURE") + "\n\n"
        + "This is a training-budget calibration result, not a Transmutor primitive-invention PASS.\n\n"
        + f"Diagnosis: **{diagnosis}**.\n\n"
        + "Only optimizer-step budget changed relative to V837j; architecture, data, seeds, optimizer family/hyperparameters, tasks, and capacity gate remained frozen.\n\n"
        + next_experiment + "\n\nFresh-audit episodes consumed: 0. Primitives promoted: 0. Primitive mining remains blocked.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
