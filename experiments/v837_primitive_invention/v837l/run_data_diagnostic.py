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
MODEL_NAMES = CONFIG["models"]
FAMILIES = [task.name for task in all_tasks()]
EXPECTED_GATE_HASH = CONFIG["historical_gate_hash"]


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


def _worker(multiplier: int, model_name: str, family: str, replicate: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    training = CONFIG["base_training"]
    n_train = int(training["train_episodes"]) * int(multiplier)
    train_start = int(training["development_seed_start"])
    train_seeds = list(range(train_start, train_start + n_train))
    validation_seeds = list(range(int(training["validation_seed_range"][0]), int(training["validation_seed_range"][1]) + 1))
    initialization_seed = deterministic_int("v837j-primary-init", family, replicate)
    result = train_sequence_model(
        model_factory=_factory(model_name, replicate),
        task=task,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        initialization_seed=initialization_seed,
        steps=int(training["steps"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        curve_steps=tuple(training["curve_steps"]),
    )
    return {
        "data_multiplier": int(multiplier),
        "model": model_name,
        "family": family,
        "replicate": int(replicate),
        "initialization_seed": int(initialization_seed),
        "train_episodes": len(train_seeds),
        "train_seed_first": train_seeds[0],
        "train_seed_last": train_seeds[-1],
        "validation_seed_first": validation_seeds[0],
        "validation_seed_last": validation_seeds[-1],
        "development_success": result.development.success_rate,
        "validation_success": result.validation.success_rate,
        "development_loss": result.development.loss,
        "validation_loss": result.validation.loss,
        "capacity_demonstrated": capacity_demonstrated(result.development.success_rate, result.validation.success_rate),
        "learning_curve": result.learning_curve,
        "resources": result.resources.to_dict(),
        "task_family_label_in_model_input": False,
        "fresh_audit_consumed": False,
        "gpu_seconds": 0.0,
    }


def _summarize(model_name: str, rows: list[dict]) -> dict:
    selected = [row for row in rows if row["model"] == model_name]
    per_family = {}
    families_passing = 0
    for family in FAMILIES:
        fr = sorted((row for row in selected if row["family"] == family), key=lambda row: row["replicate"])
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
            "validation_bootstrap": bootstrap_mean_ci(val, seed=deterministic_int("v837l-bootstrap", model_name, family, selected[0]["data_multiplier"])),
            "replicate_capacity_rate": binary_summary(passes),
            "aggregate_capacity_pass": aggregate_pass,
            "training_validation_gap_median": float(np.median(dev - val)),
            "behavior": behavior,
        }
    return {
        "parameter_count": int(selected[0]["resources"]["parameter_count"]),
        "families_passing": int(families_passing),
        "family_results": per_family,
        "resource_accounting": {
            "model_fits": len(selected),
            "optimizer_steps": int(sum(row["resources"]["optimizer_steps"] for row in selected)),
            "examples_processed": int(sum(row["resources"]["examples_processed"] for row in selected)),
            "environment_interactions": int(sum(row["resources"]["environment_steps"] for row in selected)),
            "forward_calls": int(sum(row["resources"]["forward_calls"] for row in selected)),
            "wall_seconds_sum_workers": float(sum(row["resources"]["wall_seconds"] for row in selected)),
            "cpu_seconds_sum_workers": float(sum(row["resources"]["cpu_seconds"] for row in selected)),
            "gpu_seconds": 0.0,
        },
    }


def _reuse_one_x() -> dict:
    v837j = json.loads((ROOT / "experiments/v837_primitive_invention/v837j/results.json").read_text(encoding="utf-8"))
    return {name: v837j["models"][name] for name in MODEL_NAMES}


def _run_multiplier(multiplier: int) -> tuple[list[dict], dict]:
    jobs = [(model, family, replicate) for model in MODEL_NAMES for family in FAMILIES for replicate in range(int(CONFIG["base_training"]["replicates"]))]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, multiplier, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{multiplier}x {row['model']} {row['family']} r{row['replicate']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}", flush=True)
    rows.sort(key=lambda row: (row["model"], row["family"], row["replicate"]))
    write_json(HERE / "diagnostics" / f"raw_runs_{multiplier}x.json", {"rows": rows, "fresh_audit_consumed": False})
    summary = {model: _summarize(model, rows) for model in MODEL_NAMES}
    return rows, summary


def _learned_resolved(summary: dict) -> bool:
    return any(int(summary[name]["families_passing"]) >= 4 for name in ("gru_reference", "residual_rnn_reference"))


def main() -> int:
    if gate_sha256() != EXPECTED_GATE_HASH:
        raise SystemExit("frozen V837 gate hash mismatch")
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise SystemExit("capacity criterion fingerprint mismatch")

    conditions: dict[str, dict] = {"1x": _reuse_one_x()}
    all_new_rows: list[dict] = []
    executed: list[int] = []
    for multiplier in (2, 4):
        rows, summary = _run_multiplier(multiplier)
        executed.append(multiplier)
        all_new_rows.extend(rows)
        conditions[f"{multiplier}x"] = summary
        print(f"{multiplier}x families passing: " + ", ".join(f"{name}={summary[name]['families_passing']}/5" for name in MODEL_NAMES), flush=True)
        if _learned_resolved(summary):
            break

    resolved_at = next((multiplier for multiplier in (2, 4) if f"{multiplier}x" in conditions and _learned_resolved(conditions[f"{multiplier}x"])), None)
    if resolved_at is not None:
        diagnosis = "SAMPLE_EFFICIENCY_FAILURE"
        passed = True
        classes = ["SAMPLE_EFFICIENCY_FAILURE"]
        next_experiment = "A conventional learned recurrent reference reaches >=4/5 after increasing only unique development data. Benchmark learnability is established under a calibrated data regime; V837m is scientifically justified, with sample efficiency retained as a confound relative to the original 1x regime."
    else:
        diagnosis = "BENCHMARK_LEARNABILITY_UNRESOLVED"
        passed = False
        classes = ["REFERENCE_MODEL_FAILURE", "BENCHMARK_LEARNABILITY_UNRESOLVED"]
        next_experiment = "Neither learned reference reaches >=4/5 through 4x unique development data at fixed optimizer steps. Run one parameter-capacity escalation diagnostic before changing the Transmutor cell law."

    totals = {
        "model_fits": len(all_new_rows),
        "optimizer_steps": int(sum(row["resources"]["optimizer_steps"] for row in all_new_rows)),
        "examples_processed": int(sum(row["resources"]["examples_processed"] for row in all_new_rows)),
        "environment_interactions": int(sum(row["resources"]["environment_steps"] for row in all_new_rows)),
        "forward_calls": int(sum(row["resources"]["forward_calls"] for row in all_new_rows)),
        "wall_seconds_sum_workers": float(sum(row["resources"]["wall_seconds"] for row in all_new_rows)),
        "cpu_seconds_sum_workers": float(sum(row["resources"]["cpu_seconds"] for row in all_new_rows)),
        "gpu_seconds": 0.0,
    }
    payload = {
        "version": "V837l",
        "parent": "V837k",
        "single_change": CONFIG["single_change"],
        "historical_gate_hash": EXPECTED_GATE_HASH,
        "capacity_criterion_hash": CONFIG["capacity_criterion_hash"],
        "fresh_audit_consumed": False,
        "primitive_mining_allowed": False,
        "task_family_label_allowed": False,
        "fixed_optimizer_steps": int(CONFIG["base_training"]["steps"]),
        "fixed_validation_seed_range": CONFIG["base_training"]["validation_seed_range"],
        "executed_data_multipliers": executed,
        "conditions": conditions,
        "resolved_at_data_multiplier": resolved_at,
        "compute_accounting": {"new_v837l_runs_only": totals},
        "diagnosis": diagnosis,
        "pass": passed,
        "failure_classification": classes,
        "next_experiment": next_experiment,
    }
    write_json(HERE / "results.json", payload)
    doc = HERE / ("PASS.md" if passed else "FAILURE.md")
    lines = [f"# V837l {'DIAGNOSTIC PASS' if passed else 'FAILURE'}", "", "This is a sample-efficiency calibration result, not a Transmutor primitive-invention PASS.", "", f"Diagnosis: **{diagnosis}**.", ""]
    for label, condition in conditions.items():
        lines.append(f"## {label} unique development data")
        for model in MODEL_NAMES:
            lines.append(f"- {model}: {condition[model]['families_passing']}/5")
        lines.append("")
    lines += [next_experiment, "", "Fresh-audit episodes consumed: 0. Primitives promoted: 0. Primitive mining remains blocked."]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
