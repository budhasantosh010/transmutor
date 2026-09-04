from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated, v837_capacity_criterion_sha256
from experiments.v837_primitive_invention.common.metrics import binary_summary, bootstrap_mean_ci, continuous_summary
from experiments.v837_primitive_invention.common.reference_training import train_sequence_model
from experiments.v837_primitive_invention.common.seeds import deterministic_int, gate_sha256
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import (
    _active_mask,
    _binary_entropy,
    _configure_torch,
    _lag1_autocorrelation,
    _safe_quantiles,
    _success_rate,
    _temporal_variance,
)
from experiments.v837_primitive_invention.v837o.factorial_gru import (
    CONDITION_FACTORS,
    FactorialGRUReferenceModel,
)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
CONDITIONS = list(CONFIG["conditions"])
FULL = "G0_full_dynamic"
BOTH_OFF = "G9_no_update_no_reset"


def _static_stats(values: torch.Tensor | None) -> dict | None:
    if values is None:
        return None
    arr = values.detach().cpu().numpy().astype(float).reshape(-1)
    return {
        "values": [float(x) for x in arr.tolist()],
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "inter_dimension_variance": float(np.var(arr)),
    }


def _flatten_override(trace_tensor: torch.Tensor) -> torch.Tensor:
    mean = trace_tensor.mean(dim=2, keepdim=True)
    return mean.expand_as(trace_tensor)


def _diagnostics(model: FactorialGRUReferenceModel, task, validation_seeds: list[int]) -> tuple[dict, int]:
    episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        predictions, trace = model(observations, lengths, return_trace=True)
    mask = _active_mask(lengths, trace.states.shape[1])
    updates = trace.updates[mask]
    resets = trace.resets[mask]
    states = trace.states[mask]
    candidates = trace.candidates[mask]
    update_input = trace.update_input_components[mask]
    update_state = trace.update_state_components[mask]
    reset_input = trace.reset_input_components[mask]
    reset_state = trace.reset_state_components[mask]

    diagnostics = {
        "validation_success_recomputed": _success_rate(task, predictions, targets),
        "factor_levels": {"update": model.update_level, "reset": model.reset_level},
        "update": {
            **_safe_quantiles(updates.detach().cpu().numpy().reshape(-1)),
            "entropy": _binary_entropy(updates),
            "temporal_variance": _temporal_variance(trace.updates, lengths),
            "input_conditioned_component_variance": float(update_input.var(unbiased=False).item()),
            "state_conditioned_component_variance": float(update_state.var(unbiased=False).item()),
        },
        "reset": {
            **_safe_quantiles(resets.detach().cpu().numpy().reshape(-1)),
            "entropy": _binary_entropy(resets),
            "temporal_variance": _temporal_variance(trace.resets, lengths),
            "input_conditioned_component_variance": float(reset_input.var(unbiased=False).item()),
            "state_conditioned_component_variance": float(reset_state.var(unbiased=False).item()),
        },
        "static_parameters": {
            "update": _static_stats(model.static_coefficient("update")),
            "reset": _static_stats(model.static_coefficient("reset")),
        },
        "hidden_state_norm": float(torch.linalg.vector_norm(states, dim=-1).mean().item()),
        "hidden_state_autocorrelation": _lag1_autocorrelation(trace.states, lengths),
        "candidate_state_norm": float(torch.linalg.vector_norm(candidates, dim=-1).mean().item()),
        "state_saturation_fraction": float((torch.abs(states) >= 0.95).float().mean().item()),
        "candidate_saturation_fraction": float((torch.abs(candidates) >= 0.95).float().mean().item()),
    }
    forward_calls = 1

    flattening = {}
    kwargs = {}
    if model.update_level == "static_vector":
        kwargs["update_override"] = _flatten_override(trace.updates)
        with torch.no_grad():
            p = model(observations, lengths, **kwargs)
        flattening["update_vector_to_mean"] = _success_rate(task, p, targets)
        forward_calls += 1
    if model.reset_level == "static_vector":
        kwargs = {"reset_override": _flatten_override(trace.resets)}
        with torch.no_grad():
            p = model(observations, lengths, **kwargs)
        flattening["reset_vector_to_mean"] = _success_rate(task, p, targets)
        forward_calls += 1
    if model.update_level == "static_vector" and model.reset_level == "static_vector":
        kwargs = {
            "update_override": _flatten_override(trace.updates),
            "reset_override": _flatten_override(trace.resets),
        }
        with torch.no_grad():
            p = model(observations, lengths, **kwargs)
        flattening["both_vectors_to_means"] = _success_rate(task, p, targets)
        forward_calls += 1
    diagnostics["counterfactual_flattening"] = flattening
    return diagnostics, forward_calls


def _worker(condition: str, family: str, replicate: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    training = CONFIG["training"]
    train_seeds = list(range(int(training["development_seed_range"][0]), int(training["development_seed_range"][1]) + 1))
    validation_seeds = list(range(int(training["validation_seed_range"][0]), int(training["validation_seed_range"][1]) + 1))
    initialization_seed = deterministic_int("v837j-primary-init", family, replicate)

    def factory():
        return FactorialGRUReferenceModel(
            int(CONFIG["hidden_size"]), int(CONFIG["input_dim"]), condition=condition
        )

    result = train_sequence_model(
        model_factory=factory,
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
    diagnostics, extra_forward_calls = _diagnostics(result.model, task, validation_seeds)
    result.resources.forward_calls += int(extra_forward_calls)
    return {
        "condition": condition,
        "update_level": CONDITION_FACTORS[condition][0],
        "reset_level": CONDITION_FACTORS[condition][1],
        "family": family,
        "replicate": int(replicate),
        "initialization_seed": int(initialization_seed),
        "development_success": result.development.success_rate,
        "validation_success": result.validation.success_rate,
        "development_loss": result.development.loss,
        "validation_loss": result.validation.loss,
        "capacity_demonstrated": capacity_demonstrated(result.development.success_rate, result.validation.success_rate),
        "learning_curve": result.learning_curve,
        "diagnostics": diagnostics,
        "nominal_parameter_count": int(result.model.nominal_parameter_count()),
        "active_parameter_count": int(result.model.active_parameter_count()),
        "parameter_bytes": int(result.model.parameter_bytes()),
        "resources": result.resources.to_dict(),
        "task_family_label_in_model_input": False,
        "fresh_audit_consumed": False,
        "gpu_seconds": 0.0,
    }


def summarize_condition(condition: str, rows: list[dict]) -> dict:
    selected = [row for row in rows if row["condition"] == condition]
    if not selected:
        raise ValueError(f"no rows for {condition}")
    per_family = {}
    families_passing = 0
    for family in FAMILIES:
        fr = sorted((row for row in selected if row["family"] == family), key=lambda row: row["replicate"])
        dev = np.asarray([row["development_success"] for row in fr], dtype=float)
        val = np.asarray([row["validation_success"] for row in fr], dtype=float)
        passes = np.asarray([row["capacity_demonstrated"] for row in fr], dtype=bool)
        aggregate = capacity_demonstrated(float(np.median(dev)), float(np.median(val)))
        families_passing += int(aggregate)
        per_family[family] = {
            "development": continuous_summary(dev),
            "validation": continuous_summary(val),
            "validation_bootstrap": bootstrap_mean_ci(val, seed=deterministic_int("v837o-bootstrap", condition, family)),
            "replicate_capacity_rate": binary_summary(passes),
            "aggregate_capacity_pass": bool(aggregate),
        }

    static_summary = {}
    for factor in ("update", "reset"):
        stats = [r["diagnostics"]["static_parameters"][factor] for r in selected]
        stats = [s for s in stats if s is not None]
        if stats:
            static_summary[factor] = {
                "mean": continuous_summary(np.asarray([s["mean"] for s in stats], dtype=float)),
                "std_across_dimensions": continuous_summary(np.asarray([s["std"] for s in stats], dtype=float)),
                "inter_dimension_variance": continuous_summary(np.asarray([s["inter_dimension_variance"] for s in stats], dtype=float)),
            }

    flattening = {}
    for key in ("update_vector_to_mean", "reset_vector_to_mean", "both_vectors_to_means"):
        vals = [r["diagnostics"]["counterfactual_flattening"].get(key) for r in selected]
        vals = [v for v in vals if v is not None]
        if vals:
            flattening[key] = continuous_summary(np.asarray(vals, dtype=float))

    dynamic_diagnostics = {}
    for factor in ("update", "reset"):
        vals = [r["diagnostics"][factor] for r in selected]
        dynamic_diagnostics[factor] = {
            "mean": continuous_summary(np.asarray([v["mean"] for v in vals], dtype=float)),
            "median": continuous_summary(np.asarray([v["median"] for v in vals], dtype=float)),
            "temporal_variance": continuous_summary(np.asarray([v["temporal_variance"] for v in vals], dtype=float)),
            "input_conditioned_variance": continuous_summary(np.asarray([v["input_conditioned_component_variance"] for v in vals], dtype=float)),
            "state_conditioned_variance": continuous_summary(np.asarray([v["state_conditioned_component_variance"] for v in vals], dtype=float)),
            "entropy": continuous_summary(np.asarray([v["entropy"] for v in vals], dtype=float)),
        }

    return {
        "update_level": selected[0]["update_level"],
        "reset_level": selected[0]["reset_level"],
        "nominal_parameter_count": int(selected[0]["nominal_parameter_count"]),
        "active_parameter_count": int(selected[0]["active_parameter_count"]),
        "families_passing": int(families_passing),
        "family_results": per_family,
        "factor_diagnostics": dynamic_diagnostics,
        "static_parameter_diagnostics": static_summary,
        "counterfactual_flattening": flattening,
        "resource_accounting": {
            "model_fits": len(selected),
            "optimizer_steps": int(sum(r["resources"]["optimizer_steps"] for r in selected)),
            "examples_processed": int(sum(r["resources"]["examples_processed"] for r in selected)),
            "environment_interactions": int(sum(r["resources"]["environment_steps"] for r in selected)),
            "forward_calls": int(sum(r["resources"]["forward_calls"] for r in selected)),
            "wall_seconds_sum_workers": float(sum(r["resources"]["wall_seconds"] for r in selected)),
            "cpu_seconds_sum_workers": float(sum(r["resources"]["cpu_seconds"] for r in selected)),
            "gpu_seconds": 0.0,
        },
    }


def _run_conditions(conditions: list[str]) -> list[dict]:
    replicates = int(CONFIG["training"]["replicates"])
    jobs = [(condition, family, replicate) for condition in conditions for family in FAMILIES for replicate in range(replicates)]
    rows = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{row['condition']} {row['family']} r{row['replicate']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}", flush=True)
    rows.sort(key=lambda row: (row["condition"], row["family"], row["replicate"]))
    return rows


def _positive_control(rows: list[dict]) -> dict:
    summary = summarize_condition(FULL, rows)
    historical = json.loads((ROOT / "experiments/v837_primitive_invention/v837n/results.json").read_text(encoding="utf-8"))["conditions"]["full_gru"]
    threshold = float(CONFIG["positive_control_gate"]["compatibility_drift_threshold"])
    over = 0
    drift = {}
    for family in FAMILIES:
        current = float(summary["family_results"][family]["validation"]["median"])
        prior = float(historical["family_results"][family]["validation"]["median"])
        delta = abs(current - prior)
        over += int(delta > threshold)
        drift[family] = {"v837o_median": current, "v837n_median": prior, "absolute_drift": delta}
    compatible = (
        summary["families_passing"] >= int(CONFIG["positive_control_gate"]["families_required"])
        and over <= int(CONFIG["positive_control_gate"]["max_families_over_drift_threshold"])
        and summary["nominal_parameter_count"] == int(CONFIG["reference_parameter_count"])
    )
    return {
        "compatible": bool(compatible),
        "families_passing": int(summary["families_passing"]),
        "parameter_count": int(summary["nominal_parameter_count"]),
        "families_over_drift_threshold": int(over),
        "per_family": drift,
        "summary": summary,
        "fresh_audit_consumed": False,
    }


def _assert_locks() -> None:
    if gate_sha256() != CONFIG["historical_gate_hash"]:
        raise SystemExit("historical V837 gate hash mismatch")
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise SystemExit("capacity criterion fingerprint mismatch")
    audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
    if int(audit.get("episodes_consumed", -1)) != 0:
        raise SystemExit("fresh audit already consumed; V837o cannot proceed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("full", "factorial", "all"), default="all")
    args = parser.parse_args()
    _assert_locks()
    for name in ("raw", "diagnostics", "plots"):
        (HERE / name).mkdir(exist_ok=True)
    training = CONFIG["training"]
    seed_payload = {
        "development_seeds": list(range(training["development_seed_range"][0], training["development_seed_range"][1] + 1)),
        "validation_seeds": list(range(training["validation_seed_range"][0], training["validation_seed_range"][1] + 1)),
        "initialization_seed_policy": "deterministic_int('v837j-primary-init', family, replicate)",
    }

    if args.phase in {"full", "all"}:
        rows = _run_conditions([FULL])
        write_json(HERE / "raw" / "full_gru.json", {"rows": rows, "paired_task_seeds": seed_payload, "fresh_audit_consumed": False})
        positive = _positive_control(rows)
        write_json(HERE / "diagnostics" / "full_gru_positive_control.json", positive)
        print(f"V837o G0: {positive['families_passing']}/5; compatible={positive['compatible']}", flush=True)
        if not positive["compatible"]:
            (HERE / "FAILURE.md").write_text("# V837o IMPLEMENTATION_OR_BASELINE_DRIFT\n\nThe full explicit GRU positive control failed; factorial conditions were not interpreted.\n", encoding="utf-8")
            return 2

    if args.phase in {"factorial", "all"}:
        positive_path = HERE / "diagnostics" / "full_gru_positive_control.json"
        if not positive_path.exists() or json.loads(positive_path.read_text(encoding="utf-8")).get("compatible") is not True:
            raise SystemExit("run successful --phase full before factorial conditions")
        remaining = [c for c in CONDITIONS if c != FULL]
        rows = _run_conditions(remaining)
        write_json(HERE / "raw" / "factorial_runs.json", {"rows": rows, "paired_task_seeds": seed_payload, "fresh_audit_consumed": False})
        summaries = {condition: summarize_condition(condition, rows) for condition in remaining}
        write_json(HERE / "diagnostics" / "factorial_summaries.json", summaries)
        print("families passing: " + ", ".join(f"{c}={summaries[c]['families_passing']}/5" for c in remaining), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
