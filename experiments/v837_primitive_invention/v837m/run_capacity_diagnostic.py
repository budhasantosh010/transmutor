from __future__ import annotations

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
from experiments.v837_primitive_invention.common.metrics import binary_summary, bootstrap_mean_ci, continuous_summary, paired_bootstrap_difference
from experiments.v837_primitive_invention.common.reference_training import train_sequence_model
from experiments.v837_primitive_invention.common.seeds import deterministic_int, gate_sha256
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
CONDITIONS = list(CONFIG["conditions"])
FAMILIES = [task.name for task in all_tasks()]


def _configure_torch() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _model_factory(condition: str, replicate: int):
    graph = high_capacity_generic_graph(replicate)
    spec = CONFIG["conditions"][condition]
    kwargs = {
        "state_update_mode": spec["state_update_mode"],
        "alpha_init": float(spec.get("alpha_init", 0.5)),
        "transport_rho": float(spec.get("transport_rho", 0.95)),
    }
    return graph, lambda: NeutralGraphModel(graph, obs_dim=6, state_dim=4, message_dim=4, **kwargs)


def _post_training_diagnostics(model: NeutralGraphModel, task, validation_seeds: list[int]) -> dict:
    episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, _ = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        _, trace = model(observations, lengths, return_trace=True)
    active_bt = torch.arange(trace.states.shape[1]).view(1, -1) < lengths.view(-1, 1)
    active = active_bt.unsqueeze(-1).unsqueeze(-1).expand_as(trace.states)
    state_values = trace.states[active]
    candidate_values = trace.candidate_states[active]
    state_vectors = trace.states[active_bt].reshape(-1, trace.states.shape[-1])
    state_norms = torch.linalg.vector_norm(state_vectors, dim=-1)
    transport = model.transport_diagnostics()
    return {
        "mean_state_norm": float(state_norms.mean().item()),
        "max_state_norm": float(state_norms.max().item()),
        "exploding_state_fraction_norm_gt_10": float((state_norms > 10.0).float().mean().item()),
        "vanishing_state_fraction_norm_lt_1e_3": float((state_norms < 1e-3).float().mean().item()),
        "state_abs_ge_0_95_fraction": float((torch.abs(state_values) >= 0.95).float().mean().item()),
        "tanh_candidate_saturation_fraction": float((torch.abs(candidate_values) >= 0.95).float().mean().item()),
        "transport_spectral_norm_mean": float(np.mean([row["spectral_norm"] for row in transport])) if transport else None,
        "transport_spectral_radius_mean": float(np.mean([row["spectral_radius"] for row in transport])) if transport else None,
        "transport_spectral_norm_max": float(max((row["spectral_norm"] for row in transport), default=0.0)) if transport else None,
        "alpha_mean": float(np.mean(model.state_update_coefficients())) if model.state_update_mode == "learned_leaky" else None,
    }


def _worker(condition: str, family: str, replicate: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    graph, factory = _model_factory(condition, replicate)
    training = CONFIG["training"]
    train_seeds = list(range(int(training["development_seed_range"][0]), int(training["development_seed_range"][1]) + 1))
    validation_seeds = list(range(int(training["validation_seed_range"][0]), int(training["validation_seed_range"][1]) + 1))
    initialization_seed = deterministic_int("v837m-init", family, replicate)
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
    diagnostics = _post_training_diagnostics(result.model, task, validation_seeds)
    return {
        "condition": condition,
        "family": family,
        "replicate": int(replicate),
        "initialization_seed": int(initialization_seed),
        "graph_id": graph.graph_id,
        "development_success": result.development.success_rate,
        "validation_success": result.validation.success_rate,
        "development_loss": result.development.loss,
        "validation_loss": result.validation.loss,
        "capacity_demonstrated": capacity_demonstrated(result.development.success_rate, result.validation.success_rate),
        "learning_curve": result.learning_curve,
        "stability_diagnostics": diagnostics,
        "resources": result.resources.to_dict(),
        "fresh_audit_consumed": False,
        "gpu_seconds": 0.0,
    }


def _summarize(condition: str, rows: list[dict]) -> dict:
    selected = [row for row in rows if row["condition"] == condition]
    per_family = {}
    families_passing = 0
    for family in FAMILIES:
        fr = sorted((row for row in selected if row["family"] == family), key=lambda row: row["replicate"])
        dev = np.asarray([row["development_success"] for row in fr], dtype=float)
        val = np.asarray([row["validation_success"] for row in fr], dtype=float)
        passes = np.asarray([row["capacity_demonstrated"] for row in fr], dtype=bool)
        aggregate_pass = capacity_demonstrated(float(np.median(dev)), float(np.median(val)))
        families_passing += int(aggregate_pass)
        per_family[family] = {
            "development": continuous_summary(dev),
            "validation": continuous_summary(val),
            "validation_bootstrap": bootstrap_mean_ci(val, seed=deterministic_int("v837m-bootstrap", condition, family)),
            "replicate_capacity_rate": binary_summary(passes),
            "aggregate_capacity_pass": aggregate_pass,
        }
    diag_keys = [
        "mean_state_norm", "max_state_norm", "exploding_state_fraction_norm_gt_10",
        "vanishing_state_fraction_norm_lt_1e_3", "state_abs_ge_0_95_fraction",
        "tanh_candidate_saturation_fraction", "transport_spectral_norm_mean",
        "transport_spectral_radius_mean", "transport_spectral_norm_max", "alpha_mean",
    ]
    diagnostics = {}
    for key in diag_keys:
        values = [row["stability_diagnostics"][key] for row in selected if row["stability_diagnostics"][key] is not None]
        diagnostics[key] = continuous_summary(values) if values else None
    return {
        "parameter_count": int(selected[0]["resources"]["parameter_count"]),
        "families_passing": int(families_passing),
        "family_results": per_family,
        "stability_diagnostics": diagnostics,
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


def _paired(left: str, right: str, rows: list[dict]) -> dict:
    output = {}
    for family in FAMILIES:
        a = sorted((row for row in rows if row["condition"] == left and row["family"] == family), key=lambda row: row["replicate"])
        b = sorted((row for row in rows if row["condition"] == right and row["family"] == family), key=lambda row: row["replicate"])
        output[family] = paired_bootstrap_difference(
            np.asarray([row["validation_success"] for row in a], dtype=float),
            np.asarray([row["validation_success"] for row in b], dtype=float),
            seed=deterministic_int("v837m-paired", left, right, family),
        )
    return output


def main() -> int:
    if gate_sha256() != CONFIG["historical_gate_hash"]:
        raise SystemExit("frozen V837 gate hash mismatch")
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise SystemExit("capacity criterion fingerprint mismatch")

    jobs = [(condition, family, replicate) for condition in CONDITIONS for family in FAMILIES for replicate in range(int(CONFIG["training"]["replicates"]))]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{row['condition']} {row['family']} r{row['replicate']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}", flush=True)
    rows.sort(key=lambda row: (row["condition"], row["family"], row["replicate"]))
    write_json(HERE / "diagnostics" / "raw_runs.json", {"rows": rows, "fresh_audit_consumed": False})

    conditions = {condition: _summarize(condition, rows) for condition in CONDITIONS}
    expected_counts = {name: int(spec["parameter_count_expected"]) for name, spec in CONFIG["conditions"].items()}
    for name, expected in expected_counts.items():
        actual = int(conditions[name]["parameter_count"])
        if actual != expected:
            raise SystemExit(f"parameter count mismatch for {name}: {actual} != {expected}")
    if conditions["linear_transport"]["parameter_count"] != conditions["parameter_matched_additive"]["parameter_count"]:
        raise SystemExit("transport/additive parameter control mismatch")

    linear_pass = int(conditions["linear_transport"]["families_passing"]) >= int(CONFIG["families_required"])
    additive_pass = int(conditions["parameter_matched_additive"]["families_passing"]) >= int(CONFIG["families_required"])
    paired_linear_additive = _paired("linear_transport", "parameter_matched_additive", rows)
    paired_linear_historical = _paired("linear_transport", "historical_direct", rows)

    if linear_pass and not additive_pass:
        diagnosis = "GENERAL_LINEAR_STATE_TRANSPORT_SUPPORTED"
        passed = True
        classes: list[str] = []
        next_experiment = "Freeze the V837m linear-transport cell law and rerun the original neutral structural search under the same calibrated-data policy before any motif mining."
    elif linear_pass and additive_pass:
        diagnosis = "PARAMETER_COUNT_CONFOUND"
        passed = False
        classes = ["PARAMETER_COUNT_CONFOUND"]
        next_experiment = "Both transport and its exactly parameter-matched additive control satisfy the capacity screen; do not claim linear transport mechanism. Isolate the added-capacity effect before structural search."
    else:
        diagnosis = "LINEAR_STATE_TRANSPORT_INSUFFICIENT"
        passed = False
        classes = ["REPRESENTATION_FAMILY_FAILURE_STRENGTHENED"]
        next_experiment = "Stable linear state transport does not recover >=4/5 competence under the calibrated learnable data regime. Use the V837j/l GRU gap to isolate adaptive state-update control next rather than increasing structural-search budget."

    totals = {
        "model_fits": len(rows),
        "optimizer_steps": int(sum(row["resources"]["optimizer_steps"] for row in rows)),
        "examples_processed": int(sum(row["resources"]["examples_processed"] for row in rows)),
        "environment_interactions": int(sum(row["resources"]["environment_steps"] for row in rows)),
        "forward_calls": int(sum(row["resources"]["forward_calls"] for row in rows)),
        "wall_seconds_sum_workers": float(sum(row["resources"]["wall_seconds"] for row in rows)),
        "cpu_seconds_sum_workers": float(sum(row["resources"]["cpu_seconds"] for row in rows)),
        "gpu_seconds": 0.0,
    }
    payload = {
        "version": "V837m",
        "parent": "V837l",
        "single_change": CONFIG["single_change"],
        "calibration_basis": CONFIG["calibration_basis"],
        "historical_gate_hash": CONFIG["historical_gate_hash"],
        "capacity_criterion_hash": CONFIG["capacity_criterion_hash"],
        "fresh_audit_consumed": False,
        "primitive_mining_allowed": False,
        "full_structural_search_allowed": bool(passed),
        "conditions": conditions,
        "parameter_matching": {
            "linear_transport": conditions["linear_transport"]["parameter_count"],
            "parameter_matched_additive": conditions["parameter_matched_additive"]["parameter_count"],
            "exact_match": conditions["linear_transport"]["parameter_count"] == conditions["parameter_matched_additive"]["parameter_count"],
        },
        "paired_validation_delta_linear_minus_additive": paired_linear_additive,
        "paired_validation_delta_linear_minus_historical": paired_linear_historical,
        "resource_accounting": totals,
        "pass_gate": {"families_required": 4, "development_success": 0.90, "validation_success": 0.85, "mechanism_control": "linear transport must not be explained by the parameter-matched additive control"},
        "pass": passed,
        "diagnosis": diagnosis,
        "failure_classification": classes,
        "next_experiment": next_experiment,
    }
    write_json(HERE / "results.json", payload)
    doc = HERE / ("PASS.md" if passed else "FAILURE.md")
    lines = [f"# V837m {'REPRESENTATION SCREEN PASS' if passed else 'FAILURE'}", "", "This is not a primitive-invention PASS.", "", f"Diagnosis: **{diagnosis}**.", ""]
    for condition in CONDITIONS:
        lines.append(f"- {condition}: {conditions[condition]['families_passing']}/5, {conditions[condition]['parameter_count']} parameters")
    lines += ["", next_experiment, "", "Fresh-audit episodes consumed: 0. Primitives promoted: 0. Primitive mining remains blocked."]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
