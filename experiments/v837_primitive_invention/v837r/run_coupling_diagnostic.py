from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated, v837_capacity_criterion_sha256
from experiments.v837_primitive_invention.common.reference_training import train_sequence_model
from experiments.v837_primitive_invention.common.seeds import deterministic_int, gate_sha256
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name
from experiments.v837_primitive_invention.v837r.recurrent_coupling import (
    GloballyCoupledNeutralGraphModel,
    RecurrentCouplingSpec,
    coupling_actual_macs,
    coupling_core_macs,
    coupling_scaling_complexity,
    local_recurrent_macs,
)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FROZEN = json.loads((HERE / "frozen_global_coupling_gate.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
PRIMARY = list(CONFIG["conditions"])
CONTROLS = list(CONFIG["matched_controls"])
MATCH_BY_PRIMARY = {row["matches"]: name for name, row in CONFIG["matched_controls"].items()}


def _configure_torch() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _git_blob_sha256(relative: str) -> str:
    payload = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    return hashlib.sha256(payload).hexdigest()


def _assert_science_locks() -> None:
    if gate_sha256() != CONFIG["historical_gate_hash"]:
        raise SystemExit("historical V837 gate hash mismatch")
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise SystemExit("capacity criterion fingerprint mismatch")
    if _git_blob_sha256("experiments/v837_primitive_invention/v837q/results.json") != FROZEN["parent_result_git_blob_sha256"]:
        raise SystemExit("V837q parent result changed")
    if _git_blob_sha256("experiments/v837_primitive_invention/v837l/results.json") != FROZEN["calibrated_reference_result_git_blob_sha256"]:
        raise SystemExit("V837l calibrated reference result changed")
    if _git_blob_sha256("experiments/v837_primitive_invention/v837p/results.json") != FROZEN["v837p_result_git_blob_sha256"]:
        raise SystemExit("V837p result changed")
    for key in ("dynamic_modulation_allowed", "shared_state_allowed", "structural_search_allowed", "primitive_mining_allowed", "fresh_audit_allowed"):
        if CONFIG.get(key) is not False:
            raise SystemExit(f"V837r science lock changed: {key}")
    if CONFIG.get("state_layout") != "local_10x4" or int(CONFIG.get("total_state_dim", -1)) != 40:
        raise SystemExit("V837r state-layout boundary changed")
    audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
    if int(audit.get("episodes_consumed", -1)) != 0:
        raise SystemExit("fresh-audit data has already been consumed")


def _training_seeds() -> tuple[list[int], list[int]]:
    training = CONFIG["training"]
    train = list(range(int(training["development_seed_range"][0]), int(training["development_seed_range"][1]) + 1))
    validation = list(range(int(training["validation_seed_range"][0]), int(training["validation_seed_range"][1]) + 1))
    if len(train) != int(training["train_episodes"]) or len(validation) != int(training["validation_episodes"]):
        raise RuntimeError("configured seed ranges do not match episode counts")
    return train, validation


def _coupling_spec(condition: str, replicate: int) -> RecurrentCouplingSpec:
    if condition in CONFIG["conditions"]:
        row = CONFIG["conditions"][condition]
        seed = deterministic_int(CONFIG["training"]["coupling_seed_namespace"], condition, replicate)
        return RecurrentCouplingSpec(
            mode=row["coupling_mode"],
            rank=row.get("rank"),
            cross_block_only=bool(row.get("cross_block_only", True)),
            scaling=float(CONFIG["coupling_scaling"]),
            initialization_seed=seed,
        )
    if condition in CONFIG["matched_controls"]:
        row = CONFIG["matched_controls"][condition]
        seed = deterministic_int(CONFIG["training"]["coupling_seed_namespace"], row["matches"], replicate)
        return RecurrentCouplingSpec(
            mode="parameter_matched_local",
            cross_block_only=True,
            scaling=float(CONFIG["coupling_scaling"]),
            initialization_seed=seed,
            matched_local_rank=int(row["matched_local_rank"]),
        )
    raise ValueError(f"unknown V837r condition: {condition}")


def _model_factory(condition: str, replicate: int):
    graph = high_capacity_generic_graph(replicate)
    spec = _coupling_spec(condition, replicate)
    return graph, spec, lambda: GloballyCoupledNeutralGraphModel(graph, spec, obs_dim=6)


def _success_rate(task, predictions: torch.Tensor, targets: torch.Tensor) -> float:
    pred = [float(v) for v in predictions.detach().cpu().tolist()]
    target = [float(v) for v in targets.detach().cpu().tolist()]
    return float(np.mean([bool(task.success(p, y)) for p, y in zip(pred, target)]))


def _active_mask(trace_tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return torch.arange(trace_tensor.shape[1], device=lengths.device).view(1, -1) < lengths.view(-1, 1)


def _term_norm(trace_tensor: torch.Tensor, lengths: torch.Tensor) -> float:
    active = _active_mask(trace_tensor, lengths)
    values = trace_tensor[active]
    if not values.numel():
        return 0.0
    return float(torch.linalg.vector_norm(values.reshape(values.shape[0], -1), dim=1).mean().item())


def _coupling_utilization(model: GloballyCoupledNeutralGraphModel, task, validation_seeds: list[int]) -> dict:
    episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, _ = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        _, trace = model(observations, lengths, return_trace=True)
    local_norm = _term_norm(trace.recurrent_terms, lengths)
    global_norm = _term_norm(trace.global_recurrent_terms, lengths)
    matched_norm = _term_norm(trace.matched_local_terms, lengths)
    message_norm = _term_norm(trace.message_terms, lengths)
    input_norm = _term_norm(trace.input_terms, lengths)
    added_norm = global_norm if model.coupling.mode in {"low_rank", "dense"} else matched_norm
    return {
        "global_recurrent_term_norm": global_norm,
        "matched_local_term_norm": matched_norm,
        "local_recurrent_term_norm": local_norm,
        "message_term_norm": message_norm,
        "input_term_norm": input_norm,
        "global_to_local_ratio": global_norm / (local_norm + 1e-12),
        "global_to_message_ratio": global_norm / (message_norm + 1e-12),
        "added_to_local_ratio": added_norm / (local_norm + 1e-12),
    }


def _message_and_cross_cell(model: GloballyCoupledNeutralGraphModel, task, validation_seeds: list[int]) -> dict:
    episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        baseline, baseline_trace = model(observations, lengths, return_trace=True)
        no_message = model(observations, lengths, disable_messages=True)
    baseline_success = _success_rate(task, baseline, targets)
    no_message_success = _success_rate(task, no_message, targets)
    source_rows = []
    if model.coupling.mode in {"low_rank", "dense"}:
        with torch.no_grad():
            for source_cell in range(10):
                intervened, trace = model(observations, lengths, zero_coupling_source_cell=source_cell, return_trace=True)
                state_delta = torch.abs(trace.states - baseline_trace.states)
                other_mask = torch.ones(10, dtype=torch.bool)
                other_mask[source_cell] = False
                source_rows.append({
                    "source_cell": source_cell,
                    "mean_abs_prediction_delta": float(torch.mean(torch.abs(intervened - baseline)).item()),
                    "mean_abs_other_cell_state_delta": float(state_delta[:, :, other_mask, :].mean().item()),
                    "success_delta": float(_success_rate(task, intervened, targets) - baseline_success),
                })
    return {
        "message_dependency": {
            "baseline_success": baseline_success,
            "no_message_success": no_message_success,
            "success_drop": baseline_success - no_message_success,
            "mean_abs_prediction_delta": float(torch.mean(torch.abs(no_message - baseline)).item()),
        },
        "cross_cell_influence": {
            "intervention_semantics": "zero one source cell only in the global recurrent branch while preserving its local recurrence and the historical message path",
            "per_source": source_rows,
            "mean_abs_prediction_delta": float(np.mean([r["mean_abs_prediction_delta"] for r in source_rows])) if source_rows else 0.0,
            "mean_abs_other_cell_state_delta": float(np.mean([r["mean_abs_other_cell_state_delta"] for r in source_rows])) if source_rows else 0.0,
        },
    }


def _cell_gradient_vector(model: GloballyCoupledNeutralGraphModel, cell_index: int) -> torch.Tensor:
    params = [
        model.base.cell_ws[cell_index], model.base.cell_wm[cell_index], model.base.cell_wx[cell_index],
        model.base.cell_b[cell_index], model.base.cell_wo[cell_index],
    ]
    chunks = []
    for parameter in params:
        chunks.append(torch.zeros(parameter.numel()) if parameter.grad is None else parameter.grad.detach().reshape(-1).cpu())
    return torch.cat(chunks)


def _gradient_diagnostics(model: GloballyCoupledNeutralGraphModel, task, train_seeds: list[int]) -> dict:
    episodes = [task.generate(seed, "development") for seed in train_seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    model.zero_grad(set_to_none=True)
    model.train()
    predictions = model(observations, lengths)
    loss = torch.mean((predictions - targets) ** 2)
    loss.backward()
    all_grads = [p.grad.detach().reshape(-1).cpu() for p in model.parameters() if p.grad is not None]
    global_norm = float(torch.linalg.vector_norm(torch.cat(all_grads)).item()) if all_grads else 0.0
    coupling_params = []
    if model.coupling.mode == "low_rank":
        coupling_params = [model.global_u, model.global_v]
    elif model.coupling.mode == "dense":
        coupling_params = [model.global_dense]
    elif model.coupling.mode == "parameter_matched_local":
        coupling_params = list(model.local_extra_u) + list(model.local_extra_v)
    coupling_grads = [p.grad.detach().reshape(-1).cpu() for p in coupling_params if p.grad is not None]
    coupling_grad_norm = float(torch.linalg.vector_norm(torch.cat(coupling_grads)).item()) if coupling_grads else 0.0
    vectors = [_cell_gradient_vector(model, i) for i in range(10)]
    norms = [float(torch.linalg.vector_norm(v).item()) for v in vectors]
    cosines = []
    for i in range(10):
        for j in range(i + 1, 10):
            denom = float(torch.linalg.vector_norm(vectors[i]).item() * torch.linalg.vector_norm(vectors[j]).item())
            if denom > 1e-12:
                cosines.append(float(torch.dot(vectors[i], vectors[j]).item() / denom))
    model.zero_grad(set_to_none=True)
    return {
        "global_gradient_norm": global_norm,
        "coupling_gradient_norm": coupling_grad_norm,
        "per_cell_gradient_norms": norms,
        "cell_gradient_norm_variance": float(np.var(norms)),
        "cell_gradient_cosine_mean": float(np.mean(cosines)) if cosines else None,
        "temporal_credit_diagnostic": "not_collected_in_fixed_cost_v837r_screen",
    }


def _post_training_diagnostics(model: GloballyCoupledNeutralGraphModel, task, train_seeds: list[int], validation_seeds: list[int]) -> dict:
    utilization = _coupling_utilization(model, task, validation_seeds)
    message_cross = _message_and_cross_cell(model, task, validation_seeds)
    return {
        "coupling_matrix": model.coupling_diagnostics(),
        "utilization": utilization,
        "gradient": _gradient_diagnostics(model, task, train_seeds),
        **message_cross,
        "compute": {
            "local_recurrent_macs": local_recurrent_macs(),
            "coupling_core_macs": coupling_core_macs(model.coupling),
            "coupling_actual_macs": coupling_actual_macs(model.coupling),
            "total_recurrent_macs_actual": local_recurrent_macs() + coupling_actual_macs(model.coupling),
            "approx_recurrent_flops_actual": 2 * (local_recurrent_macs() + coupling_actual_macs(model.coupling)),
            "coupling_scaling_complexity": coupling_scaling_complexity(model.coupling),
        },
    }


def _worker(condition: str, family: str, replicate: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    train_seeds, validation_seeds = _training_seeds()
    graph, spec, factory = _model_factory(condition, replicate)
    initialization_seed = deterministic_int(CONFIG["training"]["initialization_seed_namespace"], family, replicate)
    training = CONFIG["training"]
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
    diagnostics = _post_training_diagnostics(result.model, task, train_seeds, validation_seeds)
    resources = result.resources.to_dict()
    resources["training_forward_calls"] = int(resources.get("forward_calls", 0))
    # utilization trace + baseline/no-message + up to 10 source interventions + one gradient forward
    diagnostic_calls = 4 + (10 if spec.mode in {"low_rank", "dense"} else 0)
    resources["diagnostic_forward_calls"] = diagnostic_calls
    resources["forward_calls"] = int(resources.get("forward_calls", 0)) + diagnostic_calls
    return {
        "condition": condition,
        "family": family,
        "replicate": int(replicate),
        "initialization_seed": int(initialization_seed),
        "coupling_initialization_seed": int(spec.initialization_seed),
        "coupling_spec": spec.to_dict(),
        "graph_id": graph.graph_id,
        "development_seed_first": train_seeds[0],
        "development_seed_last": train_seeds[-1],
        "validation_seed_first": validation_seeds[0],
        "validation_seed_last": validation_seeds[-1],
        "development_success": result.development.success_rate,
        "validation_success": result.validation.success_rate,
        "development_loss": result.development.loss,
        "validation_loss": result.validation.loss,
        "capacity_demonstrated": capacity_demonstrated(result.development.success_rate, result.validation.success_rate),
        "learning_curve": result.learning_curve,
        "diagnostics": diagnostics,
        "parameter_count": result.model.parameter_count(),
        "added_parameter_count": result.model.added_parameter_count(),
        "parameter_bytes": result.model.parameter_bytes(),
        "resources": resources,
        "state_layout": "local_10x4",
        "total_state_dim": 40,
        "dynamic_modulation_enabled": False,
        "interaction_mode": "none",
        "shared_state_enabled": False,
        "task_family_label_in_model_input": False,
        "fresh_audit_consumed": False,
        "gpu_seconds": 0.0,
    }


def _run_jobs(jobs: list[tuple], output_path: Path) -> list[dict]:
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{row['condition']} {row['family']} r{row['replicate']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}", flush=True)
    rows.sort(key=lambda row: (row["condition"], row["family"], row["replicate"]))
    write_json(output_path, {"rows": rows, "fresh_audit_consumed": False})
    return rows


def _family_medians(rows: list[dict], condition: str) -> dict[str, dict[str, float]]:
    output = {}
    for family in FAMILIES:
        selected = [r for r in rows if r["condition"] == condition and r["family"] == family]
        output[family] = {
            "development": float(np.median([r["development_success"] for r in selected])),
            "validation": float(np.median([r["validation_success"] for r in selected])),
        }
    return output


def _families_passing(rows: list[dict], condition: str) -> int:
    medians = _family_medians(rows, condition)
    return sum(int(capacity_demonstrated(v["development"], v["validation"])) for v in medians.values())


def _mean_family_validation(rows: list[dict], condition: str) -> float:
    return float(np.mean([row["validation"] for row in _family_medians(rows, condition).values()]))


def _baseline_compatibility(rows: list[dict]) -> dict:
    observed = {family: values["validation"] for family, values in _family_medians(rows, "R0_local").items()}
    expected = FROZEN["expected_r0_validation_medians"]
    threshold = float(CONFIG["baseline_drift_threshold"]["absolute_validation_delta"])
    drifted = [family for family in FAMILIES if abs(observed[family] - float(expected[family])) > threshold]
    compatible = len(drifted) < int(CONFIG["baseline_drift_threshold"]["families_required"])
    return {
        "compatible": compatible,
        "observed_validation_medians": observed,
        "expected_validation_medians": expected,
        "absolute_deltas": {family: abs(observed[family] - float(expected[family])) for family in FAMILIES},
        "drifted_families": drifted,
        "families_passing": _families_passing(rows, "R0_local"),
        "expected_families_passing": int(FROZEN["expected_r0_families_passing"]),
        "threshold": CONFIG["baseline_drift_threshold"],
    }


def _screen_decision(rows: list[dict]) -> dict:
    trigger = CONFIG["screen_localization_trigger"]
    global_conditions = ["R2_rank2", "R3_rank4", "R5_dense_cross_block"]
    records = {}
    localization_allowed = False
    for condition in global_conditions:
        control = MATCH_BY_PRIMARY[condition]
        count = _families_passing(rows, condition)
        control_count = _families_passing(rows, control)
        mean_val = _mean_family_validation(rows, condition)
        control_mean = _mean_family_validation(rows, control)
        delta = mean_val - control_mean
        records[condition] = {
            "families_passing": count,
            "matched_control": control,
            "matched_control_families_passing": control_count,
            "mean_family_validation_median": mean_val,
            "matched_control_mean_family_validation_median": control_mean,
            "specificity_delta": delta,
        }
        if count >= int(trigger["min_families_passing"]) or delta >= float(trigger["specificity_min_mean_validation_delta"]):
            localization_allowed = True
    decision = {
        "screen_complete": True,
        "records": records,
        "localization_allowed": localization_allowed,
        "strict_stop_triggered": not localization_allowed,
        "rule": trigger,
    }
    write_json(HERE / "diagnostics" / "screen_decision.json", decision)
    return decision


def _phase_baseline() -> None:
    jobs = [("R0_local", family, replicate) for family in FAMILIES for replicate in range(int(CONFIG["training"]["replicates"]))]
    rows = _run_jobs(jobs, HERE / "raw" / "baseline_runs.json")
    compatibility = _baseline_compatibility(rows)
    write_json(HERE / "diagnostics" / "baseline_compatibility.json", compatibility)
    print("baseline compatibility:", json.dumps(compatibility, sort_keys=True), flush=True)
    if not compatibility["compatible"]:
        raise SystemExit("V837r BASELINE_DRIFT: stop before coupling interpretation")


def _phase_screen() -> None:
    path = HERE / "diagnostics" / "baseline_compatibility.json"
    if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("compatible") is not True:
        raise SystemExit("run compatible V837r baseline first")
    jobs = [
        (condition, family, replicate)
        for condition in CONFIG["screen_sequence"]
        for family in FAMILIES
        for replicate in range(int(CONFIG["training"]["replicates"]))
    ]
    rows = _run_jobs(jobs, HERE / "raw" / "screen_runs.json")
    decision = _screen_decision(rows)
    print("screen decision:", json.dumps(decision, sort_keys=True), flush=True)


def _phase_localization() -> None:
    path = HERE / "diagnostics" / "screen_decision.json"
    if not path.exists():
        raise SystemExit("run V837r screen before localization")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("localization_allowed") is not True:
        raise SystemExit("V837r strict stop rule blocks rank1/rank8 localization")
    jobs = [
        (condition, family, replicate)
        for condition in CONFIG["localization_sequence"]
        for family in FAMILIES
        for replicate in range(int(CONFIG["training"]["replicates"]))
    ]
    _run_jobs(jobs, HERE / "raw" / "localization_runs.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline", "screen", "localization"), required=True)
    args = parser.parse_args()
    _assert_science_locks()
    for name in ("raw", "diagnostics", "plots"):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    if args.phase == "baseline":
        _phase_baseline()
    elif args.phase == "screen":
        _phase_screen()
    else:
        _phase_localization()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
