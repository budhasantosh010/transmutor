from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated, v837_capacity_criterion_sha256
from experiments.v837_primitive_invention.common.reference_training import evaluate_sequence_model
from experiments.v837_primitive_invention.common.resource_accounting import ResourceAccounting, WallTimer
from experiments.v837_primitive_invention.common.seeds import deterministic_int, gate_sha256
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _configure_torch
from experiments.v837_primitive_invention.v837y.candidate_interaction import ControlledCandidateInteractionModel
from experiments.v837_primitive_invention.v837af.candidate_input_factorization import (
    CONDITIONS,
    CandidateInputFactorizationY3,
)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
AF0 = "AF0_y3_parent"
TRANSFERS = [
    "AF1_shared_candidate_input_factorization",
    "AF1F_folded_candidate_input_control",
    "AF1D_deshared_candidate_input_factorization",
]


def _git_blob_sha256(path: str) -> str:
    return hashlib.sha256(subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)).hexdigest()


def _assert_locks() -> None:
    if gate_sha256() != CONFIG["historical_gate_hash"]:
        raise SystemExit("historical V837 gate changed")
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise SystemExit("capacity criterion changed")
    frozen = {
        "experiments/v837_primitive_invention/v837y/config.json": CONFIG["v837y_config_sha256"],
        "experiments/v837_primitive_invention/v837y/candidate_interaction.py": CONFIG["v837y_model_sha256"],
        "experiments/v837_primitive_invention/v837y/results.json": CONFIG["v837y_results_sha256"],
        "experiments/v837_primitive_invention/v837y/raw/interaction_runs.json": CONFIG["v837y_raw_sha256"],
        "experiments/v837_primitive_invention/v837ab/results.json": CONFIG["v837ab_results_sha256"],
        "experiments/v837_primitive_invention/v837ac/results.json": CONFIG["v837ac_results_sha256"],
        "experiments/v837_primitive_invention/v837ad/results.json": CONFIG["v837ad_results_sha256"],
        "experiments/v837_primitive_invention/v837ad/diagnostics/decision_state.json": CONFIG["v837ad_decision_sha256"],
    }
    for path, expected in frozen.items():
        if _git_blob_sha256(path) != expected:
            raise SystemExit(f"frozen V837af dependency changed: {path}")
    ab = json.loads((ROOT / "experiments/v837_primitive_invention/v837ab/results.json").read_text(encoding="utf-8"))
    if int(ab["conditions"]["AB2_candidate_factorized_update_folded"]["families_passing"]) < 4:
        raise SystemExit("V837ab candidate-factorized sibling is not reference-sufficient")
    ac = json.loads((ROOT / "experiments/v837_primitive_invention/v837ac/results.json").read_text(encoding="utf-8"))
    if ac.get("diagnosis") != "INPUT_ORGANIZATION_TRANSFER_INSUFFICIENT":
        raise SystemExit("V837ac closure state changed")
    ad = json.loads((ROOT / "experiments/v837_primitive_invention/v837ad/diagnostics/decision_state.json").read_text(encoding="utf-8"))
    if ad.get("diagnosis") != "TEN_BY_FOUR_CANDIDATE_GEOMETRY_SUFFICIENT_IN_REFERENCE":
        raise SystemExit("V837ad frontier changed")
    audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
    if audit.get("episodes_consumed") != 0:
        raise SystemExit("fresh audit consumed")
    if (ROOT / "experiments/v837_primitive_invention/v837ae").exists():
        raise SystemExit("V837ae must remain absent")
    if (ROOT / "experiments/v837_primitive_invention/v838").exists():
        raise SystemExit("V838 exists")


def _seeds() -> tuple[list[int], list[int]]:
    tr = CONFIG["training"]
    train = list(range(tr["development_seed_range"][0], tr["development_seed_range"][1] + 1))
    validation = list(range(tr["validation_seed_range"][0], tr["validation_seed_range"][1] + 1))
    if len(train) != tr["train_episodes"] or len(validation) != tr["validation_episodes"]:
        raise RuntimeError("seed ranges do not match configured episode counts")
    return train, validation


def _base_seed(family: str, replicate: int) -> int:
    return deterministic_int(CONFIG["training"]["base_initialization_namespace"], family, replicate)


def _coupling_seed(replicate: int) -> int:
    tr = CONFIG["training"]
    return deterministic_int(tr["coupling_seed_namespace"], tr["coupling_seed_condition"], replicate)


def _projection_seed(family: str, replicate: int) -> int:
    return deterministic_int(CONFIG["training"]["projection_initialization_namespace"], family, replicate)


def _model(condition: str, family: str, replicate: int) -> CandidateInputFactorizationY3:
    return CandidateInputFactorizationY3(
        high_capacity_generic_graph(replicate),
        condition=condition,
        coupling_initialization_seed=_coupling_seed(replicate),
        projection_seed=_projection_seed(family, replicate),
    )


def _success_rate(task, predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return float(np.mean([task.success(float(p), float(t)) for p, t in zip(predictions.tolist(), targets.tolist())]))


def _active_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.arange(steps).view(1, -1) < lengths.view(-1, 1)


def _post_diagnostics(model: CandidateInputFactorizationY3, task, validation_seeds: list[int]) -> tuple[dict, int]:
    episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        baseline, trace = model(observations, lengths, return_trace=True)
        no_messages = model(observations, lengths, disable_messages=True)
    active = _active_mask(lengths, trace.input_terms.shape[1])
    active_inputs = trace.input_terms[active]
    active_candidates = trace.candidate_states[active]
    input_norms: list[float] = []
    temporal_variances: list[float] = []
    jacobian_norms: list[float] = []
    for cell in range(10):
        vals = active_inputs[:, cell, :]
        input_norms.append(float(torch.linalg.vector_norm(vals, dim=-1).mean().item()))
        temporal_variances.append(float(torch.var(vals, dim=0, unbiased=False).mean().item()))
        W, _ = model.effective_candidate_input_map(cell)
        deriv = 1.0 - active_candidates[:, cell, :] ** 2
        # J[b,o,i] = tanh'(preactivation)[b,o] * W[o,i]
        J = deriv.unsqueeze(-1) * W.detach().unsqueeze(0)
        jacobian_norms.append(float(torch.linalg.matrix_norm(J, ord="fro", dim=(-2, -1)).mean().item()))
    base_success = _success_rate(task, baseline, targets)
    no_message_success = _success_rate(task, no_messages, targets)
    return {
        "projection": model.projection_diagnostics(),
        "effective_input_maps": model.effective_input_map_diagnostics(),
        "candidate_input_term": {
            "mean_norm": float(np.mean(input_norms)),
            "per_cell_mean_norm": input_norms,
            "mean_temporal_variance": float(np.mean(temporal_variances)),
            "per_cell_temporal_variance": temporal_variances,
            "candidate_jacobian_wrt_visible_input_frobenius_mean": float(np.mean(jacobian_norms)),
            "per_cell_candidate_jacobian_wrt_visible_input_frobenius_mean": jacobian_norms,
        },
        "message_dependency": {
            "baseline_success": base_success,
            "no_message_success": no_message_success,
            "success_drop": base_success - no_message_success,
            "mean_abs_prediction_delta": float(torch.abs(baseline - no_messages).mean().item()),
        },
    }, 2


def _trajectory(model: CandidateInputFactorizationY3, step: int) -> dict:
    return {
        "step": int(step),
        "projection": model.projection_diagnostics(),
        "effective_input_maps": model.effective_input_map_diagnostics(),
    }


def _gradient_norm(parameters: list[torch.Tensor]) -> float:
    chunks = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(chunks)).item()) if chunks else 0.0


def _train(condition: str, family: str, replicate: int) -> tuple[CandidateInputFactorizationY3, dict]:
    _configure_torch()
    task = task_by_name(family)
    train_seeds, validation_seeds = _seeds()
    tr = CONFIG["training"]
    initialization_seed = _base_seed(family, replicate)
    torch.manual_seed(int(initialization_seed))
    np.random.seed(int(initialization_seed) % (2**32 - 1))
    model = _model(condition, family, replicate)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    loss_fn = nn.MSELoss()
    train_episodes = [task.generate(seed, "development") for seed in train_seeds]
    validation_episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(train_episodes)
    resources = ResourceAccounting(
        candidate_evaluations=1,
        model_fits=1,
        environment_steps=sum(len(e.observations) for e in train_episodes + validation_episodes),
        parameter_count=model.parameter_count(),
        model_parameter_bytes=model.parameter_bytes(),
    )
    curve: list[dict] = []
    trajectory: list[dict] = []
    requested = set(tr["curve_steps"])
    latest_gradient = 0.0

    def record(step: int) -> None:
        dev = evaluate_sequence_model(model, task, train_episodes)
        val = evaluate_sequence_model(model, task, validation_episodes)
        resources.forward_calls += 2
        curve.append({
            "step": int(step),
            "training_loss": dev.loss,
            "training_success": dev.success_rate,
            "validation_loss": val.loss,
            "validation_success": val.success_rate,
            "gradient_norm": latest_gradient,
        })
        trajectory.append(_trajectory(model, step))

    with WallTimer() as timer:
        if 0 in requested:
            record(0)
        for step in range(1, int(tr["steps"]) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predictions = model(observations, lengths)
            resources.forward_calls += 1
            loss = loss_fn(predictions, targets)
            loss.backward()
            latest_gradient = _gradient_norm(parameters)
            torch.nn.utils.clip_grad_norm_(parameters, float(tr["gradient_clip"]))
            optimizer.step()
            resources.optimizer_steps += 1
            resources.examples_processed += len(train_episodes)
            if step in requested:
                record(step)
        development = evaluate_sequence_model(model, task, train_episodes)
        validation = evaluate_sequence_model(model, task, validation_episodes)
        resources.forward_calls += 2
    resources.wall_seconds = timer.seconds
    resources.cpu_seconds = timer.cpu_seconds
    diagnostics, extra_calls = _post_diagnostics(model, task, validation_seeds)
    resources.forward_calls += extra_calls
    return model, {
        "initialization_seed": initialization_seed,
        "development_success": development.success_rate,
        "validation_success": validation.success_rate,
        "development_loss": development.loss,
        "validation_loss": validation.loss,
        "loss_curve": curve,
        "projection_trajectory": trajectory,
        "diagnostics": diagnostics,
        "resources": resources.to_dict(),
    }


def _worker(condition: str, family: str, replicate: int) -> dict:
    model, training = _train(condition, family, replicate)
    return {
        "version": "V837af",
        "condition": condition,
        "family": family,
        "replicate_id": replicate,
        "initialization_seed": training["initialization_seed"],
        "coupling_initialization_seed": _coupling_seed(replicate),
        "projection_initialization_seed": _projection_seed(family, replicate),
        "development_success": training["development_success"],
        "validation_success": training["validation_success"],
        "development_loss": training["development_loss"],
        "validation_loss": training["validation_loss"],
        "capacity_demonstrated": capacity_demonstrated(training["development_success"], training["validation_success"]),
        "loss_curve": training["loss_curve"],
        "projection_trajectory": training["projection_trajectory"],
        "diagnostics": training["diagnostics"],
        "parameter_count": model.parameter_count(),
        "projection_parameter_count": model.projection_parameter_count,
        "projection_specific_macs": model.projection_specific_macs,
        "recurrent_controller_macs": model.recurrent_controller_macs,
        "total_recurrent_controller_projection_macs": model.total_recurrent_controller_projection_macs,
        "resources": training["resources"],
        "processed_examples": training["resources"]["examples_processed"],
        "unique_seed_defined_episode_policy": "same 3200 family/seed episodes reused across every condition and replicate",
        "fresh_audit_consumed": False,
        "structural_search_allowed": False,
        "primitive_mining_allowed": False,
        "primitives_promoted": 0,
        "large_persistent_storage_tested": False,
        "gpu_seconds": 0.0,
        "v838_started": False,
    }


def _run(conditions: list[str]) -> list[dict]:
    jobs = [(condition, family, rep) for condition in conditions for family in FAMILIES for rep in range(CONFIG["training"]["replicates"])]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{row['condition']} {row['family']} r{row['replicate_id']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}", flush=True)
    rows.sort(key=lambda row: (row["condition"], row["family"], row["replicate_id"]))
    return rows


def _summary(rows: list[dict], condition: str) -> dict:
    output = {"families_passing": 0, "family_validation_medians": {}, "family_development_medians": {}}
    for family in FAMILIES:
        fr = [row for row in rows if row["condition"] == condition and row["family"] == family]
        development = float(np.median([row["development_success"] for row in fr]))
        validation = float(np.median([row["validation_success"] for row in fr]))
        output["family_development_medians"][family] = development
        output["family_validation_medians"][family] = validation
        output["families_passing"] += int(capacity_demonstrated(development, validation))
    return output


def _parent_guard(rows: list[dict]) -> dict:
    expected = json.loads((ROOT / "experiments/v837_primitive_invention/v837y/results.json").read_text(encoding="utf-8"))["conditions"]["Y3_global_control_rank4_candidate"]
    observed = _summary(rows, AF0)
    expected_values = {family: float(expected["family_results"][family]["validation"]["median"]) for family in FAMILIES}
    deltas = {family: abs(observed["family_validation_medians"][family] - expected_values[family]) for family in FAMILIES}
    valid = observed["families_passing"] == 3 and max(deltas.values()) <= float(CONFIG["parent_drift_threshold"])
    return {
        "parent_reproduced": bool(valid),
        "expected_families_passing": 3,
        "observed": observed,
        "expected_family_validation_medians": expected_values,
        "absolute_deltas": deltas,
        "max_absolute_delta": max(deltas.values()),
    }


def _step0_equivalence() -> dict:
    train_seeds, _ = _seeds()
    comparisons = {
        "AF1_vs_AF1F": {key: 0.0 for key in ("projected_visible_input", "candidate_input_term", "candidate", "global_gate", "next_state", "output", "prediction")},
        "AF1_vs_AF1D": {key: 0.0 for key in ("projected_visible_input", "candidate_input_term", "candidate", "global_gate", "next_state", "output", "prediction")},
    }
    rows = []
    for family in FAMILIES:
        task = task_by_name(family)
        episodes = [task.generate(seed, "development") for seed in train_seeds[:8]]
        observations, lengths, _ = episodes_to_batch(episodes)
        for rep in range(CONFIG["training"]["replicates"]):
            seed = _base_seed(family, rep)
            models = {}
            for condition in TRANSFERS:
                torch.manual_seed(seed)
                # Cast the exact stored float32 initialization to float64 for
                # the equivalence proof so only algebraic implementation
                # differences are gated, not float32 matmul association noise.
                models[condition] = _model(condition, family, rep).double()
            probe_observations = observations.double()
            with torch.no_grad():
                outputs = {condition: models[condition](probe_observations, lengths, return_trace=True) for condition in TRANSFERS}
                projected = {}
                for condition in TRANSFERS:
                    chunks = []
                    model = models[condition]
                    for t in range(probe_observations.shape[1]):
                        x_t = probe_observations[:, t, :]
                        per_cell = []
                        for cell in range(10):
                            visible = ControlledCandidateInteractionModel._visible_input(model, x_t, cell)
                            per_cell.append(model.diagnostic_projected_visible_input(visible, cell))
                        chunks.append(torch.stack(per_cell, dim=1))
                    projected[condition] = torch.stack(chunks, dim=1)
            anchor_prediction, anchor_trace = outputs["AF1_shared_candidate_input_factorization"]
            for label, condition in (
                ("AF1_vs_AF1F", "AF1F_folded_candidate_input_control"),
                ("AF1_vs_AF1D", "AF1D_deshared_candidate_input_factorization"),
            ):
                prediction, trace = outputs[condition]
                anchor_bias = torch.stack(list(models["AF1_shared_candidate_input_factorization"].base.cell_b), dim=0).view(1, 1, 10, 4)
                other_bias = torch.stack(list(models[condition].base.cell_b), dim=0).view(1, 1, 10, 4)
                active = (torch.arange(observations.shape[1]).view(1, -1) < lengths.view(-1, 1)).to(observations.dtype).view(observations.shape[0], observations.shape[1], 1, 1)
                errors = {
                    "projected_visible_input": float(torch.max(torch.abs(projected["AF1_shared_candidate_input_factorization"] - projected[condition])).item()),
                    "candidate_input_term": float(torch.max(torch.abs((anchor_trace.input_terms + active * anchor_bias) - (trace.input_terms + active * other_bias))).item()),
                    "candidate": float(torch.max(torch.abs(anchor_trace.candidate_states - trace.candidate_states)).item()),
                    "global_gate": float(torch.max(torch.abs(anchor_trace.global_gates - trace.global_gates)).item()),
                    "next_state": float(torch.max(torch.abs(anchor_trace.states - trace.states)).item()),
                    "output": float(torch.max(torch.abs(anchor_trace.outputs - trace.outputs)).item()),
                    "prediction": float(torch.max(torch.abs(anchor_prediction - prediction)).item()),
                }
                rows.append({"family": family, "replicate_id": rep, "comparison": label, "errors": errors})
                for key, value in errors.items():
                    comparisons[label][key] = max(comparisons[label][key], value)
    maximum = max(value for comp in comparisons.values() for value in comp.values())
    return {
        "step0_equivalence_proven": bool(maximum <= CONFIG["step0_tolerance"]),
        "tolerance": CONFIG["step0_tolerance"],
        "equivalence_precision": "float64 replay of exact stored float32 initialization",
        "maximum_error": maximum,
        "max_errors": comparisons,
        "comparisons": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "af0", "transfer", "all"), default="all")
    args = parser.parse_args()
    _assert_locks()
    for directory in ("raw", "diagnostics", "plots"):
        (HERE / directory).mkdir(exist_ok=True)
    if args.phase in {"preflight", "all"}:
        step0 = _step0_equivalence()
        write_json(HERE / "diagnostics" / "step0_equivalence.json", step0)
        print(json.dumps({"step0_equivalence_proven": step0["step0_equivalence_proven"], "maximum_error": step0["maximum_error"]}, indent=2))
        if not step0["step0_equivalence_proven"]:
            return 2
    if args.phase in {"af0", "all"}:
        step = HERE / "diagnostics" / "step0_equivalence.json"
        if not step.exists() or not json.loads(step.read_text(encoding="utf-8")).get("step0_equivalence_proven"):
            raise SystemExit("V837af AF0 blocked until step-zero equivalence passes")
        rows = _run([AF0])
        write_json(HERE / "raw" / "af0_runs.json", {"rows": rows, "unique_seed_defined_episodes": 3200})
        guard = _parent_guard(rows)
        write_json(HERE / "diagnostics" / "anchor_compatibility.json", guard)
        print(json.dumps(guard, indent=2))
        if not guard["parent_reproduced"]:
            return 3
    if args.phase in {"transfer", "all"}:
        guard_path = HERE / "diagnostics" / "anchor_compatibility.json"
        if not guard_path.exists() or not json.loads(guard_path.read_text(encoding="utf-8")).get("parent_reproduced"):
            raise SystemExit("V837af transfer blocked until AF0 reproduces Y3")
        rows = _run(TRANSFERS)
        write_json(HERE / "raw" / "transfer_runs.json", {"rows": rows, "unique_seed_defined_episodes": 3200})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
