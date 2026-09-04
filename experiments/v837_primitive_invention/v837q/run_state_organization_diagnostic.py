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
from experiments.v837_primitive_invention.common.reference_models import build_reference_model
from experiments.v837_primitive_invention.common.reference_training import train_sequence_model
from experiments.v837_primitive_invention.common.seeds import deterministic_int, gate_sha256
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name
from experiments.v837_primitive_invention.v837q.state_organization_models import (
    SharedStateNeutralGraphModel,
    projection_norm_error,
    standard_state_layout,
)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FROZEN = json.loads((HERE / "frozen_state_organization_gate.json").read_text(encoding="utf-8"))
PRIMARY = list(CONFIG["conditions"])
REFERENCES = list(CONFIG["references"])
FAMILIES = [task.name for task in all_tasks()]


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
    if _git_blob_sha256("experiments/v837_primitive_invention/v837p/results.json") != FROZEN["parent_result_git_blob_sha256"]:
        raise SystemExit("V837p parent result changed")
    if _git_blob_sha256("experiments/v837_primitive_invention/v837l/results.json") != FROZEN["calibrated_reference_result_git_blob_sha256"]:
        raise SystemExit("V837l calibrated reference result changed")
    if CONFIG.get("dynamic_modulation_allowed") is not False:
        raise SystemExit("V837q may not enable dynamic modulation")
    if CONFIG.get("structural_search_allowed") is not False or CONFIG.get("primitive_mining_allowed") is not False:
        raise SystemExit("V837q downstream science lock changed")
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


def _model_factory(condition: str, replicate: int, *, projection_seed: int | None = None):
    if condition in PRIMARY or condition == "Q3_shared_1x40_no_messages":
        base_condition = "Q3_shared_1x40" if condition == "Q3_shared_1x40_no_messages" else condition
        graph = high_capacity_generic_graph(replicate)
        layout = standard_state_layout(base_condition, projection_seed=int(CONFIG["projection_seed"] if projection_seed is None else projection_seed))
        return graph, lambda: SharedStateNeutralGraphModel(graph, layout, obs_dim=6, local_view_dim=4, message_dim=4)
    reference = CONFIG["references"][condition]
    return None, lambda: build_reference_model(reference["architecture"], int(reference["hidden_size"]), 6)


def _active_state_matrix(model, task, validation_seeds: list[int]) -> tuple[torch.Tensor, torch.Tensor, object]:
    episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, _ = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        _, trace = model(observations, lengths, return_trace=True)
    states = trace.states
    if states.ndim == 4:
        states = states.reshape(states.shape[0], states.shape[1], -1)
    if states.ndim != 3:
        raise RuntimeError("state trace must reduce to [B,T,D]")
    active = torch.arange(states.shape[1]).view(1, -1) < lengths.view(-1, 1)
    return states[active].detach(), lengths, trace


def _effective_rank(active_states: torch.Tensor) -> dict:
    if active_states.numel() == 0:
        return {"state_norm": 0.0, "state_variance": 0.0, "effective_rank": 0.0, "participation_ratio": 0.0, "mean_abs_pairwise_correlation": 0.0}
    matrix = active_states.float()
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, matrix.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if float(total.item()) <= 1e-12:
        effective_rank = 0.0
        participation = 0.0
    else:
        probabilities = eigenvalues / total
        positive = probabilities[probabilities > 1e-12]
        effective_rank = float(torch.exp(-(positive * torch.log(positive)).sum()).item())
        participation = float((total * total / torch.clamp(torch.sum(eigenvalues * eigenvalues), min=1e-12)).item())
    std = centered.std(dim=0, unbiased=False)
    valid = std > 1e-8
    mean_corr = 0.0
    if int(valid.sum().item()) >= 2:
        z = centered[:, valid] / std[valid].view(1, -1)
        corr = (z.T @ z) / max(1, z.shape[0])
        mask = ~torch.eye(corr.shape[0], dtype=torch.bool)
        mean_corr = float(torch.mean(torch.abs(corr[mask])).item())
    return {
        "state_norm": float(torch.linalg.vector_norm(matrix, dim=1).mean().item()),
        "state_variance": float(matrix.var(dim=0, unbiased=False).mean().item()),
        "effective_rank": effective_rank,
        "participation_ratio": participation,
        "mean_abs_pairwise_correlation": mean_corr,
    }


def _success_rate(task, predictions: torch.Tensor, targets: torch.Tensor) -> float:
    pred = [float(v) for v in predictions.detach().cpu().tolist()]
    target = [float(v) for v in targets.detach().cpu().tolist()]
    return float(np.mean([bool(task.success(p, y)) for p, y in zip(pred, target)]))


def _message_and_influence_diagnostics(model, task, validation_seeds: list[int]) -> dict:
    if not isinstance(model, SharedStateNeutralGraphModel):
        return {"message_dependency": None, "cross_cell_influence": None}
    episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        baseline = model(observations, lengths)
        no_message = model(observations, lengths, disable_messages=True)
    baseline_success = _success_rate(task, baseline, targets)
    no_message_success = _success_rate(task, no_message, targets)
    cell_rows = []
    with torch.no_grad():
        for cell_index in range(10):
            intervened = model(observations, lengths, disabled_contribution_cells={cell_index})
            cell_rows.append({
                "cell_index": cell_index,
                "mean_abs_prediction_delta": float(torch.mean(torch.abs(intervened - baseline)).item()),
                "success_rate": _success_rate(task, intervened, targets),
                "success_delta": float(_success_rate(task, intervened, targets) - baseline_success),
            })
    return {
        "message_dependency": {
            "baseline_success": baseline_success,
            "no_message_success": no_message_success,
            "success_drop": float(baseline_success - no_message_success),
            "mean_abs_prediction_delta": float(torch.mean(torch.abs(no_message - baseline)).item()),
        },
        "cross_cell_influence": {
            "intervention_semantics": "shared layouts disable only the selected cell/path write contribution; Q0 uses historical whole-cell disable as a local-state proxy",
            "per_cell": cell_rows,
            "mean_abs_prediction_delta": float(np.mean([row["mean_abs_prediction_delta"] for row in cell_rows])),
            "mean_success_delta": float(np.mean([row["success_delta"] for row in cell_rows])),
        },
    }


def _cell_gradient_vector(model: SharedStateNeutralGraphModel, cell_index: int) -> torch.Tensor:
    params = [
        model.base.cell_ws[cell_index],
        model.base.cell_wm[cell_index],
        model.base.cell_wx[cell_index],
        model.base.cell_b[cell_index],
        model.base.cell_wo[cell_index],
    ]
    pieces = []
    for parameter in params:
        if parameter.grad is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=parameter.dtype))
        else:
            pieces.append(parameter.grad.detach().reshape(-1).cpu())
    return torch.cat(pieces)


def _gradient_diagnostics(model, task, train_seeds: list[int]) -> dict:
    episodes = [task.generate(seed, "development") for seed in train_seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    model.zero_grad(set_to_none=True)
    model.train()
    predictions = model(observations, lengths)
    loss = torch.mean((predictions - targets) ** 2)
    loss.backward()
    all_grads = [parameter.grad.detach().reshape(-1).cpu() for parameter in model.parameters() if parameter.grad is not None]
    global_norm = float(torch.linalg.vector_norm(torch.cat(all_grads)).item()) if all_grads else 0.0
    if not isinstance(model, SharedStateNeutralGraphModel):
        model.zero_grad(set_to_none=True)
        return {"global_gradient_norm": global_norm, "pathway_gradient_norms": None, "group_gradient_norms": None, "within_group_gradient_cosine_mean": None}
    vectors = [_cell_gradient_vector(model, i) for i in range(10)]
    norms = [float(torch.linalg.vector_norm(vector).item()) for vector in vectors]
    group_norms = []
    cosine_values = []
    for group_index in range(model.state_layout.num_state_groups):
        members = model.state_layout.members(group_index)
        group_norms.append(float(math.sqrt(sum(norms[i] ** 2 for i in members))))
        for left_offset, left in enumerate(members):
            for right in members[left_offset + 1 :]:
                denom = float(torch.linalg.vector_norm(vectors[left]).item() * torch.linalg.vector_norm(vectors[right]).item())
                if denom > 1e-12:
                    cosine_values.append(float(torch.dot(vectors[left], vectors[right]).item() / denom))
    model.zero_grad(set_to_none=True)
    return {
        "global_gradient_norm": global_norm,
        "pathway_gradient_norms": norms,
        "pathway_gradient_norm_mean": float(np.mean(norms)),
        "pathway_gradient_norm_variance": float(np.var(norms)),
        "group_gradient_norms": group_norms,
        "within_group_gradient_cosine_mean": float(np.mean(cosine_values)) if cosine_values else None,
        "within_group_gradient_cosine_values": cosine_values,
    }


def _projection_diagnostics(model) -> dict | None:
    if not isinstance(model, SharedStateNeutralGraphModel):
        return None
    return {
        "layout": model.state_layout.to_dict(),
        "trainable_parameter_count": model.parameter_count(),
        "parameter_bytes": model.parameter_bytes(),
        "non_trainable_projection_elements": model.non_trainable_projection_elements,
        "projection_norm_error_max": max(projection_norm_error(model.projection(i)) for i in range(10)),
        "group_write_normalizations": list(model.group_write_normalizations()),
        "readout_input_width": model.readout_input_width,
    }


def _post_training_diagnostics(model, task, train_seeds: list[int], validation_seeds: list[int]) -> dict:
    active_states, _, _ = _active_state_matrix(model, task, validation_seeds)
    output = {
        "state": _effective_rank(active_states),
        "gradient": _gradient_diagnostics(model, task, train_seeds),
        "projection": _projection_diagnostics(model),
    }
    output.update(_message_and_influence_diagnostics(model, task, validation_seeds))
    return output


def _worker(condition: str, family: str, replicate: int, projection_seed: int | None = None, disable_messages: bool = False) -> dict:
    _configure_torch()
    task = task_by_name(family)
    train_seeds, validation_seeds = _training_seeds()
    graph, factory = _model_factory(condition, replicate, projection_seed=projection_seed)
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
    if disable_messages:
        # Q3-NM is trained with messages disabled, not merely evaluated that way.
        raise RuntimeError("disable_messages training requires the dedicated conditional worker")
    diagnostics = _post_training_diagnostics(result.model, task, train_seeds, validation_seeds)
    return {
        "condition": condition,
        "family": family,
        "replicate": int(replicate),
        "projection_seed": int(CONFIG["projection_seed"] if projection_seed is None else projection_seed) if condition.startswith("Q") else None,
        "initialization_seed": int(initialization_seed),
        "graph_id": None if graph is None else graph.graph_id,
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
        "parameter_count": int(sum(p.numel() for p in result.model.parameters())),
        "parameter_bytes": int(sum(p.numel() * p.element_size() for p in result.model.parameters())),
        "resources": result.resources.to_dict(),
        "task_family_label_in_model_input": False,
        "dynamic_modulation_enabled": False,
        "fresh_audit_consumed": False,
        "gpu_seconds": 0.0,
    }


def _train_with_messages_disabled(condition: str, family: str, replicate: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    train_seeds, validation_seeds = _training_seeds()
    graph, factory = _model_factory(condition, replicate)
    initialization_seed = deterministic_int(CONFIG["training"]["initialization_seed_namespace"], family, replicate)
    torch.manual_seed(initialization_seed)
    np.random.seed(initialization_seed % (2**32 - 1))
    model = factory()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["training"]["learning_rate"]), weight_decay=float(CONFIG["training"]["weight_decay"]))
    train_episodes = [task.generate(seed, "development") for seed in train_seeds]
    validation_episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(train_episodes)
    loss_fn = torch.nn.MSELoss()
    latest_gradient_norm = 0.0
    curve = []
    requested = set(int(v) for v in CONFIG["training"]["curve_steps"])
    for step in range(1, int(CONFIG["training"]["steps"]) + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        predictions = model(observations, lengths, disable_messages=True)
        loss = loss_fn(predictions, targets); loss.backward()
        grads = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
        latest_gradient_norm = float(torch.linalg.vector_norm(torch.cat(grads)).item()) if grads else 0.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(CONFIG["training"]["gradient_clip"]))
        optimizer.step()
        if step in requested:
            with torch.no_grad():
                pred = model(observations, lengths, disable_messages=True)
                curve.append({"step": step, "training_loss": float(loss_fn(pred, targets).item()), "gradient_norm": latest_gradient_norm})
    def evaluate(episodes):
        obs, lens, target = episodes_to_batch(episodes)
        with torch.no_grad():
            pred = model(obs, lens, disable_messages=True)
        return float(loss_fn(pred, target).item()), _success_rate(task, pred, target)
    dev_loss, dev_success = evaluate(train_episodes)
    val_loss, val_success = evaluate(validation_episodes)
    diagnostics = _post_training_diagnostics(model, task, train_seeds, validation_seeds)
    unique_env_steps = sum(len(ep.observations) for ep in train_episodes + validation_episodes)
    return {
        "condition": condition,
        "family": family,
        "replicate": replicate,
        "projection_seed": int(CONFIG["projection_seed"]),
        "initialization_seed": initialization_seed,
        "graph_id": graph.graph_id,
        "development_seed_first": train_seeds[0],
        "development_seed_last": train_seeds[-1],
        "validation_seed_first": validation_seeds[0],
        "validation_seed_last": validation_seeds[-1],
        "development_success": dev_success,
        "validation_success": val_success,
        "development_loss": dev_loss,
        "validation_loss": val_loss,
        "capacity_demonstrated": capacity_demonstrated(dev_success, val_success),
        "learning_curve": curve,
        "diagnostics": diagnostics,
        "parameter_count": model.parameter_count(),
        "parameter_bytes": model.parameter_bytes(),
        "resources": {
            "candidate_evaluations": 1,
            "optimizer_steps": int(CONFIG["training"]["steps"]),
            "environment_steps": unique_env_steps,
            "examples_processed": int(CONFIG["training"]["steps"]) * len(train_episodes),
            "forward_calls": int(CONFIG["training"]["steps"]) + len(requested) + 2,
            "model_fits": 1,
            "input_edges": model.input_edge_count,
            "internal_message_edges": model.internal_message_edge_count,
            "disabled_message_edges": model.internal_message_edge_count,
            "parameter_count": model.parameter_count(),
            "model_parameter_bytes": model.parameter_bytes(),
            "wall_seconds": 0.0,
            "cpu_seconds": 0.0
        },
        "task_family_label_in_model_input": False,
        "dynamic_modulation_enabled": False,
        "messages_disabled_during_training": True,
        "fresh_audit_consumed": False,
        "gpu_seconds": 0.0,
    }


def _run_jobs(jobs: list[tuple], *, worker=_worker, output_path: Path) -> list[dict]:
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(worker, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{row['condition']} {row['family']} r{row['replicate']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}", flush=True)
    rows.sort(key=lambda row: (row["condition"], row["family"], row["replicate"], row.get("projection_seed") or -1))
    write_json(output_path, {"rows": rows, "fresh_audit_consumed": False})
    return rows


def _median_by_family(rows: list[dict]) -> dict[str, float]:
    return {
        family: float(np.median([row["validation_success"] for row in rows if row["family"] == family]))
        for family in FAMILIES
    }


def _baseline_compatibility(rows: list[dict]) -> dict:
    observed = _median_by_family(rows)
    expected = FROZEN["expected_q0_validation_medians"]
    threshold = float(CONFIG["baseline_drift_threshold"]["absolute_validation_delta"])
    drifted = [family for family in FAMILIES if abs(observed[family] - float(expected[family])) > threshold]
    family_passes = sum(
        int(capacity_demonstrated(
            float(np.median([row["development_success"] for row in rows if row["family"] == family])),
            observed[family],
        ))
        for family in FAMILIES
    )
    compatible = len(drifted) < int(CONFIG["baseline_drift_threshold"]["families_required"])
    return {
        "compatible": compatible,
        "observed_validation_medians": observed,
        "expected_validation_medians": expected,
        "absolute_deltas": {family: abs(observed[family] - float(expected[family])) for family in FAMILIES},
        "drifted_families": drifted,
        "families_passing": family_passes,
        "expected_families_passing": int(FROZEN["expected_q0_families_passing"]),
        "threshold": CONFIG["baseline_drift_threshold"],
    }


def _phase_baseline() -> None:
    jobs = [("Q0_local_10x4", family, replicate) for family in FAMILIES for replicate in range(int(CONFIG["training"]["replicates"]))]
    rows = _run_jobs(jobs, output_path=HERE / "raw" / "baseline_runs.json")
    compatibility = _baseline_compatibility(rows)
    write_json(HERE / "diagnostics" / "baseline_compatibility.json", compatibility)
    print("baseline compatibility:", json.dumps(compatibility, sort_keys=True), flush=True)
    if not compatibility["compatible"]:
        raise SystemExit("V837q BASELINE_DRIFT: stop before shared-state interpretation")


def _phase_primary() -> None:
    compatibility_path = HERE / "diagnostics" / "baseline_compatibility.json"
    if not compatibility_path.exists() or json.loads(compatibility_path.read_text(encoding="utf-8")).get("compatible") is not True:
        raise SystemExit("run a compatible --phase baseline before primary V837q conditions")
    jobs = [
        (condition, family, replicate)
        for condition in ["Q1_group5_5x8", "Q2_group2_2x20", "Q3_shared_1x40", *REFERENCES]
        for family in FAMILIES
        for replicate in range(int(CONFIG["training"]["replicates"]))
    ]
    _run_jobs(jobs, output_path=HERE / "raw" / "primary_runs.json")


def _phase_conditional() -> None:
    decision_path = HERE / "diagnostics" / "decision_state.json"
    if not decision_path.exists():
        raise SystemExit("run analyze_results.py before conditional V837q controls")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("q3_representation_adequacy_pass") is not True:
        raise SystemExit("Q3 did not pass >=4/5; no-message/projection-sensitivity controls are not authorized")
    nm_jobs = [("Q3_shared_1x40_no_messages", family, replicate) for family in FAMILIES for replicate in range(int(CONFIG["training"]["replicates"]))]
    _run_jobs(nm_jobs, worker=_train_with_messages_disabled, output_path=HERE / "raw" / "q3_no_message_runs.json")
    # Primary projection seed is one of five total seeds; add four independent seeds.
    seeds = [int(CONFIG["projection_seed"]) + offset for offset in range(1, 5)]
    projection_jobs = [
        ("Q3_shared_1x40", family, replicate, projection_seed)
        for projection_seed in seeds
        for family in FAMILIES
        for replicate in range(int(CONFIG["training"]["replicates"]))
    ]
    _run_jobs(projection_jobs, output_path=HERE / "raw" / "q3_projection_sensitivity_runs.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline", "primary", "conditional", "all"), default="all")
    args = parser.parse_args()
    _assert_science_locks()
    for name in ("raw", "diagnostics", "plots"):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    if args.phase in {"baseline", "all"}:
        _phase_baseline()
    if args.phase in {"primary", "all"}:
        _phase_primary()
    if args.phase == "conditional":
        _phase_conditional()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
