from __future__ import annotations

import hashlib
import json
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
    local_recurrent_macs,
)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
CONDITIONS = list(CONFIG["conditions"])


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
    if _git_blob_sha256("experiments/v837_primitive_invention/v837r/results.json") != CONFIG["v837r_result_sha256"]:
        raise SystemExit("V837r result changed after interaction authorization")
    if _git_blob_sha256("experiments/v837_primitive_invention/v837p/results.json") != CONFIG["v837p_result_sha256"]:
        raise SystemExit("V837p result changed")
    decision = json.loads((ROOT / "experiments/v837_primitive_invention/v837r/diagnostics/decision_state.json").read_text(encoding="utf-8"))
    if decision.get("v837r_complete") is not True:
        raise SystemExit("V837r is not complete")
    if decision.get("diagnosis") != CONFIG["required_v837r_diagnosis"]:
        raise SystemExit("V837r diagnosis does not authorize this factorial")
    if decision.get("best_condition") != CONFIG["required_v837r_best_condition"]:
        raise SystemExit("V837s must use the frozen V837r best condition")
    if decision.get("interaction_followup_allowed") is not True:
        raise SystemExit("V837r interaction guard blocks V837s")
    if CONFIG.get("fresh_audit_allowed") is not False or CONFIG.get("structural_search_allowed") is not False or CONFIG.get("primitive_mining_allowed") is not False or CONFIG.get("v838_allowed") is not False:
        raise SystemExit("V837s science locks changed")
    audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
    if int(audit.get("episodes_consumed", -1)) != 0:
        raise SystemExit("fresh-audit data has already been consumed")


def _training_seeds() -> tuple[list[int], list[int]]:
    t = CONFIG["training"]
    train = list(range(int(t["development_seed_range"][0]), int(t["development_seed_range"][1]) + 1))
    val = list(range(int(t["validation_seed_range"][0]), int(t["validation_seed_range"][1]) + 1))
    if len(train) != int(t["train_episodes"]) or len(val) != int(t["validation_episodes"]):
        raise RuntimeError("V837s seed range/budget mismatch")
    return train, val


def _coupling_spec(condition: str, replicate: int) -> RecurrentCouplingSpec:
    row = CONFIG["conditions"][condition]
    if row["global_coupling"]:
        seed = deterministic_int(
            CONFIG["training"]["coupling_seed_namespace"],
            CONFIG["training"]["coupling_seed_condition_key"],
            replicate,
        )
        return RecurrentCouplingSpec(
            mode="low_rank",
            rank=int(CONFIG["coupling_rank"]),
            cross_block_only=True,
            scaling=float(CONFIG["coupling_scaling"]),
            initialization_seed=seed,
        )
    return RecurrentCouplingSpec(mode="none", cross_block_only=True, scaling=1.0, initialization_seed=0)


def _model_factory(condition: str, replicate: int):
    graph = high_capacity_generic_graph(replicate)
    coupling = _coupling_spec(condition, replicate)
    modulation = CONFIG["conditions"][condition]["state_modulation_mode"]
    return graph, coupling, lambda: GloballyCoupledNeutralGraphModel(
        graph,
        coupling,
        obs_dim=6,
        state_modulation_mode=modulation,
    )


def _success_rate(task, predictions: torch.Tensor, targets: torch.Tensor) -> float:
    p = predictions.detach().cpu().tolist()
    y = targets.detach().cpu().tolist()
    return float(np.mean([bool(task.success(float(a), float(b))) for a, b in zip(p, y)]))


def _temporal_variance(values: torch.Tensor, lengths: torch.Tensor) -> float:
    rows = []
    for i, length in enumerate(lengths.tolist()):
        length = int(length)
        if length > 1:
            rows.append(float(values[i, :length].var(dim=0, unbiased=False).mean().item()))
    return float(np.mean(rows)) if rows else 0.0


def _term_norm(tensor: torch.Tensor, lengths: torch.Tensor) -> float:
    active = torch.arange(tensor.shape[1]).view(1, -1) < lengths.view(-1, 1)
    values = tensor[active]
    return 0.0 if not values.numel() else float(torch.linalg.vector_norm(values.reshape(values.shape[0], -1), dim=1).mean().item())


def _diagnostics(model: GloballyCoupledNeutralGraphModel, task, validation_seeds: list[int]) -> dict:
    episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        pred, trace = model(observations, lengths, return_trace=True)
        no_message = model(observations, lengths, disable_messages=True)
    local_norm = _term_norm(trace.recurrent_terms, lengths)
    global_norm = _term_norm(trace.global_recurrent_terms, lengths)
    message_norm = _term_norm(trace.message_terms, lengths)
    output = {
        "coupling": model.coupling_diagnostics(),
        "local_recurrent_term_norm": local_norm,
        "global_recurrent_term_norm": global_norm,
        "message_term_norm": message_norm,
        "global_to_local_ratio": global_norm / (local_norm + 1e-12),
        "global_to_message_ratio": global_norm / (message_norm + 1e-12),
        "message_dependency": {
            "baseline_success": _success_rate(task, pred, targets),
            "no_message_success": _success_rate(task, no_message, targets),
            "mean_abs_prediction_delta": float(torch.mean(torch.abs(no_message - pred)).item()),
        },
    }
    modulators = trace.state_modulators
    if model.state_modulation_mode == "none":
        output["state_modulator"] = None
    else:
        if modulators is None:
            raise RuntimeError("V837s modulator trace missing")
        active = torch.arange(modulators.shape[1]).view(1, -1) < lengths.view(-1, 1)
        values = modulators[active].reshape(-1).detach().cpu().numpy()
        output["state_modulator"] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
            "temporal_variance": _temporal_variance(modulators, lengths),
            "near_zero_fraction": float(np.mean(values <= 0.05)),
            "near_one_fraction": float(np.mean(values >= 0.95)),
        }
    return output


def _worker(condition: str, family: str, replicate: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    train_seeds, validation_seeds = _training_seeds()
    graph, coupling, factory = _model_factory(condition, replicate)
    t = CONFIG["training"]
    init_seed = deterministic_int(t["initialization_seed_namespace"], family, replicate)
    result = train_sequence_model(
        model_factory=factory,
        task=task,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        initialization_seed=init_seed,
        steps=int(t["steps"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        gradient_clip=float(t["gradient_clip"]),
        curve_steps=tuple(t["curve_steps"]),
    )
    diagnostics = _diagnostics(result.model, task, validation_seeds)
    coupling_macs = coupling_actual_macs(coupling)
    modulator_macs = 0 if result.model.state_modulation_mode == "none" else 14 * 10
    return {
        "condition": condition,
        "family": family,
        "replicate": int(replicate),
        "initialization_seed": int(init_seed),
        "coupling_initialization_seed": int(coupling.initialization_seed),
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
        "coupling_spec": coupling.to_dict(),
        "state_modulation_mode": result.model.state_modulation_mode,
        "parameter_count": result.model.parameter_count(),
        "parameter_bytes": result.model.parameter_bytes(),
        "recurrent_macs_per_timestep": local_recurrent_macs() + coupling_macs + modulator_macs,
        "coupling_macs_per_timestep": coupling_macs,
        "modulator_macs_per_timestep": modulator_macs,
        "resources": result.resources.to_dict(),
        "task_family_label_in_model_input": False,
        "fresh_audit_consumed": False,
        "gpu_seconds": 0.0,
    }


def main() -> int:
    _assert_science_locks()
    for name in ("raw", "diagnostics", "plots"):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    jobs = [
        (condition, family, replicate)
        for condition in CONDITIONS
        for family in FAMILIES
        for replicate in range(int(CONFIG["training"]["replicates"]))
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{row['condition']} {row['family']} r{row['replicate']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}", flush=True)
    rows.sort(key=lambda row: (row["condition"], row["family"], row["replicate"]))
    write_json(HERE / "raw" / "runs.json", {"rows": rows, "fresh_audit_consumed": False})
    print(f"V837s interaction runs complete: {len(rows)} fits", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
