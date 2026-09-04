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


def _temporal_variance(values: torch.Tensor, lengths: torch.Tensor) -> float:
    per_episode: list[float] = []
    for index, length in enumerate(lengths.tolist()):
        length = int(length)
        if length <= 1:
            continue
        per_episode.append(float(values[index, :length].var(dim=0, unbiased=False).mean().item()))
    return float(np.mean(per_episode)) if per_episode else 0.0


def _post_training_diagnostics(model: NeutralGraphModel, task, validation_seeds: list[int]) -> dict:
    episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, _ = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        _, trace = model(observations, lengths, return_trace=True)
    active_bt = torch.arange(trace.states.shape[1]).view(1, -1) < lengths.view(-1, 1)
    state_vectors = trace.states[active_bt].reshape(-1, trace.states.shape[-1])
    candidate_vectors = trace.candidate_states[active_bt].reshape(-1, trace.candidate_states.shape[-1])
    output = {
        "mean_state_norm": float(torch.linalg.vector_norm(state_vectors, dim=-1).mean().item()),
        "candidate_saturation_fraction": float((torch.abs(candidate_vectors) >= 0.95).float().mean().item()),
    }
    if model.state_modulation_mode != "none":
        modulators = trace.state_modulators
        if modulators is None:
            raise RuntimeError("dynamic state-modulation trace missing")
        active_modulators = modulators[active_bt].reshape(-1)
        values = active_modulators.detach().cpu().numpy()
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
    else:
        output["state_modulator"] = None
    return output


def _assert_science_locks() -> None:
    if gate_sha256() != CONFIG["historical_gate_hash"]:
        raise SystemExit("historical V837 gate hash mismatch")
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise SystemExit("capacity criterion fingerprint mismatch")
    v837o = json.loads((ROOT / "experiments/v837_primitive_invention/v837o/results.json").read_text(encoding="utf-8"))
    if v837o.get("mechanism_diagnosis") != CONFIG["required_v837o_diagnosis"]:
        raise SystemExit("V837o does not authorize this neutral follow-up")
    if v837o.get("neutral_followup_allowed") is not True or v837o.get("neutral_followup_type") != "single_dynamic_modulator":
        raise SystemExit("V837o decision state does not authorize the single dynamic modulator")
    audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
    if int(audit.get("episodes_consumed", -1)) != 0:
        raise SystemExit("fresh audit data has already been consumed")


def _model_factory(condition: str, replicate: int):
    graph = high_capacity_generic_graph(replicate)
    spec = CONFIG["conditions"][condition]
    kwargs = {
        "state_update_mode": spec["state_update_mode"],
        "state_modulation_mode": spec["state_modulation_mode"],
        "alpha_init": float(spec.get("alpha_init", 0.5)),
    }
    return graph, lambda: NeutralGraphModel(graph, obs_dim=6, state_dim=4, message_dim=4, **kwargs)


def _worker(condition: str, family: str, replicate: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    graph, factory = _model_factory(condition, replicate)
    training = CONFIG["training"]
    train_seeds = list(range(int(training["development_seed_range"][0]), int(training["development_seed_range"][1]) + 1))
    validation_seeds = list(range(int(training["validation_seed_range"][0]), int(training["validation_seed_range"][1]) + 1))
    initialization_seed = deterministic_int("v837p-init", family, replicate)
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
        "parameter_count": int(result.model.parameter_count()),
        "parameter_bytes": int(result.model.parameter_bytes()),
        "resources": result.resources.to_dict(),
        "task_family_label_in_model_input": False,
        "fresh_audit_consumed": False,
        "gpu_seconds": 0.0,
    }


def main() -> int:
    _assert_science_locks()
    (HERE / "raw").mkdir(exist_ok=True)
    (HERE / "diagnostics").mkdir(exist_ok=True)
    (HERE / "plots").mkdir(exist_ok=True)
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
            print(
                f"{row['condition']} {row['family']} r{row['replicate']}: "
                f"dev={row['development_success']:.3f} val={row['validation_success']:.3f}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["condition"], row["family"], row["replicate"]))
    write_json(HERE / "raw" / "runs.json", {"rows": rows, "fresh_audit_consumed": False})
    print(f"V837p raw runs complete: {len(rows)} fits", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
