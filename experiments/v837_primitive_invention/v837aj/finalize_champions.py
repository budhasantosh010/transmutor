from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.v837_primitive_invention.common.reference_training import evaluate_sequence_model
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _configure_torch
from experiments.v837_primitive_invention.v837aj.af1d_structural_model import (
    build_finalization_model,
    finalization_common_seed,
    model_compute,
)
from experiments.v837_primitive_invention.v837aj.fidelity_calibration import (
    STAGE_FINALIZATION,
    final_validation_seeds,
    full_development_seeds,
)
from experiments.v837_primitive_invention.v837aj.topology import SearchTopology

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def finalization_protocol() -> dict:
    return {
        "development_episodes_per_family": 512,
        "optimizer_steps": int(CONFIG["training"]["full_steps"]),
        "final_validation_episodes_per_family": 128,
        "final_validation_data_accesses": 1,
        "proxy_weights_discarded": True,
        "finalization_namespace": "v837aj-finalize",
    }


def _success_rate(task, prediction: torch.Tensor, targets: torch.Tensor) -> float:
    return float(np.mean([task.success(float(p), float(t)) for p, t in zip(prediction.tolist(), targets.tolist())]))


def _parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        arr = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(b"\0"); digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def finalize_champion(champion_record: dict, *, engine: str) -> dict:
    family = str(champion_record["family"])
    run_index = int(champion_record["run_index"])
    champion = champion_record["champion"]
    if champion.get("selected_before_final_validation") is not True:
        raise RuntimeError("champion was not frozen before final validation")
    topology = SearchTopology.from_dict(champion["topology"])
    _configure_torch()
    task = task_by_name(family)
    model = build_finalization_model(topology, family, run_index)
    initial_hash = _parameter_hash(model)
    train_seeds = full_development_seeds()
    validation_seeds = final_validation_seeds(STAGE_FINALIZATION)
    train_episodes = [task.generate(seed, "development") for seed in train_seeds]
    # Final validation is materialized exactly once after topology freeze.
    validation_episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(train_episodes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(CONFIG["training"]["learning_rate"]),
        weight_decay=float(CONFIG["training"]["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    start_wall = time.perf_counter(); start_cpu = time.process_time()
    forward_calls = 0; backward_calls = 0
    for _ in range(int(CONFIG["training"]["full_steps"])):
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction = model(observations, lengths); forward_calls += 1
        loss = loss_fn(prediction, targets); loss.backward(); backward_calls += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(CONFIG["training"]["gradient_clip"]))
        optimizer.step()
    development = evaluate_sequence_model(model, task, train_episodes); forward_calls += 1
    val_observations, val_lengths, val_targets = episodes_to_batch(validation_episodes)
    model.eval()
    with torch.no_grad():
        validation_prediction, trace = model(val_observations, val_lengths, return_trace=True); forward_calls += 1
        no_message_prediction = model(val_observations, val_lengths, disable_messages=True); forward_calls += 1
    validation_loss = float(nn.functional.mse_loss(validation_prediction, val_targets).item())
    validation_success = _success_rate(task, validation_prediction, val_targets)
    no_message_success = _success_rate(task, no_message_prediction, val_targets)
    active = torch.arange(trace.messages.shape[1]).view(1, -1) < val_lengths.view(-1, 1)
    active_messages = trace.messages[active]
    per_cell_message_norm = [
        float(torch.linalg.vector_norm(active_messages[:, cell, :], dim=-1).mean().item())
        for cell in range(10)
    ]
    compute = model_compute(topology)
    active_timesteps = int(sum(len(ep.observations) for ep in train_episodes)) * int(CONFIG["training"]["full_steps"])
    cpu = time.process_time() - start_cpu; wall = time.perf_counter() - start_wall
    competent = (
        float(development.success_rate) >= float(CONFIG["training"]["family_development_threshold"])
        and validation_success >= float(CONFIG["training"]["family_validation_threshold"])
    )
    return {
        "version": "V837aj",
        "engine": engine,
        "family": family,
        "run_index": run_index,
        "topology": topology.to_dict(),
        "topology_id": topology.topology_id,
        "champion_search_fitness": float(champion["search_fitness"]),
        "champion_search_selection_success": float(champion["search_selection_success"]),
        "champion_selected_evaluation_index": int(champion["selected_evaluation_index"]),
        "champion_frozen_before_final_validation": True,
        "proxy_weights_discarded": True,
        "finalization_seed": int(finalization_common_seed(family, run_index)),
        "initial_parameter_hash": initial_hash,
        "development_success": float(development.success_rate),
        "development_loss": float(development.loss),
        "final_validation_success": validation_success,
        "final_validation_loss": validation_loss,
        "competent": bool(competent),
        "final_validation_data_accesses": 1,
        "final_validation_episode_count": 128,
        "message_dependence": {
            "messages_enabled_success": validation_success,
            "messages_disabled_success": no_message_success,
            "success_drop": validation_success - no_message_success,
            "mean_abs_prediction_change": float(torch.abs(validation_prediction - no_message_prediction).mean().item()),
            "per_cell_message_norm": per_cell_message_norm,
        },
        "compute": compute,
        "optimizer_steps": int(CONFIG["training"]["full_steps"]),
        "processed_examples": int(CONFIG["training"]["full_steps"]) * len(train_episodes),
        "forward_calls": forward_calls,
        "backward_calls": backward_calls,
        "environment_interactions": int(sum(len(ep.observations) for ep in train_episodes + validation_episodes)),
        "active_training_timesteps": active_timesteps,
        "modeled_active_mac_volume": active_timesteps * int(compute["total_modeled_macs_per_timestep"]),
        "cpu_seconds": float(cpu),
        "wall_seconds": float(wall),
        "gpu_seconds": 0.0,
    }


def _load_champion_runs(engine: str) -> list[dict]:
    path = HERE / "raw" / ("search_proxy_runs.json" if engine == "DIRECTED_STRUCTURAL_SEARCH" else "random_proxy_runs.json")
    if not path.is_file():
        raise RuntimeError(f"missing proxy runs for {engine}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("runs", []))


def _cache_path(engine: str, family: str, run_index: int) -> Path:
    prefix = "final_search" if engine == "DIRECTED_STRUCTURAL_SEARCH" else "final_random"
    return HERE / "raw" / "cache" / f"{prefix}_{family}_{int(run_index)}.json"


def _aggregate(engine: str) -> None:
    rows = []
    for family in ("conditional_routing", "delayed_recall", "iterative_state", "partial_observation", "variable_composition"):
        for run_index in range(10):
            path = _cache_path(engine, family, run_index)
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: (row["family"], int(row["run_index"])))
    target = HERE / "raw" / ("search_finalized.json" if engine == "DIRECTED_STRUCTURAL_SEARCH" else "random_finalized.json")
    target.write_text(json.dumps({"version":"V837aj","engine":engine,"rows":rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_finalization(engine: str, *, extension: bool) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    if extension:
        decision_path = HERE / "diagnostics/primary_decision.json"
        if not decision_path.is_file() or json.loads(decision_path.read_text(encoding="utf-8")).get("robustness_extension_required") is not True:
            raise SystemExit("V837aj robustness finalization blocked: primary machine trigger absent")
    runs = _load_champion_runs(engine)
    desired = set(range(5,10) if extension else range(0,5))
    selected = [run for run in runs if int(run["run_index"]) in desired]
    expected = 25
    if len(selected) != expected:
        raise SystemExit(f"expected {expected} {engine} champion runs, found {len(selected)}")
    jobs = []
    (HERE / "raw/cache").mkdir(parents=True, exist_ok=True)
    for run in selected:
        path = _cache_path(engine, run["family"], run["run_index"])
        if not path.is_file():
            jobs.append(run)
    if jobs:
        with ProcessPoolExecutor(max_workers=25) as pool:
            futures = {pool.submit(finalize_champion, run, engine=engine): run for run in jobs}
            for future in as_completed(futures):
                row = future.result()
                _cache_path(engine, row["family"], row["run_index"]).write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(f"final {engine} {row['family']} r{row['run_index']}: dev={row['development_success']:.6f} val={row['final_validation_success']:.6f} competent={row['competent']}", flush=True)
    _aggregate(engine)
    return 0


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("search","random"), required=True)
    parser.add_argument("--extension", action="store_true")
    args = parser.parse_args()
    engine = "DIRECTED_STRUCTURAL_SEARCH" if args.engine == "search" else "RANDOM_STRUCTURAL_SAMPLER"
    return run_finalization(engine, extension=bool(args.extension))


if __name__ == "__main__":
    raise SystemExit(main())
