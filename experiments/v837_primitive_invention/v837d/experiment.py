from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.evaluator import representation_diagnostics
from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.graph import (
    InputAccessSpec,
    build_fixed_sparse_mask,
    build_input_access_spec,
    degree_preserving_shuffled_mask,
)
from experiments.v837_primitive_invention.common.metrics import (
    binary_summary,
    bootstrap_mean_ci,
    continuous_summary,
    paired_bootstrap_difference,
)
from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.common.trainer import train_graph
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
TRAIN_SEEDS = list(range(30000, 30128))
VALIDATION_SEEDS = list(range(30200, 30328))
FAMILIES = [task.name for task in all_tasks()]


def mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(mask, dtype=np.float32).tobytes()).hexdigest()


def _access_for(replicate: int, mode: str, density: float | None = None, *, shuffled: bool = False) -> InputAccessSpec:
    n = int(CONFIG["high_capacity_graph"]["cells"])
    obs_dim = 6
    if mode == "broadcast":
        return build_input_access_spec("broadcast", obs_dim, n, density=1.0, seed=0)
    if mode == "none":
        return build_input_access_spec("none", obs_dim, n, density=0.0, seed=0)
    if mode != "fixed_sparse" or density is None:
        raise ValueError(f"unsupported access request: mode={mode} density={density}")
    mask_seed = int(CONFIG["sparse_mask_seeds"][replicate])
    base = build_fixed_sparse_mask(obs_dim, n, float(density), mask_seed)
    if shuffled:
        shuffle_seed = int(CONFIG["shuffle_mask_seeds"][replicate])
        mask = degree_preserving_shuffled_mask(base, shuffle_seed)
        seed = shuffle_seed
    else:
        mask = base
        seed = mask_seed
    spec = InputAccessSpec(
        mode="fixed_sparse",
        observation_dim=obs_dim,
        num_cells=n,
        mask=mask.tolist(),
        density=float(mask.mean()),
        seed=seed,
    )
    spec.validate()
    return spec


def train_one(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    family = str(payload["family"])
    replicate = int(payload["replicate"])
    condition = str(payload["condition"])
    density = payload.get("density")
    shuffled = bool(payload.get("shuffled", False))
    disable_messages = bool(payload.get("disable_messages", False))

    task = task_by_name(family)
    base_graph = high_capacity_generic_graph(replicate)
    graph = base_graph.clone()
    mode = "broadcast" if condition == "broadcast" else "fixed_sparse"
    graph.input_access = _access_for(replicate, mode, None if density is None else float(density), shuffled=shuffled)
    graph.validate(max_cells=16, max_edges=64)

    run_seed = deterministic_int("blocker-reference-train", family, replicate)
    # Use exactly the historical broadcast initialization seed for every paired
    # access condition. This keeps parameter initialization fixed while the
    # mask alone changes.
    init_seed = deterministic_int("train", base_graph.graph_id, task.name, run_seed)
    forward_options = {"disable_messages": True} if disable_messages else {}
    trained = train_graph(
        graph,
        task,
        TRAIN_SEEDS,
        VALIDATION_SEEDS,
        run_seed=run_seed,
        state_dim=4,
        message_dim=4,
        steps=int(CONFIG["training"]["steps"]),
        learning_rate=float(CONFIG["training"]["learning_rate"]),
        weight_decay=float(CONFIG["training"]["weight_decay"]),
        training_scope="full_adamw",
        forward_options=forward_options,
        initialization_seed_override=init_seed,
    )
    diagnostic_count = int(CONFIG["diagnostic_validation_episodes"])
    diagnostic_episodes = [task.generate(seed, "validation") for seed in VALIDATION_SEEDS[:diagnostic_count]]
    diagnostics = representation_diagnostics(
        trained.model,
        task,
        diagnostic_episodes,
        forward_options=forward_options,
        include_cell_ablations=True,
    )
    access = graph.input_access
    assert access is not None
    mask = np.asarray(access.mask, dtype=np.float32)
    return {
        "condition": condition,
        "family": family,
        "replicate": replicate,
        "requested_density": 1.0 if condition == "broadcast" else float(density),
        "effective_density": float(mask.mean()),
        "mask_seed": int(access.seed),
        "mask_sha256": mask_sha256(mask),
        "input_access": access.to_dict(),
        "graph_id": graph.graph_id,
        "base_historical_graph_id": base_graph.graph_id,
        "initialization_seed": int(init_seed),
        "development_success": trained.development.success_rate,
        "development_loss": trained.development.loss,
        "validation_success": trained.validation.success_rate,
        "validation_loss": trained.validation.loss,
        "capacity_demonstrated": capacity_demonstrated(trained.development.success_rate, trained.validation.success_rate),
        "no_message": disable_messages,
        "representation_diagnostics": diagnostics,
        "resources": trained.resources.to_dict(),
    }


def run_jobs(jobs: list[dict[str, Any]], *, workers: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_workers = int(workers or CONFIG["compute"]["workers"])
    with ProcessPoolExecutor(max_workers=min(max_workers, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(train_one, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            row = future.result()
            rows.append(row)
            print(
                f"{row['condition']} density={row['requested_density']:.3f} {row['family']} r{row['replicate']}: "
                f"dev={row['development_success']:.3f} val={row['validation_success']:.3f}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["condition"], row["requested_density"], row["family"], row["replicate"]))
    return rows


def summarize_condition(rows: list[dict[str, Any]], *, seed: int = 837) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty condition")
    per_family: dict[str, Any] = {}
    aggregate_passes = 0
    for family in FAMILIES:
        family_rows = sorted((row for row in rows if row["family"] == family), key=lambda row: row["replicate"])
        dev = np.asarray([row["development_success"] for row in family_rows], dtype=float)
        val = np.asarray([row["validation_success"] for row in family_rows], dtype=float)
        capacity = np.asarray([row["capacity_demonstrated"] for row in family_rows], dtype=bool)
        aggregate_pass = capacity_demonstrated(float(np.median(dev)), float(np.median(val)))
        aggregate_passes += int(aggregate_pass)
        per_family[family] = {
            "development": continuous_summary(dev),
            "validation": continuous_summary(val),
            "validation_bootstrap": bootstrap_mean_ci(val, seed=deterministic_int(seed, family, "validation")),
            "replicate_capacity_rate": binary_summary(capacity),
            "aggregate_capacity_pass": aggregate_pass,
        }

    by_replicate = []
    for replicate in sorted({int(row["replicate"]) for row in rows}):
        replicate_rows = [row for row in rows if int(row["replicate"]) == replicate]
        by_replicate.append(sum(int(row["capacity_demonstrated"]) for row in replicate_rows))
    diagnostic_keys = [
        "message_dependency_ratio",
        "mean_pairwise_state_corr",
        "median_pairwise_state_corr",
        "p90_pairwise_state_corr",
        "state_saturation_fraction",
        "mean_raw_ablation_effect",
        "median_raw_ablation_effect",
        "mean_message_ablation_effect",
        "median_message_ablation_effect",
        "message_magnitude",
        "raw_input_contribution_magnitude",
        "internal_message_contribution_magnitude",
        "recurrent_state_contribution_magnitude",
        "effective_active_cell_count",
    ]
    diagnostic_summary = {
        key: continuous_summary([float(row["representation_diagnostics"].get(key, 0.0)) for row in rows])
        for key in diagnostic_keys
    }
    return {
        "n_models": len(rows),
        "per_family": per_family,
        "families_passing_aggregate": aggregate_passes,
        "families_passing_per_replicate": continuous_summary(by_replicate),
        "median_families_passing_per_replicate": float(np.median(by_replicate)),
        "worst_family_median_validation": float(min(value["validation"]["median"] for value in per_family.values())),
        "effective_density": continuous_summary([row["effective_density"] for row in rows]),
        "representation_diagnostics": diagnostic_summary,
        "resource_accounting": {
            "candidate_evaluations": int(sum(row["resources"]["candidate_evaluations"] for row in rows)),
            "optimizer_steps": int(sum(row["resources"]["optimizer_steps"] for row in rows)),
            "environment_steps": int(sum(row["resources"]["environment_steps"] for row in rows)),
            "examples_processed": int(sum(row["resources"]["examples_processed"] for row in rows)),
            "forward_calls": int(sum(row["resources"].get("forward_calls", 0) for row in rows)),
            "wall_seconds_sum_workers": float(sum(row["resources"]["wall_seconds"] for row in rows)),
            "cpu_seconds_sum_workers": float(sum(row["resources"].get("cpu_seconds", 0.0) for row in rows)),
            "parameter_count": int(rows[0]["resources"]["parameter_count"]),
            "model_parameter_bytes": int(rows[0]["resources"].get("model_parameter_bytes", 0)),
            "input_edges": continuous_summary([row["resources"].get("input_edges", 0) for row in rows]),
            "internal_message_edges": int(rows[0]["resources"].get("internal_message_edges", 0)),
            "disabled_message_edges": int(rows[0]["resources"].get("disabled_message_edges", 0)),
        },
    }


def paired_family_deltas(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], *, seed: int = 837) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in FAMILIES:
        left = sorted((row for row in left_rows if row["family"] == family), key=lambda row: row["replicate"])
        right = sorted((row for row in right_rows if row["family"] == family), key=lambda row: row["replicate"])
        left_map = {int(row["replicate"]): float(row["validation_success"]) for row in left}
        right_map = {int(row["replicate"]): float(row["validation_success"]) for row in right}
        replicates = sorted(set(left_map) & set(right_map))
        a = np.asarray([left_map[index] for index in replicates], dtype=float)
        b = np.asarray([right_map[index] for index in replicates], dtype=float)
        output[family] = paired_bootstrap_difference(a, b, seed=deterministic_int(seed, family, "paired"))
    return output
