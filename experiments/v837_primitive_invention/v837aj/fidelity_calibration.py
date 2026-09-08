from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from experiments.v837_primitive_invention.common.reference_training import evaluate_sequence_model
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _configure_torch
from experiments.v837_primitive_invention.v837aj.af1d_structural_model import build_candidate_model, model_compute
from experiments.v837_primitive_invention.v837aj.topology import SearchTopology, calibration_panel

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = ("conditional_routing", "delayed_recall", "iterative_state", "partial_observation", "variable_composition")

STAGE_ANCHOR = "ANCHOR_REPRODUCTION"
STAGE_CALIBRATION = "CALIBRATION"
STAGE_SEARCH = "SEARCH"
STAGE_RANDOM_SEARCH = "RANDOM_SEARCH"
STAGE_FINALIZATION = "FINALIZATION"

SEARCH_TRAIN_POOL = tuple(range(10000, 10384))
SEARCH_SELECTION_POOL = tuple(range(10384, 10512))
FINAL_VALIDATION_POOL = tuple(range(20000, 20128))
FULL_DEVELOPMENT_POOL = tuple(range(10000, 10512))
FRESH_AUDIT_POOL = set(range(90000, 90500))


class FinalValidationLeakageError(RuntimeError):
    pass


def search_train_seeds(count: int) -> list[int]:
    count = int(count)
    if count < 1 or count > len(SEARCH_TRAIN_POOL):
        raise ValueError("invalid SEARCH_TRAIN count")
    return list(SEARCH_TRAIN_POOL[:count])


def search_selection_seeds(count: int = 128) -> list[int]:
    count = int(count)
    if count < 1 or count > len(SEARCH_SELECTION_POOL):
        raise ValueError("invalid SEARCH_SELECTION count")
    return list(SEARCH_SELECTION_POOL[:count])


def full_development_seeds() -> list[int]:
    return list(FULL_DEVELOPMENT_POOL)


def final_validation_seeds(stage: str) -> list[int]:
    if stage not in {STAGE_ANCHOR, STAGE_FINALIZATION}:
        raise FinalValidationLeakageError(f"STRUCTURAL_SEARCH_VALIDATION_LEAKAGE: {stage} requested FINAL_VALIDATION")
    return list(FINAL_VALIDATION_POOL)


def assert_data_partitions() -> dict:
    train = set(SEARCH_TRAIN_POOL)
    selection = set(SEARCH_SELECTION_POOL)
    final = set(FINAL_VALIDATION_POOL)
    full = set(FULL_DEVELOPMENT_POOL)
    all_task = full | final
    return {
        "search_train_pool": [min(train), max(train)],
        "search_train_count_per_family": len(train),
        "search_selection_pool": [min(selection), max(selection)],
        "search_selection_count_per_family": len(selection),
        "final_validation": [min(final), max(final)],
        "final_validation_count_per_family": len(final),
        "train_selection_disjoint": not bool(train & selection),
        "development_final_disjoint": not bool(full & final),
        "full_development_exact": sorted(full) == list(range(10000, 10512)),
        "union_unique_family_seed_episodes": len(all_task) * len(FAMILIES),
        "fresh_audit_overlap": len(all_task & FRESH_AUDIT_POOL),
        "exact": (
            sorted(train) == list(range(10000, 10384))
            and sorted(selection) == list(range(10384, 10512))
            and sorted(final) == list(range(20000, 20128))
        ),
    }


def _success_rate(task, prediction: torch.Tensor, targets: torch.Tensor) -> float:
    return float(np.mean([task.success(float(p), float(t)) for p, t in zip(prediction.tolist(), targets.tolist())]))


def _parameter_hash(model: torch.nn.Module) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        arr = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(b"\0"); digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def train_proxy_candidate(
    topology: SearchTopology,
    *,
    family: str,
    run_index: int,
    candidate_slot: int,
    fidelity: str,
    stage: str,
) -> dict:
    if stage not in {STAGE_CALIBRATION, STAGE_SEARCH, STAGE_RANDOM_SEARCH}:
        raise ValueError(f"proxy training invalid stage: {stage}")
    if fidelity not in CONFIG["fidelities"]:
        raise ValueError(f"unknown fidelity {fidelity}")
    spec = CONFIG["fidelities"][fidelity]
    train_seeds = search_train_seeds(int(spec["train_count"]))
    selection_seeds = search_selection_seeds(int(spec["selection_count"]))
    if set(train_seeds) & set(selection_seeds):
        raise RuntimeError("search train/selection overlap")
    if set(train_seeds + selection_seeds) & set(FINAL_VALIDATION_POOL):
        raise FinalValidationLeakageError("STRUCTURAL_SEARCH_VALIDATION_LEAKAGE")
    _configure_torch()
    task = task_by_name(family)
    model = build_candidate_model(topology, family, int(run_index), int(candidate_slot))
    initial_hash = _parameter_hash(model)
    train_episodes = [task.generate(seed, "development") for seed in train_seeds]
    selection_episodes = [task.generate(seed, "development") for seed in selection_seeds]
    observations, lengths, targets = episodes_to_batch(train_episodes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(CONFIG["training"]["learning_rate"]),
        weight_decay=float(CONFIG["training"]["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    start_wall = time.perf_counter(); start_cpu = time.process_time()
    backward_calls = 0; forward_calls = 0
    for _ in range(int(spec["steps"])):
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction = model(observations, lengths); forward_calls += 1
        loss = loss_fn(prediction, targets); loss.backward(); backward_calls += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(CONFIG["training"]["gradient_clip"]))
        optimizer.step()
    development = evaluate_sequence_model(model, task, train_episodes); forward_calls += 1
    selection = evaluate_sequence_model(model, task, selection_episodes); forward_calls += 1
    wall = time.perf_counter() - start_wall; cpu = time.process_time() - start_cpu
    compute = model_compute(topology)
    active_timesteps = int(sum(len(ep.observations) for ep in train_episodes)) * int(spec["steps"])
    fitness = float(selection.loss + float(CONFIG["lambda_edges"]) * topology.edge_count)
    environment_interactions = int(sum(len(ep.observations) for ep in train_episodes + selection_episodes))
    return {
        "version": "V837aj",
        "stage": stage,
        "family": family,
        "run_index": int(run_index),
        "candidate_initialization_slot": int(candidate_slot),
        "fidelity": fidelity,
        "topology": topology.to_dict(),
        "topology_id": topology.topology_id,
        "initial_parameter_hash": initial_hash,
        "train_count": len(train_seeds),
        "selection_count": len(selection_seeds),
        "steps": int(spec["steps"]),
        "development_loss": float(development.loss),
        "development_success": float(development.success_rate),
        "selection_loss": float(selection.loss),
        "selection_success": float(selection.success_rate),
        "fitness": fitness,
        "edge_penalty": float(CONFIG["lambda_edges"]) * topology.edge_count,
        "compute": compute,
        "optimizer_steps": int(spec["steps"]),
        "processed_examples": int(spec["steps"]) * len(train_episodes),
        "forward_calls": forward_calls,
        "backward_calls": backward_calls,
        "environment_interactions": environment_interactions,
        "active_training_timesteps": active_timesteps,
        "modeled_active_mac_volume": active_timesteps * int(compute["total_modeled_macs_per_timestep"]),
        "cpu_seconds": float(cpu),
        "wall_seconds": float(wall),
        "gpu_seconds": 0.0,
        "final_validation_accessed": False,
    }


def _average_tie_ranks(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman_rho(a: Iterable[float], b: Iterable[float]) -> float:
    ra, rb = _average_tie_ranks(a), _average_tie_ranks(b)
    if len(ra) < 2 or np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall_tau_b(a: Iterable[float], b: Iterable[float]) -> float:
    a = list(a); b = list(b)
    concordant = discordant = ties_a = ties_b = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da = np.sign(a[i] - a[j]); db = np.sign(b[i] - b[j])
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif da == db:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return float((concordant - discordant) / denom) if denom else 0.0


def pairwise_order_accuracy(a: Iterable[float], b: Iterable[float]) -> float:
    a = list(a); b = list(b); score = total = 0.0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da = np.sign(a[i] - a[j]); db = np.sign(b[i] - b[j]); total += 1
            if da == db:
                score += 1.0
            elif da == 0 or db == 0:
                score += 0.5
    return float(score / total) if total else 1.0


def topk_recall(proxy: list[float], target: list[float], k: int) -> float:
    proxy_ids = set(np.argsort(np.asarray(proxy), kind="mergesort")[:k].tolist())
    target_ids = set(np.argsort(np.asarray(target), kind="mergesort")[:k].tolist())
    return len(proxy_ids & target_ids) / float(k)


def aggregate_calibration_scores(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["fidelity"], row["family"], row["panel_id"])
        grouped.setdefault(key, []).append(row)
    output: dict[str, dict[str, dict[str, float]]] = {}
    for (fidelity, family, panel_id), group in grouped.items():
        output.setdefault(fidelity, {}).setdefault(family, {})[panel_id] = float(np.mean([r["fitness"] for r in group]))
    return output


def fidelity_metrics(rows: list[dict]) -> dict:
    aggregates = aggregate_calibration_scores(rows)
    panel_ids = list(CONFIG["calibration"]["panel_ids"])
    target = aggregates.get("F4", {})
    metrics: dict[str, dict] = {}
    for fidelity in ("F_LEGACY", "F0", "F1", "F2", "F3"):
        family_rows = {}
        for family in FAMILIES:
            proxy_values = [aggregates[fidelity][family][panel_id] for panel_id in panel_ids]
            target_values = [target[family][panel_id] for panel_id in panel_ids]
            family_rows[family] = {
                "spearman_rho": spearman_rho(proxy_values, target_values),
                "kendall_tau_b": kendall_tau_b(proxy_values, target_values),
                "pairwise_order_accuracy": pairwise_order_accuracy(proxy_values, target_values),
                "top4_recall": topk_recall(proxy_values, target_values, 4),
                "top2_recall": topk_recall(proxy_values, target_values, 2),
            }
        metrics[fidelity] = {
            "families": family_rows,
            "median_spearman_rho": float(np.median([v["spearman_rho"] for v in family_rows.values()])),
            "median_kendall_tau": float(np.median([v["kendall_tau_b"] for v in family_rows.values()])),
            "minimum_family_kendall_tau": float(min(v["kendall_tau_b"] for v in family_rows.values())),
            "median_pairwise_order_accuracy": float(np.median([v["pairwise_order_accuracy"] for v in family_rows.values()])),
            "median_top4_recall": float(np.median([v["top4_recall"] for v in family_rows.values()])),
            "minimum_family_top4_recall": float(min(v["top4_recall"] for v in family_rows.values())),
            "median_top2_recall": float(np.median([v["top2_recall"] for v in family_rows.values()])),
        }
    return {"aggregated_fitness": aggregates, "metrics": metrics}


def fidelity_passes(metric: dict) -> bool:
    gates = CONFIG["fidelity_gates"]
    return (
        metric["median_spearman_rho"] >= float(gates["median_spearman_rho"])
        and metric["median_kendall_tau"] >= float(gates["median_kendall_tau"])
        and metric["minimum_family_kendall_tau"] >= float(gates["minimum_family_kendall_tau"])
        and metric["median_pairwise_order_accuracy"] >= float(gates["median_pairwise_order_accuracy"])
        and metric["median_top4_recall"] >= float(gates["median_top4_recall"])
        and metric["minimum_family_top4_recall"] >= float(gates["minimum_family_top4_recall"])
    )


def select_cheapest_fidelity(metrics: dict[str, dict]) -> str | None:
    for fidelity in ("F0", "F1", "F2", "F3"):
        if fidelity_passes(metrics[fidelity]):
            return fidelity
    return None


def calibration_panel_payload() -> dict:
    panel = calibration_panel()
    return {
        "task_independent": True,
        "created_before_training_results": True,
        "topologies": {panel_id: topology.to_dict() for panel_id, topology in panel.items()},
    }
