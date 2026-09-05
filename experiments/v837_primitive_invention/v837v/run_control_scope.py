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
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _success_rate
from experiments.v837_primitive_invention.v837v.control_scope import DOMAIN_SPECS, NeutralControlScopeModel

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
CONDITIONS = list(CONFIG["conditions"])


def _blob_hash(path: str) -> str:
    return hashlib.sha256(subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)).hexdigest()


def _configure_torch() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _active_bt(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.arange(steps).view(1, -1) < lengths.view(-1, 1)


def _safe_corr(values: np.ndarray) -> float:
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return 0.0
    corr = np.corrcoef(values, rowvar=False)
    upper = corr[np.triu_indices(corr.shape[0], k=1)]
    upper = upper[np.isfinite(upper)]
    return float(np.mean(upper)) if upper.size else 0.0


def _effective_rank(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        return 1.0
    cov = np.cov(matrix, rowvar=False)
    vals = np.linalg.eigvalsh(np.atleast_2d(cov))
    vals = np.clip(vals, 0.0, None)
    total = float(vals.sum())
    if total <= 1e-12:
        return 1.0
    p = vals / total
    entropy = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    return float(np.exp(entropy))


def _gate_stats(gates: torch.Tensor, lengths: torch.Tensor, sources: tuple[int, ...]) -> dict:
    active = _active_bt(lengths, gates.shape[1])
    src = gates[:, :, list(sources), 0]
    flat = src[active].reshape(-1).detach().cpu().numpy()
    temporal = []
    disagreement = []
    for b, length in enumerate(lengths.tolist()):
        if length > 1:
            temporal.append(float(src[b, :length].var(dim=0, unbiased=False).mean().item()))
        if len(sources) > 1:
            disagreement.extend(src[b, :length].var(dim=1, unbiased=False).detach().cpu().tolist())
    return {
        "mean": float(np.mean(flat)), "median": float(np.median(flat)), "std": float(np.std(flat)),
        "p10": float(np.quantile(flat, 0.1)), "p90": float(np.quantile(flat, 0.9)),
        "temporal_variance": float(np.mean(temporal)) if temporal else 0.0,
        "near_zero_fraction": float(np.mean(flat <= 0.05)), "near_one_fraction": float(np.mean(flat >= 0.95)),
        "between_domain_gate_disagreement": float(np.mean(disagreement)) if disagreement else 0.0,
    }


def _state_diagnostics(trace, lengths: torch.Tensor) -> dict:
    states = trace.states
    candidates = trace.candidate_states
    gates = trace.state_modulators[..., 0]
    active = _active_bt(lengths, states.shape[1])
    flat_states = states[active].reshape(-1, states.shape[2], states.shape[3])
    flat_matrix = flat_states.reshape(flat_states.shape[0], -1).detach().cpu().numpy()
    norms = torch.linalg.vector_norm(flat_states, dim=-1).detach().cpu().numpy()
    pair_cos = []
    for sample in flat_states:
        normalized = torch.nn.functional.normalize(sample, dim=-1)
        sim = normalized @ normalized.T
        vals = sim[torch.triu_indices(sim.shape[0], sim.shape[1], offset=1).unbind()]
        pair_cos.append(float(vals.mean().item()))

    prev = torch.zeros_like(states)
    prev[:, 1:] = states[:, :-1]
    candidate_delta = torch.linalg.vector_norm(candidates - prev, dim=-1)
    realized_change = (1.0 - gates) * candidate_delta
    actual_change = torch.linalg.vector_norm(states - prev, dim=-1)
    active_realized = realized_change[active].detach().cpu().numpy()
    active_actual = actual_change[active].detach().cpu().numpy()
    state_change_corr = _safe_corr(active_actual)
    realized_corr = _safe_corr(active_realized)

    return {
        "pairwise_state_cosine_similarity": float(np.mean(pair_cos)) if pair_cos else 0.0,
        "state_covariance_trace": float(np.trace(np.cov(flat_matrix, rowvar=False))) if flat_matrix.shape[0] > 1 else 0.0,
        "effective_state_rank": _effective_rank(flat_matrix),
        "mean_state_norm": float(np.mean(norms)),
        "mean_state_change_magnitude": float(np.mean(active_actual)),
        "cross_cell_state_change_correlation": state_change_corr,
        "cross_cell_realized_update_correlation": realized_corr,
        "mean_candidate_update_magnitude": float(np.mean(candidate_delta[active].detach().cpu().numpy())),
        "mean_realized_update_magnitude": float(np.mean(active_realized)),
    }


def _diagnostics(model: NeutralControlScopeModel, task, val_seeds: list[int], condition: str) -> dict:
    episodes = [task.generate(seed, "validation") for seed in val_seeds]
    obs, lengths, targets = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        pred, trace = model(obs, lengths, return_trace=True)
        pred_no_message = model(obs, lengths, disable_messages=True)
    normal_success = _success_rate(task, pred, targets)
    no_message_success = _success_rate(task, pred_no_message, targets)
    spec = model.control_domain_spec
    result = {
        "source_gate": _gate_stats(trace.state_modulators, lengths, spec.source_cells),
        "state_coherence": _state_diagnostics(trace, lengths),
        "message_ablation": {
            "validation_success_normal": normal_success,
            "validation_success_no_message": no_message_success,
            "performance_drop": normal_success - no_message_success,
        },
    }
    if condition == "V0_10_domains":
        collapse = {}
        with torch.no_grad():
            for name in ("V1_5_domains", "V2_2_domains", "V3_1_domain"):
                p = model(obs, lengths, domain_spec_override=DOMAIN_SPECS[name])
                collapse[name] = _success_rate(task, p, targets)
        result["counterfactual_domain_collapse"] = collapse
    return result


def _assert_authorization() -> None:
    if gate_sha256() != CONFIG["historical_gate_hash"]:
        raise SystemExit("historical gate changed")
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise SystemExit("capacity criterion changed")
    if _blob_hash("experiments/v837_primitive_invention/v837u/results.json") != CONFIG["v837u_result_sha256"]:
        raise SystemExit("V837u result changed")
    if _blob_hash("experiments/v837_primitive_invention/v837u/diagnostics/decision_state.json") != CONFIG["v837u_decision_sha256"]:
        raise SystemExit("V837u decision changed")
    decision = json.loads((ROOT / "experiments/v837_primitive_invention/v837u/diagnostics/decision_state.json").read_text(encoding="utf-8"))
    if decision.get("diagnosis") != "DYNAMIC_SCALAR_CARRY_INSUFFICIENT" or decision.get("representation_adequacy_pass") is not False:
        raise SystemExit("V837v requires the frozen failed V837u scalar-carry frontier")
    audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
    if audit.get("episodes_consumed") != 0:
        raise SystemExit("fresh audit consumed")
    if CONFIG.get("gate_pooling") is not False or CONFIG.get("global_controller") is not False:
        raise SystemExit("V837v primary experiment may not pool gates or introduce a global controller")


def _worker(condition: str, family: str, replicate: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    graph = high_capacity_generic_graph(replicate)
    tr = CONFIG["training"]
    train_seeds = list(range(tr["development_seed_range"][0], tr["development_seed_range"][1] + 1))
    val_seeds = list(range(tr["validation_seed_range"][0], tr["validation_seed_range"][1] + 1))
    init_seed = deterministic_int(tr["initialization_namespace"], family, replicate)
    spec = DOMAIN_SPECS[condition]
    result = train_sequence_model(
        model_factory=lambda: NeutralControlScopeModel(graph, domain_spec=spec, obs_dim=6),
        task=task, train_seeds=train_seeds, validation_seeds=val_seeds,
        initialization_seed=init_seed, steps=tr["steps"], learning_rate=tr["learning_rate"],
        weight_decay=tr["weight_decay"], gradient_clip=tr["gradient_clip"], curve_steps=tuple(tr["curve_steps"]),
    )
    model = result.model
    diag = _diagnostics(model, task, val_seeds, condition)
    return {
        "version": "V837v", "condition": condition, "family": family, "replicate": replicate,
        "domain_count": spec.domain_count, "domain_assignment": [list(d) for d in spec.domains], "source_cells": list(spec.source_cells),
        "nominal_controller_count": model.nominal_controller_count, "active_controller_count": model.active_controller_count,
        "nominal_controller_parameters": model.nominal_controller_parameter_count, "active_controller_parameters": model.active_controller_parameter_count,
        "development_success": result.development.success_rate, "validation_success": result.validation.success_rate,
        "development_loss": result.development.loss, "validation_loss": result.validation.loss,
        "capacity_pass": capacity_demonstrated(result.development.success_rate, result.validation.success_rate),
        "nominal_parameters": model.parameter_count(), "active_parameters": model.parameter_count() - (model.nominal_controller_parameter_count - model.active_controller_parameter_count),
        "controller_macs_per_timestep": model.controller_macs_per_timestep, "base_macs_per_timestep": 160,
        "total_macs_per_timestep": model.total_recurrent_controller_macs_per_timestep,
        "diagnostics": diag, "resources": result.resources.to_dict(), "gpu_seconds": 0.0,
        "processed_examples": result.resources.examples_processed, "unique_seed_policy": "same 3200 family/seed episodes reused across conditions and replicates",
        "development_seed_range": tr["development_seed_range"], "validation_seed_range": tr["validation_seed_range"],
        "model_init_seed": init_seed, "controller_information_scope": "source_cell_local_only",
        "gate_pooling": False, "global_state_visibility": False, "task_family_label_in_model_input": False,
        "fresh_audit_consumed": False,
    }


def _run_jobs(conditions: list[str], output: Path) -> list[dict]:
    jobs = [(c, f, r) for c in conditions for f in FAMILIES for r in range(CONFIG["training"]["replicates"])]
    rows = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, *job): job for job in jobs}
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(f"{row['condition']} {row['family']} r{row['replicate']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}", flush=True)
    rows.sort(key=lambda r: (r["condition"], r["family"], r["replicate"]))
    write_json(output, {"rows": rows, "unique_seed_defined_episodes": 3200, "fresh_audit_consumed": False})
    return rows


def _baseline_guard(rows: list[dict]) -> dict:
    medians = {}
    drifted = []
    for family in FAMILIES:
        vals = [r["validation_success"] for r in rows if r["family"] == family]
        median = float(np.median(vals))
        expected = float(CONFIG["expected_v0_validation_medians"][family])
        delta = median - expected
        medians[family] = {"observed": median, "expected": expected, "absolute_difference": abs(delta), "difference": delta}
        if abs(delta) > CONFIG["baseline_drift_threshold"]:
            drifted.append(family)
    families_passing = sum(capacity_demonstrated(
        float(np.median([r["development_success"] for r in rows if r["family"] == family])),
        float(np.median([r["validation_success"] for r in rows if r["family"] == family])),
    ) for family in FAMILIES)
    compatible = families_passing == CONFIG["expected_v0_families_passing"] and len(drifted) < 2
    return {"compatible": bool(compatible), "families_passing": int(families_passing), "drifted_families": drifted, "family_medians": medians}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline", "full"), default="full")
    args = parser.parse_args()
    _assert_authorization()
    for name in ("raw", "diagnostics", "plots"):
        (HERE / name).mkdir(exist_ok=True)

    baseline_path = HERE / "raw" / "v0_runs.json"
    if baseline_path.exists():
        baseline_rows = json.loads(baseline_path.read_text(encoding="utf-8"))["rows"]
    else:
        baseline_rows = _run_jobs(["V0_10_domains"], baseline_path)
    guard = _baseline_guard(baseline_rows)
    write_json(HERE / "diagnostics" / "baseline_compatibility.json", guard)
    print(json.dumps(guard, indent=2))
    if not guard["compatible"]:
        raise SystemExit("CONTROL_SCOPE_BASELINE_DRIFT")
    if args.phase == "baseline":
        return 0

    remaining_path = HERE / "raw" / "shared_scope_runs.json"
    if remaining_path.exists():
        remaining_rows = json.loads(remaining_path.read_text(encoding="utf-8"))["rows"]
    else:
        remaining_rows = _run_jobs(CONDITIONS[1:], remaining_path)
    all_rows = baseline_rows + remaining_rows
    all_rows.sort(key=lambda r: (r["condition"], r["family"], r["replicate"]))
    write_json(HERE / "raw" / "runs.json", {"rows": all_rows, "unique_seed_defined_episodes": 3200, "fresh_audit_consumed": False})
    print(f"V837v raw runs complete: {len(all_rows)} fits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
