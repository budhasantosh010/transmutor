from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
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
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _configure_torch
from experiments.v837_primitive_invention.v837t.gru_dynamic_granularity import DynamicGranularityGRU
from experiments.v837_primitive_invention.v837ab.input_factorization import (
    CONDITIONS,
    HIDDEN_SIZE,
    INPUT_DIM,
    InputFactorizationT2,
    max_trace_errors,
)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
AB0 = "AB0_exact_factorized_t2"
OTHERS = [c for c in CONFIG["conditions"] if c != AB0]


def _git_blob_sha256(path: str) -> str:
    return hashlib.sha256(subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)).hexdigest()


def _assert_locks() -> None:
    if gate_sha256() != CONFIG["historical_gate_hash"]:
        raise SystemExit("historical V837 gate changed")
    if v837_capacity_criterion_sha256() != CONFIG["capacity_criterion_hash"]:
        raise SystemExit("capacity criterion changed")
    frozen = {
        "experiments/v837_primitive_invention/v837t/config.json": CONFIG["v837t_config_sha256"],
        "experiments/v837_primitive_invention/v837t/gru_dynamic_granularity.py": CONFIG["v837t_model_sha256"],
        "experiments/v837_primitive_invention/v837t/results.json": CONFIG["v837t_results_sha256"],
        "experiments/v837_primitive_invention/v837aa/results.json": CONFIG["v837aa_results_sha256"],
        "experiments/v837_primitive_invention/v837aa/diagnostics/decision_state.json": CONFIG["v837aa_decision_sha256"],
    }
    for path, expected in frozen.items():
        actual = _git_blob_sha256(path)
        if actual != expected:
            raise SystemExit(f"frozen parent changed: {path}: {actual} != {expected}")
    aa = json.loads((ROOT / "experiments/v837_primitive_invention/v837aa/diagnostics/decision_state.json").read_text(encoding="utf-8"))
    if aa.get("candidate_law_diagnosis") != "GENUINELY_DIVERSE_CANDIDATE_LAWS" or aa.get("recommended_next_axis") != "NEXT_AXIS_SHARED_INPUT_REPRESENTATION":
        raise SystemExit("V837aa does not authorize input-factorization localization")
    audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
    if audit.get("episodes_consumed") != 0:
        raise SystemExit("fresh audit consumed")
    if (ROOT / "experiments/v837_primitive_invention/v838").exists():
        raise SystemExit("V838 exists")


def _seeds() -> tuple[list[int], list[int]]:
    tr = CONFIG["training"]
    train = list(range(tr["development_seed_range"][0], tr["development_seed_range"][1] + 1))
    validation = list(range(tr["validation_seed_range"][0], tr["validation_seed_range"][1] + 1))
    return train, validation


def _init_seed(family: str, replicate: int) -> int:
    return deterministic_int(CONFIG["training"]["initialization_namespace"], family, replicate)


def _norm(t: torch.Tensor | None) -> float:
    if t is None:
        return 0.0
    return float(torch.linalg.vector_norm(t.detach()).item())


def _initial_signature(model: InputFactorizationT2) -> dict:
    output = {"projection_weight": None, "projection_bias": None, "effective": {}}
    if model.projection_active:
        output["projection_weight"] = model.projection_weight.detach().clone()
        output["projection_bias"] = model.projection_bias.detach().clone()
    for gate in ("n", "z"):
        w, b = model.effective_input_map(gate)
        output["effective"][gate] = (w.detach().clone(), b.detach().clone())
    return output


def _trajectory(model: InputFactorizationT2, initial: dict, step: int) -> dict:
    row = {"step": int(step), **model.factorization_snapshot(), "distance_from_initialization": {}}
    if model.projection_active:
        row["distance_from_initialization"]["projection_weight"] = _norm(model.projection_weight - initial["projection_weight"])
        row["distance_from_initialization"]["projection_bias"] = _norm(model.projection_bias - initial["projection_bias"])
    else:
        row["distance_from_initialization"]["projection_weight"] = 0.0
        row["distance_from_initialization"]["projection_bias"] = 0.0
    for gate in ("n", "z"):
        w, b = model.effective_input_map(gate)
        iw, ib = initial["effective"][gate]
        row["distance_from_initialization"][f"effective_{gate}_weight"] = _norm(w - iw)
        row["distance_from_initialization"][f"effective_{gate}_bias"] = _norm(b - ib)
    return row


def _gradient_norm(parameters) -> float:
    chunks = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(chunks)).item()) if chunks else 0.0


def _train(condition: str, family: str, replicate: int) -> tuple[InputFactorizationT2, dict]:
    _configure_torch()
    task = task_by_name(family)
    train_seeds, validation_seeds = _seeds()
    tr = CONFIG["training"]
    initialization_seed = _init_seed(family, replicate)
    torch.manual_seed(int(initialization_seed)); np.random.seed(int(initialization_seed) % (2**32 - 1))
    model = InputFactorizationT2(condition=condition)
    initial = _initial_signature(model)
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(tr["learning_rate"]), weight_decay=float(tr["weight_decay"]))
    loss_fn = nn.MSELoss()
    train_episodes = [task.generate(seed, "development") for seed in train_seeds]
    validation_episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(train_episodes)
    env_steps = sum(len(ep.observations) for ep in train_episodes + validation_episodes)
    resources = ResourceAccounting(
        candidate_evaluations=1, optimizer_steps=0, environment_steps=env_steps, examples_processed=0,
        model_fits=1, parameter_count=model.nominal_parameter_count(), model_parameter_bytes=model.parameter_bytes(),
    )
    requested = sorted(set(int(s) for s in tr["curve_steps"]))
    curve: list[dict] = []
    trajectories: list[dict] = []
    latest_gradient_norm = 0.0

    def record(step: int) -> None:
        dev = evaluate_sequence_model(model, task, train_episodes)
        val = evaluate_sequence_model(model, task, validation_episodes)
        resources.forward_calls += 2
        curve.append({"step": int(step), "training_loss": dev.loss, "training_success": dev.success_rate, "validation_loss": val.loss, "validation_success": val.success_rate, "gradient_norm": latest_gradient_norm})
        trajectories.append(_trajectory(model, initial, step))

    with WallTimer() as timer:
        if 0 in requested:
            record(0)
        for step in range(1, int(tr["steps"]) + 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            pred = model(observations, lengths); resources.forward_calls += 1
            loss = loss_fn(pred, targets); loss.backward()
            latest_gradient_norm = _gradient_norm(params)
            torch.nn.utils.clip_grad_norm_(params, float(tr["gradient_clip"])); optimizer.step()
            resources.optimizer_steps += 1; resources.examples_processed += len(train_episodes)
            if step in requested:
                record(step)
        development = evaluate_sequence_model(model, task, train_episodes)
        validation = evaluate_sequence_model(model, task, validation_episodes)
        resources.forward_calls += 2
    resources.wall_seconds = timer.seconds; resources.cpu_seconds = timer.cpu_seconds
    return model, {
        "initialization_seed": initialization_seed,
        "development_success": development.success_rate, "validation_success": validation.success_rate,
        "development_loss": development.loss, "validation_loss": validation.loss,
        "loss_curve": curve, "factorization_trajectory": trajectories, "resources": resources.to_dict(),
        "processed_examples": resources.examples_processed,
    }


def _projection_gradient_decomposition(model: InputFactorizationT2, family: str, replicate: int) -> tuple[dict, int, int]:
    if model.condition != AB0:
        return {"applicable": False}, 0, 0
    task = task_by_name(family); train_seeds, _ = _seeds()
    episodes = [task.generate(seed, "development") for seed in train_seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    loss_fn = nn.MSELoss()

    def grad(*, detach_candidate: bool, detach_update: bool) -> torch.Tensor:
        model.zero_grad(set_to_none=True); model.train()
        pred = model(observations, lengths, detach_projection_for_candidate=detach_candidate, detach_projection_for_update=detach_update)
        loss = loss_fn(pred, targets); loss.backward()
        chunks = []
        for p in (model.projection_weight, model.projection_bias):
            chunks.append(torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1).clone())
        return torch.cat(chunks)

    full = grad(detach_candidate=False, detach_update=False)
    candidate = grad(detach_candidate=False, detach_update=True)
    update = grad(detach_candidate=True, detach_update=False)
    residual = full - candidate - update
    denom = max(float(torch.linalg.vector_norm(full).item()), 1e-12)
    c_norm = float(torch.linalg.vector_norm(candidate).item()); u_norm = float(torch.linalg.vector_norm(update).item())
    cosine = float(torch.dot(candidate, update).item() / max(c_norm * u_norm, 1e-12))
    model.zero_grad(set_to_none=True)
    return {
        "applicable": True,
        "family": family, "replicate_id": replicate,
        "grad_candidate_norm": c_norm,
        "grad_update_norm": u_norm,
        "candidate_update_cosine": cosine,
        "grad_full_norm": float(torch.linalg.vector_norm(full).item()),
        "sum_residual_norm": float(torch.linalg.vector_norm(residual).item()),
        "sum_residual_relative": float(torch.linalg.vector_norm(residual).item() / denom),
        "sum_identity_within_1e_5_relative": bool(torch.linalg.vector_norm(residual).item() / denom <= 1e-5),
    }, 3, 3


def _worker(condition: str, family: str, replicate: int) -> dict:
    model, training = _train(condition, family, replicate)
    grad_diag, diagnostic_forward, diagnostic_backward = _projection_gradient_decomposition(model, family, replicate)
    resources = dict(training["resources"])
    resources["diagnostic_forward_calls"] = diagnostic_forward
    resources["diagnostic_backward_calls"] = diagnostic_backward
    return {
        "version": "V837ab", "condition": condition, "family": family, "replicate_id": replicate,
        "model_init_seed": training["initialization_seed"],
        "development_success": training["development_success"], "validation_success": training["validation_success"],
        "development_loss": training["development_loss"], "validation_loss": training["validation_loss"],
        "capacity_demonstrated": capacity_demonstrated(training["development_success"], training["validation_success"]),
        "loss_curve": training["loss_curve"], "factorization_trajectory": training["factorization_trajectory"],
        "projection_diagnostics_final": model.projection_diagnostics(), "projection_gradient_decomposition": grad_diag,
        "nominal_parameters": model.nominal_parameter_count(), "trainable_parameters": model.trainable_parameter_count(),
        "active_parameters": model.active_parameter_count(), "executed_parameters": model.executed_parameter_count(),
        "executed_frozen_parameters": model.executed_frozen_parameter_count(),
        "projection_specific_macs": model.projection_specific_macs,
        "active_reference_macs_per_timestep": model.active_reference_macs_per_timestep,
        "resources": resources, "processed_examples": training["processed_examples"],
        "unique_seed_defined_episode_policy": "same 3200 family/seed episodes reused across all conditions and replicates",
        "fresh_audit_consumed": False, "structural_search_allowed": False, "primitive_mining_allowed": False,
        "gpu_seconds": 0.0, "v838_started": False,
    }


def _run(conditions: list[str]) -> list[dict]:
    jobs = [(c, f, r) for c in conditions for f in FAMILIES for r in range(CONFIG["training"]["replicates"])]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            print(f"{row['condition']} {row['family']} r{row['replicate_id']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}", flush=True)
    rows.sort(key=lambda r: (r["condition"], r["family"], r["replicate_id"]))
    return rows


def _summary(rows: list[dict], condition: str) -> dict:
    result = {"families_passing": 0, "family_development_medians": {}, "family_validation_medians": {}}
    for family in FAMILIES:
        fr = [r for r in rows if r["condition"] == condition and r["family"] == family]
        dev = float(np.median([r["development_success"] for r in fr])); val = float(np.median([r["validation_success"] for r in fr]))
        result["family_development_medians"][family] = dev; result["family_validation_medians"][family] = val
        result["families_passing"] += int(capacity_demonstrated(dev, val))
    return result


def _historical_t2() -> dict:
    results = json.loads((ROOT / "experiments/v837_primitive_invention/v837t/results.json").read_text(encoding="utf-8"))
    row = results["conditions"]["T2_scalarized_update_no_reset"]
    return {
        "families_passing": int(row["families_passing"]),
        "family_validation_medians": {family: float(v["validation"]["median"]) for family, v in row["family_results"].items()},
        "family_development_medians": {family: float(v["development"]["median"]) for family, v in row["family_results"].items()},
    }


def _ab0_guard(rows: list[dict]) -> dict:
    observed = _summary(rows, AB0); expected = _historical_t2()
    deltas = {family: abs(observed["family_validation_medians"][family] - expected["family_validation_medians"][family]) for family in FAMILIES}
    valid = observed["families_passing"] >= CONFIG["representation_family_gate"] and max(deltas.values()) <= 0.10
    return {"reference_baseline_valid": bool(valid), "expected": expected, "observed": observed, "validation_median_absolute_deltas": deltas, "max_validation_median_absolute_delta": max(deltas.values())}


def _reference_equivalence() -> dict:
    torch.manual_seed(731); source = DynamicGranularityGRU(HIDDEN_SIZE, INPUT_DIM, condition="T2_scalarized_update_no_reset")
    optimizer = torch.optim.AdamW(source.parameters(), lr=0.005, weight_decay=0.0001)
    torch.manual_seed(732); observations = torch.randn(16, 9, INPUT_DIM); lengths = torch.tensor([9,9,9,9,8,8,8,8,7,7,6,6,5,5,4,3]); targets = torch.tanh(torch.randn(16))
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True); pred = source(observations, lengths); loss = torch.mean((pred-targets)**2); loss.backward(); optimizer.step()
    folded = InputFactorizationT2.from_trained_t2(source, condition="AB1_fully_folded_equivalent")
    errors = max_trace_errors(source, folded, observations, lengths)
    maximum = max(errors.values())
    return {"function_class_equivalence_proven": bool(maximum <= CONFIG["folding_tolerance"]), "max_abs_errors": errors, "maximum_error": maximum, "tolerance": CONFIG["folding_tolerance"], "probe_training_steps": 3, "probe_uses_task_episodes": False, "conclusion": "PROJECTION DOES NOT ADD FUNCTION-CLASS CAPACITY" if maximum <= CONFIG["folding_tolerance"] else "FOLDING_EQUIVALENCE_FAILURE"}


def _step0_equivalence() -> dict:
    train_seeds, _ = _seeds(); rows = []
    maxima = {condition: {"candidate": 0.0, "update": 0.0, "state": 0.0, "prediction": 0.0} for condition in ("AB1_fully_folded_equivalent","AB2_candidate_factorized_update_folded","AB3_candidate_folded_update_factorized","AB4_frozen_shared_projection")}
    for family in FAMILIES:
        task = task_by_name(family); episodes = [task.generate(seed, "development") for seed in train_seeds[:8]]; observations, lengths, _ = episodes_to_batch(episodes)
        for replicate in range(CONFIG["training"]["replicates"]):
            seed = _init_seed(family, replicate)
            torch.manual_seed(seed); anchor = InputFactorizationT2(condition=AB0)
            with torch.no_grad(): pa, ta = anchor(observations, lengths, return_trace=True)
            for condition in maxima:
                torch.manual_seed(seed); model = InputFactorizationT2(condition=condition)
                with torch.no_grad(): pb, tb = model(observations, lengths, return_trace=True)
                errors = {
                    "candidate": float(torch.max(torch.abs(ta.candidates-tb.candidates)).item()),
                    "update": float(torch.max(torch.abs(ta.updates-tb.updates)).item()),
                    "state": float(torch.max(torch.abs(ta.states-tb.states)).item()),
                    "prediction": float(torch.max(torch.abs(pa-pb)).item()),
                }
                for k,v in errors.items(): maxima[condition][k] = max(maxima[condition][k],v)
                rows.append({"family": family, "replicate_id": replicate, "condition": condition, "errors": errors})
    tolerance = CONFIG["step0_tolerance"]
    passed = all(max(errs.values()) <= tolerance for errs in maxima.values())
    return {"step0_equivalence_proven": bool(passed), "tolerance": tolerance, "max_errors_by_condition": maxima, "comparisons": rows, "ab5_excluded_by_design": True}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("preflight","ab0","others","all"), default="all"); args = parser.parse_args()
    _assert_locks()
    for name in ("raw","diagnostics","plots"): (HERE/name).mkdir(exist_ok=True)
    if args.phase in {"preflight","all"}:
        reference = _reference_equivalence(); step0 = _step0_equivalence()
        write_json(HERE/"diagnostics/reference_equivalence.json", reference); write_json(HERE/"diagnostics/step0_equivalence.json", step0)
        print(json.dumps({"reference_equivalence": reference["function_class_equivalence_proven"], "step0_equivalence": step0["step0_equivalence_proven"], "reference_max_error": reference["maximum_error"]}, indent=2), flush=True)
        if not reference["function_class_equivalence_proven"] or not step0["step0_equivalence_proven"]: return 2
    if args.phase in {"ab0","all"}:
        for required in (HERE/"diagnostics/reference_equivalence.json",HERE/"diagnostics/step0_equivalence.json"):
            if not required.exists(): raise SystemExit("preflight equivalence evidence missing")
        rows = _run([AB0]); write_json(HERE/"raw/ab0_runs.json", {"rows":rows,"unique_seed_defined_episodes":3200})
        guard = _ab0_guard(rows); write_json(HERE/"diagnostics/reference_baseline_guard.json",guard); print(json.dumps(guard,indent=2),flush=True)
        if not guard["reference_baseline_valid"]:
            (HERE/"FAILURE.md").write_text("# V837ab INPUT_FACTORIZATION_REFERENCE_BASELINE_DRIFT\n\nAB0 failed the frozen >=4/5 T2 positive-control gate. No factorization interpretation or V837ac authorization is permitted.\n",encoding="utf-8"); return 3
    if args.phase in {"others","all"}:
        guard_path=HERE/"diagnostics/reference_baseline_guard.json"
        if not guard_path.exists() or not json.loads(guard_path.read_text(encoding="utf-8")).get("reference_baseline_valid"): raise SystemExit("AB1-AB5 blocked until AB0 reproduces T2")
        rows=_run(OTHERS); write_json(HERE/"raw/factorization_runs.json", {"rows":rows,"unique_seed_defined_episodes":3200})
    return 0


if __name__ == "__main__": raise SystemExit(main())
