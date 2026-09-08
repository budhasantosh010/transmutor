from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.reference_training import evaluate_sequence_model
from experiments.v837_primitive_invention.common.resource_accounting import ResourceAccounting, WallTimer
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _configure_torch
from experiments.v837_primitive_invention.v837ai.af1d_sample_efficiency import (
    CONFIG, CONDITION, FAMILIES, architecture_lock, historical_v837l_lock,
    development_seeds, validation_seeds, build_af1d, trainable_tensor_fingerprint,
    data_nesting_diagnostic, ai4_reuse_diagnostic, initialization_pairing_diagnostic,
    historical_v837l_comparison,
)

HERE = Path(__file__).resolve().parent
CURVE_STEPS = set(CONFIG["training"]["curve_steps"])


def _forbidden_absence() -> dict:
    names = ("v837ae", "v837ag", "v837ah", "v838")
    status = {name: not (ROOT / "experiments/v837_primitive_invention" / name).exists() for name in names}
    return {"all_absent": all(status.values()), "variants": status}


def run_preflight() -> int:
    for d in ("raw", "diagnostics", "plots"):
        (HERE / d).mkdir(exist_ok=True)
    lock = architecture_lock()
    nesting = data_nesting_diagnostic()
    v837l_lock = historical_v837l_lock()
    anchor, rows = ai4_reuse_diagnostic()
    pairing = initialization_pairing_diagnostic(rows)
    historical = historical_v837l_comparison()
    absence = _forbidden_absence()
    protected_snapshot = {
        "start_sha": CONFIG["required_start_sha"],
        "protected_through": "V837af",
        "architecture_source_hashes": CONFIG["architecture_hashes"],
        "historical_v837l_hashes": CONFIG["historical_v837l_hashes"],
        "forbidden_stage_absence": absence,
    }
    write_json(HERE / "diagnostics/architecture_lock.json", lock)
    write_json(HERE / "diagnostics/data_nesting.json", nesting)
    write_json(HERE / "diagnostics/ai4_anchor_reuse.json", anchor)
    write_json(HERE / "diagnostics/initialization_pairing.json", pairing)
    write_json(HERE / "diagnostics/historical_v837l_comparison.json", historical)
    write_json(HERE / "diagnostics/protected_historical_snapshot.json", protected_snapshot)
    write_json(HERE / "raw/ai4_reused_anchor.json", {
        "source_version":"V837af", "source_sha":CONFIG["source_sha"], "source_condition":CONDITION,
        "source_raw_sha256":CONFIG["architecture_hashes"]["v837af_raw_transfer"], "rows":rows,
    })
    ok = (
        lock["compatible"] and nesting["exact"] and nesting["strict_nesting"]
        and nesting["development_validation_overlap"] == 0 and not any(nesting["duplicates"].values())
        and v837l_lock["compatible"] and anchor["compatible"] and pairing["pairing_exact"]
        and pairing["parameter_count_constant"] and pairing["macs_constant"]
        and pairing["ten_projection_copies_identical_at_step0"] and historical["compatible"]
        and absence["all_absent"]
    )
    print(json.dumps({
        "architecture_lock": lock["compatible"], "data_nesting": nesting["strict_nesting"],
        "ai4_anchor": anchor["compatible"], "initialization_pairing": pairing["pairing_exact"],
        "historical_v837l": historical["compatible"], "forbidden_stages_absent": absence["all_absent"],
        "preflight_pass": bool(ok),
    }, indent=2))
    return 0 if ok else 2


def _success_rate(task, prediction: torch.Tensor, targets: torch.Tensor) -> float:
    return float(np.mean([task.success(float(p), float(t)) for p, t in zip(prediction.tolist(), targets.tolist())]))


def _active_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.arange(steps).view(1, -1) < lengths.view(-1, 1)


def _gradient_norm(parameters) -> float:
    chunks = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(chunks)).item()) if chunks else 0.0


def _trajectory(model, step: int) -> dict:
    return {"step": int(step), "projection": model.projection_diagnostics(), "effective_input_maps": model.effective_input_map_diagnostics()}


def _controller_stats(trace, lengths: torch.Tensor) -> dict:
    gates = trace.global_gates
    if gates is None:
        raise RuntimeError("AF1D must have global scalar controller")
    mask = _active_mask(lengths, gates.shape[1])
    values = gates.squeeze(-1)[mask].detach().cpu().numpy().astype(float)
    return {
        "gate_mean": float(np.mean(values)), "gate_median": float(np.median(values)),
        "temporal_variance": float(np.var(values)), "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)),
        "carry_fraction": float(np.mean(values)), "rewrite_fraction": float(1.0 - np.mean(values)), "active_gate_observations": int(values.size),
    }


def _post_diagnostics(model, task, val_seeds: list[int]) -> tuple[dict, int]:
    episodes = [task.generate(seed, "validation") for seed in val_seeds]
    observations, lengths, targets = episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad():
        baseline, trace = model(observations, lengths, return_trace=True)
        no_message = model(observations, lengths, disable_messages=True)
    success = _success_rate(task, baseline, targets)
    no_message_success = _success_rate(task, no_message, targets)
    active = _active_mask(lengths, trace.input_terms.shape[1])
    candidate_active = trace.candidate_states[active]
    input_active = trace.input_terms[active]
    per_cell = []
    jacobian_norms = []
    for cell in range(10):
        W_eff, _ = model.effective_candidate_input_map(cell)
        terms = input_active[:, cell, :]
        candidates = candidate_active[:, cell, :]
        deriv = 1.0 - candidates.pow(2)
        jac = deriv.unsqueeze(-1) * W_eff.detach().unsqueeze(0)
        jac_norm = float(torch.linalg.vector_norm(jac, dim=(1, 2)).mean().item())
        jacobian_norms.append(jac_norm)
        per_cell.append({
            "cell": cell,
            "input_term_norm_mean": float(torch.linalg.vector_norm(terms, dim=-1).mean().item()),
            "input_term_temporal_variance": float(torch.var(terms, dim=0, unbiased=False).mean().item()),
            "candidate_visible_input_jacobian_frobenius_mean": jac_norm,
        })
    return {
        "projection": model.projection_diagnostics(),
        "effective_input_maps": model.effective_input_map_diagnostics(),
        "candidate_input": {
            "per_cell": per_cell,
            "mean_input_term_norm": float(np.mean([r["input_term_norm_mean"] for r in per_cell])),
            "mean_temporal_variance": float(np.mean([r["input_term_temporal_variance"] for r in per_cell])),
            "mean_candidate_visible_input_jacobian_frobenius": float(np.mean(jacobian_norms)),
        },
        "message_dependency": {
            "baseline_success": success, "no_message_success": no_message_success,
            "success_drop": success - no_message_success,
            "mean_abs_prediction_delta": float(torch.abs(no_message - baseline).mean().item()),
        },
        "controller": _controller_stats(trace, lengths),
    }, 2


def _train(multiplier: int, family: str, replicate: int) -> dict:
    _configure_torch()
    task = task_by_name(family)
    tr = CONFIG["training"]
    train_seeds = development_seeds(multiplier)
    val_seeds = validation_seeds()
    model = build_af1d(family, replicate)
    init_hash = trainable_tensor_fingerprint(model)
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(tr["learning_rate"]), weight_decay=float(tr["weight_decay"]))
    loss_fn = nn.MSELoss()
    train_episodes = [task.generate(seed, "development") for seed in train_seeds]
    validation_episodes = [task.generate(seed, "validation") for seed in val_seeds]
    observations, lengths, targets = episodes_to_batch(train_episodes)
    active_timesteps_per_step = int(sum(len(ep.observations) for ep in train_episodes))
    resources = ResourceAccounting(
        candidate_evaluations=1, model_fits=1,
        environment_steps=sum(len(ep.observations) for ep in train_episodes + validation_episodes),
        parameter_count=model.parameter_count(), model_parameter_bytes=model.parameter_bytes(),
    )
    curve=[]; projection_trajectory=[]; latest_gradient_norm=0.0
    def record(step: int) -> None:
        dev = evaluate_sequence_model(model, task, train_episodes)
        val = evaluate_sequence_model(model, task, validation_episodes)
        resources.forward_calls += 2
        curve.append({"step":int(step),"development_loss":dev.loss,"development_success":dev.success_rate,"validation_loss":val.loss,"validation_success":val.success_rate,"gradient_norm":latest_gradient_norm})
        projection_trajectory.append(_trajectory(model, step))
    with WallTimer() as timer:
        record(0)
        for step in range(1, int(tr["steps"])+1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            prediction = model(observations, lengths); resources.forward_calls += 1
            loss = loss_fn(prediction, targets); loss.backward()
            latest_gradient_norm = _gradient_norm(params)
            torch.nn.utils.clip_grad_norm_(params, float(tr["gradient_clip"])); optimizer.step()
            resources.optimizer_steps += 1; resources.examples_processed += len(train_episodes)
            if step in CURVE_STEPS: record(step)
        development = evaluate_sequence_model(model, task, train_episodes)
        validation = evaluate_sequence_model(model, task, validation_episodes)
        resources.forward_calls += 2
    resources.wall_seconds = timer.seconds; resources.cpu_seconds = timer.cpu_seconds
    diagnostics, extra_calls = _post_diagnostics(model, task, val_seeds); resources.forward_calls += extra_calls
    modeled_active_training_timesteps = int(active_timesteps_per_step * int(tr["steps"]))
    modeled_mac_volume = int(modeled_active_training_timesteps * model.total_recurrent_controller_projection_macs)
    return {
        "version":"V837ai", "data_multiplier":int(multiplier), "regime":f"{multiplier}x", "condition":CONDITION,
        "family":family, "replicate_id":int(replicate),
        "initialization_seed":int(__import__('experiments.v837_primitive_invention.v837ai.af1d_sample_efficiency',fromlist=['base_seed']).base_seed(family,replicate)),
        "coupling_initialization_seed":int(__import__('experiments.v837_primitive_invention.v837ai.af1d_sample_efficiency',fromlist=['coupling_seed']).coupling_seed(replicate)),
        "projection_initialization_seed":int(__import__('experiments.v837_primitive_invention.v837ai.af1d_sample_efficiency',fromlist=['projection_seed']).projection_seed(family,replicate)),
        "initialization_fingerprint":init_hash,
        "train_episodes":len(train_seeds), "train_seed_first":train_seeds[0], "train_seed_last":train_seeds[-1],
        "validation_episodes":len(val_seeds), "validation_seed_first":val_seeds[0], "validation_seed_last":val_seeds[-1],
        "development_success":development.success_rate, "validation_success":validation.success_rate,
        "development_loss":development.loss, "validation_loss":validation.loss,
        "capacity_demonstrated":capacity_demonstrated(development.success_rate, validation.success_rate),
        "learning_curve":curve, "projection_trajectory":projection_trajectory, "diagnostics":diagnostics,
        "parameter_count":model.parameter_count(), "active_parameter_count":model.parameter_count(),
        "projection_parameter_count":model.projection_parameter_count,
        "projection_specific_macs":model.projection_specific_macs,
        "active_macs_per_timestep":model.total_recurrent_controller_projection_macs,
        "active_training_timesteps_per_optimizer_step":active_timesteps_per_step,
        "modeled_active_training_timesteps":modeled_active_training_timesteps,
        "modeled_training_recurrent_mac_volume":modeled_mac_volume,
        "resources":resources.to_dict(), "processed_examples":resources.examples_processed,
        "fresh_audit_consumed":False,"structural_search_executed":False,"primitive_mining_allowed":False,"v838_started":False,"gpu_seconds":0.0,
    }


def _worker(multiplier: int, family: str, replicate: int) -> dict:
    return _train(multiplier, family, replicate)


def _run_multiplier(multiplier: int) -> list[dict]:
    jobs=[(multiplier,family,rep) for family in FAMILIES for rep in range(5)]
    rows=[]
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures={pool.submit(_worker,*job):job for job in jobs}
        for future in as_completed(futures):
            row=future.result(); rows.append(row)
            print(f"AI{multiplier} {row['family']} r{row['replicate_id']}: dev={row['development_success']:.6f} val={row['validation_success']:.6f}",flush=True)
    rows.sort(key=lambda r:(r["family"],r["replicate_id"]))
    return rows


def _require_preflight() -> None:
    checks={
        "architecture_lock.json":"compatible",
        "ai4_anchor_reuse.json":"compatible",
        "initialization_pairing.json":"pairing_exact",
        "historical_v837l_comparison.json":"compatible",
    }
    for name,key in checks.items():
        path=HERE/"diagnostics"/name
        if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get(key) is not True:
            raise SystemExit(f"V837ai execution blocked by missing/failed preflight: {name}:{key}")
    nesting=json.loads((HERE/"diagnostics/data_nesting.json").read_text(encoding="utf-8"))
    if nesting.get("strict_nesting") is not True or nesting.get("exact") is not True:
        raise SystemExit("V837ai execution blocked by data nesting failure")


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("preflight","1x","2x"),required=True)
    args=parser.parse_args()
    if args.phase=="preflight": return run_preflight()
    _require_preflight()
    multiplier=1 if args.phase=="1x" else 2
    rows=_run_multiplier(multiplier)
    write_json(HERE/"raw"/f"ai{multiplier}_runs.json",{
        "version":"V837ai","regime":f"{multiplier}x","rows":rows,
        "development_episodes_per_family":128*multiplier,"validation_episodes_per_family":128,
        "fresh_audit_consumed":False,
    })
    return 0

if __name__=="__main__": raise SystemExit(main())
