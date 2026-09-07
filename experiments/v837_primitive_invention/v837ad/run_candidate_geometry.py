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
from experiments.v837_primitive_invention.tasks import all_tasks, task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _configure_torch
from experiments.v837_primitive_invention.v837ad.candidate_recurrent_geometry import (
    CONDITION_SPECS, H40, SPARSE_TOPOLOGIES, CandidateRecurrentGeometryGRU,
    candidate_mask, mask_integrity, raw_initialization_signature,
)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FAMILIES = [task.name for task in all_tasks()]
AD0 = "AD0_H13_dense"
AD1 = "AD1_H40_dense"
GEOMETRY = list(CONFIG["stage_b_conditions"])
ROBUSTNESS = list(CONFIG["robustness_conditions"])


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
        "experiments/v837_primitive_invention/v837ab/config.json": CONFIG["v837ab_config_sha256"],
        "experiments/v837_primitive_invention/v837ab/results.json": CONFIG["v837ab_results_sha256"],
        "experiments/v837_primitive_invention/v837ac/results.json": CONFIG["v837ac_results_sha256"],
        "experiments/v837_primitive_invention/v837ac/diagnostics/decision_state.json": CONFIG["v837ac_decision_sha256"],
    }
    for path, expected in frozen.items():
        actual = _git_blob_sha256(path)
        if actual != expected:
            raise SystemExit(f"frozen parent changed: {path}: {actual} != {expected}")
    ac = json.loads((ROOT / "experiments/v837_primitive_invention/v837ac/diagnostics/decision_state.json").read_text(encoding="utf-8"))
    if ac.get("diagnosis") != "INPUT_ORGANIZATION_TRANSFER_INSUFFICIENT" or ac.get("representation_adequacy_pass") is not False:
        raise SystemExit("V837ac frontier incompatible")
    audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
    if audit.get("episodes_consumed") != 0:
        raise SystemExit("fresh audit consumed")
    if (ROOT / "experiments/v837_primitive_invention/v838").exists():
        raise SystemExit("V838 exists")


def _seeds() -> tuple[list[int], list[int]]:
    tr = CONFIG["training"]
    return (
        list(range(tr["development_seed_range"][0], tr["development_seed_range"][1] + 1)),
        list(range(tr["validation_seed_range"][0], tr["validation_seed_range"][1] + 1)),
    )


def _init_seed(family: str, replicate: int) -> int:
    return deterministic_int(CONFIG["training"]["initialization_namespace"], family, replicate)


def _covariance_metrics(values: torch.Tensor) -> dict:
    x = values.detach().reshape(-1, values.shape[-1]).float().cpu()
    if x.shape[0] > 512:
        x = x[:512]
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(int(x.shape[0]) - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    total = float(eig.sum().item())
    sq = float(torch.sum(eig * eig).item())
    participation = total * total / max(sq, 1e-12)
    threshold = float(eig.max().item()) * 1e-3 if eig.numel() else 0.0
    effective_rank = int(torch.sum(eig > threshold).item())
    corr = torch.corrcoef(x.T) if x.shape[0] > 1 else torch.eye(x.shape[1])
    offdiag = corr[~torch.eye(corr.shape[0], dtype=torch.bool)] if corr.shape[0] > 1 else torch.tensor([0.0])
    return {
        "effective_rank": effective_rank,
        "participation_ratio": float(participation),
        "norm_mean": float(torch.linalg.vector_norm(x, dim=1).mean().item()),
        "mean_abs_offdiag_correlation": float(torch.nan_to_num(offdiag, nan=0.0).abs().mean().item()),
    }


def _gate_metrics(updates: torch.Tensor) -> dict:
    # Scalarized update is repeated across H; inspect one coordinate.
    g = updates.detach()[..., 0].reshape(-1).float().cpu()
    return {
        "carry_fraction_mean": float(g.mean().item()),
        "rewrite_fraction_mean": float((1.0 - g).mean().item()),
        "temporal_variance": float(torch.var(g, unbiased=False).item()),
        "p10": float(torch.quantile(g, 0.10).item()),
        "p90": float(torch.quantile(g, 0.90).item()),
        "near_zero_fraction": float((g < 0.05).float().mean().item()),
        "near_one_fraction": float((g > 0.95).float().mean().item()),
    }


def _jacobian_metrics(model: CandidateRecurrentGeometryGRU, candidates: torch.Tensor) -> dict:
    w = model.candidate_weight_effective().detach().float().cpu()
    flat = candidates.detach().reshape(-1, model.hidden_size).float().cpu()[:32]
    spectra=[]; frobs=[]; ranks=[]
    for c in flat:
        j = (1.0 - c * c).unsqueeze(1) * w
        s = torch.linalg.svdvals(j)
        spectra.append(float(s.max().item()) if s.numel() else 0.0)
        frobs.append(float(torch.linalg.vector_norm(j).item()))
        ranks.append(int(torch.linalg.matrix_rank(j).item()))
    return {
        "samples": len(spectra),
        "spectral_norm_median": float(np.median(spectra)) if spectra else 0.0,
        "frobenius_norm_median": float(np.median(frobs)) if frobs else 0.0,
        "effective_rank_median": float(np.median(ranks)) if ranks else 0.0,
    }


def _trace_diagnostics(model: CandidateRecurrentGeometryGRU, observations: torch.Tensor, lengths: torch.Tensor) -> dict:
    with torch.no_grad():
        _, trace = model(observations[:32], lengths[:32], return_trace=True)
    state = _covariance_metrics(trace.states)
    candidate = _covariance_metrics(trace.candidates)
    hidden_norm = float(torch.linalg.vector_norm(trace.candidate_hidden_terms.reshape(-1, model.hidden_size), dim=1).mean().item())
    input_norm = float(torch.linalg.vector_norm(trace.candidate_input_terms.reshape(-1, model.hidden_size), dim=1).mean().item())
    hidden_var = float(torch.var(trace.candidate_hidden_terms, unbiased=False).item())
    update_hidden_norm = float(torch.linalg.vector_norm(trace.update_hidden_terms.reshape(-1, model.hidden_size), dim=1).mean().item())
    return {
        "state": state,
        "candidate": candidate,
        "gate": _gate_metrics(trace.updates),
        "candidate_recurrent_influence": {
            "hidden_candidate_norm_mean": hidden_norm,
            "hidden_candidate_temporal_variance": hidden_var,
            "candidate_input_to_hidden_norm_ratio": input_norm / max(hidden_norm, 1e-12),
        },
        "update_hidden_norm_mean": update_hidden_norm,
        "candidate_jacobian": _jacobian_metrics(model, trace.candidates),
    }


def _gradient_metrics(model: CandidateRecurrentGeometryGRU) -> dict:
    h = model.hidden_size
    wi = model.weight_ih.grad.detach() if model.weight_ih.grad is not None else torch.zeros_like(model.weight_ih)
    wh = model.weight_hh.grad.detach() if model.weight_hh.grad is not None else torch.zeros_like(model.weight_hh)
    proj = []
    for p in (model.input_projection_weight, model.input_projection_bias):
        if p.grad is not None: proj.append(p.grad.detach().reshape(-1))
    readout=[]
    for p in (model.readout_weight, model.readout_bias):
        if p.grad is not None: readout.append(p.grad.detach().reshape(-1))
    candidate_raw_grad = wh[2*h:3*h]
    active = candidate_raw_grad * model.candidate_mask
    masked = candidate_raw_grad * (1.0 - model.candidate_mask)
    return {
        "candidate_input_slice_norm": float(torch.linalg.vector_norm(wi[2*h:3*h]).item()),
        "candidate_recurrent_active_grad_norm": float(torch.linalg.vector_norm(active).item()),
        "candidate_recurrent_masked_grad_norm": float(torch.linalg.vector_norm(masked).item()),
        "candidate_recurrent_masked_grad_max_abs": float(torch.max(torch.abs(masked)).item()) if masked.numel() else 0.0,
        "update_input_slice_norm": float(torch.linalg.vector_norm(wi[h:2*h]).item()),
        "update_recurrent_slice_norm": float(torch.linalg.vector_norm(wh[h:2*h]).item()),
        "input_projection_grad_norm": float(torch.linalg.vector_norm(torch.cat(proj)).item()) if proj else 0.0,
        "readout_grad_norm": float(torch.linalg.vector_norm(torch.cat(readout)).item()) if readout else 0.0,
    }


def _train(condition: str, family: str, replicate: int) -> tuple[CandidateRecurrentGeometryGRU, dict]:
    _configure_torch(); task = task_by_name(family); train_seeds, validation_seeds = _seeds(); tr = CONFIG["training"]
    init_seed = _init_seed(family, replicate)
    torch.manual_seed(int(init_seed)); np.random.seed(int(init_seed) % (2**32 - 1))
    model = CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS[condition])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr["learning_rate"]), weight_decay=float(tr["weight_decay"]))
    loss_fn = nn.MSELoss()
    train_episodes = [task.generate(seed, "development") for seed in train_seeds]
    validation_episodes = [task.generate(seed, "validation") for seed in validation_seeds]
    observations, lengths, targets = episodes_to_batch(train_episodes)
    env_steps = sum(len(ep.observations) for ep in train_episodes + validation_episodes)
    resources = ResourceAccounting(candidate_evaluations=1, optimizer_steps=0, environment_steps=env_steps, examples_processed=0, model_fits=1, parameter_count=model.nominal_parameter_count(), model_parameter_bytes=model.parameter_bytes())
    requested = sorted(set(int(s) for s in tr["curve_steps"])); curve=[]; latest_grad={}

    def record(step: int) -> None:
        dev = evaluate_sequence_model(model, task, train_episodes); resources.forward_calls += 1
        diag = _trace_diagnostics(model, observations, lengths); resources.forward_calls += 1
        curve.append({
            "step": int(step), "development_loss": dev.loss, "development_success": dev.success_rate,
            "candidate_matrix_norm": model.geometry_diagnostics()["frobenius_norm"],
            "state_effective_rank": diag["state"]["effective_rank"], "state_participation_ratio": diag["state"]["participation_ratio"],
            "candidate_effective_rank": diag["candidate"]["effective_rank"], "candidate_participation_ratio": diag["candidate"]["participation_ratio"],
            "gate": diag["gate"],
        })

    with WallTimer() as timer:
        if 0 in requested: record(0)
        for step in range(1, int(tr["steps"]) + 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            pred = model(observations, lengths); resources.forward_calls += 1
            loss = loss_fn(pred, targets); loss.backward()
            if step == int(tr["steps"]): latest_grad = _gradient_metrics(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr["gradient_clip"])); optimizer.step()
            resources.optimizer_steps += 1; resources.examples_processed += len(train_episodes)
            if step in requested: record(step)
        development = evaluate_sequence_model(model, task, train_episodes); validation = evaluate_sequence_model(model, task, validation_episodes); resources.forward_calls += 2
        final_diag = _trace_diagnostics(model, observations, lengths); resources.forward_calls += 1
    resources.wall_seconds=timer.seconds; resources.cpu_seconds=timer.cpu_seconds
    return model, {
        "initialization_seed": init_seed,
        "development_success": development.success_rate, "validation_success": validation.success_rate,
        "development_loss": development.loss, "validation_loss": validation.loss,
        "learning_trajectory": curve, "trace_diagnostics": final_diag, "gradient_diagnostics": latest_grad,
        "resources": resources.to_dict(),
    }


def _worker(condition: str, family: str, replicate: int) -> dict:
    model, row = _train(condition, family, replicate)
    return {
        "version":"V837ad", "condition":condition, "family":family, "replicate_id":replicate,
        "model_init_seed":row["initialization_seed"], "development_success":row["development_success"], "validation_success":row["validation_success"],
        "development_loss":row["development_loss"], "validation_loss":row["validation_loss"],
        "capacity_demonstrated":capacity_demonstrated(row["development_success"],row["validation_success"]),
        "learning_trajectory":row["learning_trajectory"], "trace_diagnostics":row["trace_diagnostics"], "gradient_diagnostics":row["gradient_diagnostics"],
        "candidate_matrix_geometry":model.geometry_diagnostics(),
        "nominal_parameters":model.nominal_parameter_count(), "trainable_raw_parameters":model.trainable_raw_parameter_count(), "active_parameters":model.active_parameter_count(),
        "active_candidate_recurrent_weights":model.active_candidate_recurrent_weights, "masked_candidate_recurrent_weights":model.masked_candidate_recurrent_weights,
        "candidate_recurrent_macs":model.candidate_recurrent_macs, "total_active_macs_per_timestep":model.total_active_macs_per_timestep, "parameter_bytes":model.parameter_bytes(),
        "resources":row["resources"], "unique_seed_defined_episode_policy":"same 3200 family/seed episodes reused across all conditions and replicates",
        "fresh_audit_consumed":False, "structural_search_allowed":False, "primitive_mining_allowed":False, "large_persistent_storage_tested":False, "v838_started":False, "gpu_seconds":0.0,
    }


def _run(conditions: list[str]) -> list[dict]:
    jobs=[(c,f,r) for c in conditions for f in FAMILIES for r in range(int(CONFIG["training"]["replicates"]))]; rows=[]
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        futures={pool.submit(_worker,*job):job for job in jobs}
        for future in as_completed(futures):
            row=future.result(); rows.append(row); print(f"{row['condition']} {row['family']} r{row['replicate_id']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}",flush=True)
    rows.sort(key=lambda r:(r["condition"],r["family"],r["replicate_id"])); return rows


def _summary(rows: list[dict], condition: str) -> dict:
    result={"families_passing":0,"family_development_medians":{},"family_validation_medians":{}}
    for family in FAMILIES:
        fr=[r for r in rows if r["condition"]==condition and r["family"]==family]
        if len(fr)!=int(CONFIG["training"]["replicates"]): raise SystemExit(f"incomplete rows for {condition}/{family}")
        dev=float(np.median([r["development_success"] for r in fr])); val=float(np.median([r["validation_success"] for r in fr]))
        result["family_development_medians"][family]=dev; result["family_validation_medians"][family]=val; result["families_passing"] += int(capacity_demonstrated(dev,val))
    return result


def _historical_t2() -> dict:
    r=json.loads((ROOT/"experiments/v837_primitive_invention/v837t/results.json").read_text(encoding="utf-8"))["conditions"]["T2_scalarized_update_no_reset"]
    return {"families_passing":int(r["families_passing"]),"family_validation_medians":{f:float(v["validation"]["median"]) for f,v in r["family_results"].items()},"family_development_medians":{f:float(v["development"]["median"]) for f,v in r["family_results"].items()}}


def _anchor_guard(rows: list[dict]) -> dict:
    observed=_summary(rows,AD0); expected=_historical_t2(); deltas={f:abs(observed["family_validation_medians"][f]-expected["family_validation_medians"][f]) for f in FAMILIES}
    valid=observed["families_passing"]>=int(CONFIG["representation_family_gate"]) and max(deltas.values())<=float(CONFIG["anchor_drift_tolerance"])
    return {"ad0_anchor_valid":bool(valid),"expected":expected,"observed":observed,"absolute_validation_median_deltas":deltas,"max_absolute_delta":max(deltas.values())}


def _write_mask_integrity() -> dict:
    rows={name:mask_integrity(spec) for name,spec in CONDITION_SPECS.items()}
    sparse=[candidate_mask(CONDITION_SPECS[f"AD4S_S{i}"]) for i in range(5)]
    unique=all(not torch.equal(sparse[i],sparse[j]) for i in range(5) for j in range(i+1,5))
    valid=unique
    for i in range(5):
        x=rows[f"AD4S_S{i}"]; valid &= x["active_weights"]==160 and set(x["fan_in"])=={4} and set(x["fan_out"])=={4} and x["same_logical_4d_block_edges"]==0 and x["strongly_connected"]
    valid &= rows["AD4_H40_10x4"]["active_weights"]==160
    result={"valid":bool(valid),"all_sparse_topologies_unique":bool(unique),"conditions":rows,"task_independent":True,"masks_nontrainable":True}
    write_json(HERE/"diagnostics/mask_integrity.json",result); return result


def _paired_h40_preflight() -> dict:
    checks=[]; valid=True
    for family in FAMILIES:
        for replicate in range(int(CONFIG["training"]["replicates"])):
            seed=_init_seed(family,replicate); signatures=[]
            for condition in [AD1,*GEOMETRY,*ROBUSTNESS]:
                torch.manual_seed(seed); signatures.append((condition,raw_initialization_signature(CandidateRecurrentGeometryGRU(spec=CONDITION_SPECS[condition]))))
            ref=signatures[0][1]; equal={}
            for condition,sig in signatures[1:]:
                ok=all(torch.equal(ref[k],sig[k]) for k in ref); equal[condition]=ok; valid &= ok
            checks.append({"family":family,"replicate_id":replicate,"raw_initialization_equal":equal})
    result={"valid":bool(valid),"checks":checks,"only_candidate_mask_differs":bool(valid)}
    write_json(HERE/"diagnostics/h40_pairing.json",result); return result


def _write_width_gate(rows: list[dict]) -> dict:
    anchor=json.loads((HERE/"diagnostics/anchor_compatibility.json").read_text(encoding="utf-8")); ad1=_summary(rows,AD1); allowed=bool(anchor.get("ad0_anchor_valid") and ad1["families_passing"]>=int(CONFIG["representation_family_gate"]))
    result={"ad0_anchor_valid":bool(anchor.get("ad0_anchor_valid")),"ad1_dense_h40":ad1,"ad1_dense_h40_families":int(ad1["families_passing"]),"geometry_stage_allowed":allowed,"diagnosis":"WIDTH_GATE_PASS" if allowed else "DENSE_H40_REFERENCE_INADEQUATE","next_axis":None if allowed else "HIDDEN_WIDTH_LOCALIZATION"}
    write_json(HERE/"diagnostics/width_gate.json",result); return result


def _write_robustness_trigger(rows: list[dict]) -> dict:
    ad4=_summary(rows,"AD4_H40_10x4"); s0=_summary(rows,"AD4S_S0"); trigger=ad4["families_passing"]<4 and s0["families_passing"]>=4
    result={"ad4_families":ad4["families_passing"],"ad4s_s0_families":s0["families_passing"],"robustness_required":bool(trigger)}
    write_json(HERE/"diagnostics/sparse_robustness_gate.json",result); return result


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=("preflight","ad0","ad1","geometry","robustness"),required=True); args=parser.parse_args(); _assert_locks()
    for name in ("raw","diagnostics","plots"): (HERE/name).mkdir(exist_ok=True)
    if args.phase=="preflight":
        mask=_write_mask_integrity(); pair=_paired_h40_preflight(); print(json.dumps({"mask_integrity":mask["valid"],"h40_pairing":pair["valid"]},indent=2)); return 0 if mask["valid"] and pair["valid"] else 2
    if args.phase=="ad0":
        for p in (HERE/"diagnostics/mask_integrity.json",HERE/"diagnostics/h40_pairing.json"):
            if not p.exists(): raise SystemExit("preflight evidence missing")
        rows=_run([AD0]); write_json(HERE/"raw/ad0_runs.json",{"rows":rows,"unique_seed_defined_episodes":3200}); guard=_anchor_guard(rows); write_json(HERE/"diagnostics/anchor_compatibility.json",guard); print(json.dumps(guard,indent=2)); return 0 if guard["ad0_anchor_valid"] else 3
    if args.phase=="ad1":
        anchor_path=HERE/"diagnostics/anchor_compatibility.json"
        if not anchor_path.exists() or not json.loads(anchor_path.read_text(encoding="utf-8")).get("ad0_anchor_valid"): raise SystemExit("AD1 blocked until AD0 anchor passes")
        rows=_run([AD1]); write_json(HERE/"raw/ad1_runs.json",{"rows":rows,"unique_seed_defined_episodes":3200}); gate=_write_width_gate(rows); print(json.dumps(gate,indent=2)); return 0
    if args.phase=="geometry":
        gate_path=HERE/"diagnostics/width_gate.json"
        if not gate_path.exists() or not json.loads(gate_path.read_text(encoding="utf-8")).get("geometry_stage_allowed"): raise SystemExit("geometry stage blocked by width gate")
        rows=_run(GEOMETRY); write_json(HERE/"raw/geometry_runs.json",{"rows":rows,"unique_seed_defined_episodes":3200}); trigger=_write_robustness_trigger(rows); print(json.dumps(trigger,indent=2)); return 0
    gate_path=HERE/"diagnostics/sparse_robustness_gate.json"
    if not gate_path.exists() or not json.loads(gate_path.read_text(encoding="utf-8")).get("robustness_required"): raise SystemExit("S1-S4 robustness blocked: AD4 local failure + S0 sparse success trigger not satisfied")
    rows=_run(ROBUSTNESS); write_json(HERE/"raw/robustness_runs.json",{"rows":rows,"unique_seed_defined_episodes":3200}); return 0


if __name__=="__main__": raise SystemExit(main())
