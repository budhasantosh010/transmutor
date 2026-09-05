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
from experiments.v837_primitive_invention.v837u.dynamic_control import NeutralDynamicCarryModel

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/"config.json").read_text(encoding="utf-8"))
CONDITIONS=list(CONFIG["conditions"])
FAMILIES=[t.name for t in all_tasks()]


def _blob_hash(path:str)->str:
    return hashlib.sha256(subprocess.check_output(["git","show",f"HEAD:{path}"],cwd=ROOT)).hexdigest()


def _configure_torch():
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass


def _temporal_variance(values:torch.Tensor,lengths:torch.Tensor)->float:
    vals=[]
    for i,l in enumerate(lengths.tolist()):
        if l>1: vals.append(float(values[i,:l].var(dim=0,unbiased=False).mean().item()))
    return float(np.mean(vals)) if vals else 0.0


def _diagnostics(model:NeutralDynamicCarryModel,task,val_seeds:list[int])->dict:
    episodes=[task.generate(seed,"validation") for seed in val_seeds]
    obs,lengths,_=episodes_to_batch(episodes)
    model.eval()
    with torch.no_grad(): _,trace=model(obs,lengths,return_trace=True)
    active=torch.arange(trace.states.shape[1]).view(1,-1)<lengths.view(-1,1)
    states=trace.states[active].reshape(-1,4)
    candidates=trace.candidate_states[active].reshape(-1,4)
    out={
        "mean_state_norm":float(torch.linalg.vector_norm(states,dim=-1).mean().item()),
        "state_variance":float(states.var(unbiased=False).item()),
        "candidate_saturation_fraction":float((torch.abs(candidates)>=0.95).float().mean().item()),
    }
    if model.controller_output_dim:
        mods=trace.state_modulators
        flat=mods[active].reshape(-1).detach().cpu().numpy()
        out["state_modulator"]={
            "mean":float(np.mean(flat)),"median":float(np.median(flat)),"std":float(np.std(flat)),
            "p10":float(np.quantile(flat,.1)),"p90":float(np.quantile(flat,.9)),
            "temporal_variance":_temporal_variance(mods,lengths),
            "near_zero_fraction":float(np.mean(flat<=.05)),"near_one_fraction":float(np.mean(flat>=.95)),
        }
    else: out["state_modulator"]=None
    return out


def _assert_authorization():
    if gate_sha256()!=CONFIG["historical_gate_hash"]: raise SystemExit("historical gate changed")
    if v837_capacity_criterion_sha256()!=CONFIG["capacity_criterion_hash"]: raise SystemExit("capacity criterion changed")
    if _blob_hash("experiments/v837_primitive_invention/v837t/results.json")!=CONFIG["v837t_result_sha256"]: raise SystemExit("V837t result changed")
    if _blob_hash("experiments/v837_primitive_invention/v837t/diagnostics/decision_state.json")!=CONFIG["v837t_decision_sha256"]: raise SystemExit("V837t decision changed")
    if _blob_hash("experiments/v837_primitive_invention/v837p/results.json")!=CONFIG["v837p_result_sha256"]: raise SystemExit("V837p result changed")
    d=json.loads((ROOT/"experiments/v837_primitive_invention/v837t/diagnostics/decision_state.json").read_text(encoding="utf-8"))
    if d.get("v837t_complete") is not True or d.get("positive_controls_pass") is not True or d.get("neutral_followup_allowed") is not True: raise SystemExit("V837u not authorized")
    if d.get("authorized_v837u_mode")!=CONFIG["authorized_mode"]: raise SystemExit("V837u mode differs from machine authorization")
    audit=json.loads((ROOT/"experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
    if audit.get("episodes_consumed")!=0: raise SystemExit("fresh audit consumed")


def _worker(condition:str,family:str,replicate:int)->dict:
    _configure_torch(); task=task_by_name(family); graph=high_capacity_generic_graph(replicate); tr=CONFIG["training"]
    train_seeds=list(range(tr["development_seed_range"][0],tr["development_seed_range"][1]+1)); val_seeds=list(range(tr["validation_seed_range"][0],tr["validation_seed_range"][1]+1))
    init_seed=deterministic_int("v837p-init",family,replicate)
    result=train_sequence_model(
        model_factory=lambda:NeutralDynamicCarryModel(graph,condition=condition,obs_dim=6),task=task,train_seeds=train_seeds,validation_seeds=val_seeds,
        initialization_seed=init_seed,steps=tr["steps"],learning_rate=tr["learning_rate"],weight_decay=tr["weight_decay"],gradient_clip=tr["gradient_clip"],curve_steps=tuple(tr["curve_steps"]),
    )
    diag=_diagnostics(result.model,task,val_seeds)
    return {
        "version":"V837u","authorized_mode":CONFIG["authorized_mode"],"condition":condition,"family":family,"replicate_id":replicate,
        "development_seed_range":tr["development_seed_range"],"validation_seed_range":tr["validation_seed_range"],"model_init_seed":init_seed,
        "development_success":result.development.success_rate,"validation_success":result.validation.success_rate,
        "development_loss":result.development.loss,"validation_loss":result.validation.loss,"capacity_demonstrated":capacity_demonstrated(result.development.success_rate,result.validation.success_rate),
        "loss_curve":result.learning_curve,"parameter_count":result.model.parameter_count(),"parameter_bytes":result.model.parameter_bytes(),
        "controller_output_dimension":result.model.controller_output_dim,"controller_parameter_count":result.model.controller_parameter_count,
        "controller_macs_per_timestep":result.model.controller_macs_per_timestep,"base_recurrent_macs_per_timestep":160,
        "total_recurrent_controller_macs_per_timestep":160+result.model.controller_macs_per_timestep,
        "state_modulation_location":result.model.state_modulation_location,
        "multiplicative":condition in {"U1_v837p_scalar_candidate","U2_dynamic_scalar_carry","U2C_scalar_scale_candidate_control"},
        "scalarized_control":False,"additive_control":False,"diagnostics":diag,"resources":result.resources.to_dict(),
        "unique_seed_defined_episode_policy":"same 3200 family/seed episodes reused across all conditions and replicates",
        "task_family_label_in_model_input":False,"fresh_audit_consumed":False,"gpu_seconds":0.0,
    }


def main()->int:
    _assert_authorization()
    for n in ("raw","diagnostics","plots"): (HERE/n).mkdir(exist_ok=True)
    jobs=[(c,f,r) for c in CONDITIONS for f in FAMILIES for r in range(CONFIG["training"]["replicates"])]
    rows=[]
    with ProcessPoolExecutor(max_workers=min(10,os.cpu_count() or 1)) as pool:
        futures={pool.submit(_worker,*j):j for j in jobs}
        for fut in as_completed(futures):
            row=fut.result(); rows.append(row); print(f"{row['condition']} {row['family']} r{row['replicate_id']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}",flush=True)
    rows.sort(key=lambda r:(r["condition"],r["family"],r["replicate_id"]))
    write_json(HERE/"raw"/"runs.json",{"rows":rows,"unique_seed_defined_episodes":3200,"reuse_policy":"paired reuse across conditions and replicates","fresh_audit_consumed":False})
    print(f"V837u raw runs complete: {len(rows)} fits")
    return 0


if __name__=="__main__": raise SystemExit(main())
