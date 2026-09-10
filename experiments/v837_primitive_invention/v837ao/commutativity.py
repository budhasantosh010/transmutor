from __future__ import annotations

import numpy as np
import torch

from experiments.v837_primitive_invention.v837an.an_a_causal_macrovariables import _eligibility
from experiments.v837_primitive_invention.v837an.causal_interventions import build_patch_plan
from experiments.v837_primitive_invention.v837an.instrumented_af1d import run_instrumented
from experiments.v837_primitive_invention.v837an.trace_cache import pair_traces

from .abstract_rollout import next_canonical_state, rollout_from_setpoint
from .canonical_reader import read_k1
from .canonical_writer import set_state
from .k1_backend import component_for_phase
from .metrics import causal_recovery
from .phase_backends import representative_phase_timesteps
from .setpoint_eval import evaluate_setpoints
from .setpoint_grid import family_grid
from .trajectory_eval import expected_future, phase_at_timestep, read_state_at, semantic_timesteps

HORIZONS=(1,2,4,8)


def _expected_next(ep,t,z):
    family=ep.family
    if family in {"conditional_routing","delayed_recall"}:return float(z)
    nt=t+1
    if family=="iterative_state":return next_canonical_state(family,z,{"x":float(np.asarray(ep.causal_inputs["x_t"])[nt-1])})
    return next_canonical_state(family,z,{"gain":float(np.asarray(ep.causal_inputs["gain"])[nt-1]),"drive":float(np.asarray(ep.causal_inputs["drive"])[nt-1])})


def evaluate_natural_commutativity(backend:dict,seeds:list[int],partition_name:str)->dict:
    family=backend["family"];R=float(family_grid(family)["semantic_range"]);data=pair_traces(backend["organism_id"],family,seeds);mask,_=_eligibility(data,family);states=data["base_trace"].states.detach().cpu().numpy().reshape(len(data["pairs"]),data["base_trace"].states.shape[1],40);errs=[];cons=[];n=0
    for i in np.flatnonzero(mask):
        ep=data["pairs"][i].base_episode;times=semantic_timesteps(ep)
        for t,nt in zip(times[:-1],times[1:]):
            p0=phase_at_timestep(ep,t);p1=phase_at_timestep(ep,nt)
            if p0 is None or p1 is None:continue
            z=read_state_at(backend,states[i,t],p0);got=read_state_at(backend,states[i,nt],p1);expected=_expected_next(ep,t,z);err=(got-expected)/max(R,1e-12);errs.append(err);desired=expected-z;actual=got-z;cons.append(float(abs(desired)<1e-10 and abs(actual)<=.10*R or abs(desired)>=1e-10 and np.sign(desired)==np.sign(actual)));n+=1
    a=np.asarray(errs,dtype=np.float64);metrics={"normalized_rmse":float(np.sqrt(np.mean(a*a))) if len(a) else float("inf"),"median_abs_normalized_error":float(np.median(np.abs(a))) if len(a) else float("inf"),"direction_sign_consistency":float(np.mean(cons)) if cons else 0.0,"transitions":n};passed=metrics["normalized_rmse"]<=.10 and metrics["median_abs_normalized_error"]<=.075 and metrics["direction_sign_consistency"]>=.90
    return {"organism_id":backend["organism_id"],"family":family,"engine":backend["engine"],"variant":backend["variant"],"partition":partition_name,"metrics":metrics,"pass":bool(passed),"failure_code":None if passed else "CANONICAL_DYNAMICS_ONE_STEP_FAIL"}


def evaluate_interventional_commutativity(backend:dict,seeds:list[int],partition_name:str)->dict:
    oid=backend["organism_id"];family=backend["family"];grid=family_grid(family);R=float(grid["semantic_range"]);data=pair_traces(oid,family,seeds);mask,_=_eligibility(data,family);states=data["base_trace"].states.detach().cpu().numpy().reshape(len(data["pairs"]),data["base_trace"].states.shape[1],40);samples=[]
    for i in np.flatnonzero(mask):
        ep=data["pairs"][i].base_episode
        for phase,t in representative_phase_timesteps(ep).items():
            comp=component_for_phase(backend,phase)
            if not comp.get("valid"):continue
            s=states[i,t].astype(np.float64);z=float(read_k1(s[None,:],comp["reader"])[0]);target=float(max(grid["targets"],key=lambda x:abs(float(x)-z)))
            samples.append({"episode_index":int(i),"phase":phase,"timestep":int(t),"target":target,"patched":set_state(s,target,comp["reader"],np.asarray(comp["writer"])),"base_prediction":float(data["base_prediction"][i].item()),"abstract_target":float(rollout_from_setpoint(family,ep,t,target)["final_target"])})
    if not samples:return {"organism_id":oid,"family":family,"engine":backend["engine"],"variant":backend["variant"],"partition":partition_name,"pass":False,"failure_code":"CANONICAL_DYNAMICS_ROLLOUT_FAIL"}
    idx=np.asarray([s["episode_index"] for s in samples]);times=np.asarray([s["timestep"] for s in samples]);obs=data["base_obs"][torch.as_tensor(idx,dtype=torch.long)];lens=data["base_lengths"][torch.as_tensor(idx,dtype=torch.long)];plan=build_patch_plan("STATE40",np.stack([s["patched"] for s in samples]),times)
    with torch.no_grad():pred,trace=run_instrumented(data["model"],obs,lens,patch_plan=plan,return_trace=True)
    pred=pred.detach().cpu().numpy().astype(np.float64);tr=trace.states.detach().cpu().numpy().reshape(len(samples),trace.states.shape[1],40);one=[];roll=[];horizon_rows=[]
    for j,s in enumerate(samples):
        ep=data["pairs"][s["episode_index"]].base_episode;expected=expected_future(ep,s["timestep"],s["target"]);future=[t for t in semantic_timesteps(ep) if t>s["timestep"]]
        if future:
            nt=future[0];phase=phase_at_timestep(ep,nt)
            if phase is not None:one.append((read_state_at(backend,tr[j,nt],phase)-expected[nt])/max(R,1e-12))
        for h in HORIZONS:
            if not future:continue
            ix=min(h-1,len(future)-1);tt=future[ix];phase=phase_at_timestep(ep,tt)
            if phase is None:continue
            err=(read_state_at(backend,tr[j,tt],phase)-expected[tt])/max(R,1e-12);roll.append(err);horizon_rows.append({"horizon":h,"error":float(abs(err))})
    expected_final=np.asarray([s["abstract_target"] for s in samples]);base=np.asarray([s["base_prediction"] for s in samples]);rec=causal_recovery(base,expected_final,pred)
    aux=evaluate_setpoints(backend,seeds,partition_name=partition_name+"_CONTROL",random_controls=True)
    one_arr=np.asarray(one,dtype=np.float64);roll_arr=np.asarray(roll,dtype=np.float64);metrics={"one_step_normalized_rmse":float(np.sqrt(np.mean(one_arr*one_arr))) if len(one_arr) else 0.0,"multi_step_trajectory_nrmse":float(np.sqrt(np.mean(roll_arr*roll_arr))) if len(roll_arr) else 0.0,"final_abstract_output_recovery":float(np.median(rec)),"counterfactual_task_success":float(aux.get("metrics",{}).get("counterfactual_task_success",0.0)),"random_control_margin":float(aux.get("metrics",{}).get("control_margin",-1e9)),"median_ood_ratio":float(aux.get("metrics",{}).get("median_ood_ratio",1e9)),"interventions":len(samples)}
    passed=metrics["one_step_normalized_rmse"]<=.10 and metrics["multi_step_trajectory_nrmse"]<=.15 and metrics["final_abstract_output_recovery"]>=.80 and metrics["counterfactual_task_success"]>=.75 and metrics["random_control_margin"]>=.20 and metrics["median_ood_ratio"]<=2.0
    return {"organism_id":oid,"family":family,"engine":backend["engine"],"variant":backend["variant"],"partition":partition_name,"metrics":metrics,"horizon_rows":horizon_rows,"pass":bool(passed),"failure_code":None if passed else "CANONICAL_DYNAMICS_ROLLOUT_FAIL"}


def evaluate_dynamics(backend:dict,seeds:list[int],partition_name:str="AO_DYNAMICS_SELECT")->dict:
    natural=evaluate_natural_commutativity(backend,seeds,partition_name);inter=evaluate_interventional_commutativity(backend,seeds,partition_name);return {"organism_id":backend["organism_id"],"family":backend["family"],"engine":backend["engine"],"variant":backend["variant"],"natural":natural,"interventional":inter,"pass":bool(natural["pass"] and inter["pass"]),"failure_code":None if natural["pass"] and inter["pass"] else (natural.get("failure_code") if not natural["pass"] else inter.get("failure_code"))}
