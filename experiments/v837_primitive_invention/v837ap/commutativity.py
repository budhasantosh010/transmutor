from __future__ import annotations

import numpy as np
import torch

from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837an.an_a_causal_macrovariables import _eligibility
from experiments.v837_primitive_invention.v837an.causal_interventions import build_patch_plan
from experiments.v837_primitive_invention.v837an.instrumented_af1d import run_instrumented
from experiments.v837_primitive_invention.v837an.trace_cache import pair_traces
from experiments.v837_primitive_invention.v837ao.phase_backends import representative_phase_timesteps
from experiments.v837_primitive_invention.v837ao.trajectory_eval import phase_at_timestep, semantic_timesteps

from .abstract_rollout import next_state, rollout_from_setpoint
from .geometry_runtime import read_state, set_state
from .metrics import causal_recovery
from .setpoint_eval import d90_for, evaluate_geometry
from .setpoint_grid import family_grid
from .utils import HERE, write_json

HORIZONS=(1,2,4,8)
BINARY={"conditional_routing","delayed_recall"}


def _expected_next(ep,t,z):
    family=ep.family
    if family in BINARY:return float(z)
    nt=int(t)+1
    if family=="iterative_state":
        return next_state(family,z,{"x":float(np.asarray(ep.causal_inputs["x_t"])[nt-1])})
    return next_state(family,z,{"gain":float(np.asarray(ep.causal_inputs["gain"])[nt-1]),"drive":float(np.asarray(ep.causal_inputs["drive"])[nt-1])})


def evaluate_natural_commutativity(geometry:dict,q:np.ndarray,eval_seeds:list[int],partition_name:str="AP_DYNAMICS")->dict:
    oid=geometry["organism_id"];family=geometry["family"];Q=np.asarray(q,dtype=np.float64).reshape(40,-1);R=float(family_grid(family)["semantic_range"])
    data=pair_traces(oid,family,eval_seeds);mask,_=_eligibility(data,family);states=data["base_trace"].states.detach().cpu().numpy().reshape(len(data["pairs"]),data["base_trace"].states.shape[1],40)
    errs=[];direction=[];rows=[]
    for i in np.flatnonzero(mask):
        ep=data["pairs"][i].base_episode;times=semantic_timesteps(ep)
        for t,nt in zip(times[:-1],times[1:]):
            p0=phase_at_timestep(ep,t);p1=phase_at_timestep(ep,nt)
            if p0 is None or p1 is None:continue
            z=read_state(geometry,Q,states[i,t],p0);got=read_state(geometry,Q,states[i,nt],p1);expected=_expected_next(ep,t,z);err=(got-expected)/max(R,1e-12);errs.append(err)
            desired=expected-z;actual=got-z
            ok=abs(desired)<1e-10 and abs(actual)<=.10*R or abs(desired)>=1e-10 and np.sign(desired)==np.sign(actual)
            direction.append(float(ok));rows.append({"seed":int(data["pairs"][i].base_seed),"t":int(t),"next_t":int(nt),"phase":p0,"next_phase":p1,"z":float(z),"expected":float(expected),"observed":float(got),"normalized_error":float(err)})
    a=np.asarray(errs,dtype=np.float64)
    metrics={"one_step_semantic_nrmse":float(np.sqrt(np.mean(a*a))) if len(a) else float("inf"),"median_abs_normalized_error":float(np.median(np.abs(a))) if len(a) else float("inf"),"direction_consistency":float(np.mean(direction)) if direction else 0.0,"transitions":len(rows)}
    passed=bool(metrics["one_step_semantic_nrmse"]<=.10 and metrics["median_abs_normalized_error"]<=.075 and metrics["direction_consistency"]>=.90)
    return {"version":"V837ap","organism_id":oid,"family":family,"engine":geometry["engine"],"k":int(geometry["k"]),"chart_family":geometry["chart_family"],"phase_atlas":bool(geometry.get("phase_atlas")),"partition":partition_name,"metrics":metrics,"rows":rows,"pass":passed,"failure_code":None if passed else "CANONICAL_DYNAMICS_ONE_STEP_FAIL"}


def evaluate_interventional_commutativity(geometry:dict,q:np.ndarray,eval_seeds:list[int],partition_name:str="AP_DYNAMICS", *, chart_fit_seeds:list[int]|None=None, writer_fit_seeds:list[int]|None=None)->dict:
    oid=geometry["organism_id"];family=geometry["family"];Q=np.asarray(q,dtype=np.float64).reshape(40,-1);grid=family_grid(family);R=float(grid["semantic_range"]);d90=d90_for(oid,family,Q,writer_fit_seeds);task=task_by_name(family)
    data=pair_traces(oid,family,eval_seeds);mask,_=_eligibility(data,family);states=data["base_trace"].states.detach().cpu().numpy().reshape(len(data["pairs"]),data["base_trace"].states.shape[1],40);samples=[]
    for i in np.flatnonzero(mask):
        ep=data["pairs"][i].base_episode
        for phase,t in representative_phase_timesteps(ep).items():
            s=states[i,t].astype(np.float64);z=read_state(geometry,Q,s,phase);target=float(max(grid["targets"],key=lambda x:abs(float(x)-z)))
            sr=set_state(geometry,Q,s,phase,target,d90)
            if not sr.get("valid"):continue
            abstract=rollout_from_setpoint(family,ep,t,target)
            samples.append({"episode_index":int(i),"phase":phase,"timestep":int(t),"target":target,"patched":np.asarray(sr["state"]),"base_prediction":float(data["base_prediction"][i].item()),"abstract_target":float(abstract["final_target"]),"expected_trajectory":abstract["trajectory"]})
    if not samples:return {"version":"V837ap","organism_id":oid,"family":family,"engine":geometry["engine"],"partition":partition_name,"pass":False,"failure_code":"CANONICAL_DYNAMICS_ROLLOUT_FAIL"}
    idx=np.asarray([s["episode_index"] for s in samples],dtype=np.int64);times=np.asarray([s["timestep"] for s in samples],dtype=np.int64);obs=data["base_obs"][torch.as_tensor(idx,dtype=torch.long)];lens=data["base_lengths"][torch.as_tensor(idx,dtype=torch.long)];plan=build_patch_plan("STATE40",np.stack([s["patched"] for s in samples]),times)
    with torch.no_grad():pred,trace=run_instrumented(data["model"],obs,lens,patch_plan=plan,return_trace=True)
    pred=pred.detach().cpu().numpy().astype(np.float64);tr=trace.states.detach().cpu().numpy().reshape(len(samples),trace.states.shape[1],40)
    one=[];all_multi=[];horizon_rows=[]
    for j,s in enumerate(samples):
        ep=data["pairs"][s["episode_index"]].base_episode;future=[t for t in semantic_timesteps(ep) if t>s["timestep"]];z=float(s["target"]);expected_by_t={}
        for tt in future:
            if family in BINARY:z=float(z)
            elif family=="iterative_state":z=next_state(family,z,{"x":float(np.asarray(ep.causal_inputs["x_t"])[tt-1])})
            else:z=next_state(family,z,{"gain":float(np.asarray(ep.causal_inputs["gain"])[tt-1]),"drive":float(np.asarray(ep.causal_inputs["drive"])[tt-1])})
            expected_by_t[tt]=float(z)
        if future:
            nt=future[0];phase=phase_at_timestep(ep,nt)
            if phase is not None:one.append((read_state(geometry,Q,tr[j,nt],phase)-expected_by_t[nt])/max(R,1e-12))
        for h in HORIZONS:
            if not future:continue
            pos=min(h-1,len(future)-1);tt=future[pos];phase=phase_at_timestep(ep,tt)
            if phase is None:continue
            err=(read_state(geometry,Q,tr[j,tt],phase)-expected_by_t[tt])/max(R,1e-12);all_multi.append(err);horizon_rows.append({"horizon":h,"error":float(abs(err))})
    expected_final=np.asarray([s["abstract_target"] for s in samples],dtype=np.float64);base=np.asarray([s["base_prediction"] for s in samples],dtype=np.float64);rec=causal_recovery(base,expected_final,pred);success=np.asarray([task.success(float(p),float(t)) for p,t in zip(pred,expected_final)],dtype=float)
    # Reuse exact frozen SET/control gate on the same partition for random-control margin and OOD.
    setdiag=evaluate_geometry(geometry,Q,eval_partition=partition_name,random_controls_count=32,chart_fit_seeds=chart_fit_seeds,writer_fit_seeds=writer_fit_seeds)
    onea=np.asarray(one,dtype=np.float64);multia=np.asarray(all_multi,dtype=np.float64)
    metrics={"one_step_semantic_nrmse":float(np.sqrt(np.mean(onea*onea))) if len(onea) else 0.0,"multi_step_trajectory_nrmse":float(np.sqrt(np.mean(multia*multia))) if len(multia) else 0.0,"final_abstract_output_recovery":float(np.median(rec)),"counterfactual_task_success":float(np.mean(success)),"control_margin":float(min(setdiag.get("metrics",{}).get("norm_random_margin",-1e9),setdiag.get("metrics",{}).get("random_subspace_margin",-1e9),setdiag.get("metrics",{}).get("shuffled_margin",-1e9))),"median_ood_ratio":float(setdiag.get("metrics",{}).get("median_ood_ratio",1e9)),"interventions":len(samples)}
    passed=bool(metrics["one_step_semantic_nrmse"]<=.10 and metrics["multi_step_trajectory_nrmse"]<=.15 and metrics["final_abstract_output_recovery"]>=.80 and metrics["counterfactual_task_success"]>=.75 and metrics["control_margin"]>=.20 and metrics["median_ood_ratio"]<=2.0)
    return {"version":"V837ap","organism_id":oid,"family":family,"engine":geometry["engine"],"k":int(geometry["k"]),"chart_family":geometry["chart_family"],"phase_atlas":bool(geometry.get("phase_atlas")),"partition":partition_name,"metrics":metrics,"horizon_rows":horizon_rows,"pass":passed,"failure_code":None if passed else ("PHASE_DECODING_WITHOUT_CAUSAL_CHART_TRANSITION" if geometry.get("phase_atlas") else "CANONICAL_DYNAMICS_ROLLOUT_FAIL")}


def evaluate_dynamics(geometry:dict,q:np.ndarray,eval_seeds:list[int],partition_name:str="AP_DYNAMICS", *, chart_fit_seeds:list[int]|None=None, writer_fit_seeds:list[int]|None=None)->dict:
    natural=evaluate_natural_commutativity(geometry,q,eval_seeds,partition_name);inter=evaluate_interventional_commutativity(geometry,q,eval_seeds,partition_name,chart_fit_seeds=chart_fit_seeds,writer_fit_seeds=writer_fit_seeds)
    return {"version":"V837ap","organism_id":geometry["organism_id"],"family":geometry["family"],"engine":geometry["engine"],"k":int(geometry["k"]),"chart_family":geometry["chart_family"],"phase_atlas":bool(geometry.get("phase_atlas")),"natural":natural,"interventional":inter,"pass":bool(natural["pass"] and inter["pass"]),"failure_code":None if natural["pass"] and inter["pass"] else (natural.get("failure_code") if not natural["pass"] else inter.get("failure_code"))}


def save_commutativity(rows:list[dict])->None:
    write_json(HERE/"raw/commutativity_results.json",{"version":"V837ap","rows":rows})
    write_json(HERE/"diagnostics/natural_commutativity.json",{"version":"V837ap","rows":[r["natural"] for r in rows]})
    write_json(HERE/"diagnostics/interventional_commutativity.json",{"version":"V837ap","rows":[r["interventional"] for r in rows]})
