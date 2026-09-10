from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np

from .authorization import COUNTERFACTUALS, SUCCESS_TOLERANCE
from .oracle_macrostate import InstrumentedEpisode, instrument_episode, primary_value
from .phase_masks import primary_timestep
from .utils import HERE, sha256_json, write_json


@dataclass
class CounterfactualPair:
    family: str
    base_seed: int
    base_episode: InstrumentedEpisode
    counterfactual_episode: InstrumentedEpisode
    intervention_kind: str
    semantic_delta: float
    primary_phase: int


def _clone(ep:InstrumentedEpisode,*,obs:np.ndarray,target:float,macro:dict,causal:dict|None=None,latent:dict|None=None,metadata:dict|None=None)->InstrumentedEpisode:
    return InstrumentedEpisode(ep.family,ep.seed,ep.split,np.asarray(obs,dtype=np.float32),float(target),list(ep.phase_labels),{k:np.asarray(v).copy() for k,v in macro.items()},dict(ep.causal_inputs if causal is None else causal),dict(ep.latent_variables if latent is None else latent),dict(ep.metadata if metadata is None else metadata))


def make_counterfactual(family:str,seed:int,split:str="development")->CounterfactualPair:
    if split not in {"development","validation"}:
        raise ValueError(f"V837an counterfactual split must be development/validation, got {split}")
    base=instrument_episode(family,int(seed),split);obs=base.observations.copy();macro={k:v.copy() for k,v in base.macrostate_traces.items()}
    if family=="conditional_routing":
        c=float(base.causal_inputs["control"]);a=float(base.causal_inputs["payload_a"]);b=float(base.causal_inputs["payload_b"]);cp=-c;obs[1,0]=cp;target=a if cp>0 else b
        macro["ROUTING_CONTROL_STATE"][1:]=cp;macro["ROUTING_SELECTED_VALUE"][3:]=target
        causal={**base.causal_inputs,"control":cp,"selected_payload":target};meta={**base.metadata,"control":cp}
        cf=_clone(base,obs=obs,target=target,macro=macro,causal=causal,metadata=meta)
    elif family=="delayed_recall":
        value=float(base.causal_inputs["value"]);vp=-value;obs[1,0]=vp
        for key in ("RECALL_MAINTAIN","RECALL_WRITE","RECALL_READ"):macro[key][1:]=vp
        cf=_clone(base,obs=obs,target=vp,macro=macro,causal={"value":vp})
    elif family=="iterative_state":
        values=np.asarray(base.causal_inputs["x_t"],dtype=np.float64).copy();j=(len(values)-1)//2;values[j]=-values[j];obs[1+j,0]=values[j]
        state=0.0;before=[];after=[];running=[0.0]
        for x in values:before.append(state);state=.65*state+.35*float(x);after.append(state);running.append(state)
        macro["ITERATIVE_RUNNING_STATE"]=np.asarray(running);causal={"x_t":values,"s_before":np.asarray(before),"s_after":np.asarray(after)}
        cf=_clone(base,obs=obs,target=state,macro=macro,causal=causal)
    elif family=="partial_observation":
        rho=float(base.causal_inputs["rho"]);z0=-float(base.latent_variables["initial_z0"]);eps=np.asarray(base.causal_inputs["innovation"]);noise=np.asarray(base.causal_inputs["observation_noise"])
        z=z0;z_after=[];z_before=[];observed=[]
        for i,(e,n) in enumerate(zip(eps,noise)):
            z_before.append(z);z=rho*z+float(e);z_after.append(z);raw=z+float(n);observed.append(raw);obs[1+i,0]=float(np.tanh(raw))
        target=float(np.tanh(rho*z));macro["PARTIAL_LATENT_Z"]=np.asarray([z0]+z_after);macro["PARTIAL_PREDICTIVE_MEAN"]=np.asarray([np.tanh(rho*z0)]+[np.tanh(rho*v) for v in z_after])
        latent={**base.latent_variables,"initial_z0":z0,"z_before":np.asarray(z_before),"z_after":np.asarray(z_after),"observed_raw":np.asarray(observed)};meta={**base.metadata,"hidden_target":target}
        cf=_clone(base,obs=obs,target=target,macro=macro,latent=latent,metadata=meta)
    elif family=="variable_composition":
        initial=-float(base.causal_inputs["initial_value"]);gains=np.asarray(base.causal_inputs["gain"]);drives=np.asarray(base.causal_inputs["drive"]);obs[1,0]=initial
        value=initial;before=[];after=[];running=[initial]
        for g,d in zip(gains,drives):before.append(value);value=float(np.tanh(float(g)*value+float(d)));after.append(value);running.append(value)
        macro["COMPOSITION_RUNNING_STATE"]=np.asarray(running);causal={**base.causal_inputs,"initial_value":initial,"v_before":np.asarray(before),"v_after":np.asarray(after)}
        cf=_clone(base,obs=obs,target=value,macro=macro,causal=causal)
    else: raise KeyError(family)
    t=primary_timestep(base);semantic_delta=primary_value(cf,t)-primary_value(base,t)
    return CounterfactualPair(family,int(seed),base,cf,COUNTERFACTUALS[family],float(semantic_delta),t)


def partial_rho_mirror(seed:int)->CounterfactualPair:
    base=instrument_episode("partial_observation",int(seed),"development");rho=float(base.causal_inputs["rho"]);mirrored=1.66-rho
    # Diagnostic only. Reconstruct with identical z0, innovations, observation noise and nuisance rows.
    obs=base.observations.copy();z=float(base.latent_variables["initial_z0"]);zs=[];raws=[]
    for i,(e,n) in enumerate(zip(base.causal_inputs["innovation"],base.causal_inputs["observation_noise"])):
        z=mirrored*z+float(e);zs.append(z);raw=z+float(n);raws.append(raw);obs[1+i,0]=float(np.tanh(raw))
    target=float(np.tanh(mirrored*z));macro={"PARTIAL_LATENT_Z":np.asarray([base.latent_variables["initial_z0"]]+zs),"PARTIAL_PREDICTIVE_MEAN":np.asarray([np.tanh(mirrored*base.latent_variables["initial_z0"] )]+[np.tanh(mirrored*v) for v in zs])}
    cf=_clone(base,obs=obs,target=target,macro=macro,causal={**base.causal_inputs,"rho":mirrored},latent={**base.latent_variables,"z_after":np.asarray(zs),"observed_raw":np.asarray(raws)},metadata={**base.metadata,"rho":mirrored,"hidden_target":target})
    t=primary_timestep(base);return CounterfactualPair("partial_observation",int(seed),base,cf,"RHO_MIRROR",float(primary_value(cf,t)-primary_value(base,t)),t)


def verify_counterfactual_constructors()->dict:
    summaries=[]
    for family in sorted(COUNTERFACTUALS):
        changed_target=0;max_unchanged_error=0.0
        for seed in range(10000,10512):
            p=make_counterfactual(family,seed)
            if len(p.base_episode.observations)!=len(p.counterfactual_episode.observations):raise RuntimeError("COUNTERFACTUAL_EPISODE_INVALID:length")
            if not np.isfinite(p.counterfactual_episode.observations).all() or not np.isfinite(p.counterfactual_episode.target):raise RuntimeError("COUNTERFACTUAL_EPISODE_INVALID:nonfinite")
            if abs(p.base_episode.target-p.counterfactual_episode.target)>=SUCCESS_TOLERANCE[family]:changed_target+=1
        summaries.append({"family":family,"intervention":COUNTERFACTUALS[family],"seeds":512,"meaningful_target_displacement":changed_target})
    payload={"version":"V837an","pass":True,"families":summaries,"task_seed_source":"historical seeds only","new_random_task_seeds":0,"fresh_audit_consumed":False}
    write_json(HERE/"raw/counterfactual_specifications.json",payload);write_json(HERE/"diagnostics/counterfactual_validity.json",payload);return payload


if __name__=="__main__":
    import json;print(json.dumps(verify_counterfactual_constructors(),indent=2))
