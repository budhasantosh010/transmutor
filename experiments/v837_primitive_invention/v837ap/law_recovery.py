from __future__ import annotations

import json
import numpy as np

from experiments.v837_primitive_invention.v837an.an_a_causal_macrovariables import _eligibility
from experiments.v837_primitive_invention.v837an.trace_cache import pair_traces
from experiments.v837_primitive_invention.v837ao.trajectory_eval import phase_at_timestep, semantic_timesteps

from .authorization import POWERED_FAMILIES
from .data_roles import seeds
from .freeze_geometry import freeze_family_geometries
from .geometry_runtime import read_state
from .geometry_store import load_discovery_geometry
from .projected_causal_spaces import space_map
from .source_folds import freeze_source_folds
from .utils import HERE, write_json


def _mse(a,b):
    x=np.asarray(a,dtype=np.float64);y=np.asarray(b,dtype=np.float64);return float(np.mean((x-y)**2)) if len(x) else float("inf")


def _organism_law(geom:dict,q:np.ndarray)->dict:
    family=geom["family"];oid=geom["organism_id"];data=pair_traces(oid,family,seeds("REUSED_HISTORICAL_VALIDATION"));mask,_=_eligibility(data,family);states=data["base_trace"].states.detach().cpu().numpy().reshape(len(data["pairs"]),data["base_trace"].states.shape[1],40)
    if family=="iterative_state":
        X=[];y=[]
        for i in np.flatnonzero(mask):
            ep=data["pairs"][i].base_episode;times=semantic_timesteps(ep)
            for t,nt in zip(times[:-1],times[1:]):
                p0=phase_at_timestep(ep,t);p1=phase_at_timestep(ep,nt)
                if p0 is None or p1 is None:continue
                z=read_state(geom,q,states[i,t],p0);zn=read_state(geom,q,states[i,nt],p1);x=float(np.asarray(ep.causal_inputs["x_t"])[nt-1]);X.append([z,x,1.0]);y.append(zn)
        coef=np.linalg.lstsq(np.asarray(X),np.asarray(y),rcond=None)[0] if X else np.asarray([np.nan]*3)
        return {"family":family,"organism_id":oid,"law":"z_next=a*z+b*x+c","coefficients":{"a":float(coef[0]),"b":float(coef[1]),"c":float(coef[2])},"expected":{"a":.65,"b":.35,"c":0.0},"samples":len(y)}
    if family=="variable_composition":
        X=[];y=[]
        for i in np.flatnonzero(mask):
            ep=data["pairs"][i].base_episode;times=semantic_timesteps(ep)
            for t,nt in zip(times[:-1],times[1:]):
                p0=phase_at_timestep(ep,t);p1=phase_at_timestep(ep,nt)
                if p0 is None or p1 is None:continue
                z=read_state(geom,q,states[i,t],p0);zn=np.clip(read_state(geom,q,states[i,nt],p1),-.999999,.999999);g=float(np.asarray(ep.causal_inputs["gain"])[nt-1]);d=float(np.asarray(ep.causal_inputs["drive"])[nt-1]);X.append([g*z,d,1.0]);y.append(float(np.arctanh(zn)))
        coef=np.linalg.lstsq(np.asarray(X),np.asarray(y),rcond=None)[0] if X else np.asarray([np.nan]*3)
        return {"family":family,"organism_id":oid,"law":"atanh(z_next)=a*(g*z)+b*d+c","coefficients":{"a":float(coef[0]),"b":float(coef[1]),"c":float(coef[2])},"expected":{"a":1.0,"b":1.0,"c":0.0},"samples":len(y)}
    if family=="delayed_recall":
        z0=[];z1=[]
        for i in np.flatnonzero(mask):
            ep=data["pairs"][i].base_episode;times=semantic_timesteps(ep)
            for t,nt in zip(times[:-1],times[1:]):
                p0=phase_at_timestep(ep,t);p1=phase_at_timestep(ep,nt)
                if p0 is None or p1 is None:continue
                z0.append(read_state(geom,q,states[i,t],p0));z1.append(read_state(geom,q,states[i,nt],p1))
        a=np.asarray(z0);b=np.asarray(z1);decay=float(np.dot(a,b)/max(np.dot(a,a),1e-12)) if len(a) else 0.0;aff=np.linalg.lstsq(np.column_stack([a,np.ones(len(a))]),b,rcond=None)[0] if len(a) else np.asarray([0.,0.]);scores={"HOLD":_mse(b,a),"DECAY":_mse(b,decay*a),"RESET":_mse(b,np.zeros_like(a)),"AFFINE":_mse(b,aff[0]*a+aff[1])};winner=min(scores,key=scores.get)
        return {"family":family,"organism_id":oid,"program_scores":scores,"winner":winner,"expected_winner":"HOLD","samples":len(a)}
    if family=="conditional_routing":
        truth=[];preds={"SELECT":[],"ALWAYS_A":[],"ALWAYS_B":[],"LINEAR_MIXTURE":[]}
        for i in np.flatnonzero(mask):
            ep=data["pairs"][i].base_episode;phase="POST_CONTROL";t=1;z=read_state(geom,q,states[i,t],phase);a=float(ep.causal_inputs["payload_a"]);b=float(ep.causal_inputs["payload_b"]);truth.append(float(ep.target));preds["SELECT"].append(a if z>0 else b);preds["ALWAYS_A"].append(a);preds["ALWAYS_B"].append(b);preds["LINEAR_MIXTURE"].append(.5*(a+b))
        scores={k:_mse(truth,v) for k,v in preds.items()};winner=min(scores,key=scores.get)
        return {"family":family,"organism_id":oid,"program_scores":scores,"winner":winner,"expected_winner":"SELECT","samples":len(truth)}
    raise KeyError(family)


def run_law_recovery()->dict:
    frozen=freeze_family_geometries();folds=freeze_source_folds();spaces=space_map();families={};rows=[]
    for family in POWERED_FAMILIES:
        fspec=frozen["families"].get(family)
        if fspec is None:
            families[family]={"run":False,"reason":"NO_VALID_CANONICAL_TRAJECTORY"};continue
        winner=fspec["selection_evidence"];org=[]
        for oid in folds["families"][family]["discovery"]:
            geom=load_discovery_geometry(family,oid,winner);q=np.asarray(spaces[(family,oid,int(winner["k"]))]["q"],dtype=np.float64);r=_organism_law(geom,q);org.append(r);rows.append(r)
        families[family]={"run":True,"organisms":len(org),"rows":org}
    payload={"version":"V837ap","stage":"AP17_TRANSITION_LAW_AUDIT","non_gating":True,"cannot_rescue_family":True,"cannot_upgrade_failed_family":True,"claim_restriction":"canonical trajectories make the known primitive law readily recoverable; no autonomous symbolic primitive invention claim","families":families,"rows":rows}
    write_json(HERE/"raw/law_recovery.json",payload);write_json(HERE/"diagnostics/law_recovery.json",payload);return payload

if __name__=="__main__":
    print(json.dumps(run_law_recovery(),indent=2,default=str))
