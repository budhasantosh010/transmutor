from __future__ import annotations

import numpy as np

from .abstract_rollout import next_canonical_state
from .canonical_reader import read_k1
from .k1_backend import component_for_phase


def phase_at_timestep(ep, t: int) -> str | None:
    family=ep.family;t=int(t)
    if family=="conditional_routing": return {1:"POST_CONTROL",2:"POST_PAYLOAD_A",3:"POST_PAYLOAD_B"}.get(t)
    if family=="delayed_recall":
        delay=int(ep.latent_variables["delay"])
        if t==1:return "POST_WRITE"
        if t==1+delay:return "PRE_READ"
        if 2<=t<1+delay:return "MID_DELAY"
        return None
    if family in {"iterative_state","variable_composition"}:
        count=int(ep.latent_variables["length"] if family=="iterative_state" else ep.latent_variables["depth"])
        if t<1 or t>count:return None
        frac=(t-0.5)/max(count,1)
        return "EARLY" if frac<1/3 else ("MIDDLE" if frac<2/3 else "LATE")
    return None


def semantic_timesteps(ep) -> list[int]:
    if ep.family=="conditional_routing":return [1,2,3]
    if ep.family=="delayed_recall":return list(range(1,2+int(ep.latent_variables["delay"])))
    count=int(ep.latent_variables["length"] if ep.family=="iterative_state" else ep.latent_variables["depth"])
    return list(range(1,count+1))


def read_state_at(backend:dict,state40:np.ndarray,phase:str)->float:
    comp=component_for_phase(backend,phase)
    return float(read_k1(np.asarray(state40,dtype=np.float64).reshape(1,40),comp["reader"])[0])


def expected_future(ep,start_t:int,z_star:float)->dict[int,float]:
    z=float(z_star);out={int(start_t):z}
    family=ep.family
    for t in semantic_timesteps(ep):
        if t<=start_t:continue
        if family in {"conditional_routing","delayed_recall"}:z=float(z)
        elif family=="iterative_state":
            x=float(np.asarray(ep.causal_inputs["x_t"])[t-1]);z=next_canonical_state(family,z,{"x":x})
        elif family=="variable_composition":
            g=float(np.asarray(ep.causal_inputs["gain"])[t-1]);d=float(np.asarray(ep.causal_inputs["drive"])[t-1]);z=next_canonical_state(family,z,{"gain":g,"drive":d})
        out[t]=float(z)
    return out


def trajectory_nrmse(backend:dict,ep,start_t:int,z_star:float,trace_states:np.ndarray,semantic_range:float)->float:
    expected=expected_future(ep,start_t,z_star);errs=[]
    for t,z in expected.items():
        if t<=start_t:continue
        phase=phase_at_timestep(ep,t)
        if phase is None:continue
        got=read_state_at(backend,trace_states[t],phase);errs.append((got-z)/max(float(semantic_range),1e-12))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0
