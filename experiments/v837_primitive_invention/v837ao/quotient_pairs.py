from __future__ import annotations

import numpy as np

from experiments.v837_primitive_invention.v837an.an_a_causal_macrovariables import _eligibility
from experiments.v837_primitive_invention.v837an.trace_cache import pair_traces

from .canonical_reader import read_k1
from .k1_backend import component_for_phase
from .phase_backends import PHASES, representative_phase_timesteps
from .residual_projector import residual
from .setpoint_grid import family_grid


def select_quotient_pairs(backend:dict,seeds:list[int],partition_name:str)->dict:
    oid=backend["organism_id"];family=backend["family"];R=float(family_grid(family)["semantic_range"]);data=pair_traces(oid,family,seeds);mask,_=_eligibility(data,family);eligible=np.flatnonzero(mask)
    states=data["base_trace"].states.detach().cpu().numpy().reshape(len(data["pairs"]),data["base_trace"].states.shape[1],40);rows=[]
    for phase in PHASES[family]:
        comp=component_for_phase(backend,phase)
        if not comp.get("valid"):continue
        local=[]
        for i in eligible:
            ep=data["pairs"][i].base_episode;t=representative_phase_timesteps(ep).get(phase)
            if t is None:continue
            s=states[i,t].astype(np.float64);z=float(read_k1(s[None,:],comp["reader"])[0]);r=residual(s,comp)
            local.append({"episode_index":int(i),"seed":int(data["pairs"][i].base_seed),"timestep":int(t),"state":s,"z":z,"residual":r})
        for a in local:
            candidates=[b for b in local if b["episode_index"]!=a["episode_index"] and abs(a["z"]-b["z"])<=.25*R]
            if not candidates:continue
            donor=max(candidates,key=lambda b:float(np.linalg.norm(a["residual"]-b["residual"])))
            rows.append({"phase":phase,"anchor_episode_index":a["episode_index"],"anchor_seed":a["seed"],"anchor_timestep":a["timestep"],"anchor_z":a["z"],"donor_episode_index":donor["episode_index"],"donor_seed":donor["seed"],"donor_timestep":donor["timestep"],"donor_z":donor["z"],"semantic_distance":abs(a["z"]-donor["z"]),"residual_distance":float(np.linalg.norm(a["residual"]-donor["residual"]))})
    return {"organism_id":oid,"family":family,"engine":backend["engine"],"variant":backend["variant"],"partition":partition_name,"rule":"same organism/family/frozen phase; |za-zb|<=0.25Rz; maximum residual distance","rows":rows,"pair_count":len(rows)}
