from __future__ import annotations

import numpy as np

from experiments.v837_primitive_invention.v837an.an_a_causal_macrovariables import _eligibility
from experiments.v837_primitive_invention.v837an.trace_cache import pair_traces
from experiments.v837_primitive_invention.v837ao.phase_backends import PHASES, representative_phase_timesteps

from .geometry_runtime import read_state
from .setpoint_grid import family_grid


def select_quotient_pairs(geometry: dict, q: np.ndarray, seeds: list[int], partition_name: str = "AP_QUOTIENT") -> dict:
    oid=geometry["organism_id"]; family=geometry["family"]; Q=np.asarray(q,dtype=np.float64).reshape(40,-1); R=float(family_grid(family)["semantic_range"])
    data=pair_traces(oid,family,seeds); mask,_=_eligibility(data,family); eligible=np.flatnonzero(mask)
    states=data["base_trace"].states.detach().cpu().numpy().reshape(len(data["pairs"]),data["base_trace"].states.shape[1],40)
    rows=[]
    for phase in PHASES[family]:
        local=[]
        for i in eligible:
            ep=data["pairs"][i].base_episode; t=representative_phase_timesteps(ep).get(phase)
            if t is None: continue
            s=states[i,t].astype(np.float64); z=read_state(geometry,Q,s,phase); h=s@Q
            local.append({"episode_index":int(i),"seed":int(data["pairs"][i].base_seed),"timestep":int(t),"z":float(z),"h":h,"state":s})
        for a in local:
            candidates=[b for b in local if b["episode_index"]!=a["episode_index"] and abs(a["z"]-b["z"])<=0.10*R]
            if not candidates: continue
            donor=max(candidates,key=lambda b:float(np.linalg.norm(a["h"]-b["h"])))
            rows.append({
                "phase":phase,"anchor_episode_index":a["episode_index"],"anchor_seed":a["seed"],"anchor_timestep":a["timestep"],"anchor_z":a["z"],
                "donor_episode_index":donor["episode_index"],"donor_seed":donor["seed"],"donor_timestep":donor["timestep"],"donor_z":donor["z"],
                "semantic_distance":abs(a["z"]-donor["z"]),"projected_state_distance":float(np.linalg.norm(a["h"]-donor["h"])),"state_distance":float(np.linalg.norm(a["state"]-donor["state"]))
            })
    return {"version":"V837ap","organism_id":oid,"family":family,"engine":geometry["engine"],"k":int(geometry["k"]),"chart_family":geometry["chart_family"],"phase_atlas":bool(geometry.get("phase_atlas")),"partition":partition_name,"rule":"same organism/family/phase; |Ea-Eb|<=0.10Rz; maximize Euclidean projected-state distance without outcome use","outcome_used_for_pair_selection":False,"rows":rows,"pair_count":len(rows)}
