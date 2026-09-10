from __future__ import annotations

import itertools
import json
import numpy as np

from experiments.v837_primitive_invention.v837an.an_a_causal_macrovariables import _eligibility
from experiments.v837_primitive_invention.v837an.trace_cache import pair_traces
from experiments.v837_primitive_invention.v837ao.phase_backends import representative_phase_timesteps

from .authorization import POWERED_FAMILIES
from .data_roles import seeds
from .geometry_runtime import read_state
from .setpoint_grid import family_grid
from .utils import HERE, read_json, write_json


def _semantic_map(record:dict)->dict:
    geom=record["geometry"];q=np.asarray(record["q"],dtype=np.float64);family=record["family"];data=pair_traces(record["organism_id"],family,seeds("REUSED_HISTORICAL_VALIDATION"));mask,_=_eligibility(data,family);states=data["base_trace"].states.detach().cpu().numpy().reshape(len(data["pairs"]),data["base_trace"].states.shape[1],40);out={}
    for i in np.flatnonzero(mask):
        ep=data["pairs"][i].base_episode
        for phase,t in representative_phase_timesteps(ep).items():
            out[(int(data["pairs"][i].base_seed),phase)]=read_state(geom,q,states[i,t],phase)
    return out


def run_cross_organism_agreement()->dict:
    held=read_json(HERE/"raw/heldout_calibration_frontier.json");raw=read_json(HERE/"raw/heldout_backend_results.json");families={};rows=[]
    for family in POWERED_FAMILIES:
        info=held["families"].get(family,{});n=info.get("minimum_successful_n")
        if n is None:
            families[family]={"pass":False,"reason":"NO_HELDOUT_FAMILY_PASS","median":None,"p90":None};continue
        passing_ids={r["organism_id"] for r in raw.get("rows",[]) if r.get("family")==family and int(r.get("calibration_n",-1))==int(n) and r.get("pass")}
        records=[r for r in raw.get("backend_records",[]) if r.get("family")==family and int(r.get("calibration_n",-1))==int(n) and r.get("valid") and r.get("organism_id") in passing_ids]
        maps={r["organism_id"]:_semantic_map(r) for r in records};R=float(family_grid(family)["semantic_range"]);vals=[];groups={"DIRECTED↔DIRECTED":[],"RANDOM↔RANDOM":[],"DIRECTED↔RANDOM":[]}
        for a,b in itertools.combinations(records,2):
            ma=maps[a["organism_id"]];mb=maps[b["organism_id"]];common=sorted(set(ma).intersection(mb));pair=[abs(ma[k]-mb[k])/max(R,1e-12) for k in common];vals.extend(pair)
            ea=a["engine"];eb=b["engine"]
            if ea==eb=="DIRECTED_STRUCTURAL_SEARCH":group="DIRECTED↔DIRECTED"
            elif ea==eb=="RANDOM_STRUCTURAL_SAMPLER":group="RANDOM↔RANDOM"
            else:group="DIRECTED↔RANDOM"
            groups[group].extend(pair);rows.append({"family":family,"calibration_n":n,"organism_a":a["organism_id"],"organism_b":b["organism_id"],"engine_pair":group,"matched_states":len(pair),"median":float(np.median(pair)) if pair else None,"p90":float(np.quantile(pair,.90)) if pair else None})
        med=float(np.median(vals)) if vals else float("inf");p90=float(np.quantile(vals,.90)) if vals else float("inf");passed=bool(vals and med<=.10 and p90<=.25)
        families[family]={"pass":passed,"calibration_n":n,"organisms":len(records),"matched_disagreements":len(vals),"median":med,"p90":p90,"median_pairwise_disagreement":med,"p90_pairwise_disagreement":p90,"groups":{g:{"n":len(v),"median":float(np.median(v)) if v else None,"p90":float(np.quantile(v,.90)) if v else None} for g,v in groups.items()},"physical_alignment_performed":False}
    payload={"version":"V837ap","stage":"AP15_CROSS_ORGANISM_AGREEMENT","families":families,"rows":rows,"physical_state_alignment":False,"q_alignment":False,"microstate_comparison_used":False,"state_alignment_used":False,"q_alignment_used":False,"coefficient_comparison_as_criterion":False,"passing_families":sum(1 for v in families.values() if v.get("pass"))}
    write_json(HERE/"raw/cross_organism_agreement.json",payload);write_json(HERE/"diagnostics/cross_organism_agreement.json",payload);return payload

if __name__=="__main__":
    print(json.dumps(run_cross_organism_agreement(),indent=2,default=str))
