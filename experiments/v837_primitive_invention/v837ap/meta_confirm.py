from __future__ import annotations

import json
import math

import numpy as np

from .authorization import POWERED_FAMILIES
from .candidate_selection import family_gate
from .commutativity import evaluate_dynamics
from .data_roles import seeds
from .geometry_eval import evaluate_reader_geometry
from .geometry_store import load_discovery_geometry
from .projected_causal_spaces import space_map
from .quotient_eval import evaluate_quotient
from .setpoint_eval import evaluate_geometry
from .source_folds import freeze_source_folds
from .utils import HERE, read_json, write_json


def evaluate_full_no_refit(family: str, winner: dict, partition: str) -> list[dict]:
    spaces=space_map();folds=freeze_source_folds();rows=[]
    for idx,oid in enumerate(folds["families"][family]["discovery"],1):
        geom=load_discovery_geometry(family,oid,winner);q=np.asarray(spaces[(family,oid,int(winner["k"]))]["q"],dtype=np.float64)
        reader=evaluate_reader_geometry(geom,q,seeds(partition),partition)
        if reader.get("pass"):
            setp=evaluate_geometry(geom,q,eval_partition=partition)
        else:
            setp={"organism_id":oid,"family":family,"engine":geom.get("engine"),"pass":False,"failure_code":"READER_GATE_FAIL","metrics":{}}
        if reader.get("pass") and setp.get("pass"):
            quot=evaluate_quotient(geom,q,seeds(partition),partition)
            dyn=evaluate_dynamics(geom,q,seeds(partition),partition)
        else:
            quot={"pass":False,"failure_code":"UPSTREAM_GATE_FAIL"};dyn={"pass":False,"failure_code":"UPSTREAM_GATE_FAIL"}
        passed=bool(reader.get("pass") and setp.get("pass") and quot.get("pass") and dyn.get("pass"))
        rows.append({"organism_id":oid,"family":family,"engine":geom.get("engine"),"reader":reader,"setpoint":setp,"quotient":quot,"dynamics":dyn,"pass":passed})
        print(f"V837ap {partition} {family} {idx}/{len(folds['families'][family]['discovery'])} pass={passed}",flush=True)
    return rows


def run_meta_confirm()->dict:
    selection=read_json(HERE/"raw/discovery_family_geometry_winners.json");closure=read_json(HERE/"raw/model_complexity_adjudication.json");results=[];confirmed={}
    for family in POWERED_FAMILIES:
        winner=selection["family_winners"].get(family);closed=bool(closure["families"].get(family,{}).get("pass"))
        if winner is None:
            confirmed[family]=None;results.append({"family":family,"candidate":None,"partition":"AP_META_CONFIRM","no_refit":True,"pass":False,"reason":"NO_AP7_DISCOVERY_WINNER"});continue
        if not closed:
            confirmed[family]=None;results.append({"family":family,"candidate":winner,"partition":"AP_META_CONFIRM","no_refit":True,"pass":False,"reason":"DISCOVERY_QUOTIENT_OR_DYNAMICS_FAILED","no_fallback":True});continue
        rows=evaluate_full_no_refit(family,winner,"AP_META_CONFIRM");gate=family_gate(rows,"pass");cand={**winner,"meta_gate":gate} if gate["pass"] else None;confirmed[family]=cand
        results.append({"family":family,"candidate":winner,"partition":"AP_META_CONFIRM","no_refit":True,"no_fallback":True,"organism_results":rows,"aggregate":gate,"pass":bool(cand)})
    payload={"version":"V837ap","stage":"AP11_META_CONFIRM","partition":"AP_META_CONFIRM","no_refit":True,"no_geometry_fallback":True,"results":results,"confirmed_families":confirmed,"meta_confirmed_families":sum(v is not None for v in confirmed.values())}
    write_json(HERE/"raw/meta_confirmation.json",payload);write_json(HERE/"diagnostics/meta_confirmation.json",payload);return payload

if __name__=="__main__":
    print(json.dumps(run_meta_confirm(),indent=2,default=str))
