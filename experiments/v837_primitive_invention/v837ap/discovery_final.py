from __future__ import annotations

import json

from .authorization import POWERED_FAMILIES
from .candidate_selection import family_gate
from .meta_confirm import evaluate_full_no_refit
from .utils import HERE, read_json, write_json


def run_discovery_final()->dict:
    meta=read_json(HERE/"raw/meta_confirmation.json");rows=[];survivors={}
    for family in POWERED_FAMILIES:
        cand=meta["confirmed_families"].get(family)
        if cand is None:
            survivors[family]=None;rows.append({"family":family,"candidate":None,"partition":"AP_DISCOVERY_FINAL","no_refit":True,"pass":False,"reason":"NO_META_CONFIRMED_CANDIDATE"});continue
        organism_rows=evaluate_full_no_refit(family,cand,"AP_DISCOVERY_FINAL");gate=family_gate(organism_rows,"pass");survivors[family]={**cand,"discovery_final_gate":gate} if gate["pass"] else None
        rows.append({"family":family,"candidate":cand,"partition":"AP_DISCOVERY_FINAL","no_refit":True,"organism_results":organism_rows,"aggregate":gate,"pass":gate["pass"]})
    payload={"version":"V837ap","stage":"AP12_DISCOVERY_FINAL","partition":"AP_DISCOVERY_FINAL","no_refit":True,"no_fallback":True,"results":rows,"surviving_families":survivors,"survivor_count":sum(v is not None for v in survivors.values())}
    write_json(HERE/"raw/discovery_final.json",payload);write_json(HERE/"raw/discovery_final_confirmation.json",payload);write_json(HERE/"diagnostics/discovery_final_confirmation.json",payload);return payload

if __name__=="__main__":
    print(json.dumps(run_discovery_final(),indent=2,default=str))
