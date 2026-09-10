from __future__ import annotations

import numpy as np

from .controls import evaluate_controls
from .data_roles import seeds
from .interaction_eval import evaluate_interactions, interaction_bundles
from .localization import verify_frozen_units
from .natural_interventions import build_compound_cases, build_first_order_cases
from .operator_eval import evaluate_cases, operator_pass
from .operator_metrics import family_gate
from .source_folds import discovery_ids
from .utils import HERE, read_json, write_json


def _composition_pass(oid:str,family:str,seed_values:list[int])->dict:
    got=evaluate_cases(oid,family,build_compound_cases(family,seed_values));m=got["metrics"]
    passed=bool(m["response_nrmse"]<=.10 and m["direction_agreement"]>=.85 and m["perturbed_task_success"]>=.80)
    return {"metrics":m,"pass":passed}


def run_meta_confirm()->dict:
    discovery=read_json(HERE/"raw/operator_discovery.json");loc=read_json(HERE/"raw/localization_results.json");discovery_comp=read_json(HERE/"raw/composition_results.json");orders={f:int(discovery["families"][f]["operator_order"]) for f in discovery.get("accepted_families",[])}
    payload={"version":"V837aq","stage":"AQ_META_CONFIRM","families":{},"rows":[],"no_operator_refit":True,"no_fallback":True}
    fit_cases={f:build_first_order_cases(f,seeds("AQ_OPERATOR_FIT")) for f in discovery.get("accepted_families",[])}
    meta_cases={f:build_first_order_cases(f,seeds("AQ_META_CONFIRM")) for f in discovery.get("accepted_families",[])}
    for family in discovery.get("accepted_families",[]):
        loc_rows={r["organism_id"]:r for r in loc["families"].get(family,{}).get("organism_rows",[])};rows=[]
        bundles=interaction_bundles(family,seeds("AQ_META_CONFIRM"))
        for oid in discovery_ids(family):
            fit=evaluate_cases(oid,family,fit_cases[family]);meta=evaluate_cases(oid,family,meta_cases[family]);op,failed=operator_pass(meta["metrics"]);ctrl=evaluate_controls(family,fit["rows"],meta["rows"],oid);inter=evaluate_interactions(oid,family,bundles);comp=_composition_pass(oid,family,seeds("AQ_META_CONFIRM"));winner=loc_rows.get(oid,{}).get("winner")
            if winner:
                lmeta=verify_frozen_units(oid,family,winner["units"],seeds("AQ_META_CONFIRM"))
            else:lmeta={"pass":False,"reason":"no compact localization winner"}
            operator_ok=bool(op and ctrl["pass"] and inter["pass"]);program_ok=bool(operator_ok and comp["pass"]);row={"organism_id":oid,"family":family,"engine":meta["engine"],"operator_order":orders[family],"operator_metrics":meta["metrics"],"operator_failed":failed,"controls":ctrl,"interaction":inter,"composition":comp,"compact_localization_meta":lmeta,"operator_pass":operator_ok,"program_pass":program_ok,"pass":operator_ok};rows.append(row);payload["rows"].append(row)
        operator_gate=family_gate([{"pass":r["operator_pass"],"engine":r["engine"]} for r in rows]);program_gate=family_gate([{"pass":r["program_pass"],"engine":r["engine"]} for r in rows]);compact_gate=family_gate([{"pass":bool(r["compact_localization_meta"].get("pass")),"engine":r["engine"]} for r in rows]);discovery_comp_pass=bool(discovery_comp.get("families",{}).get(family,{}).get("pass"));program_closed=bool(discovery_comp_pass and program_gate["pass"]);payload["families"][family]={"operator_meta_gate":operator_gate,"program_meta_gate":program_gate,"discovery_composition_pass":discovery_comp_pass,"compact_localization_meta_gate":compact_gate,"operator_confirmed":operator_gate["pass"],"program_closed":program_closed,"pass":operator_gate["pass"],"operator_order":orders[family]}
    payload["confirmed_families"]=[f for f,v in payload["families"].items() if v["operator_confirmed"]];payload["confirmed_family_count"]=len(payload["confirmed_families"]);payload["program_closed_families"]=[f for f,v in payload["families"].items() if v["program_closed"]];payload["program_closed_family_count"]=len(payload["program_closed_families"])
    write_json(HERE/"raw/meta_confirmation.json",payload);write_json(HERE/"diagnostics/meta_confirmation.json",payload);return payload


if __name__=="__main__":
    import json;print(json.dumps(run_meta_confirm(),indent=2))
