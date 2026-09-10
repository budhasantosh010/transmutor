from __future__ import annotations

import math
import numpy as np

from .controls import evaluate_controls
from .data_roles import seeds
from .freeze_operator_contracts import OUT, freeze_operator_contracts
from .interaction_eval import evaluate_interactions, interaction_bundles
from .natural_interventions import build_compound_cases, build_first_order_cases
from .operator_eval import evaluate_cases, operator_pass
from .operator_metrics import family_gate
from .source_folds import holdout_ids
from .utils import HERE, read_json, write_json


def _composition(oid:str,family:str)->dict:
    got=evaluate_cases(oid,family,build_compound_cases(family,seeds("AQ_HELDOUT_EVAL")));m=got["metrics"];return {"metrics":m,"pass":bool(m["response_nrmse"]<=.10 and m["direction_agreement"]>=.85 and m["perturbed_task_success"]>=.80)}


def _cross(rows:list[dict])->dict:
    good=[r for r in rows if r["pass"]]
    if len(good)<2:return {"pass":False,"organisms":len(good),"median_response_nrmse":None,"max_response_nrmse":None}
    vals=[r["operator_metrics"]["response_nrmse"] for r in good];return {"pass":True,"organisms":len(good),"median_response_nrmse":float(np.median(vals)),"max_response_nrmse":float(max(vals)),"state_alignment_used":False,"q_alignment_used":False}


def run_heldout_confirmation()->dict:
    if not OUT.is_file():raise RuntimeError("V837AQ_HELDOUT_BLOCKED_UNTIL_OPERATOR_CONTRACT_FREEZE")
    frozen=read_json(OUT);payload={"version":"V837aq","stage":"AQ10_HELDOUT_ORGANISM_CONFIRMATION","contract_sha256":frozen["contract_sha256"],"families":{},"rows":[],"operator_refit":False,"hyperparameter_search":False,"hidden_state_alignment":False}
    for family in frozen.get("confirmed_families",[]):
        fit_cases=build_first_order_cases(family,seeds("AQ_OPERATOR_FIT"));held_cases=build_first_order_cases(family,seeds("AQ_HELDOUT_EVAL"));bundles=interaction_bundles(family,seeds("AQ_HELDOUT_EVAL"));rows=[]
        for oid in holdout_ids(family):
            fit=evaluate_cases(oid,family,fit_cases);held=evaluate_cases(oid,family,held_cases);op,failed=operator_pass(held["metrics"]);ctrl=evaluate_controls(family,fit["rows"],held["rows"],oid);inter=evaluate_interactions(oid,family,bundles);comp=_composition(oid,family);operator_ok=bool(op and ctrl["pass"] and inter["pass"]);program_ok=bool(operator_ok and comp["pass"]);row={"organism_id":oid,"family":family,"engine":held["engine"],"operator_metrics":held["metrics"],"operator_failed":failed,"controls":ctrl,"interaction":inter,"composition":comp,"operator_pass":operator_ok,"program_pass":program_ok,"pass":operator_ok};rows.append(row);payload["rows"].append(row);print(f"V837aq heldout {family} {oid[:10]} operator={operator_ok} program={program_ok} nrmse={held['metrics']['response_nrmse']:.4f}",flush=True)
        operator_passing=[r for r in rows if r["operator_pass"]];operator_engines={r["engine"] for r in operator_passing};gate={"organisms":len(rows),"required":3,"passing":len(operator_passing),"directed":sum(r["engine"]=="DIRECTED_STRUCTURAL_SEARCH" for r in operator_passing),"random":sum(r["engine"]=="RANDOM_STRUCTURAL_SAMPLER" for r in operator_passing)};gate["pass"]=bool(len(operator_passing)>=3 and "DIRECTED_STRUCTURAL_SEARCH" in operator_engines and "RANDOM_STRUCTURAL_SAMPLER" in operator_engines)
        program_passing=[r for r in rows if r["program_pass"]];program_engines={r["engine"] for r in program_passing};program_gate={"organisms":len(rows),"required":3,"passing":len(program_passing),"directed":sum(r["engine"]=="DIRECTED_STRUCTURAL_SEARCH" for r in program_passing),"random":sum(r["engine"]=="RANDOM_STRUCTURAL_SAMPLER" for r in program_passing)};program_gate["pass"]=bool(len(program_passing)>=3 and "DIRECTED_STRUCTURAL_SEARCH" in program_engines and "RANDOM_STRUCTURAL_SAMPLER" in program_engines)
        payload["families"][family]={"operator_gate":gate,"program_gate":program_gate,"cross_organism":_cross(rows),"operator_pass":gate["pass"],"program_pass":program_gate["pass"],"pass":gate["pass"]}
    payload["heldout_confirmed_families"]=[f for f,v in payload["families"].items() if v["operator_pass"]];payload["heldout_confirmed_family_count"]=len(payload["heldout_confirmed_families"]);payload["heldout_program_closed_families"]=[f for f,v in payload["families"].items() if v["program_pass"]];payload["heldout_program_closed_family_count"]=len(payload["heldout_program_closed_families"])
    write_json(HERE/"raw/heldout_confirmation.json",payload);write_json(HERE/"diagnostics/heldout_confirmation.json",payload);return payload


if __name__=="__main__":
    import json;print(json.dumps(run_heldout_confirmation(),indent=2))
