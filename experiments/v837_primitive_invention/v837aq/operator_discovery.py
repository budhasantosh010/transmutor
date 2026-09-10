from __future__ import annotations

import json
import math
from collections import defaultdict
import numpy as np

from .authorization import POWERED_FAMILIES
from .controls import evaluate_controls
from .data_roles import seeds
from .failure_ledger import add, make_entry
from .interaction_eval import run_interaction_program
from .natural_interventions import build_first_order_cases
from .operator_eval import evaluate_cases, operator_pass
from .operator_metrics import RANGE, family_gate
from .source_folds import discovery_ids
from .utils import HERE, ROOT, read_json, write_json

RECON=ROOT/"experiments/v837_primitive_invention/v837ak/raw/reconstruction_results.json"


def _compact(result:dict)->dict:
    return {k:v for k,v in result.items() if k!="rows"}


def _cross_agreement(rows:list[dict])->dict:
    passing=[r for r in rows if r.get("pass")]
    if len(passing)<2:return {"pass":False,"median_prediction_std_normalized":None,"p90_pairwise_disagreement_normalized":None,"organisms":len(passing)}
    arrays=[np.asarray([x["predicted_response"] for x in r["select_rows"]],dtype=float) for r in passing];n=min(map(len,arrays));A=np.stack([a[:n] for a in arrays]);std=np.std(A,axis=0)/RANGE;pair=[]
    for i in range(len(A)):
        for j in range(i+1,len(A)):pair.extend((np.abs(A[i]-A[j])/RANGE).tolist())
    med=float(np.median(std));p90=float(np.quantile(pair,.9)) if pair else 0.0
    return {"pass":bool(med<=.10 and p90<=.20),"median_prediction_std_normalized":med,"p90_pairwise_disagreement_normalized":p90,"organisms":len(passing),"state_alignment_used":False,"q_alignment_used":False}


def _incompetent_rows(family:str)->list[dict]:
    pop=read_json(RECON)["rows"];return [r for r in pop if not r.get("competent") and r["family"]==family]


def run_operator_discovery()->dict:
    from .reality_gate import run_reality_gate
    reality=run_reality_gate()
    if not reality["pass"]:
        payload={"version":"V837aq","stage":"AQ3_AQ5","blocked":True,"reason":"PROGRAM_OPERATOR_MEASUREMENT_INVALID","families":{},"operator_family_count":0};write_json(HERE/"raw/operator_discovery.json",payload);return payload
    payload={"version":"V837aq","stage":"AQ3_AQ5_FIRST_ORDER_AND_EQUIVALENCE","families":{},"rows":[],"incompetent_controls":[],"fresh_audit_consumed":False}
    eligible=[]
    for family in POWERED_FAMILIES:
        fit_cases=build_first_order_cases(family,seeds("AQ_OPERATOR_FIT"));select_cases=build_first_order_cases(family,seeds("AQ_OPERATOR_SELECT"));rows=[]
        for idx,oid in enumerate(discovery_ids(family),1):
            fit=evaluate_cases(oid,family,fit_cases);sel=evaluate_cases(oid,family,select_cases);op_pass,failed=operator_pass(sel["metrics"],reality=False);ctrl=evaluate_controls(family,fit["rows"],sel["rows"],oid);passed=bool(op_pass and ctrl["pass"]);failed_all=list(failed)+[f"control:{x}" for x in ctrl["failed_conditions"]]
            row={**_compact(sel),"fit_metrics":fit["metrics"],"controls":ctrl,"operator_gate_pass":op_pass,"pass":passed,"failed_conditions":failed_all,"select_rows":sel["rows"]};rows.append(row);payload["rows"].append({k:v for k,v in row.items() if k!="select_rows"})
            print(f"V837aq operator {family} {idx}/{len(discovery_ids(family))} pass={passed} nrmse={sel['metrics']['response_nrmse']:.4f} corr={sel['metrics']['pearson']:.3f} ctrl={ctrl['pass']}",flush=True)
            if not passed:
                add(make_entry(failure_id=f"V837aq-AQ3-{family}-{oid[:14]}",stage="AQ3_AQ5_OPERATOR_EQUIVALENCE",branch="FIRST_ORDER_OPERATOR",family=family,organism=oid,operator_order=1,metrics=sel["metrics"],matched_control_metrics=ctrl,acceptance_gate={"response_nrmse_max":.10,"pearson_min":.85,"direction_min":.85,"task_success_min":.80,"control_margins":.03},failed_conditions=failed_all,scientific_interpretation="This competent organism did not satisfy the complete oracle-relative first-order operator plus specificity-control gate.",confounds_ruled_out=["ordinary benchmark competence alone","hidden-state coordinate alignment","search-engine negative labeling"],confounds_remaining=["second-order interaction if oracle requires it","family-specific temporal context"],artifact_paths=["experiments/v837_primitive_invention/v837aq/raw/operator_discovery.json"]))
        write_json(HERE/f"raw/operator_response_tensors/{family}.json",{"version":"V837aq","family":family,"partition":"AQ_OPERATOR_SELECT","organisms":[{"organism_id":r["organism_id"],"engine":r["engine"],"pass":r["pass"],"rows":r["select_rows"]} for r in rows]})
        fg=family_gate(rows);cross=_cross_agreement(rows)
        inc=[]
        for src in _incompetent_rows(family):
            oid=src["organism_id"];fit=evaluate_cases(oid,family,fit_cases);sel=evaluate_cases(oid,family,select_cases);op,_=operator_pass(sel["metrics"]);ctrl=evaluate_controls(family,fit["rows"],sel["rows"],oid);inc_pass=bool(op and ctrl["pass"]);inc.append({"organism_id":oid,"family":family,"engine":src["engine"],"final_validation_success":src["final_validation_success"],"metrics":sel["metrics"],"controls":ctrl,"pass":inc_pass})
        payload["incompetent_controls"].extend(inc)
        if inc:
            inc_rate=float(np.mean([r["pass"] for r in inc]));adv=fg["pass_fraction"]-inc_rate;inc_specific=adv>=.25
        else:inc_rate=None;adv=None;inc_specific=True
        family_pass=bool(fg["pass"] and cross["pass"] and inc_specific)
        payload["families"][family]={"first_order_gate":fg,"cross_organism":cross,"historical_incompetent_available":len(inc),"historical_incompetent_pass_rate":inc_rate,"competent_minus_incompetent_pass_rate":adv,"incompetent_specificity_pass":inc_specific,"pass_before_interaction":family_pass}
        if family_pass:eligible.append(family)
    # AQ4 is an oracle-property audit for every powered family. A family may
    # still fail AQ3, but we must not silently skip evidence that second-order
    # interaction is required by the task law.
    interactions=run_interaction_program(list(POWERED_FAMILIES))
    accepted=[]
    for family in POWERED_FAMILIES:
        base=payload["families"][family];inter=interactions["families"].get(family,{"pass":False,"second_order_required":False,"gate":{"pass":False}});base["interaction"]=inter;base["operator_order"]=2 if inter.get("second_order_required") else 1;base["pass"]=bool(base["pass_before_interaction"] and inter.get("pass"));
        if base["pass"]:accepted.append(family)
    payload["accepted_families"]=accepted;payload["operator_family_count"]=len(accepted);payload["interaction_artifact"]="raw/interaction_results.json"
    # Detailed response tensors are stored one file per family; this summary remains reviewable.
    write_json(HERE/"raw/operator_discovery.json",payload);write_json(HERE/"diagnostics/operator_equivalence.json",payload);write_json(HERE/"diagnostics/incompetent_controls.json",{"version":"V837aq","rows":payload["incompetent_controls"],"all_historical_incompetents_registered_in":"raw/incompetent_control_registry.json"})
    return payload


if __name__=="__main__":print(json.dumps(run_operator_discovery(),indent=2))
