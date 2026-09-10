from __future__ import annotations

import json

from .data_roles import seeds
from .failure_ledger import add, make_entry
from .natural_interventions import build_first_order_cases
from .operator_eval import evaluate_cases, operator_pass
from .operator_metrics import family_gate
from .source_folds import discovery_ids
from .utils import HERE, write_json


def run_reality_gate() -> dict:
    family="iterative_state";cases=build_first_order_cases(family,seeds("AQ_REALITY_GATE"),reality=True);rows=[]
    for i,oid in enumerate(discovery_ids(family),1):
        result=evaluate_cases(oid,family,cases);passed,failed=operator_pass(result["metrics"],reality=True);result["pass"]=passed;result["failed_conditions"]=failed;result["rows"]=result["rows"]
        rows.append(result);print(f"V837aq reality {i}/{len(discovery_ids(family))} {oid[:10]} pass={passed} nrmse={result['metrics']['response_nrmse']:.5f} corr={result['metrics']['pearson']:.4f} gain={result['metrics']['gain_ratio']:.4f}",flush=True)
        if not passed:
            add(make_entry(failure_id=f"V837aq-AQ2-{oid[:16]}",stage="AQ2_OPERATOR_REALITY_GATE",branch="ITERATIVE_IMPULSE_RESPONSE",family=family,organism=oid,intervention="INPUT_PERTURB",operator_order=1,metrics=result["metrics"],acceptance_gate={"nrmse_max":.08,"pearson_min":.90,"direction_min":.90,"task_success_min":.85,"gain_ratio":[.75,1.25]},failed_conditions=failed,scientific_interpretation="This organism did not recover the known iterative-state finite-horizon impulse-response law under legitimate input-channel perturbations.",confounds_ruled_out=["hidden-state SET","neural coordinate alignment","fresh audit"],confounds_remaining=["organism approximation error","operator measurement validity"],next_justified_experiment="Use the predeclared family-level kill gate; do not tune thresholds."))
    gate=family_gate(rows);payload={"version":"V837aq","stage":"AQ2_OPERATOR_REALITY_GATE","oracle_kernel":"0.35*(0.65**(h-1))*delta","case_count":len(cases),"organisms":rows,"family_gate":gate,"pass":gate["pass"],"kill_switch_triggered":not gate["pass"],"fresh_audit_consumed":False}
    write_json(HERE/"raw/operator_reality_gate.json",payload);write_json(HERE/"diagnostics/operator_reality_gate.json",payload)
    return payload


if __name__=="__main__": print(json.dumps(run_reality_gate(),indent=2))
