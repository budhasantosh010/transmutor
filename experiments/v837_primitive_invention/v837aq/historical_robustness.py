from __future__ import annotations

from .controls import evaluate_controls
from .data_roles import seeds
from .freeze_operator_contracts import OUT
from .natural_interventions import build_first_order_cases
from .operator_eval import evaluate_cases, operator_pass
from .operator_metrics import family_gate
from .source_folds import discovery_ids
from .utils import HERE, read_json, write_json


def run_historical_robustness()->dict:
    if not OUT.is_file():raise RuntimeError("V837AQ_HISTORICAL_ROBUSTNESS_REQUIRES_FROZEN_OPERATOR_CONTRACTS")
    frozen=read_json(OUT);payload={"version":"V837aq","stage":"REUSED_HISTORICAL_VALIDATION_ROBUSTNESS","descriptive_only":True,"can_rescue_failed_discovery_or_meta":False,"families":{},"rows":[]}
    fit_seeds=seeds("AQ_OPERATOR_FIT");hist=seeds("REUSED_HISTORICAL_VALIDATION")
    for family in frozen.get("confirmed_families",[]):
        fit_cases=build_first_order_cases(family,fit_seeds);hist_cases=build_first_order_cases(family,hist,split="validation");rows=[]
        for oid in discovery_ids(family):
            fit=evaluate_cases(oid,family,fit_cases);got=evaluate_cases(oid,family,hist_cases);op,failed=operator_pass(got["metrics"]);ctrl=evaluate_controls(family,fit["rows"],got["rows"],oid);passed=bool(op and ctrl["pass"]);row={"organism_id":oid,"family":family,"engine":got["engine"],"metrics":got["metrics"],"controls":ctrl,"failed_conditions":failed,"pass":passed};rows.append(row);payload["rows"].append(row)
        gate=family_gate(rows);payload["families"][family]={"gate":gate,"pass":gate["pass"],"label":"REUSED_HISTORICAL_EPISODE_VALIDATION"}
    write_json(HERE/"raw/historical_validation_robustness.json",payload);write_json(HERE/"diagnostics/historical_validation_robustness.json",payload);return payload


if __name__=="__main__":
    import json;print(json.dumps(run_historical_robustness(),indent=2))
