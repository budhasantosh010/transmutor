from __future__ import annotations

import json
import numpy as np

from .composition_eval import evaluate_composition_organism
from .controls import evaluate_frozen_controls
from .data_roles import seeds
from .freeze_operator_contracts import freeze_operator_contracts
from .interaction_eval import evaluate_interactions, interaction_bundles
from .natural_interventions import build_first_order_cases
from .operator_discovery import _cross_agreement
from .operator_eval import evaluate_cases, operator_pass
from .operator_metrics import family_gate
from .source_folds import holdout_ids
from .utils import HERE, write_json


def run_heldout()->dict:
    frozen=freeze_operator_contracts();payload={"version":"V837aq","stage":"AQ10_HELDOUT_OPERATOR_CONFIRMATION","frozen_contract_sha256":frozen["frozen_sha256"],"no_operator_refit":True,"no_control_refit":True,"no_hidden_state_or_q_alignment":True,"families":{},"rows":[],"fresh_audit_consumed":False}
    for family,spec in frozen["families"].items():
        cases=build_first_order_cases(family,seeds("AQ_HELDOUT_EVAL"));rows=[]
        for i,oid in enumerate(holdout_ids(family),1):
            res=evaluate_cases(oid,family,cases);op,failed=operator_pass(res["metrics"]);ctrl=evaluate_frozen_controls(family,res["rows"],spec["frozen_control_baseline"]);inter=evaluate_interactions(oid,family,interaction_bundles(family,seeds("AQ_HELDOUT_EVAL")));comp=evaluate_composition_organism(oid,family,seeds("AQ_HELDOUT_EVAL"),int(spec["operator_order"]));op_pass=bool(op and ctrl["pass"] and inter["pass"]);row={"organism_id":oid,"family":family,"engine":res["engine"],"metrics":res["metrics"],"controls":ctrl,"interaction":inter,"composition":comp,"operator_pass":op_pass,"pass":op_pass,"failed_conditions":failed+[f"control:{x}" for x in ctrl["failed_conditions"]],"select_rows":res["rows"]};rows.append(row);print(f"V837aq heldout {family} {i}/{len(holdout_ids(family))} operator={op_pass} composition={comp['pass']}",flush=True)
        gate=family_gate(rows);cross=_cross_agreement(rows);operator_pass=bool(gate["pass"] and cross["pass"]);comp_rows=[{**r,"pass":bool(r["operator_pass"] and r["composition"]["pass"])} for r in rows];comp_gate=family_gate(comp_rows);payload["families"][family]={"operator_gate":gate,"cross_organism":cross,"operator_pass":operator_pass,"composition_gate":comp_gate,"composition_pass":comp_gate["pass"],"discovery_composition_pass":spec["composition_pass"]};payload["rows"].extend([{k:v for k,v in r.items() if k!="select_rows"} for r in rows])
        write_json(HERE/f"raw/heldout_response_tensors/{family}.json",{"version":"V837aq","family":family,"contract_sha256":frozen["frozen_sha256"],"organisms":[{"organism_id":r["organism_id"],"engine":r["engine"],"operator_pass":r["operator_pass"],"rows":r["select_rows"]} for r in rows]})
    payload["operator_validated_families"]=[f for f,v in payload["families"].items() if v["operator_pass"]];payload["composition_validated_families"]=[f for f,v in payload["families"].items() if v["operator_pass"] and v["composition_pass"]];write_json(HERE/"raw/heldout_operator_results.json",payload);write_json(HERE/"diagnostics/heldout_operator_confirmation.json",payload);return payload


if __name__=="__main__":print(json.dumps(run_heldout(),indent=2))
