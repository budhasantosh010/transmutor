from __future__ import annotations

import json
from .authorization import POWERED_FAMILIES
from .data_roles import seeds
from .natural_interventions import build_first_order_cases
from .operator_eval import evaluate_cases
from .source_folds import discovery_ids
from .utils import HERE, write_json


def export_response_tensors()->dict:
    summary={"version":"V837aq","partition":"AQ_OPERATOR_SELECT","families":{}}
    for family in POWERED_FAMILIES:
        cases=build_first_order_cases(family,seeds("AQ_OPERATOR_SELECT"));orgs=[]
        for oid in discovery_ids(family):
            r=evaluate_cases(oid,family,cases);orgs.append({"organism_id":oid,"engine":r["engine"],"rows":r["rows"]})
        write_json(HERE/f"raw/operator_response_tensors/{family}.json",{"version":"V837aq","family":family,"partition":"AQ_OPERATOR_SELECT","organisms":orgs});summary["families"][family]={"organisms":len(orgs),"rows":sum(len(o["rows"]) for o in orgs)}
    write_json(HERE/"diagnostics/operator_response_tensor_export.json",summary);return summary

if __name__=="__main__":print(json.dumps(export_response_tensors(),indent=2))
