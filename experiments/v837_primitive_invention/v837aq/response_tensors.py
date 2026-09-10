from __future__ import annotations

from collections import Counter

from .authorization import POWERED_FAMILIES
from .data_roles import seeds
from .natural_interventions import build_first_order_cases
from .operator_eval import evaluate_cases
from .source_folds import discovery_ids
from .utils import HERE, write_json


def materialize_response_tensors()->dict:
    manifest={"version":"V837aq","stage":"AQ3_CAUSAL_RESPONSE_TENSORS","families":{},"coordinate_free":True,"axes":["intervention","phase","magnitude","context/history","horizon"]}
    for family in POWERED_FAMILIES:
        cases=build_first_order_cases(family,seeds("AQ_OPERATOR_SELECT"));rows=[]
        for oid in discovery_ids(family):
            got=evaluate_cases(oid,family,cases)
            for r in got["rows"]:rows.append({"organism_id":oid,"engine":got["engine"],**r})
        payload={"version":"V837aq","family":family,"axes":manifest["axes"],"organisms":len(discovery_ids(family)),"rows":rows,"intervention_counts":dict(Counter(r["intervention"] for r in rows)),"state_alignment_used":False,"q_alignment_used":False}
        name=f"response_tensor_{family}.json";write_json(HERE/"raw"/name,payload);manifest["families"][family]={"artifact":f"raw/{name}","rows":len(rows),"organisms":len(discovery_ids(family))}
    write_json(HERE/"raw/response_tensor_manifest.json",manifest);write_json(HERE/"diagnostics/response_tensor_manifest.json",manifest);return manifest


if __name__=="__main__":
    import json;print(json.dumps(materialize_response_tensors(),indent=2))
