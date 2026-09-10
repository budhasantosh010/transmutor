from __future__ import annotations

from .utils import HERE, read_json, sha256_json, write_json

OUT=HERE/"raw/frozen_program_operator_contracts.json"


def freeze_operator_contracts()->dict:
    if OUT.is_file():return read_json(OUT)
    discovery=read_json(HERE/"raw/operator_discovery.json");meta=read_json(HERE/"raw/meta_confirmation.json");loc=read_json(HERE/"raw/localization_results.json");comp=read_json(HERE/"raw/composition_results.json");interaction=read_json(HERE/"raw/interaction_results.json")
    specs={}
    for family in ("conditional_routing","delayed_recall","iterative_state","variable_composition"):
        if family not in meta.get("confirmed_families",[]):specs[family]=None;continue
        order=int(discovery["families"][family]["operator_order"]);specs[family]={
            "family":family,"operator_order":order,"semantic_contract":{
                "conditional_routing":"SELECT(control,payload_A,payload_B)","delayed_recall":"READ(HOLD^n(WRITE(value)))","iterative_state":"z_next=0.65*z+0.35*x",
            }[family],
            "response_object":"finite-horizon semantic intervention-response tensor",
            "coordinate_free":True,"oracle_relative":True,"hidden_state_alignment":False,"q_alignment":False,
            "second_order_required":bool(order==2),"interaction_gate":interaction["families"].get(family),
            "composition_gate":comp["families"].get(family),"compositionally_closed_in_discovery":bool(comp["families"].get(family,{}).get("pass")),"program_closed_in_meta":bool(meta["families"].get(family,{}).get("program_closed")),"compact_spatiotemporal_support":bool(loc["families"].get(family,{}).get("pass")),
            "localization_summary":{k:v for k,v in loc["families"].get(family,{}).items() if k!="organism_rows"},
            "meta_confirmed_operator":True,"heldout_results_read_before_freeze":False,
        }
    payload={"version":"V837aq","stage":"AQ_PRE_HELDOUT_FREEZE","family_contracts":specs,"confirmed_families":[f for f,v in specs.items() if v],"heldout_results_read":False,"fresh_audit_consumed":False,"primitives_promoted":0,"primitive_archive_population":False,"v838_started":False}
    payload["contract_sha256"]=sha256_json(payload);write_json(OUT,payload);write_json(HERE/"diagnostics/operator_contract_freeze.json",payload);return payload


if __name__=="__main__":
    import json;print(json.dumps(freeze_operator_contracts(),indent=2))
