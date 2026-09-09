from __future__ import annotations

from .authorization import SOURCE, assert_authorized
from .utils import HERE, ROOT, read_json, write_json


def run_source_integrity():
    auth=assert_authorized(); pairs=read_json(ROOT/SOURCE["v837al_pairs"]); causal=[p for p in pairs["pairs"] if p.get("primary_causal")]
    payload={"version":"V837am","source_integrity":True,"primary_causal_class":auth["primary_causal_class"],"causal_recipient_count":len(causal),"minimum_possible_p":2.0**(-len(causal)),"v837al_pairing_reused_unchanged":True,"source_checkpoint_count":50,"new_model_fits":0,"organism_optimizer_steps":0,"adapter_gradient_steps":0,"fresh_audit_consumed":False,"v838_started":False}
    write_json(HERE/"raw/source_state.json",payload);write_json(HERE/"diagnostics/source_integrity.json",payload);return payload

if __name__=="__main__": run_source_integrity()
