from __future__ import annotations

from .utils import HERE, ROOT, START_SHA, git_blob_sha256, write_json
from .source_contracts import AQ_RESULTS_REL,AQ_GATE_REL,AQ_OPERATOR_CONTRACT_REL,AQ_PROGRAM_CONTRACT_REL,AQ_FOLDS_REL,aq_import_snapshot,load_aq_contracts

PROTECTED=(AQ_RESULTS_REL,AQ_GATE_REL,AQ_OPERATOR_CONTRACT_REL,AQ_PROGRAM_CONTRACT_REL,AQ_FOLDS_REL)

def verify_source_integrity() -> dict:
    c=load_aq_contracts(); snap=aq_import_snapshot()
    rows={p:{"start_blob_sha256":git_blob_sha256(p,START_SHA),"head_blob_sha256":git_blob_sha256(p,"HEAD")} for p in PROTECTED}
    changed=[p for p,d in rows.items() if d["start_blob_sha256"]!=d["head_blob_sha256"]]
    if changed: raise RuntimeError(f"V837AR_PROTECTED_AQ_DRIFT:{changed}")
    payload={"version":"V837ar","pass":True,"start_sha":START_SHA,"protected":rows,"aq_import":snap,"fold_sha256":c["folds"]["fold_sha256"],
             "source_architecture_frozen":True,"source_organisms_frozen":True,"new_source_model_fits":0,"source_optimizer_steps":0,"fresh_audit_consumed":False,"v838_started":False}
    write_json(HERE/"raw/source_state.json",payload);write_json(HERE/"raw/v837aq_contract_import.json",snap);write_json(HERE/"raw/frozen_source_folds.json",c["folds"]);write_json(HERE/"diagnostics/source_integrity.json",payload);write_json(HERE/"diagnostics/aq_contract_integrity.json",snap)
    return payload
