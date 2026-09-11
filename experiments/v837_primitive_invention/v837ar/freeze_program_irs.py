from __future__ import annotations
from .utils import HERE,read_json,sha256_json,write_json

def freeze_program_irs():
    meta=read_json(HERE/"raw/meta_confirmation.json");families={f:(d["candidate"]["ir"] if d.get("pass") and d.get("candidate") else None) for f,d in meta["families"].items()}
    payload={"version":"V837ar","stage":"AR9_PROGRAM_IR_FREEZE","families":families,"meta_no_refit":True,"final_unseen_words_opened":False,"reused_aq_heldout_opened":False}
    freeze_sha=sha256_json(payload);payload["freeze_sha256"]=freeze_sha;write_json(HERE/"raw/frozen_canonical_program_irs.json",payload);write_json(HERE/"diagnostics/program_ir_freeze.json",{"version":"V837ar","freeze_sha256":freeze_sha,"family_nulls":{f:v is None for f,v in families.items()}});return payload
