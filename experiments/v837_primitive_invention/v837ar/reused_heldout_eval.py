from __future__ import annotations
import math
from .utils import HERE,read_json,write_json
from .semantic_execution import reused_heldout_ids
from .unseen_word_eval import evaluate_organism,family_gate

def run_reused_heldout():
    frozen=read_json(HERE/"raw/frozen_canonical_program_irs.json");out={"version":"V837ar","label":"REUSED_AQ_HELDOUT_ORGANISM_CONFIRMATION","freeze_sha256":frozen["freeze_sha256"],"no_refit":True,"no_gain_calibration":True,"no_bias_calibration":True,"families":{}}
    for fam,ir in frozen["families"].items():
        if ir is None:out["families"][fam]={"pass":False,"reason":"NULL_AT_FREEZE","rows":[],"gate":{"pass":False}};continue
        ids=reused_heldout_ids(fam);rows=[evaluate_organism(fam,ir,oid) for oid in ids];required=int(math.ceil(.75*len(rows)));passing=[r for r in rows if r["pass"]];engines=sorted({r["engine"] for r in passing});gate={"organisms":len(rows),"required":required,"passing":len(passing),"pass_fraction":len(passing)/len(rows) if rows else 0.0,"passing_engines":engines,"both_engines":len(engines)>=2,"pass":bool(rows and len(passing)>=required and len(engines)>=2)};out["families"][fam]={"pass":gate["pass"],"gate":gate,"rows":rows}
    write_json(HERE/"raw/reused_aq_heldout_results.json",out);write_json(HERE/"diagnostics/reused_heldout_confirmation.json",out);return out
