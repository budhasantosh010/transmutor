from __future__ import annotations

import json
from .authorization import assert_authorized, SOURCE_FILES
from .utils import HERE, ROOT, read_json, sha256_file, write_json


def run_source_integrity():
    auth=assert_authorized(); gate=read_json(HERE/"frozen_interface_alignment_gate.json")
    conf=read_json(SOURCE_FILES["confirmed"]); causal=read_json(SOURCE_FILES["causal"]); rec=read_json(SOURCE_FILES["reconstruction"])
    classes=[]
    for c in conf["classes"]:
        if not c.get("confirmed"): continue
        rows=[r for r in c.get("confirmation_rows",[]) if r.get("compatible") and r.get("competent")]
        classes.append({"class_id":c["class_id"],"stream":c["stream"],"size":c["size"],"support":len({r["organism_id"] for r in rows}),"compatible_occurrences":rows,"primary_causal":c["class_id"]==gate["primary_causal_class_id"]})
    source={"version":"V837al","confirmed_class_count":len(classes),"causal_class_count":1,"classes":classes,"primary_causal_class_id":gate["primary_causal_class_id"],"source_checkpoint_count":len(rec["rows"])}
    write_json(HERE/"raw/source_classes.json",source)
    diag={"version":"V837al","source_integrity":True,"source_hashes":gate["source_hashes"],"checkpoint_hashes":gate["checkpoint_hashes"],"checkpoint_count":len(rec["rows"]),"confirmed_classes":len(classes),"causal_classes":sum(c["primary_causal"] for c in classes),"new_model_fits":0,"optimizer_steps":0,"fresh_audit_consumed":False,"v838_started":False}
    write_json(HERE/"diagnostics/source_integrity.json",diag); return source


def main()->int:
    s=run_source_integrity(); print(json.dumps({"source_integrity":True,"confirmed_classes":s["confirmed_class_count"],"causal_class":s["primary_causal_class_id"]},indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
