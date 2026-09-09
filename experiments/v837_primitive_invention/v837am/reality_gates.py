from __future__ import annotations

import json

from .authorization import CONTEXT_SETS,PARTITIONS,assert_authorized
from .context_features import NAME_TO_PORT,context_tensor
from .interface_data import collect_trace,trace_integrity
from .utils import HERE,read_json,seeds,write_json


def run():
    assert_authorized();dev_names=["AM_A_FIT","AM_A_SELECT","AM_B_FIT","AM_B_SELECT","AM_C_FIT","AM_C_SELECT","META_CONFIRM","FINAL_DEV_CONFIRM"];blocks={n:set(seeds(*PARTITIONS[n])) for n in dev_names}
    for i,a in enumerate(dev_names):
        for b in dev_names[i+1:]:
            if blocks[a]&blocks[b]:raise RuntimeError("V837AM_DATA_PARTITION_OVERLAP")
    validation=set(seeds(*PARTITIONS["FINAL_VALIDATION"]));audit=set(seeds(*PARTITIONS["FRESH_AUDIT"]));all_dev=set().union(*blocks.values())
    if all_dev&validation or all_dev&audit or validation&audit:raise RuntimeError("V837AM_DATA_PARTITION_LEAKAGE")
    lock={"version":"V837am","development_blocks":PARTITIONS,"all_development_disjoint":True,"final_validation_not_used_before_freeze":True,"fresh_audit_consumed":False}
    write_json(HERE/"diagnostics/data_partition_lock.json",lock)
    pairs=read_json(HERE/"raw/frozen_pairs.json")["pairs"];p=pairs[0];rec=p["recipient"];don=p["same_class_donor"];r=collect_trace(rec["organism_id"],rec["nodes"],seeds(*PARTITIONS["AM_A_FIT"]));d=collect_trace(don["organism_id"],don["nodes"],seeds(*PARTITIONS["AM_A_FIT"]));trace_integrity(r,d)
    dims={}
    for c in CONTEXT_SETS:
        for port in ("state","external_messages","global_term","projected_input","gate","output"):
            x=context_tensor(r,rec["nodes"],0,c,port,1);dims[f"{c}:{port}"]=x.shape[-1]
            raw_name={"state":"PREV_STATE","external_messages":"EXTERNAL_MESSAGE","global_term":"GLOBAL_TERM","projected_input":"PROJECTED_INPUT","gate":"GATE"}.get(port)
            if raw_name and raw_name in CONTEXT_SETS[c]:
                baseline=context_tensor(r,rec["nodes"],0,c,"output",1).shape[-1];raw_dim={"PREV_STATE":4,"EXTERNAL_MESSAGE":4,"GLOBAL_TERM":4,"PROJECTED_INPUT":6,"GATE":1}[raw_name]
                if x.shape[-1]!=baseline-raw_dim:raise RuntimeError("V837AM_SELF_PORT_EXCLUSION_FAILURE")
    context={"version":"V837am","recipient_observable_only":True,"no_future_state":True,"no_target":True,"no_task_label":True,"no_success_indicator":True,"self_port_exclusion":True,"normalization_fit_only":True,"sample_dimensions":dims,"trace_masks_identical":True}
    write_json(HERE/"diagnostics/context_feature_integrity.json",context);return {"data":lock,"context":context}

if __name__=="__main__":print(json.dumps(run(),indent=2))
