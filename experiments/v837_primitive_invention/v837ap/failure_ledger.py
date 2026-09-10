from __future__ import annotations

from typing import Any
from .utils import HERE, ROOT, START_SHA, git_blob_bytes, now_stamps, read_json, sha256_json, write_json

RAW=HERE/"raw/failure_ledger.json"; DIAG=HERE/"diagnostics/failure_ledger.json"; CENTRAL=ROOT/"docs/V837_FAILURE_LEDGER.md"
REQUIRED=("failure_id","version","stage","branch","family","organism","phase","carrier_dimension","semantic_dimension","chart_family","chart_degree_rank","writer_family","fit_partition","selection_partition","parameter_count","stored_bytes","mac_estimate","metrics","matched_control_metrics","ood_metrics","acceptance_gate","failed_conditions","distance_from_threshold","scientific_interpretation","confounds_ruled_out","confounds_remaining","result_status","do_not_repeat_unchanged","next_justified_experiment","reproduction_command","artifact_paths","artifact_hashes","failure_type")

def initialize()->dict:
    payload=read_json(RAW) if RAW.is_file() else {"version":"V837ap","append_only":True,"entries":[]}
    write_json(RAW,payload); write_json(DIAG,payload)
    return payload

def add(entry:dict,append_central:bool=True)->dict:
    missing=[k for k in REQUIRED if k not in entry]
    if missing: raise RuntimeError(f"V837AP_FAILURE_ENTRY_INCOMPLETE:{missing}")
    if entry["failure_type"] not in {"SCIENTIFIC_FAILURE","ENGINEERING_FAILURE"}: raise ValueError("bad failure_type")
    payload=initialize()
    if any(e["failure_id"]==entry["failure_id"] for e in payload["entries"]): return payload
    payload["entries"].append(entry); write_json(RAW,payload); write_json(DIAG,payload)
    if append_central:
        with CENTRAL.open("a",encoding="utf-8",newline="\n") as h:
            h.write(f"## {entry['failure_id']} — V837ap / {entry['stage']}\n\n")
            h.write(f"- **Type:** {entry['failure_type']} / {entry['result_status']}\n")
            h.write(f"- **Family / organism:** {entry['family']} / {entry['organism']}\n")
            h.write(f"- **Geometry:** k={entry['carrier_dimension']} semantic_dim={entry['semantic_dimension']} {entry['chart_family']} {entry['writer_family']}\n")
            h.write(f"- **Failed conditions:** {entry['failed_conditions']}\n")
            h.write(f"- **Interpretation:** {entry['scientific_interpretation']}\n")
            h.write(f"- **Do not repeat unchanged:** {entry['do_not_repeat_unchanged']}\n")
            h.write(f"- **Artifacts:** {', '.join(entry['artifact_paths'])}\n\n")
    return payload

def make_entry(*,failure_id:str,stage:str,branch:str,family:str|None,organism:str|None,phase:str|None,carrier_dimension:int|None,chart_family:str|None,chart_degree_rank:Any,writer_family:str|None,fit_partition:str|None,selection_partition:str|None,parameter_count:int|None,stored_bytes:int|None,mac_estimate:Any,metrics:Any,matched_control_metrics:Any=None,ood_metrics:Any=None,acceptance_gate:Any=None,failed_conditions:Any=None,distance_from_threshold:Any=None,scientific_interpretation:str="",confounds_ruled_out:Any=None,confounds_remaining:Any=None,result_status:str="DEFINITIVE_WITHIN_FROZEN_SCOPE",next_justified_experiment:str="Follow the frozen V837ap complexity ladder.",reproduction_command:str="python scripts/reproduce_v837_recovery.py --variant v837ap --stage analyze --execute",artifact_paths:list[str]|None=None,artifact_hashes:dict|None=None,failure_type:str="SCIENTIFIC_FAILURE")->dict:
    utc,local=now_stamps()
    return {"failure_id":failure_id,"version":"V837ap","stage":stage,"branch":branch,"family":family,"organism":organism,"phase":phase,"carrier_dimension":carrier_dimension,"semantic_dimension":1,"chart_family":chart_family,"chart_degree_rank":chart_degree_rank,"writer_family":writer_family,"fit_partition":fit_partition,"selection_partition":selection_partition,"parameter_count":parameter_count,"stored_bytes":stored_bytes,"mac_estimate":mac_estimate,"metrics":metrics,"matched_control_metrics":matched_control_metrics or {},"ood_metrics":ood_metrics or {},"acceptance_gate":acceptance_gate or {},"failed_conditions":failed_conditions or [],"distance_from_threshold":distance_from_threshold or {},"scientific_interpretation":scientific_interpretation,"confounds_ruled_out":confounds_ruled_out or [],"confounds_remaining":confounds_remaining or [],"result_status":result_status,"do_not_repeat_unchanged":True,"next_justified_experiment":next_justified_experiment,"reproduction_command":reproduction_command,"artifact_paths":artifact_paths or [],"artifact_hashes":artifact_hashes or {"metrics_sha256":sha256_json(metrics)},"failure_type":failure_type,"timestamp_utc":utc,"timestamp_local":local,"source_sha":START_SHA}


def sync_central_ledger()->dict:
    """Rebuild the evolving central ledger as exact START content + V837ap appendices."""
    base=git_blob_bytes("docs/V837_FAILURE_LEDGER.md",START_SHA).decode("utf-8").rstrip("\n")+"\n"
    entries=initialize()["entries"]
    blocks=[]
    for entry in entries:
        blocks.append(
            f"\n## {entry['failure_id']} — V837ap / {entry['stage']}\n\n"
            f"- **Type:** {entry['failure_type']} / {entry['result_status']}\n"
            f"- **Family / organism:** {entry['family']} / {entry['organism']}\n"
            f"- **Geometry:** k={entry['carrier_dimension']} semantic_dim={entry['semantic_dimension']} {entry['chart_family']} {entry['writer_family']}\n"
            f"- **Failed conditions:** {entry['failed_conditions']}\n"
            f"- **Interpretation:** {entry['scientific_interpretation']}\n"
            f"- **Do not repeat unchanged:** {entry['do_not_repeat_unchanged']}\n"
            f"- **Artifacts:** {', '.join(entry['artifact_paths'])}\n"
        )
    CENTRAL.write_text(base+"".join(blocks),encoding="utf-8",newline="\n")
    return {"version":"V837ap","base_start_sha":START_SHA,"entries_appended":len(entries),"earlier_content_exact":True}

