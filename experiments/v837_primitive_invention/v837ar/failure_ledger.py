from __future__ import annotations

from typing import Any
from .utils import HERE, ROOT, START_SHA, canonical_json, git_blob_sha256, read_json, sha256_json, write_json

RAW = HERE / "raw/failure_ledger.json"
DIAG = HERE / "diagnostics/failure_ledger.json"
CENTRAL = ROOT / "docs/V837_FAILURE_LEDGER.md"
REQUIRED = (
    "failure_id","version","stage","family","organism_fold","candidate_ir","candidate_granularity","parameter_count","serialized_bytes",
    "operator_order","predictive_rank","fit_evidence","selection_evidence","oracle_relative_metrics","organism_relative_metrics","composition_metrics",
    "control_metrics","failed_gate","distance_from_gate","result_status","what_this_rules_out","what_remains_alive","do_not_retry_unchanged",
    "reproduction_command","artifact_path_hash","failure_type"
)


def initialize() -> dict:
    p = read_json(RAW) if RAW.is_file() else {"version":"V837ar","append_only":True,"entries":[]}
    write_json(RAW,p); write_json(DIAG,p); return p


def make_entry(*, failure_id:str, stage:str, family:str|None=None, organism_fold:str|None=None, candidate_ir:str|None=None,
               candidate_granularity:str|None=None, parameter_count:int|None=None, serialized_bytes:int|None=None,
               operator_order:int|None=None, predictive_rank:int|None=None, fit_evidence:Any=None, selection_evidence:Any=None,
               oracle_relative_metrics:Any=None, organism_relative_metrics:Any=None, composition_metrics:Any=None, control_metrics:Any=None,
               failed_gate:str="", distance_from_gate:Any=None, result_status:str="DEFINITIVE_WITHIN_FROZEN_SCOPE",
               what_this_rules_out:str="", what_remains_alive:str="", reproduction_command:str="python scripts/reproduce_v837ar.py --execute",
               artifact_path_hash:Any=None, failure_type:str="SCIENTIFIC_FAILURE") -> dict:
    return {
        "failure_id":failure_id,"version":"V837ar","stage":stage,"family":family,"organism_fold":organism_fold,
        "candidate_ir":candidate_ir,"candidate_granularity":candidate_granularity,"parameter_count":parameter_count,"serialized_bytes":serialized_bytes,
        "operator_order":operator_order,"predictive_rank":predictive_rank,"fit_evidence":fit_evidence or {},"selection_evidence":selection_evidence or {},
        "oracle_relative_metrics":oracle_relative_metrics or {},"organism_relative_metrics":organism_relative_metrics or {},"composition_metrics":composition_metrics or {},
        "control_metrics":control_metrics or {},"failed_gate":failed_gate,"distance_from_gate":distance_from_gate or {},"result_status":result_status,
        "what_this_rules_out":what_this_rules_out,"what_remains_alive":what_remains_alive,"do_not_retry_unchanged":True,
        "reproduction_command":reproduction_command,"artifact_path_hash":artifact_path_hash or {},"failure_type":failure_type,"source_sha":START_SHA,
    }


def add(entry:dict) -> dict:
    missing=[k for k in REQUIRED if k not in entry]
    if missing: raise RuntimeError(f"V837AR_FAILURE_ENTRY_INCOMPLETE:{missing}")
    if entry["failure_type"] not in {"SCIENTIFIC_FAILURE","ENGINEERING_FAILURE","INVALID_EVIDENCE"}: raise ValueError("V837AR_BAD_FAILURE_TYPE")
    p=initialize()
    if any(e["failure_id"]==entry["failure_id"] for e in p["entries"]): return p
    p["entries"].append(entry); write_json(RAW,p); write_json(DIAG,p); return p


def sync_central_ledger() -> dict:
    import subprocess
    # Preserve the historical ledger as exact Git blob bytes. This avoids any
    # locale/newline recoding of old entries on Windows; V837ar may append but
    # must never rewrite prior failure memory.
    base = subprocess.check_output(["git", "show", f"{START_SHA}:docs/V837_FAILURE_LEDGER.md"], cwd=ROOT).rstrip(b"\r\n") + b"\n"
    entries = initialize()["entries"]
    blocks=[]
    for e in entries:
        blocks.append(f"\n## {e['failure_id']} - V837ar / {e['stage']}\n\n- **Type:** {e['failure_type']} / {e['result_status']}\n- **Family / fold:** {e['family']} / {e['organism_fold']}\n- **Candidate / granularity:** {e['candidate_ir']} / {e['candidate_granularity']}\n- **Failed gate:** {e['failed_gate']}\n- **Rules out:** {e['what_this_rules_out']}\n- **Remains alive:** {e['what_remains_alive']}\n- **Do not retry unchanged:** {e['do_not_retry_unchanged']}\n- **Reproduce:** `{e['reproduction_command']}`\n")
    CENTRAL.write_bytes(base + "".join(blocks).encode("utf-8"))
    return {"version":"V837ar","base_start_sha":START_SHA,"entries_appended":len(entries),"earlier_content_exact":True}
