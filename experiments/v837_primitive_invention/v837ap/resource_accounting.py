from __future__ import annotations

from pathlib import Path
from .utils import HERE, read_json, write_json


def _rows(path:str)->int:
    p=HERE/path
    if not p.is_file():return 0
    x=read_json(p)
    if isinstance(x,dict):
        if isinstance(x.get("rows"),list):return len(x["rows"])
        if isinstance(x.get("entries"),list):return len(x["entries"])
    return 0


def build_resource_accounting()->dict:
    held=read_json(HERE/"raw/heldout_calibration_frontier.json") if (HERE/"raw/heldout_calibration_frontier.json").is_file() else {}
    payload={
        "version":"V837ap",
        "new_source_model_fits":0,"source_optimizer_steps":0,"source_training_examples":0,"source_architecture_changes":0,
        "backend_gradient_steps":0,"gpu_seconds":0,"fresh_audit_episodes":0,"v838_started":False,"primitives_promoted":0,
        "projected_subspace_rows":_rows("raw/projected_subspace_hashes.json"),
        "reader_candidate_rows":_rows("diagnostics/chart_conditioning.json"),
        "setpoint_evaluation_rows":_rows("raw/setpoint_results.json"),
        "quotient_evaluation_rows":_rows("raw/quotient_results.json"),
        "commutativity_evaluation_rows":_rows("raw/commutativity_results.json"),
        "heldout_evaluation_rows":len(held.get("rows",[])) if isinstance(held,dict) else 0,
        "failure_entries":_rows("raw/failure_ledger.json"),
        "heldout_wall_seconds":held.get("wall_seconds") if isinstance(held,dict) else None,
        "cpu_seconds":None,"cpu_seconds_exact":False,
        "wall_seconds_total":None,"wall_seconds_total_exact":False,
        "note":"Exact per-process CPU/wall accounting is unavailable for stages executed through resumable pytest drivers; measured stage wall time is retained where emitted. No GPU execution or source-model training occurred.",
    }
    write_json(HERE/"v837ap_resource_accounting.json",payload);write_json(HERE/"diagnostics/resource_accounting.json",payload);return payload

if __name__=="__main__":
    import json;print(json.dumps(build_resource_accounting(),indent=2))
