from __future__ import annotations

import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; HERE=ROOT/"experiments/v837_primitive_invention/v837u"
EXPECTED=["U0_historical_direct","U1_v837p_scalar_candidate","U2_dynamic_scalar_carry","U2C_scalar_scale_candidate_control"]

def blob(path): return hashlib.sha256(subprocess.check_output(["git","show",f"HEAD:{path}"],cwd=ROOT)).hexdigest()

def validate():
    c=json.loads((HERE/"config.json").read_text(encoding="utf-8")); assert c["experiment"]=="V837u" and c["parent"]=="V837t" and c["authorized_mode"]=="DYNAMIC_SCALAR_CARRY" and c["conditions"]==EXPECTED
    d=json.loads((ROOT/"experiments/v837_primitive_invention/v837t/diagnostics/decision_state.json").read_text(encoding="utf-8")); assert d["neutral_followup_allowed"] is True and d["authorized_v837u_mode"]==c["authorized_mode"]
    assert blob("experiments/v837_primitive_invention/v837t/results.json")==c["v837t_result_sha256"]; assert blob("experiments/v837_primitive_invention/v837t/diagnostics/decision_state.json")==c["v837t_decision_sha256"]
    assert c["state_layout"]=="local_10x4" and c["total_state_dim"]==40 and c["graph_cells"]==10 and c["graph_edges"]==55 and c["global_coupling_allowed"] is False and c["shared_state_allowed"] is False
    tr=c["training"]; assert tr["steps"]==192 and tr["train_episodes"]==512 and tr["validation_episodes"]==128 and tr["replicates"]==5 and c["unique_seed_defined_episodes"]==3200
    for k in ("fresh_audit_consumed","structural_search_allowed","primitive_mining_allowed","v838_started"): assert c[k] is False
    raw=HERE/"raw/runs.json"
    if raw.exists():
        p=json.loads(raw.read_text(encoding="utf-8")); rows=p["rows"]; assert len(rows)==100 and p["unique_seed_defined_episodes"]==3200
        for r in rows:
            assert r["authorized_mode"]==c["authorized_mode"] and r["development_seed_range"]==[10000,10511] and r["validation_seed_range"]==[20000,20127] and r["resources"]["optimizer_steps"]==192 and r["resources"]["examples_processed"]==98304 and r["fresh_audit_consumed"] is False and r["task_family_label_in_model_input"] is False
        counts={r["condition"]:r["parameter_count"] for r in rows}; assert counts["U0_historical_direct"]==856 and counts["U2_dynamic_scalar_carry"]==1006 and counts["U2C_scalar_scale_candidate_control"]==1006
    rp=HERE/"results.json"
    if rp.exists():
        r=json.loads(rp.read_text(encoding="utf-8")); assert r["version"]=="V837u" and r["authorized_mode"]==c["authorized_mode"] and set(r["conditions"])==set(EXPECTED); assert r["resource_accounting"]["model_fits"]==100 and r["resource_accounting"]["optimizer_steps"]==19200 and r["resource_accounting"]["processed_examples"]==9830400 and r["resource_accounting"]["unique_seed_defined_episodes"]==3200
        dd=json.loads((HERE/"diagnostics/decision_state.json").read_text(encoding="utf-8")); assert dd["v837u_complete"] is True and dd["authorized_mode"]==c["authorized_mode"] and dd["fresh_audit_consumed"] is False and dd["v838_started"] is False
    print("V837 neutral dynamic followup validation: PASS")
if __name__=="__main__": validate()
