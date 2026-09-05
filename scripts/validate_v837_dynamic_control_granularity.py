from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"experiments/v837_primitive_invention/v837t"
EXPECTED=["T0_full_vector_gru","T1_vector_update_no_reset","T2_scalarized_update_no_reset","T3_no_update_vector_reset","T4_no_update_scalarized_reset","T5_dual_scalarized"]
MODES={"DYNAMIC_SCALAR_CARRY","POST_TRANSFORM_SCALAR_MODULATION","DUAL_SCALAR_DYNAMIC_PATHWAYS","DYNAMIC_VECTOR_STATE_MODULATION"}


def blob_hash(path:str)->str:
    return hashlib.sha256(subprocess.check_output(["git","show",f"HEAD:{path}"],cwd=ROOT)).hexdigest()


def validate():
    c=json.loads((HERE/"config.json").read_text(encoding="utf-8")); g=json.loads((HERE/"frozen_dynamic_granularity_gate.json").read_text(encoding="utf-8"))
    assert c["experiment"]=="V837t" and c["parent"]=="V837s" and c["conditions"]==EXPECTED
    assert c["data_regime"]=="4x_unique" and c["unique_seed_defined_episodes"]==3200
    tr=c["training"]; assert tr["steps"]==192 and tr["train_episodes"]==512 and tr["validation_episodes"]==128 and tr["replicates"]==5
    assert tr["development_seed_range"]==[10000,10511] and tr["validation_seed_range"]==[20000,20127]
    assert tr["learning_rate"]==0.005 and tr["weight_decay"]==0.0001 and tr["gradient_clip"]==5.0
    assert c["historical_gate_hash"]=="a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867"
    assert c["capacity_criterion_hash"]=="7178eed701ad50a298f172e867c73db47c03ecb28767de2add61feb34a61a3aa"
    assert blob_hash("experiments/v837_primitive_invention/v837s/results.json")==c["v837s_result_sha256"]
    assert blob_hash("experiments/v837_primitive_invention/v837o/results.json")==c["v837o_result_sha256"]
    assert g["scalarization"]=="mean(sigmoid(logits), hidden_dimension) then broadcast" and g["same_vector_gate_network_retained"] is True
    for k in ("fresh_audit_consumed","structural_search_allowed","primitive_mining_allowed","v838_started"): assert c[k] is False
    for filename,conditions in (("anchor_runs.json",set(c["positive_control_conditions"])),("scalarized_runs.json",set(EXPECTED)-set(c["positive_control_conditions"]))):
        p=HERE/"raw"/filename
        if not p.exists(): continue
        payload=json.loads(p.read_text(encoding="utf-8")); rows=payload["rows"]; assert len(rows)==75
        assert set(r["condition"] for r in rows)==conditions
        assert payload["seed_policy"]["unique_seed_defined_episodes"]==3200
        seen=set()
        for r in rows:
            key=(r["condition"],r["family"],r["replicate_id"]); assert key not in seen; seen.add(key)
            assert r["development_seed_range"]==[10000,10511] and r["validation_seed_range"]==[20000,20127]
            assert r["nominal_parameter_count"]==875 and r["optimizer_steps"]==192 and r["processed_examples"]==98304
            assert r["fresh_audit_consumed"] is False and r["task_family_label_in_model_input"] is False
    rp=HERE/"results.json"
    if rp.exists():
        r=json.loads(rp.read_text(encoding="utf-8")); assert r["version"]=="V837t" and r["parent"]=="V837s" and r["unique_seed_defined_episodes"]==3200
        assert set(r["conditions"])==set(EXPECTED); assert r["resource_accounting"]["model_fits"]==150 and r["resource_accounting"]["optimizer_steps"]==28800 and r["resource_accounting"]["processed_examples"]==14745600
        d=json.loads((HERE/"diagnostics/decision_state.json").read_text(encoding="utf-8")); assert d["v837t_complete"] is True
        if d["positive_controls_pass"]: assert d["authorized_v837u_mode"] in MODES and d["neutral_followup_allowed"] is True
        else: assert d["authorized_v837u_mode"] is None and d["neutral_followup_allowed"] is False
        assert d["fresh_audit_consumed"] is False and d["v838_started"] is False
        u=ROOT/"experiments/v837_primitive_invention/v837u"
        if u.exists():
            assert d["neutral_followup_allowed"] is True and d["authorized_v837u_mode"] in MODES
        assert r["diagnosis"] == "DYNAMIC_VECTOR_GRANULARITY_NOT_REQUIRED"
        assert r["authorized_v837u_mode"] == "DYNAMIC_SCALAR_CARRY"
        expected_counts={"T0_full_vector_gru":5,"T1_vector_update_no_reset":5,"T2_scalarized_update_no_reset":4,"T3_no_update_vector_reset":5,"T4_no_update_scalarized_reset":3,"T5_dual_scalarized":3}
        assert {name:int(row["families_passing"]) for name,row in r["conditions"].items()} == expected_counts
        assert d["diagnosis"] == r["diagnosis"] and d["authorized_v837u_mode"] == r["authorized_v837u_mode"]
    print("V837 dynamic control granularity validation: PASS")


if __name__=="__main__": validate()
