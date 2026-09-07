from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"experiments/v837_primitive_invention/v837ab"
EXPECTED=["AB0_exact_factorized_t2","AB1_fully_folded_equivalent","AB2_candidate_factorized_update_folded","AB3_candidate_folded_update_factorized","AB4_frozen_shared_projection","AB5_naive_projection_free"]
ALLOWED={"INPUT_PROJECTION_AXIS_CLOSED","COMPOSED_EFFECTIVE_INPUT_INITIALIZATION_SUFFICIENT","FIXED_INPUT_PRECONDITIONING_SUFFICIENT","CANDIDATE_INPUT_FACTORIZATION_SUFFICIENT","UPDATE_CONTROLLER_INPUT_FACTORIZATION_SUFFICIENT","SINGLE_PATH_INPUT_FACTORIZATION_SUFFICIENT","JOINT_CANDIDATE_CONTROLLER_INPUT_FACTORIZATION_REQUIRED","INPUT_FACTORIZATION_REFERENCE_BASELINE_DRIFT"}


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def git_sha(path): return hashlib.sha256(subprocess.check_output(["git","show",f"HEAD:{path}"],cwd=ROOT)).hexdigest()


def main()->int:
    if not HERE.exists(): return 0
    config=load(HERE/"config.json"); gate=load(HERE/"frozen_input_factorization_gate.json")
    if config.get("experiment")!="V837ab" or config.get("parent")!="V837aa" or config.get("reference_condition")!="T2_scalarized_update_no_reset": raise ValueError("V837ab identity/reference mismatch")
    if config.get("conditions")!=EXPECTED: raise ValueError("V837ab condition set/order changed")
    tr=config["training"]
    if (tr["steps"],tr["train_episodes"],tr["validation_episodes"],tr["replicates"])!=(192,512,128,5): raise ValueError("V837ab training regime changed")
    if tr["development_seed_range"]!=[10000,10511] or tr["validation_seed_range"]!=[20000,20127] or tr["initialization_namespace"]!="v837j-primary-init": raise ValueError("V837ab seeds/init namespace changed")
    if config.get("unique_seed_defined_episodes")!=3200 or config.get("data_regime")!="4x_unique": raise ValueError("V837ab data regime changed")
    if any(config.get(k) is not False for k in ("fresh_audit_consumed","structural_search_allowed","primitive_mining_allowed","v838_started")): raise ValueError("V837ab science lock changed")
    if gate.get("frozen_before_results") is not True or gate.get("primary_comparison")!="AB0_exact_factorized_t2 vs AB1_fully_folded_equivalent": raise ValueError("V837ab frozen gate changed")
    frozen={"experiments/v837_primitive_invention/v837t/config.json":config["v837t_config_sha256"],"experiments/v837_primitive_invention/v837t/gru_dynamic_granularity.py":config["v837t_model_sha256"],"experiments/v837_primitive_invention/v837t/results.json":config["v837t_results_sha256"],"experiments/v837_primitive_invention/v837aa/results.json":config["v837aa_results_sha256"],"experiments/v837_primitive_invention/v837aa/diagnostics/decision_state.json":config["v837aa_decision_sha256"]}
    for p,h in frozen.items():
        if git_sha(p)!=h: raise ValueError(f"frozen parent hash changed: {p}")
    audit=load(ROOT/"experiments/v837_primitive_invention/audit/audit_results.json")
    if audit.get("episodes_consumed")!=0: raise ValueError("fresh audit consumed")
    if (ROOT/"experiments/v837_primitive_invention/v838").exists(): raise ValueError("V838 exists")
    ref=HERE/"diagnostics/reference_equivalence.json"; step=HERE/"diagnostics/step0_equivalence.json"
    if ref.exists():
        r=load(ref)
        if r.get("function_class_equivalence_proven") is not True or float(r.get("maximum_error",1))>1e-6: raise ValueError("folding equivalence failed")
    if step.exists():
        s=load(step)
        if s.get("step0_equivalence_proven") is not True: raise ValueError("step0 equivalence failed")
        for condition in EXPECTED[1:5]:
            if condition not in s.get("max_errors_by_condition",{}) or max(float(v) for v in s["max_errors_by_condition"][condition].values())>1e-6: raise ValueError(f"step0 drift: {condition}")
    raw0=HERE/"raw/ab0_runs.json"
    if raw0.exists():
        rows=load(raw0)["rows"]
        if len(rows)!=25: raise ValueError("AB0 must contain 25 fits")
        for row in rows:
            if row["condition"]!=EXPECTED[0] or row["processed_examples"]!=98304 or row["resources"]["optimizer_steps"]!=192 or row.get("fresh_audit_consumed") is not False: raise ValueError("AB0 row budget/lock mismatch")
    rawrest=HERE/"raw/factorization_runs.json"
    if rawrest.exists():
        rows=load(rawrest)["rows"]
        if len(rows)!=125 or {r["condition"] for r in rows}!=set(EXPECTED[1:]): raise ValueError("AB1-AB5 run set incomplete")
        if any(r["processed_examples"]!=98304 or r["resources"]["optimizer_steps"]!=192 or r.get("fresh_audit_consumed") is not False for r in rows): raise ValueError("AB1-AB5 row budget/lock mismatch")
    results_path=HERE/"results.json"
    if results_path.exists():
        result=load(results_path); decision=load(HERE/"diagnostics/decision_state.json")
        if result.get("function_class_equivalence_proven") is not True or result.get("step0_equivalence_proven") is not True: raise ValueError("V837ab results lost equivalence proof")
        if set(result.get("conditions",{}))!=set(EXPECTED): raise ValueError("V837ab results missing conditions")
        diagnosis=result.get("diagnosis")
        if diagnosis not in ALLOWED or decision.get("diagnosis")!=diagnosis: raise ValueError("V837ab diagnosis invalid")
        expected_mode=gate["authorization_map"].get(diagnosis)
        if result.get("authorized_v837ac_mode")!=expected_mode or decision.get("authorized_v837ac_mode")!=expected_mode: raise ValueError("V837ab authorization mismatch")
        if decision.get("reference_baseline_valid") is not True and diagnosis!="INPUT_FACTORIZATION_REFERENCE_BASELINE_DRIFT": raise ValueError("invalid baseline interpretation")
        res=result.get("resource_accounting",{})
        if int(res.get("model_fits",-1))!=150 or int(res.get("optimizer_steps",-1))!=28800 or int(res.get("processed_training_examples",-1))!=14745600 or int(res.get("unique_seed_defined_episodes",-1))!=3200: raise ValueError("V837ab resource accounting mismatch")
        if result.get("fresh_audit_consumed") is not False or result.get("structural_search_allowed") is not False or result.get("primitive_mining_allowed") is not False or result.get("v838_started") is not False: raise ValueError("V837ab result science lock changed")
    print("V837ab input-factorization validation: PASS"); return 0

if __name__=="__main__": raise SystemExit(main())
