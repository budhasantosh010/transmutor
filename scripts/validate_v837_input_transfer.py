from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; HERE=ROOT/"experiments/v837_primitive_invention/v837ac"
EXPECTED=["AC0_y3_parent","AC1_controller_input_factorization","AC1F_folded_control"]
ALLOWED={"SHARED_INPUT_FACTORIZATION_SPECIFICALLY_SUFFICIENT","INPUT_EFFECTIVE_MAPPING_SUFFICIENT","INPUT_FACTORIZATION_SUFFICIENT_SHAREDNESS_NOT_ESTABLISHED","INPUT_ORGANIZATION_TRANSFER_INSUFFICIENT","INPUT_ORGANIZATION_TRANSFER_HARMFUL"}
def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def gsha(p): return hashlib.sha256(subprocess.check_output(["git","show",f"HEAD:{p}"],cwd=ROOT)).hexdigest()
def main():
    if not HERE.exists(): return 0
    c=load(HERE/"config.json"); gate=load(HERE/"frozen_input_transfer_gate.json"); ab=load(ROOT/"experiments/v837_primitive_invention/v837ab/diagnostics/decision_state.json")
    if c.get("experiment")!="V837ac" or c.get("authorized_mode")!="TRAINABLE_CONTROLLER_INPUT_FACTORIZATION" or ab.get("authorized_v837ac_mode")!=c["authorized_mode"] or ab.get("neutral_transfer_allowed") is not True: raise ValueError("V837ac authorization mismatch")
    if c.get("conditions")!=EXPECTED or c.get("deshared_control_applicable") is not False or gate.get("ac1d_required") is not False: raise ValueError("V837ac controller-only condition/control set changed")
    tr=c["training"]
    if (tr["steps"],tr["train_episodes"],tr["validation_episodes"],tr["replicates"])!=(192,512,128,5) or tr["development_seed_range"]!=[10000,10511] or tr["validation_seed_range"]!=[20000,20127]: raise ValueError("V837ac training/data changed")
    if c.get("unique_seed_defined_episodes")!=3200 or c.get("data_regime")!="4x_unique": raise ValueError("V837ac unique data changed")
    if any(c.get(k) is not False for k in ("fresh_audit_consumed","structural_search_allowed","primitive_mining_allowed","v838_started")) or c.get("primitives_promoted")!=0: raise ValueError("V837ac initial science locks changed")
    frozen={"experiments/v837_primitive_invention/v837ab/results.json":c["v837ab_results_sha256"],"experiments/v837_primitive_invention/v837ab/diagnostics/decision_state.json":c["v837ab_decision_sha256"],"experiments/v837_primitive_invention/v837y/config.json":c["v837y_config_sha256"],"experiments/v837_primitive_invention/v837y/candidate_interaction.py":c["v837y_model_sha256"],"experiments/v837_primitive_invention/v837y/results.json":c["v837y_results_sha256"],"experiments/v837_primitive_invention/v837y/raw/interaction_runs.json":c["v837y_raw_sha256"]}
    for p,h in frozen.items():
        if gsha(p)!=h: raise ValueError(f"V837ac frozen dependency changed: {p}")
    audit=load(ROOT/"experiments/v837_primitive_invention/audit/audit_results.json")
    if audit.get("episodes_consumed")!=0 or (ROOT/"experiments/v837_primitive_invention/v838").exists(): raise ValueError("V837ac audit/V838 lock violated")
    step=HERE/"diagnostics/step0_equivalence.json"
    if step.exists():
        s=load(step)
        if s.get("step0_equivalence_proven") is not True or max(float(v) for v in s.get("max_errors",{}).values())>1e-6 or s.get("ac1d_applicable") is not False: raise ValueError("V837ac step0 equivalence/control semantics failed")
    ac0=HERE/"raw/ac0_runs.json"
    if ac0.exists():
        rows=load(ac0)["rows"]
        if len(rows)!=25 or any(r["condition"]!=EXPECTED[0] or r["processed_examples"]!=98304 or r["resources"]["optimizer_steps"]!=192 for r in rows): raise ValueError("V837ac AC0 run set invalid")
        guard=load(HERE/"diagnostics/parent_compatibility.json")
        if guard.get("parent_reproduced") is not True or guard.get("observed",{}).get("families_passing")!=3: raise ValueError("V837ac Y3 parent reproduction invalid")
    transfer=HERE/"raw/transfer_runs.json"
    if transfer.exists():
        rows=load(transfer)["rows"]
        if len(rows)!=50 or {r["condition"] for r in rows}!=set(EXPECTED[1:]) or any(r["processed_examples"]!=98304 or r["resources"]["optimizer_steps"]!=192 for r in rows): raise ValueError("V837ac transfer run set invalid")
    result=HERE/"results.json"
    if result.exists():
        r=load(result); d=load(HERE/"diagnostics/decision_state.json")
        if set(r.get("conditions",{}))!=set(EXPECTED) or r.get("diagnosis") not in ALLOWED or d.get("diagnosis")!=r.get("diagnosis"): raise ValueError("V837ac result/diagnosis invalid")
        if r.get("authorized_mode")!=c["authorized_mode"] or d.get("authorized_mode")!=c["authorized_mode"]: raise ValueError("V837ac result authorization changed")
        passed=bool(r.get("representation_adequacy_pass"));
        if r.get("sample_efficiency_retest_allowed") is not passed or r.get("structural_search_allowed") is not passed or r.get("primitive_mining_allowed") is not False or r.get("fresh_audit_consumed") is not False or r.get("v838_started") is not False: raise ValueError("V837ac downstream locks inconsistent")
        res=r.get("resource_accounting",{})
        if int(res.get("model_fits",-1))!=75 or int(res.get("optimizer_steps",-1))!=14400 or int(res.get("processed_training_examples",-1))!=7372800 or int(res.get("unique_seed_defined_episodes",-1))!=3200: raise ValueError("V837ac resource accounting mismatch")
    print("V837ac input-transfer validation: PASS"); return 0
if __name__=="__main__": raise SystemExit(main())
