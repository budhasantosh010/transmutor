from __future__ import annotations
import json
from .candidate_runtime import evaluate_candidate
from .failure_ledger import add,make_entry
from .utils import HERE,read_json,seeds,sha256_file,write_json

DEV=seeds(10448,10511);VAL=seeds(20000,20127)
def run_dev_confirm():
    frozen=read_json(HERE/"raw/selected_final_hypothesis.json");selected=frozen.get("selected")
    if selected is None:
        p={"version":"V837am","stage":"FINAL_DEV_CONFIRM","run":False,"reason":"NO_META_CONFIRMED_CANDIDATE","pass":False};write_json(HERE/"raw/final_dev_confirmation.json",p);return p
    before=frozen["selected_final_hypothesis_sha256"];ev=evaluate_candidate(selected["branch"],selected["config"],DEV,.01);after=read_json(HERE/"raw/selected_final_hypothesis.json")["selected_final_hypothesis_sha256"];
    if before!=after:raise RuntimeError("V837AM_FINAL_HYPOTHESIS_MUTATED")
    p={"version":"V837am","stage":"FINAL_DEV_CONFIRM","run":True,"seeds":[10448,10511],"no_refit":True,"selected_hash":before,"metrics":ev["aggregate"],"pass":bool(ev["aggregate"]["pass"])};write_json(HERE/"raw/final_dev_confirmation.json",p)
    if not p["pass"]:add(make_entry(failure_id="V837am-FINAL-DEV-CONFIRMATION",version="V837am",stage="FINAL_DEV_CONFIRM",hypothesis="Frozen meta-confirmed explanation replicates on FINAL_DEV_CONFIRM.",why="Final development firewall before validation.",implementation=selected,data="10448-10511",fit_seeds=[],selection_seeds=[10448,10511],controls=["SAME_CLASS","DIFFERENT_CLASS_REFIT","RANDOMIZED_REFIT","RANDOMIZED_FIXED_ADAPTER"],params=selected['economics']['adapter_parameters'],macs=selected['economics']['adapter_macs'],metrics=p['metrics'],gate={"p_max":.01,"output_ratio":.75,"state_ratio":.85,"beat_both":.60,"random_refit_ratio_min":1.20},failed=["V837AM_DEV_CONFIRMATION_FAILURE"],distance={},status="definitive within tested scope",failure_type="SCIENTIFIC_FAILURE",ruled=["meta-only replication"],remaining=["final validation remains untouched"],meaning="Frozen candidate failed the final development confirmation and cannot access validation.",uncertainty="Whether a newly designed future hypothesis could work.",next_action="Close V837am without validation; no second-best retry",reproduce="python scripts/reproduce_v837_recovery.py --variant v837am --stage dev-confirm",artifacts=["experiments/v837_primitive_invention/v837am/raw/final_dev_confirmation.json"]));return p
    return p
def run_final_validation():
    dev=read_json(HERE/"raw/final_dev_confirmation.json");frozen=read_json(HERE/"raw/selected_final_hypothesis.json");selected=frozen.get("selected")
    if not dev.get("run") or not dev.get("pass"):
        p={"version":"V837am","stage":"FINAL_VALIDATION","run":False,"reason":"FINAL_DEV_CONFIRM_REQUIRED","pass":False};write_json(HERE/"raw/final_validation.json",p);write_json(HERE/"diagnostics/final_validation.json",p);return p
    before=frozen["selected_final_hypothesis_sha256"];ev=evaluate_candidate(selected["branch"],selected["config"],VAL,.01);after=read_json(HERE/"raw/selected_final_hypothesis.json")["selected_final_hypothesis_sha256"];
    if before!=after:raise RuntimeError("V837AM_VALIDATION_SELECTION_LEAKAGE")
    p={"version":"V837am","stage":"FINAL_VALIDATION","run":True,"seeds":[20000,20127],"no_refit":True,"selected_hash":before,"metrics":ev["aggregate"],"pass":bool(ev["aggregate"]["pass"]),"generalization_diagnostic":"deferred until primary result frozen"};write_json(HERE/"raw/final_validation.json",p);write_json(HERE/"diagnostics/final_validation.json",p)
    if not p["pass"]:add(make_entry(failure_id="V837am-FINAL-VALIDATION",version="V837am",stage="FINAL_VALIDATION",hypothesis="Frozen V837am explanation generalizes to historical held-out validation without refit.",why="Final causal-class held-out test.",implementation=selected,data="20000-20127",fit_seeds=[],selection_seeds=[20000,20127],controls=["SAME_CLASS","DIFFERENT_CLASS_REFIT","RANDOMIZED_REFIT","RANDOMIZED_FIXED_ADAPTER"],params=selected['economics']['adapter_parameters'],macs=selected['economics']['adapter_macs'],metrics=p['metrics'],gate={"p_max":.01,"output_ratio":.75,"state_ratio":.85,"beat_both":.60,"random_refit_ratio_min":1.20},failed=["FINAL_VALIDATION_GATE_FAIL"],distance={},status="definitive within tested scope",failure_type="SCIENTIFIC_FAILURE",ruled=["development selection and meta-confirmation"],remaining=["future untested hypothesis families"],meaning="The frozen candidate did not generalize to the held-out validation set; no second-best retry is permitted.",uncertainty="A different future phase may define a new hypothesis before validation.",next_action="Close V837am with final-validation failure",reproduce="python scripts/reproduce_v837_recovery.py --variant v837am --stage final",artifacts=["experiments/v837_primitive_invention/v837am/raw/final_validation.json"]))
    return p
if __name__=="__main__":run_dev_confirm();print(json.dumps(run_final_validation(),indent=2))
