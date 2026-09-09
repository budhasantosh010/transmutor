from __future__ import annotations
import json
from .candidate_runtime import evaluate_candidate
from .compute_accounting import primitive_accounting
from .expanded_primitive import expanded_pair,ranking_index
from .failure_ledger import add,make_entry
from .utils import HERE,read_json,seeds,sha256_json,write_json

META=seeds(10384,10447)
def _winner(branch):return read_json(HERE/f"raw/am_{ {'AM-A':'a','AM-B':'b','AM-C':'c'}[branch] }_selection.json").get("winner")
def _economics(branch,w):
    p=read_json(HERE/"raw/frozen_pairs.json")["pairs"][0];idx=ranking_index()
    if branch=="AM-B":occ=expanded_pair(p,w["boundary"],idx)["same_class_donor"]
    elif branch=="AM-C" and w["boundary"]!="ORIGINAL":occ=expanded_pair(p,w["boundary"],idx)["same_class_donor"]
    else:occ=p["same_class_donor"]
    prim=primitive_accounting(occ);return {**prim,"adapter_parameters":w.get("adapter_parameters",0),"adapter_macs":w.get("adapter_macs",0),"added_cells":w.get("added_cells",0),"history":w.get("history",1),"total_macs":prim["primitive_macs_per_timestep"]+w.get("adapter_macs",0)}
def run_meta():
    results=[]
    for b in ("AM-A","AM-B","AM-C"):
        w=_winner(b)
        if w is None:results.append({"branch":b,"winner":None,"run":False,"pass":False,"reason":"NO_SELECT_PASS"});continue
        ev=evaluate_candidate(b,w,META,.05);econ=_economics(b,w);row={"branch":b,"winner":w,"run":True,"pass":bool(ev["aggregate"]["pass"]),"metrics":ev["aggregate"],"economics":econ,"refit":False};results.append(row)
        if not row["pass"]:
            add(make_entry(failure_id=f"V837am-META-{b}",version="V837am",stage="META_CONFIRM",hypothesis=f"{b} development winner replicates without refit on an independent development block.",why="Prevent branch-select overfit before choosing the final explanation.",implementation=w,data="META_CONFIRM 10384-10447",fit_seeds=[],selection_seeds=[10384,10447],controls=["SAME_CLASS","DIFFERENT_CLASS_REFIT","RANDOMIZED_REFIT","RANDOMIZED_FIXED_ADAPTER"],params=econ['adapter_parameters'],macs=econ['adapter_macs'],metrics=row['metrics'],gate={"branch_gate":True,"p_max":.05},failed=["META_CONFIRM_GATE_FAIL"],distance={"p_excess":max(0,(row['metrics'].get('p') or 1)-.05)},status="definitive within tested scope",failure_type="SCIENTIFIC_FAILURE",ruled=["branch-select-only fit"],remaining=["other branch winner if any"],meaning="The branch winner did not independently replicate on META_CONFIRM.",uncertainty="Other branch winners may survive.",next_action="Select only surviving meta-confirmed candidates",reproduce="python scripts/reproduce_v837_recovery.py --variant v837am --stage meta",artifacts=["experiments/v837_primitive_invention/v837am/raw/meta_confirmation.json"]))
    passed=[r for r in results if r.get("pass")];prio={"AM-B":0,"AM-A":1,"AM-C":2}
    def cost(r):e=r["economics"];return (e["total_macs"],e["adapter_parameters"],e["added_cells"],e["history"],prio[r["branch"]])
    selected=min(passed,key=cost) if passed else None;payload={"version":"V837am","stage":"META_CONFIRM","seeds":[10384,10447],"no_refit":True,"pass_count":len(passed),"results":results,"selected":selected};write_json(HERE/"raw/meta_confirmation.json",payload);write_json(HERE/"diagnostics/meta_confirmation.json",payload)
    freeze={"version":"V837am","frozen_before_final_dev_confirm":True,"frozen_before_final_validation":True,"selected":None if selected is None else {"branch":selected['branch'],"config":selected['winner'],"meta_metrics":selected['metrics'],"economics":selected['economics']}};freeze["selected_final_hypothesis_sha256"]=sha256_json(freeze);write_json(HERE/"raw/selected_final_hypothesis.json",freeze);write_json(HERE/"diagnostics/final_hypothesis_freeze.json",freeze);return payload
if __name__=="__main__":print(json.dumps(run_meta(),indent=2)[:5000])
