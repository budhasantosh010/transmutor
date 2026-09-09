from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837ak.boundary_substitution import run_boundary_substitution
from experiments.v837_primitive_invention.v837ak.boundary_traces import run_full_probe
from experiments.v837_primitive_invention.v837ak.intervention_runtime import randomized_primitive,run_intervened
from experiments.v837_primitive_invention.v837ak.ported_primitive import PortedPrimitive
from experiments.v837_primitive_invention.v837ak.reconstruct_organisms import load_reconstructed_model
from experiments.v837_primitive_invention.v837ak.utils import HERE,one_sided_paired_sign_permutation,read_json,write_json

CONFIRM=list(range(20064,20128))


def _success(task,pred,target)->float:return float(np.mean([task.success(float(p),float(t)) for p,t in zip(pred.tolist(),target.tolist())]))


def run_closed_loop()->dict:
    boundary_path=HERE/"raw/boundary_substitution_results.json"
    if not boundary_path.is_file():run_boundary_substitution()
    boundary=read_json(boundary_path);causal={c["class_id"]:c for c in read_json(HERE/"raw/causal_results.json")["classes"]};recon={r["organism_id"]:r for r in read_json(HERE/"raw/reconstruction_results.json")["rows"]}
    eligible=[c for c in boundary["classes"] if c["boundary_interchangeable"] and causal.get(c["class_id"],{}).get("causal_specificity_pass")]
    model_cache={};trace_cache={};classes=[]
    for c in eligible:
        rows=[]
        for pair in c["pairs"]:
            recipient=pair["recipient"];same=pair["same_class_donor"];different=pair["different_class_donor"];roid=recipient["organism_id"]
            for occ in (recipient,same,different):
                oid=occ["organism_id"]
                if oid not in model_cache:model_cache[oid]=load_reconstructed_model(recon[oid])[0]
            if roid not in trace_cache:trace_cache[roid]=run_full_probe(roid,CONFIRM,cache_key="confirmation64")
            trace=trace_cache[roid];obs,lengths,targets=trace.observations,trace.lengths,trace.targets;task=task_by_name(recipient["family"]);rmodel=model_cache[roid];rnodes=recipient["nodes"]
            with torch.no_grad():base_pred,base_trace=run_intervened(rmodel,obs,lengths,return_trace=True)
            baseline=_success(task,base_pred,targets);selfp=PortedPrimitive(rmodel,rnodes)
            with torch.no_grad():self_pred,self_trace=run_intervened(rmodel,obs,lengths,recipient_nodes=rnodes,donor=selfp,return_trace=True)
            self_pred_err=float(torch.max(torch.abs(self_pred-base_pred)).item());self_state_err=float(torch.max(torch.abs(self_trace.states-base_trace.states)).item())
            if max(self_pred_err,self_state_err)>1e-6:raise RuntimeError("CLOSED_LOOP_TRANSPLANT_IMPLEMENTATION_INVALID")
            sprim=PortedPrimitive(model_cache[same["organism_id"]],same["nodes"]);dprim=PortedPrimitive(model_cache[different["organism_id"]],different["nodes"]);rand=randomized_primitive(sprim,int(same["organism_id"][:12],16)%2_000_000_000)
            with torch.no_grad():sp=run_intervened(rmodel,obs,lengths,recipient_nodes=rnodes,donor=sprim);dp=run_intervened(rmodel,obs,lengths,recipient_nodes=rnodes,donor=dprim);rp=run_intervened(rmodel,obs,lengths,recipient_nodes=rnodes,donor=rand)
            ss,ds,rs=_success(task,sp,targets),_success(task,dp,targets),_success(task,rp,targets)
            rows.append({"recipient":recipient,"same_class_donor":same,"different_class_donor":different,"baseline_success":baseline,"self_transplant_prediction_max_error":self_pred_err,"self_transplant_state_max_error":self_state_err,"same_class_success":ss,"same_class_drop":baseline-ss,"different_class_success":ds,"randomized_success":rs})
        if rows:
            same=np.array([r["same_class_success"] for r in rows]);diff=np.array([r["different_class_success"] for r in rows]);rand=np.array([r["randomized_success"] for r in rows]);drops=np.array([r["same_class_drop"] for r in rows]);ret=float(np.mean(same>=0.85));improvement=np.minimum(same-diff,same-rand);p=one_sided_paired_sign_permutation(improvement.tolist())
            passed=float(np.median(drops))<=0.05 and ret>=0.60 and float(np.median(same))>=float(np.median(diff))+0.10 and float(np.median(same))>=float(np.median(rand))+0.10 and p<=0.01
            gate={"pass":bool(passed),"median_same_class_drop":float(np.median(drops)),"retain_success_fraction":ret,"median_same_class_success":float(np.median(same)),"median_different_class_success":float(np.median(diff)),"median_randomized_success":float(np.median(rand)),"p":p}
        else:gate={"pass":False,"p":1.0}
        classes.append({"class_id":c["class_id"],"stream":c["stream"],"size":c["size"],"rows":rows,"closed_loop_interchangeability":gate,"closed_loop_interchangeable":gate["pass"],"whole_system_control":int(c["size"])==10})
    payload={"version":"V837ak","stage":"AK9","run":bool(eligible),"eligible_class_count":len(eligible),"classes":classes,"closed_loop_interchangeable_count":sum(c["closed_loop_interchangeable"] for c in classes),"non_whole_system_interchangeable_count":sum(c["closed_loop_interchangeable"] and int(c["size"])<10 for c in classes)}
    write_json(HERE/"raw/closed_loop_substitution_results.json",payload);write_json(HERE/"diagnostics/closed_loop_interchangeability.json",payload);return payload


def main()->int:
    p=run_closed_loop();print(json.dumps({"run":p["run"],"eligible_classes":p["eligible_class_count"],"closed_loop":p["closed_loop_interchangeable_count"],"non_whole":p["non_whole_system_interchangeable_count"]},indent=2));return 0

if __name__=="__main__":raise SystemExit(main())
