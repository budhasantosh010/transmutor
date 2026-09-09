from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837ak.boundary_traces import active_mask, run_full_probe
from experiments.v837_primitive_invention.v837ak.confirm_candidates import confirm_candidates
from experiments.v837_primitive_invention.v837ak.intervention_runtime import run_intervened
from experiments.v837_primitive_invention.v837ak.reconstruct_organisms import _load_rows, load_reconstructed_model
from experiments.v837_primitive_invention.v837ak.structural_signatures import structural_signature
from experiments.v837_primitive_invention.v837ak.utils import DIRECTED,RANDOM,HERE,one_sided_paired_sign_permutation,read_json,write_json

SEEDS=list(range(20064,20128))


def _success(task,pred,target)->float:return float(np.mean([task.success(float(p),float(t)) for p,t in zip(pred.tolist(),target.tolist())]))


def _select_representatives(c:dict)->list[dict]:
    rows=[r for r in c.get("confirmation_rows",[]) if r.get("compatible")]
    selected=[];remaining=list(rows)
    while remaining and len(selected)<8:
        fams={x["family"] for x in selected};engs={x["engine"] for x in selected}
        remaining.sort(key=lambda r:(-(r["family"] not in fams),-(r["engine"] not in engs),r["organism_id"],r["occurrence_id"]))
        selected.append(remaining.pop(0))
    return selected


def _activity(trace,nodes)->float:
    m=active_mask(trace);x=trace.states[:,:,list(nodes),:].numpy()[m.numpy()];return float(np.mean(np.linalg.norm(x.reshape(len(x),-1),axis=1))) if len(x) else 0.0


def _boundary_degree(sig:dict)->int:
    keys=("external_incoming_same_step_per_cell","external_incoming_recurrent_per_cell","external_outgoing_same_step_per_cell","external_outgoing_recurrent_per_cell")
    return sum(sum(sig[k]) for k in keys)


def _shams(source_row:dict,trace,nodes)->list[dict]:
    k=len(nodes);target_sig=structural_signature(source_row["topology"],nodes);target_e=len(target_sig["internal_same_step_edges"])+len(target_sig["internal_recurrent_edges"]);target_r=len(target_sig["internal_recurrent_edges"]);target_b=_boundary_degree(target_sig);target_a=_activity(trace,nodes)
    candidates=[]
    for combo in itertools.combinations(range(10),k):
        if tuple(combo)==tuple(nodes):continue
        sig=structural_signature(source_row["topology"],combo);e=len(sig["internal_same_step_edges"])+len(sig["internal_recurrent_edges"]);r=len(sig["internal_recurrent_edges"]);b=_boundary_degree(sig);a=_activity(trace,combo)
        edge_delta=abs(e-target_e);rec_delta=abs(r-target_r)
        candidates.append((edge_delta,rec_delta,abs(b-target_b),abs(a-target_a),combo,e,r,b,a))
    # Frozen matching order: exact edge/recurrent counts first, then +/-1,
    # then +/-2. Never silently relax beyond the declared +/-2 rule.
    chosen=[];seen=set()
    for tolerance in (0,1,2):
        eligible=[x for x in candidates if x[0]<=tolerance and x[1]<=tolerance and x[4] not in seen]
        eligible.sort(key=lambda x:(x[0]+x[1],x[2],x[3],x[4]))
        for x in eligible:
            if x[4] in seen:continue
            seen.add(x[4]);chosen.append((tolerance,*x))
            if len(chosen)>=5:break
        if len(chosen)>=5:break
    return [{"nodes":list(x[5]),"relaxation_level":x[0],"edge_count_delta":x[1],"recurrent_count_delta":x[2],"internal_edges":x[6],"recurrent_edges":x[7],"boundary_degree":x[8],"activity_norm":x[9]} for x in chosen[:5]]


def _divergence(hybrid:torch.Tensor,baseline:torch.Tensor,lengths:torch.Tensor,nodes_excluded:set[int])->float:
    cells=[i for i in range(10) if i not in nodes_excluded]
    if not cells:return 0.0
    mask=(torch.arange(baseline.shape[1]).view(1,-1)<lengths.view(-1,1));a=hybrid[:,:,cells,:][mask];b=baseline[:,:,cells,:][mask]
    rmse=float(torch.sqrt(torch.mean((a-b)**2)).item());return rmse/(float(torch.std(b).item())+1e-8)


def run_causal()->dict:
    conf_path=HERE/"raw/confirmed_candidate_classes.json"
    if not conf_path.is_file():confirm_candidates()
    confirmed=read_json(conf_path);source={r["organism_id"]:r for r in _load_rows()};recon={r["organism_id"]:r for r in read_json(HERE/"raw/reconstruction_results.json")["rows"]}
    classes=[];model_cache={};trace_cache={};batch_cache={}
    for c in confirmed["classes"]:
        if not c["confirmed"]:continue
        # Size-10 is a whole-organism diagnostic control. There is no distinct
        # same-size sham subset inside a ten-cell organism, so it is not eligible
        # for the five-sham causal-specificity gate.
        if int(c["size"])==10:
            classes.append({"class_id":c["class_id"],"stream":c["stream"],"size":c["size"],"representatives":[],"median_causal_specificity":0.0,"positive_fraction":0.0,"one_sided_p":1.0,"median_real_lesion_drop":None,"distribution_shift_fraction":0.0,"intervention_distribution_shift":False,"causal_specificity_pass":False,"causal_evidence_downgraded":False,"whole_system_control":True,"causal_status":"WHOLE_SYSTEM_CONTROL_NO_SAME_SIZE_SHAM"})
            continue
        reps=_select_representatives(c);rep_rows=[]
        for rep in reps:
            oid=rep["organism_id"];nodes=tuple(rep["nodes"]);rr=recon[oid];sr=source[oid]
            if oid not in model_cache:model_cache[oid]=load_reconstructed_model(rr)[0]
            if oid not in trace_cache:trace_cache[oid]=run_full_probe(oid,SEEDS,cache_key="confirmation64")
            model=model_cache[oid];trace=trace_cache[oid];task=task_by_name(rr["family"])
            obs,lengths,targets=trace.observations,trace.lengths,trace.targets
            with torch.no_grad():
                base_pred,base_trace=run_intervened(model,obs,lengths,return_trace=True);real_pred,real_trace=run_intervened(model,obs,lengths,freeze_nodes=nodes,return_trace=True)
            baseline=_success(task,base_pred,targets);real=_success(task,real_pred,targets);delta_real=baseline-real
            shams=_shams(sr,trace,nodes);sham_rows=[]
            for sham in shams:
                with torch.no_grad():sp,st=run_intervened(model,obs,lengths,freeze_nodes=sham["nodes"],return_trace=True)
                ss=_success(task,sp,targets);sham_rows.append({**sham,"success":ss,"drop":baseline-ss,"outside_state_divergence":_divergence(st.states,base_trace.states,lengths,set(sham["nodes"])),"message_divergence":_divergence(st.messages,base_trace.messages,lengths,set(sham["nodes"]))})
            complete=len(sham_rows)==5
            median_sham=float(np.median([x["drop"] for x in sham_rows])) if sham_rows else 0.0;C=delta_real-median_sham if complete else 0.0;real_div=_divergence(real_trace.states,base_trace.states,lengths,set(nodes));sham_div=float(np.median([x["outside_state_divergence"] for x in sham_rows])) if sham_rows else 0.0;shift=complete and real_div>2.0*max(sham_div,1e-12)
            rep_rows.append({"organism_id":oid,"engine":rr["engine"],"family":rr["family"],"nodes":list(nodes),"baseline_success":baseline,"real_lesion_success":real,"real_lesion_drop":delta_real,"shams":sham_rows,"sham_control_complete":complete,"sham_count":len(sham_rows),"median_sham_drop":median_sham,"causal_specificity":C,"real_outside_state_divergence":real_div,"median_sham_outside_state_divergence":sham_div,"message_divergence":_divergence(real_trace.messages,base_trace.messages,lengths,set(nodes)),"intervention_distribution_shift":bool(shift)})
        controls_complete=bool(rep_rows) and all(r["sham_control_complete"] for r in rep_rows);Cs=[r["causal_specificity"] for r in rep_rows];median_c=float(np.median(Cs)) if Cs else 0.0;positive=float(np.mean(np.asarray(Cs)>0)) if Cs else 0.0;p=one_sided_paired_sign_permutation(Cs);shift_frac=float(np.mean([r["intervention_distribution_shift"] for r in rep_rows])) if rep_rows else 0.0
        causal_pass=controls_complete and median_c>=0.03 and positive>=0.60 and p<=0.05
        downgraded=shift_frac>0.5
        classes.append({"class_id":c["class_id"],"stream":c["stream"],"size":c["size"],"representatives":rep_rows,"matched_sham_controls_complete":controls_complete,"median_causal_specificity":median_c,"positive_fraction":positive,"one_sided_p":p,"median_real_lesion_drop":float(np.median([r["real_lesion_drop"] for r in rep_rows])) if rep_rows else 0.0,"distribution_shift_fraction":shift_frac,"intervention_distribution_shift":downgraded,"causal_specificity_pass":causal_pass,"causal_evidence_downgraded":causal_pass and downgraded,"causal_status":"PASS" if causal_pass else ("SHAM_CONTROL_INCOMPLETE" if not controls_complete else "CAUSAL_GATE_FAIL")})
    payload={"version":"V837ak","stage":"AK7","classes":classes,"causally_specific_count":sum(c["causal_specificity_pass"] for c in classes)}
    write_json(HERE/"raw/causal_results.json",payload);write_json(HERE/"diagnostics/causal_specificity.json",payload);write_json(HERE/"diagnostics/intervention_distribution_shift.json",{"version":"V837ak","classes":[{"class_id":c["class_id"],"distribution_shift_fraction":c["distribution_shift_fraction"],"flag":c["intervention_distribution_shift"]} for c in classes]})
    return payload


def main()->int:
    p=run_causal();print(json.dumps({"classes":len(p["classes"]),"causally_specific":p["causally_specific_count"]},indent=2));return 0

if __name__=="__main__":raise SystemExit(main())
