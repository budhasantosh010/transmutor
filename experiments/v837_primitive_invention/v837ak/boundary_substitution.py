from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from experiments.v837_primitive_invention.v837ak.boundary_traces import active_mask,boundary_streams,run_full_probe
from experiments.v837_primitive_invention.v837ak.causal_interventions import run_causal
from experiments.v837_primitive_invention.v837ak.dynamic_fingerprint import load_size_matrix
from experiments.v837_primitive_invention.v837ak.intervention_runtime import randomized_primitive
from experiments.v837_primitive_invention.v837ak.ported_primitive import PortedPrimitive
from experiments.v837_primitive_invention.v837ak.reconstruct_organisms import load_reconstructed_model
from experiments.v837_primitive_invention.v837ak.utils import DIRECTED,RANDOM,HERE,normalize,normalized_rmse,one_sided_paired_sign_permutation,read_json,write_json

DISCOVERY=list(range(20000,20064));CONFIRM=list(range(20064,20128))


def _orthogonal_map(x:np.ndarray,y:np.ndarray)->np.ndarray:
    xc=x-x.mean(0,keepdims=True);yc=y-y.mean(0,keepdims=True);u,_,vt=np.linalg.svd(xc.T@yc,full_matrices=False);return u@vt


def _fit_alignment(rec_trace,don_trace,rec_nodes,don_nodes)->dict:
    mr=active_mask(rec_trace).numpy();md=active_mask(don_trace).numpy()
    if mr.shape!=md.shape or not np.array_equal(mr,md):raise RuntimeError("V837AK_ALIGNMENT_EPISODE_MASK_MISMATCH")
    state=[];message=[];output=[]
    for rn,dn in zip(rec_nodes,don_nodes):
        state.append(_orthogonal_map(rec_trace.states[:,:,rn,:].numpy()[mr],don_trace.states[:,:,dn,:].numpy()[md]))
        message.append(_orthogonal_map(rec_trace.messages[:,:,rn,:].numpy()[mr],don_trace.messages[:,:,dn,:].numpy()[md]))
        output.append(_orthogonal_map(rec_trace.outputs[:,:,rn,:].numpy()[mr],don_trace.outputs[:,:,dn,:].numpy()[md]))
    return {"state":state,"message":message,"output":output}


def _aligned_replay(donor:PortedPrimitive,streams,recipient_states,adapters):
    transformed={k:v.clone() for k,v in streams.items()}
    for li in range(len(donor.nodes)):
        qs=torch.tensor(adapters["state"][li],dtype=recipient_states.dtype);qm=torch.tensor(adapters["message"][li],dtype=recipient_states.dtype)
        transformed["external_messages"][:,:,li,:]=transformed["external_messages"][:,:,li,:]@qm
        transformed["global_terms"][:,:,li,:]=transformed["global_terms"][:,:,li,:]@qs
    rs=recipient_states.clone()
    for li in range(len(donor.nodes)):rs[:,:,li,:]=rs[:,:,li,:]@torch.tensor(adapters["state"][li],dtype=rs.dtype)
    replay=donor.replay_teacher_forced(transformed,rs)
    for li in range(len(donor.nodes)):
        qs=torch.tensor(adapters["state"][li],dtype=replay.states.dtype);qo=torch.tensor(adapters["output"][li],dtype=replay.outputs.dtype)
        replay.states[:,:,li,:]=replay.states[:,:,li,:]@qs.T;replay.candidates[:,:,li,:]=replay.candidates[:,:,li,:]@qs.T;replay.outputs[:,:,li,:]=replay.outputs[:,:,li,:]@qo.T
    return replay


def _nrmses(replay,trace,nodes)->dict:
    mask=active_mask(trace).numpy();idx=list(nodes)
    target_s=trace.states[:,:,idx,:].numpy()[mask];target_c=trace.candidates[:,:,idx,:].numpy()[mask];target_o=trace.outputs[:,:,idx,:].numpy()[mask]
    return {"state_nrmse":normalized_rmse(replay.states.numpy()[mask],target_s),"candidate_nrmse":normalized_rmse(replay.candidates.numpy()[mask],target_c),"output_nrmse":normalized_rmse(replay.outputs.numpy()[mask],target_o)}


def _self_errors(replay,trace,nodes)->dict:
    idx=list(nodes);mask=active_mask(trace)
    return {"state_max_abs_error":float(torch.max(torch.abs(replay.states-trace.states[:,:,idx,:])).item()),"candidate_max_abs_error":float(torch.max(torch.abs(replay.candidates[mask]-trace.candidates[:,:,idx,:][mask])).item()),"output_max_abs_error":float(torch.max(torch.abs(replay.outputs-trace.outputs[:,:,idx,:])).item())}


def _choose_same_class(c,recipient):
    members=[m for m in c.get("confirmation_rows",[]) if m.get("compatible") and m["organism_id"]!=recipient["organism_id"]]
    members.sort(key=lambda m:(-(m["family"]==recipient["family"]),-(m["engine"]!=recipient["engine"]),m["organism_id"],m["occurrence_id"]));return members[0] if members else None


def _choose_different_class(classes,c,recipient,census_rows,dynamic_cache):
    # Prefer another frozen candidate class of the same size, regardless of
    # whether that control class itself survived held-out confirmation.
    options=[]
    for other in classes:
        if other["class_id"]==c["class_id"] or int(other["size"])!=int(c["size"]):continue
        for m in other.get("confirmation_rows",[]):
            if m.get("compatible") and m["organism_id"]!=recipient["organism_id"]:options.append((m,other["class_id"],"frozen_candidate"))
    if not options:
        k=int(c["size"])
        if c["stream"]=="S":
            for m in census_rows:
                if not m.get("competent") or int(m["size"])!=k or m["organism_id"]==recipient["organism_id"]:continue
                if m.get("structural_class_id")==c["class_id"]:continue
                options.append((m,m.get("structural_class_id","DIFFERENT_STRUCTURAL_CLASS"),"full_census_outside_structural_class"))
        else:
            if k not in dynamic_cache:
                thresholds=read_json(HERE/"diagnostics/fingerprint_thresholds.json")["sizes"][str(k)]
                records,raw=load_size_matrix(k,"mean",competent_only=True)
                mat=normalize(raw,np.asarray(thresholds["median"]),np.asarray(thresholds["iqr"]))
                dynamic_cache[k]=(records,mat)
            records,mat=dynamic_cache[k];medoid=np.asarray(c["discovery_medoid"]);scale=math.sqrt(mat.shape[1]);tau=float(c["tau"])
            for m,vec in zip(records,mat):
                if m["organism_id"]==recipient["organism_id"]:continue
                if float(np.linalg.norm(vec-medoid)/scale)<=tau+1e-12:continue
                options.append((m,"OUTSIDE_FROZEN_DYNAMIC_CLASS","full_census_outside_dynamic_radius"))
    options.sort(key=lambda x:(-(x[0]["family"]==recipient["family"]),-(x[0]["engine"]!=recipient["engine"]),x[1],x[0]["organism_id"],x[0]["occurrence_id"]));return options[0] if options else (None,None,None)


def _class_gate(rows,key="same"):
    if not rows:return {"pass":False,"median_output":None,"median_state":None,"p":1.0,"beat_both_fraction":0.0}
    same_o=np.array([r[key]["output_nrmse"] for r in rows]);same_s=np.array([r[key]["state_nrmse"] for r in rows]);diff_o=np.array([r["different"]["output_nrmse"] for r in rows]);rand_o=np.array([r["randomized"]["output_nrmse"] for r in rows]);diff_s=np.array([r["different"]["state_nrmse"] for r in rows]);rand_s=np.array([r["randomized"]["state_nrmse"] for r in rows])
    beat=(same_o<diff_o)&(same_o<rand_o);improvement=np.minimum(diff_o-same_o,rand_o-same_o);p=one_sided_paired_sign_permutation(improvement.tolist())
    passed=float(np.median(same_o))<=0.75*float(np.median(diff_o)) and float(np.median(same_o))<=0.75*float(np.median(rand_o)) and float(np.median(same_s))<=0.85*float(np.median(diff_s)) and float(np.median(same_s))<=0.85*float(np.median(rand_s)) and float(np.mean(beat))>=0.60 and p<=0.01
    return {"pass":bool(passed),"median_output":float(np.median(same_o)),"median_state":float(np.median(same_s)),"different_median_output":float(np.median(diff_o)),"randomized_median_output":float(np.median(rand_o)),"different_median_state":float(np.median(diff_s)),"randomized_median_state":float(np.median(rand_s)),"beat_both_fraction":float(np.mean(beat)),"p":p}


def run_boundary_substitution()->dict:
    causal_path=HERE/"raw/causal_results.json"
    if not causal_path.is_file():run_causal()
    confirmed=read_json(HERE/"raw/confirmed_candidate_classes.json");causal=read_json(causal_path);causal_map={c["class_id"]:c for c in causal["classes"]};recon={r["organism_id"]:r for r in read_json(HERE/"raw/reconstruction_results.json")["rows"]}
    model_cache={};confirm_trace={};discovery_trace={};classes=[];dynamic_cache={};census_rows=read_json(HERE/"raw/cache/subset_occurrences.json")["rows"]
    for c in confirmed["classes"]:
        if not c.get("confirmed"):continue
        reps=causal_map.get(c["class_id"],{}).get("representatives",[])[:8];pair_rows=[]
        for rep in reps:
            recipient={k:rep[k] for k in ("organism_id","engine","family","nodes")};same=_choose_same_class(c,recipient);different,diff_class,diff_source=_choose_different_class(confirmed["classes"],c,recipient,census_rows,dynamic_cache)
            if same is None or different is None:continue
            for occ in (recipient,same,different):
                oid=occ["organism_id"]
                if oid not in model_cache:model_cache[oid]=load_reconstructed_model(recon[oid])[0]
                if oid not in confirm_trace:confirm_trace[oid]=run_full_probe(oid,CONFIRM,cache_key="confirmation64")
            roid=recipient["organism_id"];rmodel=model_cache[roid];rtrace=confirm_trace[roid];rnodes=recipient["nodes"];streams=boundary_streams(rmodel,rtrace,rnodes);rstates=rtrace.states[:,:,list(rnodes),:]
            selfp=PortedPrimitive(rmodel,rnodes);self_replay=selfp.replay_teacher_forced(streams,rstates);selferr=_self_errors(self_replay,rtrace,rnodes)
            if max(selferr.values())>1e-6:raise RuntimeError("BOUNDARY_REPLAY_INVALID")
            smodel=model_cache[same["organism_id"]];sprim=PortedPrimitive(smodel,same["nodes"]);sreplay=sprim.replay_teacher_forced(streams,rstates);same_metrics=_nrmses(sreplay,rtrace,rnodes)
            dmodel=model_cache[different["organism_id"]];dprim=PortedPrimitive(dmodel,different["nodes"]);dreplay=dprim.replay_teacher_forced(streams,rstates);diff_metrics=_nrmses(dreplay,rtrace,rnodes)
            rand=randomized_primitive(sprim,int(same["organism_id"][:12],16)%2_000_000_000);rreplay=rand.replay_teacher_forced(streams,rstates);rand_metrics=_nrmses(rreplay,rtrace,rnodes)
            aligned_metrics=None;alignment_available=False
            if same["family"]==recipient["family"]:
                for oid in (roid,same["organism_id"]):
                    if oid not in discovery_trace:discovery_trace[oid]=run_full_probe(oid,DISCOVERY,cache_key="discovery64")
                adapters=_fit_alignment(discovery_trace[roid],discovery_trace[same["organism_id"]],rnodes,same["nodes"]);aligned=_aligned_replay(sprim,streams,rstates,adapters);aligned_metrics=_nrmses(aligned,rtrace,rnodes);alignment_available=True
            pair_rows.append({"recipient":recipient,"same_class_donor":same,"different_class_donor":{**different,"class_id":diff_class,"control_source":diff_source},"self_replay":selferr,"same":same_metrics,"different":diff_metrics,"randomized":rand_metrics,"aligned_same":aligned_metrics,"alignment_available":alignment_available})
        gate=_class_gate(pair_rows,"same");aligned_rows=[r for r in pair_rows if r["aligned_same"] is not None];aligned_gate=_class_gate(aligned_rows,"aligned_same") if aligned_rows else {"pass":False,"p":1.0}
        classes.append({"class_id":c["class_id"],"stream":c["stream"],"size":c["size"],"pairs":pair_rows,"boundary_interchangeability":gate,"boundary_interchangeable":gate["pass"],"orthogonal_alignment":aligned_gate,"basis_alignment_rescues":not gate["pass"] and bool(aligned_gate.get("pass"))})
    payload={"version":"V837ak","stage":"AK8","classes":classes,"boundary_interchangeable_count":sum(c["boundary_interchangeable"] for c in classes),"basis_alignment_rescue_count":sum(c["basis_alignment_rescues"] for c in classes)}
    write_json(HERE/"raw/boundary_substitution_results.json",payload);write_json(HERE/"diagnostics/boundary_interchangeability.json",payload);write_json(HERE/"diagnostics/interface_alignment.json",{"version":"V837ak","classes":[{"class_id":c["class_id"],"unaligned":c["boundary_interchangeability"],"aligned":c["orthogonal_alignment"],"basis_alignment_rescues":c["basis_alignment_rescues"]} for c in classes]})
    return payload


def main()->int:
    p=run_boundary_substitution();print(json.dumps({"classes":len(p["classes"]),"boundary_interchangeable":p["boundary_interchangeable_count"],"basis_alignment_rescue":p["basis_alignment_rescue_count"]},indent=2));return 0

if __name__=="__main__":raise SystemExit(main())
