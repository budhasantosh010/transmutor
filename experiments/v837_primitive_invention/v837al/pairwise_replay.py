from __future__ import annotations

import numpy as np
import torch

from experiments.v837_primitive_invention.v837ak.intervention_runtime import randomized_primitive
from experiments.v837_primitive_invention.v837ak.ported_primitive import PortedPrimitive
from experiments.v837_primitive_invention.v837ak.utils import normalized_rmse, one_sided_paired_sign_permutation
from .aligned_primitive import AlignedPortedPrimitive
from .interface_ports import collect_interface_trace, load_model
from .utils import seeds

SELECT=seeds(10064,10127); TEST=seeds(20000,20127)


def replay_metrics(replay,trace,nodes):
    mask=(torch.arange(trace.states.shape[1]).view(1,-1)<trace.lengths.view(-1,1)).numpy();idx=list(nodes)
    ts=trace.states[:,:,idx,:].numpy()[mask];tc=trace.candidates[:,:,idx,:].numpy()[mask];to=trace.outputs[:,:,idx,:].numpy()[mask]
    return {"state_nrmse":normalized_rmse(replay.states.numpy()[mask],ts),"candidate_nrmse":normalized_rmse(replay.candidates.numpy()[mask],tc),"output_nrmse":normalized_rmse(replay.outputs.numpy()[mask],to)}


def evaluate_pair(pair,bundles,scope,seeds_used,split):
    rec=pair["recipient"];same=pair["same_class_donor"];diff=pair["different_class_donor"]
    rt=collect_interface_trace(rec["organism_id"],rec["nodes"],seeds_used,split);rm,_=load_model(rec["organism_id"]);rp=PortedPrimitive(rm,rec["nodes"])
    def run(occ,bundle,random=False):
        m,_=load_model(occ["organism_id"]);p=PortedPrimitive(m,occ["nodes"])
        if random:p=randomized_primitive(p,int(same["organism_id"][:12],16)%2_000_000_000)
        r=AlignedPortedPrimitive(p,rp).replay_teacher_forced(rt,rec["nodes"],bundle,scope);return replay_metrics(r,rt,rec["nodes"])
    empty={"ports":{}}
    sb=bundles.get("same",empty);db=bundles.get("different",empty)
    return {"class_id":pair["class_id"],"primary_causal":pair["primary_causal"],"recipient_organism_id":rec["organism_id"],"family":rec["family"],"engine":rec["engine"],"same":run(same,sb),"different":run(diff,db),"randomized":run(same,sb,True)}


def aggregate(rows,significance=False,powered=True):
    if not rows:return {"pass":False,"row_count":0,"median_output":None,"median_state":None,"beat_both_fraction":0.0,"p":1.0}
    so=np.asarray([r["same"]["output_nrmse"] for r in rows]);ss=np.asarray([r["same"]["state_nrmse"] for r in rows]);do=np.asarray([r["different"]["output_nrmse"] for r in rows]);ds=np.asarray([r["different"]["state_nrmse"] for r in rows]);ro=np.asarray([r["randomized"]["output_nrmse"] for r in rows]);rs=np.asarray([r["randomized"]["state_nrmse"] for r in rows]);beat=(so<do)&(so<ro);imp=np.minimum(do-so,ro-so);p=one_sided_paired_sign_permutation(imp.tolist()) if significance else None
    effect=float(np.median(so))<=0.75*float(np.median(do)) and float(np.median(so))<=0.75*float(np.median(ro)) and float(np.median(ss))<=0.85*float(np.median(ds)) and float(np.median(ss))<=0.85*float(np.median(rs)) and float(np.mean(beat))>=0.60
    passed=effect and ((p is not None and p<=0.01 and powered) if significance else True)
    return {"pass":bool(passed),"effect_gate_pass":bool(effect),"row_count":len(rows),"median_output":float(np.median(so)),"median_state":float(np.median(ss)),"different_median_output":float(np.median(do)),"randomized_median_output":float(np.median(ro)),"different_median_state":float(np.median(ds)),"randomized_median_state":float(np.median(rs)),"beat_both_fraction":float(np.mean(beat)),"p":p,"strong_statistical_pass":bool(passed) if significance else None}
