from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib, json
import numpy as np
import torch
from torch.nn import functional as F

from experiments.v837_primitive_invention.v837ak.intervention_runtime import randomized_primitive
from experiments.v837_primitive_invention.v837ak.ported_primitive import PortedPrimitive
from experiments.v837_primitive_invention.v837ak.utils import normalized_rmse, one_sided_paired_sign_permutation
from .conditional_adapter import fit_map,fit_gate_map,apply_tensor,inverse_tensor,apply_gate_tensor,apply_np,apply_gate_np,conditioning
from .context_features import active_context,active_gate_context,context_tensor,gate_context_tensor
from .interface_data import collect_trace,load_model,mask,port_arrays,randomized_trace,trace_integrity
from .utils import HERE,sha256_file,write_json

PORTS=("state","external_messages","global_term","projected_input","gate","output")

@dataclass
class ContextReplay:
    states:torch.Tensor;candidates:torch.Tensor;outputs:torch.Tensor;donor_native_states:torch.Tensor;donor_native_outputs:torch.Tensor

def _rand_seed(pair):return int(hashlib.sha256((pair["recipient"]["organism_id"]+pair["same_class_donor"]["organism_id"]).encode()).hexdigest()[:8],16)%2_000_000_000

def _context_np(trace,nodes,li,context_set,port,history):
    if context_set in (None,"NONE"):return np.zeros((int(mask(trace).sum()),0),dtype=np.float64)
    if port=="gate":return active_gate_context(trace,nodes,context_set,history)
    return active_context(trace,nodes,li,context_set,port,history)

def _fit_bundle(rec_trace,don_trace,rec_nodes,don_nodes,family,context_set,history,rec_select=None,don_select=None,randomized=False):
    trace_integrity(rec_trace,don_trace);rp=port_arrays(rec_trace,rec_nodes);dp=port_arrays(don_trace,don_nodes);ports={};k=len(rec_nodes)
    for port in ("state","external_messages","global_term","projected_input"):
        maps=[]
        for li,(x,y) in enumerate(zip(rp[port],dp[port])):
            c=_context_np(rec_trace,rec_nodes,li,context_set,port,history);m=fit_map(x,y,c,family,state=(port=="state"));maps.append(m)
        ports[port]=maps
    maps=[]
    for li,(x,y) in enumerate(zip(dp["output"],rp["output"])):
        c=_context_np(rec_trace,rec_nodes,li,context_set,"output",history);maps.append(fit_map(x,y,c,family,False))
    ports["output"]=maps
    cg=_context_np(rec_trace,rec_nodes,0,context_set,"gate",history);ports["gate"]=fit_gate_map(rp["gate"][0],dp["gate"][0],cg,family)
    if rec_select is not None and don_select is not None:
        rs=port_arrays(rec_select,rec_nodes);ds=port_arrays(don_select,don_nodes);trace_integrity(rec_select,don_select)
        for port in ("state","external_messages","global_term","projected_input"):
            for li,(m,x,y) in enumerate(zip(ports[port],rs[port],ds[port])):
                c=_context_np(rec_select,rec_nodes,li,context_set,port,history);m["select_mse"]=float(np.mean((apply_np(x,c,m)-y)**2))
                if port=="state":m["conditioning"]=conditioning(c,m);m["valid"]=m["conditioning"]["valid"]
        for li,(m,x,y) in enumerate(zip(ports["output"],ds["output"],rs["output"])):
            c=_context_np(rec_select,rec_nodes,li,context_set,"output",history);m["select_mse"]=float(np.mean((apply_np(x,c,m)-y)**2))
        c=_context_np(rec_select,rec_nodes,0,context_set,"gate",history);ports["gate"]["select_mse"]=float(np.mean((apply_gate_np(rs["gate"][0],c,ports["gate"])-ds["gate"][0])**2))
    valid=all(m.get("valid",True) for x in ports.values() for m in (x if isinstance(x,list) else [x]))
    params=sum(m.get("parameters",0) for x in ports.values() for m in (x if isinstance(x,list) else [x]));macs=sum(m.get("macs",0) for x in ports.values() for m in (x if isinstance(x,list) else [x]))
    return {"family":family,"context_set":context_set,"history":history,"ports":ports,"valid":valid,"parameters":params,"macs":macs,"randomized_refit":randomized}

def fit_pair_roles(pair,family,context_set,history,fit_seeds,select_seeds,cache_path:Path):
    if cache_path.is_file():
        prior=json.loads(cache_path.read_text(encoding="utf-8"))
        if prior.get("cache_schema")==2:return prior
    rec=pair["recipient"];same=pair["same_class_donor"];diff=pair["different_class_donor"];rn=rec["nodes"]
    rf=collect_trace(rec["organism_id"],rn,fit_seeds);rs=collect_trace(rec["organism_id"],rn,select_seeds)
    sf=collect_trace(same["organism_id"],same["nodes"],fit_seeds);ss=collect_trace(same["organism_id"],same["nodes"],select_seeds)
    df=collect_trace(diff["organism_id"],diff["nodes"],fit_seeds);ds=collect_trace(diff["organism_id"],diff["nodes"],select_seeds)
    seed=_rand_seed(pair);rrf=randomized_trace(sf,same,seed);rrs=randomized_trace(ss,same,seed)
    roles={"same":_fit_bundle(rf,sf,rn,same["nodes"],family,context_set,history,rs,ss),"different_refit":_fit_bundle(rf,df,rn,diff["nodes"],family,context_set,history,rs,ds),"randomized_refit":_fit_bundle(rf,rrf,rn,same["nodes"],family,context_set,history,rs,rrs,True)}
    out={"cache_schema":2,"gate_context_mode":"motif_mean_position_invariant","randomized_projected_input":"randomized_projection_parameters","family":family,"context_set":context_set,"history":history,"roles":roles,"random_seed":seed};cache_path.parent.mkdir(parents=True,exist_ok=True);cache_path.write_text(json.dumps(out,sort_keys=True,separators=(",",":")),encoding="utf-8");return out

def _map(bundle,port,li):
    x=bundle["ports"][port];return x[li] if isinstance(x,list) else x

def enabled_bundle_cost(bundle,enabled_ports):
    params=macs=0
    for port in enabled_ports:
        maps=bundle["ports"][port]
        for m in (maps if isinstance(maps,list) else [maps]):
            params+=int(m.get("parameters",0));macs+=int(m.get("macs",0))
    return {"parameters":params,"macs":macs}

def replay(trace,recipient_nodes,donor_occ,bundle,enabled_ports,context_set,history,randomize=False,random_seed=0):
    dm,_=load_model(donor_occ["organism_id"]);donor=PortedPrimitive(dm,donor_occ["nodes"])
    if randomize:donor=randomized_primitive(donor,random_seed)
    rnodes=list(recipient_nodes);B,T=trace.observations.shape[:2];dtype=trace.observations.dtype;rstates=trace.states[:,:,rnodes,:];prev=torch.zeros_like(rstates);prev[:,1:]=rstates[:,:-1]
    context_cache={}
    if context_set not in (None,"NONE"):
        for port in enabled_ports:
            if port=="gate":context_cache[port]=[gate_context_tensor(trace,rnodes,context_set,history)]
            else:context_cache[port]=[context_tensor(trace,rnodes,li,context_set,port,history) for li in range(len(rnodes))]
    zero_context=torch.zeros(B,0,dtype=dtype)
    def cget(port,li,t):
        if context_set in (None,"NONE"):return zero_context
        if port=="gate":return context_cache[port][0][:,t,:]
        return context_cache[port][li][:,t,:]
    native_s=[];native_o=[];states=[];cands=[];outs=[]
    for t in range(T):
        cur_ns=[];cur_no=[];cur_s=[];cur_c=[];cur_o=[];x_t=trace.observations[:,t,:]
        for li,orig in enumerate(donor.nodes):
            rs=prev[:,t,li,:];cs=cget("state",li,t) if "state" in enabled_ports else zero_context;ds=apply_tensor(rs,cs,_map(bundle,"state",li)) if "state" in enabled_ports else rs
            msg=torch.zeros(B,4,dtype=dtype)
            for edge in donor.internal_edges:
                if edge["dst"]!=orig:continue
                src=donor.node_to_local[edge["src"]]
                if (not edge["recurrent"]) and src<len(cur_no):source=cur_no[src]
                else:
                    prs=prev[:,t,src,:];csrc=cget("state",src,t) if "state" in enabled_ports else zero_context;dns=apply_tensor(prs,csrc,_map(bundle,"state",src)) if "state" in enabled_ports else prs;source=dns@donor.wo[src].T
                msg=msg+edge["weight"]*source
            ext=trace.external_messages[:,t,li,:];glob=trace.global_terms[:,t,li,:]
            if "external_messages" in enabled_ports:ext=apply_tensor(ext,cget("external_messages",li,t),_map(bundle,"external_messages",li))
            if "global_term" in enabled_ports:glob=apply_tensor(glob,cget("global_term",li,t),_map(bundle,"global_term",li))
            if "projected_input" in enabled_ports:projected=apply_tensor(trace.projected_inputs[:,t,li,:],cget("projected_input",li,t),_map(bundle,"projected_input",li))
            else:projected=F.linear(x_t,donor.proj_w[li],donor.proj_b[li])
            cand=torch.tanh(ds@donor.ws[li].T+glob+(ext+msg)@donor.wm[li].T+projected@donor.wx[li].T+donor.b[li]);gate=trace.gates[:,t,:]
            if "gate" in enabled_ports:gate=apply_gate_tensor(gate,cget("gate",0,t),_map(bundle,"gate",0))
            prop=gate*ds+(1-gate)*cand;ono=prop@donor.wo[li].T;active=(t<trace.lengths).to(dtype).unsqueeze(1)
            if "state" in enabled_ports:
                sr=inverse_tensor(prop,cs,_map(bundle,"state",li));cr=inverse_tensor(cand,cs,_map(bundle,"state",li))
            else:sr=prop;cr=cand
            if "output" in enabled_ports:orr=apply_tensor(ono,cget("output",li,t),_map(bundle,"output",li))
            else:orr=ono
            sr=active*sr+(1-active)*rs;orr=active*orr+(1-active)*trace.outputs[:,t,rnodes[li],:]
            cur_ns.append(prop);cur_no.append(ono);cur_s.append(sr);cur_c.append(cr);cur_o.append(orr)
        native_s.append(torch.stack(cur_ns,1));native_o.append(torch.stack(cur_no,1));states.append(torch.stack(cur_s,1));cands.append(torch.stack(cur_c,1));outs.append(torch.stack(cur_o,1))
    return ContextReplay(torch.stack(states,1),torch.stack(cands,1),torch.stack(outs,1),torch.stack(native_s,1),torch.stack(native_o,1))
def metrics(rep,trace,nodes):
    m=mask(trace).numpy();idx=list(nodes);return {"state_nrmse":normalized_rmse(rep.states.numpy()[m],trace.states[:,:,idx,:].numpy()[m]),"candidate_nrmse":normalized_rmse(rep.candidates.numpy()[m],trace.candidates[:,:,idx,:].numpy()[m]),"output_nrmse":normalized_rmse(rep.outputs.numpy()[m],trace.outputs[:,:,idx,:].numpy()[m])}
def evaluate_pair(pair,fits,ports,context_set,history,seeds_used,split="development"):
    rec=pair["recipient"];same=pair["same_class_donor"];diff=pair["different_class_donor"];rt=collect_trace(rec["organism_id"],rec["nodes"],seeds_used,split);setattr(rt,"_recipient_oid",rec["organism_id"]);seed=fits["random_seed"]
    s=replay(rt,rec["nodes"],same,fits["roles"]["same"],ports,context_set,history,False,seed);d=replay(rt,rec["nodes"],diff,fits["roles"]["different_refit"],ports,context_set,history,False,seed);rr=replay(rt,rec["nodes"],same,fits["roles"]["randomized_refit"],ports,context_set,history,True,seed);rf=replay(rt,rec["nodes"],same,fits["roles"]["same"],ports,context_set,history,True,seed)
    return {"recipient_organism_id":rec["organism_id"],"family":rec["family"],"engine":rec["engine"],"same":metrics(s,rt,rec["nodes"]),"different_refit":metrics(d,rt,rec["nodes"]),"randomized_refit":metrics(rr,rt,rec["nodes"]),"randomized_fixed_adapter":metrics(rf,rt,rec["nodes"])}
def aggregate(rows,p_threshold=None):
    if not rows:return {"pass":False,"row_count":0}
    so=np.array([r["same"]["output_nrmse"] for r in rows]);ss=np.array([r["same"]["state_nrmse"] for r in rows]);do=np.array([r["different_refit"]["output_nrmse"] for r in rows]);ds=np.array([r["different_refit"]["state_nrmse"] for r in rows]);ro=np.array([r["randomized_refit"]["output_nrmse"] for r in rows]);rs=np.array([r["randomized_refit"]["state_nrmse"] for r in rows]);rfo=np.array([r["randomized_fixed_adapter"]["output_nrmse"] for r in rows]);beat=(so<do)&(so<ro);improve=np.minimum(do-so,ro-so);p=one_sided_paired_sign_permutation(improve.tolist()) if p_threshold is not None else None
    med={"same_output":float(np.median(so)),"same_state":float(np.median(ss)),"different_output":float(np.median(do)),"different_state":float(np.median(ds)),"random_refit_output":float(np.median(ro)),"random_refit_state":float(np.median(rs)),"random_fixed_output":float(np.median(rfo))};base=med["same_output"]<=.75*med["different_output"] and med["same_output"]<=.75*med["random_refit_output"] and med["same_state"]<=.85*med["different_state"] and med["same_state"]<=.85*med["random_refit_state"] and float(np.mean(beat))>=.60;anti=med["random_refit_output"]>=1.20*med["same_output"];passed=base and anti and (p is None or p<=p_threshold)
    return {**med,"row_count":len(rows),"beat_both_fraction":float(np.mean(beat)),"effect_gate_pass":bool(base),"adapter_cheating_gate_pass":bool(anti),"p":p,"pass":bool(passed)}
