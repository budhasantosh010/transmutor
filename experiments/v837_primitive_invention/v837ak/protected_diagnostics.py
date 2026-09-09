from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
import torch

from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837ak.boundary_traces import run_full_probe
from experiments.v837_primitive_invention.v837ak.intervention_runtime import run_intervened
from experiments.v837_primitive_invention.v837ak.reconstruct_organisms import _load_rows,load_reconstructed_model
from experiments.v837_primitive_invention.v837ak.utils import HERE,read_json,sha256_json,write_json

CONFIRM=list(range(20064,20128))


def _success(task,pred,target):return float(np.mean([task.success(float(p),float(t)) for p,t in zip(pred.tolist(),target.tolist())]))


def _path_signatures(topology:dict):
    edges=[(int(e["src"]),int(e["dst"]),bool(e["recurrent"])) for e in topology["edges"]]
    paths=[]
    for e in edges:paths.append((e,))
    frontier=[(e,) for e in edges]
    for _ in range(3):
        nxt=[]
        for path in frontier:
            last=path[-1]
            for e in edges:
                if last[1]==e[0]:
                    candidate=path+(e,);paths.append(candidate);nxt.append(candidate)
        frontier=nxt
    out=[]
    for p in paths:
        nodes=[]
        for e in p:
            if e[0] not in nodes:nodes.append(e[0])
            if e[1] not in nodes:nodes.append(e[1])
        order={n:i for i,n in enumerate(sorted(nodes))}
        sig={"edge_count":len(p),"edges":[[order[e[0]],order[e[1]],e[2]] for e in p],"temporal_pattern":["R" if e[2] else "S" for e in p]}
        out.append((sha256_json(sig),sig,p))
    return out


def routing_diagnostic()->dict:
    recon={r["organism_id"]:r for r in read_json(HERE/"raw/reconstruction_results.json")["rows"] if r["competent"]};source={r["organism_id"]:r for r in _load_rows()};classes=defaultdict(lambda:defaultdict(list));signatures={}
    for oid,row in recon.items():
        for cid,sig,path in _path_signatures(source[oid]["topology"]):
            signatures[cid]=sig;classes[cid][oid].append(path)
    ranked=[]
    for cid,by_org in classes.items():
        reps={oid:sorted(paths,key=lambda p:(len(p),p))[0] for oid,paths in by_org.items()};ranked.append({"class_id":cid,"signature":signatures[cid],"support":len(reps),"representatives":reps})
    ranked.sort(key=lambda c:(-c["support"],c["signature"]["edge_count"],c["class_id"]));tested=[]
    for c in [x for x in ranked if x["support"]>=6][:6]:
        drops=[];flows=[];rows=[]
        for oid,path in list(sorted(c["representatives"].items()))[:8]:
            rr=recon[oid];model,_,_=load_reconstructed_model(rr);trace=run_full_probe(oid,CONFIRM,cache_key="confirmation64");task=task_by_name(rr["family"])
            disabled=set(path)
            with torch.no_grad():base=run_intervened(model,trace.observations,trace.lengths);cut=run_intervened(model,trace.observations,trace.lengths,disabled_edges=disabled)
            b=_success(task,base,trace.targets);s=_success(task,cut,trace.targets);drop=b-s
            edge_flow=[]
            prev=torch.zeros_like(trace.outputs);prev[:,1:]=trace.outputs[:,:-1]
            for src,dst,rec in path:
                idx=next(i for i,e in enumerate(model.graph.edges) if int(e.src)==src and int(e.dst)==dst and bool(e.recurrent)==rec);source_out=prev[:,:,src,:] if rec else trace.outputs[:,:,src,:];edge_flow.append(float(torch.mean(torch.linalg.vector_norm(model.base.edge_weights[idx].detach()*source_out,dim=-1)).item()))
            flow=float(np.mean(edge_flow));drops.append(drop);flows.append(flow);rows.append({"organism_id":oid,"family":rr["family"],"baseline_success":b,"edge_cut_success":s,"drop":drop,"message_flow_magnitude":flow,"path":[list(e) for e in path]})
        tested.append({**{k:c[k] for k in ("class_id","signature","support")},"median_edge_cut_drop":float(np.median(drops)) if drops else 0.0,"median_message_flow":float(np.median(flows)) if flows else 0.0,"rows":rows})
    strong=[c for c in tested if c["median_edge_cut_drop"]>=0.03 and c["median_message_flow"]>0]
    payload={"version":"V837ak","diagnostic_only":True,"enumerated_class_count":len(ranked),"tested_class_count":len(tested),"classes":tested,"hypothesis_supported":bool(strong),"diagnosis":"ROUTING_PRIMITIVE_HYPOTHESIS_SUPPORTED" if strong else "ROUTING_PRIMITIVE_HYPOTHESIS_NOT_SUPPORTED"}
    write_json(HERE/"raw/routing_diagnostic.json",payload);return payload


def distributed_mode_diagnostic()->dict:
    rel=read_json(HERE/"diagnostics/fingerprint_reliability.json");dyn=read_json(HERE/"raw/dynamic_classes_discovery.json")
    whole=[c for c in dyn.get("classes",[]) if int(c["size"])==10 and (c.get("general_eligible") or c.get("family_specific_eligible"))]
    payload={"version":"V837ak","diagnostic_only":True,"whole_system_fingerprint_reliable":bool(rel["sizes"]["10"]["eligible"]),"whole_system_dynamic_classes":len(whole),"whole_system_supports":[c["support"] for c in whole],"global_latent_hypothesis":bool(rel["sizes"]["10"]["eligible"] and whole),"diagnosis":"GLOBAL_LATENT_DYNAMICAL_PRIMITIVE_HYPOTHESIS" if rel["sizes"]["10"]["eligible"] and whole else "GLOBAL_LATENT_DYNAMICAL_PRIMITIVE_HYPOTHESIS_NOT_SUPPORTED"}
    write_json(HERE/"raw/distributed_mode_diagnostic.json",payload);return payload
