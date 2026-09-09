from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence
import hashlib

import torch

from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837ak.boundary_traces import external_raw_messages
from experiments.v837_primitive_invention.v837ak.reconstruct_organisms import load_reconstructed_model
from experiments.v837_primitive_invention.v837ak.intervention_runtime import randomized_primitive
from experiments.v837_primitive_invention.v837ak.ported_primitive import PortedPrimitive
from .utils import HERE, ROOT, read_json

CACHE=HERE/"raw/cache/traces"
_ROWS=None;_MODELS={};_TRACES={}

@dataclass
class InterfaceTrace:
    observations:torch.Tensor;lengths:torch.Tensor;targets:torch.Tensor;states:torch.Tensor;candidates:torch.Tensor;outputs:torch.Tensor;external_messages:torch.Tensor;global_terms:torch.Tensor;projected_inputs:torch.Tensor;gates:torch.Tensor


def reconstruction_rows():
    global _ROWS
    if _ROWS is None:_ROWS={r["organism_id"]:r for r in read_json(ROOT/"experiments/v837_primitive_invention/v837ak/raw/reconstruction_results.json")["rows"]}
    return _ROWS

def load_model(oid):
    if oid not in _MODELS:
        row=reconstruction_rows()[oid];model=load_reconstructed_model(row)[0];model.eval();_MODELS[oid]=(model,row)
    return _MODELS[oid]
def _key(seeds,split,nodes):return hashlib.sha256(f"{split}:{','.join(map(str,seeds))}:{','.join(map(str,nodes))}".encode()).hexdigest()[:16]

def collect_trace(oid:str,nodes:Sequence[int],seeds:Sequence[int],split:str="development"):
    nodes=tuple(sorted(map(int,nodes)));key=(oid,_key(seeds,split,nodes))
    if key in _TRACES:return _TRACES[key]
    path=CACHE/f"{oid}__{key[1]}.pt"
    if path.is_file():
        d=torch.load(path,map_location="cpu",weights_only=False);tr=InterfaceTrace(**d);_TRACES[key]=tr;return tr
    model,row=load_model(oid);task=task_by_name(row["family"]);eps=[task.generate(int(s),split) for s in seeds];obs,lengths,targets=episodes_to_batch(eps)
    with torch.no_grad():_,full=model(obs,lengths,return_trace=True)
    projected=[]
    with torch.no_grad():
        for t in range(obs.shape[1]): projected.append(torch.stack([model._visible_input(obs[:,t,:],n) for n in nodes],1))
    class Proxy:pass
    p=Proxy();p.outputs=full.outputs.detach().cpu();p.messages=full.messages.detach().cpu();p.states=full.states.detach().cpu();p.lengths=lengths.detach().cpu()
    ext=external_raw_messages(model,p,nodes)
    tr=InterfaceTrace(obs.detach().cpu(),lengths.detach().cpu(),targets.detach().cpu(),full.states.detach().cpu(),full.candidate_states.detach().cpu(),full.outputs.detach().cpu(),ext.detach().cpu(),(full.global_recurrent_terms[:,:,list(nodes),:]+full.matched_local_terms[:,:,list(nodes),:]).detach().cpu(),torch.stack(projected,1).detach().cpu(),full.global_gates.detach().cpu())
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(tr.__dict__,path);_TRACES[key]=tr;return tr

def mask(trace):return torch.arange(trace.states.shape[1]).view(1,-1)<trace.lengths.view(-1,1)
def port_arrays(trace,nodes):
    m=mask(trace);nodes=list(nodes)
    return {"state":[trace.states[:,:,n,:][m].numpy() for n in nodes],"external_messages":[trace.external_messages[:,:,i,:][m].numpy() for i in range(len(nodes))],"global_term":[trace.global_terms[:,:,i,:][m].numpy() for i in range(len(nodes))],"projected_input":[trace.projected_inputs[:,:,i,:][m].numpy() for i in range(len(nodes))],"gate":[trace.gates[m].numpy()],"output":[trace.outputs[:,:,n,:][m].numpy() for n in nodes]}

def trace_integrity(a,b):
    if mask(a).shape!=mask(b).shape or not torch.equal(mask(a),mask(b)):raise RuntimeError("ALIGNMENT_EPISODE_MASK_MISMATCH")

def randomized_trace(donor_trace,donor_occ,seed:int):
    model,_=load_model(donor_occ["organism_id"]);p=PortedPrimitive(model,donor_occ["nodes"]);rp=randomized_primitive(p,seed)
    streams={"observations":donor_trace.observations,"lengths":donor_trace.lengths,"external_messages":donor_trace.external_messages,"global_terms":donor_trace.global_terms,"gates":donor_trace.gates}
    native=donor_trace.states[:,:,list(donor_occ["nodes"]),:];replay=rp.replay_teacher_forced(streams,native)
    projected=[]
    for t in range(donor_trace.observations.shape[1]):
        x=donor_trace.observations[:,t,:]
        projected.append(torch.stack([torch.nn.functional.linear(x,rp.proj_w[li],rp.proj_b[li]) for li in range(len(rp.nodes))],dim=1))
    randomized_projected=torch.stack(projected,dim=1)
    return replace(donor_trace,states=_replace_selected(donor_trace.states,donor_occ["nodes"],replay.states),candidates=_replace_selected(donor_trace.candidates,donor_occ["nodes"],replay.candidates),outputs=_replace_selected(donor_trace.outputs,donor_occ["nodes"],replay.outputs),projected_inputs=randomized_projected)
def _replace_selected(full,nodes,local):
    x=full.clone()
    for li,n in enumerate(nodes):x[:,:,n,:]=local[:,:,li,:]
    return x
