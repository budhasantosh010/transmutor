from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import hashlib

import torch

from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837ak.boundary_traces import active_mask, external_raw_messages
from experiments.v837_primitive_invention.v837ak.reconstruct_organisms import load_reconstructed_model
from .utils import HERE, read_json

CACHE=HERE/"raw/cache/traces"
_ROW_CACHE:dict[str,dict]|None=None
_MODEL_CACHE:dict[str,tuple]= {}
_TRACE_CACHE:dict[tuple[str,str],"InterfaceTrace"]={}

@dataclass
class InterfaceTrace:
    observations:torch.Tensor
    lengths:torch.Tensor
    targets:torch.Tensor
    states:torch.Tensor
    candidates:torch.Tensor
    outputs:torch.Tensor
    external_messages:torch.Tensor
    global_terms:torch.Tensor
    projected_inputs:torch.Tensor
    gates:torch.Tensor


def reconstruction_rows():
    global _ROW_CACHE
    if _ROW_CACHE is None:
        _ROW_CACHE={r["organism_id"]:r for r in read_json(HERE.parent/"v837ak/raw/reconstruction_results.json")["rows"]}
    return _ROW_CACHE


def load_model(organism_id:str):
    if organism_id not in _MODEL_CACHE:
        row=reconstruction_rows()[organism_id]
        model=load_reconstructed_model(row)[0];model.eval()
        _MODEL_CACHE[organism_id]=(model,row)
    return _MODEL_CACHE[organism_id]


def _key(seeds,split,nodes):
    raw=f"{split}:{','.join(map(str,seeds))}:{','.join(map(str,nodes))}".encode();return hashlib.sha256(raw).hexdigest()[:16]


def collect_interface_trace(organism_id:str,nodes:Sequence[int],seeds:Sequence[int],split:str)->InterfaceTrace:
    nodes=tuple(sorted(int(n) for n in nodes)); row=reconstruction_rows()[organism_id]; key=_key(seeds,split,nodes); cache_id=(organism_id,key)
    if cache_id in _TRACE_CACHE:return _TRACE_CACHE[cache_id]
    path=CACHE/f"{organism_id}__{key}.pt"
    if path.is_file():
        x=torch.load(path,map_location="cpu",weights_only=False);trace=InterfaceTrace(**x);_TRACE_CACHE[cache_id]=trace;return trace
    model,_=load_model(organism_id); task=task_by_name(row["family"]); eps=[task.generate(int(s),split) for s in seeds]; obs,lengths,targets=episodes_to_batch(eps)
    model.eval()
    with torch.no_grad(): _,tr=model(obs,lengths,return_trace=True)
    # Build explicit projected-input boundary independently for each selected cell.
    projected=[]
    with torch.no_grad():
        for t in range(obs.shape[1]):
            projected.append(torch.stack([model._visible_input(obs[:,t,:],n) for n in nodes],dim=1))
    projected_inputs=torch.stack(projected,dim=1).detach().cpu()
    class Proxy: pass
    proxy=Proxy(); proxy.outputs=tr.outputs.detach().cpu(); proxy.messages=tr.messages.detach().cpu(); proxy.states=tr.states.detach().cpu(); proxy.lengths=lengths.detach().cpu()
    ext=external_raw_messages(model,proxy,nodes)
    out=InterfaceTrace(observations=obs.detach().cpu(),lengths=lengths.detach().cpu(),targets=targets.detach().cpu(),states=tr.states.detach().cpu(),candidates=tr.candidate_states.detach().cpu(),outputs=tr.outputs.detach().cpu(),external_messages=ext.detach().cpu(),global_terms=(tr.global_recurrent_terms[:,:,list(nodes),:]+tr.matched_local_terms[:,:,list(nodes),:]).detach().cpu(),projected_inputs=projected_inputs,gates=tr.global_gates.detach().cpu())
    path.parent.mkdir(parents=True,exist_ok=True); torch.save(out.__dict__,path);_TRACE_CACHE[cache_id]=out;return out


def active_rows(trace:InterfaceTrace,tensor:torch.Tensor)->torch.Tensor:
    mask=(torch.arange(trace.states.shape[1]).view(1,-1)<trace.lengths.view(-1,1))
    return tensor[mask]


def port_matrices(trace:InterfaceTrace,nodes:Sequence[int])->dict:
    nodes=list(nodes); mask=(torch.arange(trace.states.shape[1]).view(1,-1)<trace.lengths.view(-1,1))
    return {
      "state":[trace.states[:,:,n,:][mask].numpy() for n in nodes],
      "external_messages":[trace.external_messages[:,:,i,:][mask].numpy() for i in range(len(nodes))],
      "global_term":[trace.global_terms[:,:,i,:][mask].numpy() for i in range(len(nodes))],
      "projected_input":[trace.projected_inputs[:,:,i,:][mask].numpy() for i in range(len(nodes))],
      "gate":[trace.gates[mask].numpy()],
      "output":[trace.outputs[:,:,n,:][mask].numpy() for n in nodes],
    }


def trace_integrity(rec:InterfaceTrace,don:InterfaceTrace)->None:
    mr=(torch.arange(rec.states.shape[1]).view(1,-1)<rec.lengths.view(-1,1)); md=(torch.arange(don.states.shape[1]).view(1,-1)<don.lengths.view(-1,1))
    if mr.shape!=md.shape or not torch.equal(mr,md): raise RuntimeError("ALIGNMENT_EPISODE_MASK_MISMATCH")
