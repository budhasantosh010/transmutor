from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.v837ak.reconstruct_organisms import load_reconstructed_model
from experiments.v837_primitive_invention.v837r.recurrent_coupling import LOCAL_STATE_DIM, MESSAGE_DIM, NUM_CELLS

from .authorization import SOURCE
from .oracle_macrostate import instrument_episode
from .utils import HERE, ROOT, read_json, write_json


@dataclass
class TensorPatch:
    values: torch.Tensor
    mask: torch.Tensor | None = None


@dataclass
class PatchPlan:
    state: dict[int, TensorPatch] = field(default_factory=dict)
    output: dict[int, TensorPatch] = field(default_factory=dict)
    message: dict[int, TensorPatch] = field(default_factory=dict)
    global_term: dict[int, TensorPatch] = field(default_factory=dict)
    gate: dict[int, TensorPatch] = field(default_factory=dict)


@dataclass
class AF1DTrace:
    states: torch.Tensor
    outputs: torch.Tensor
    messages: torch.Tensor
    global_terms: torch.Tensor
    local_terms: torch.Tensor
    input_terms: torch.Tensor
    gate: torch.Tensor
    coupling_factor4: torch.Tensor

    def carrier(self, name: str) -> torch.Tensor:
        if name == "STATE40": return self.states.flatten(2)
        if name == "OUTPUT40": return self.outputs.flatten(2)
        if name == "MESSAGE40": return self.messages.flatten(2)
        if name == "GLOBAL40": return self.global_terms.flatten(2)
        if name == "GATE1": return self.gate
        if name == "COUPLING_FACTOR4": return self.coupling_factor4
        raise KeyError(name)


def _masked_replace(current: torch.Tensor, patch: TensorPatch | None) -> torch.Tensor:
    if patch is None:
        return current
    values = patch.values.to(device=current.device, dtype=current.dtype)
    if values.shape != current.shape:
        values = values.expand_as(current)
    if patch.mask is None:
        return values
    mask = patch.mask.to(device=current.device)
    while mask.ndim < current.ndim:
        mask = mask.unsqueeze(0)
    mask = mask.expand_as(current)
    return torch.where(mask, values, current)


def run_instrumented(model, observations: torch.Tensor, lengths: torch.Tensor | None = None, *, patch_plan: PatchPlan | None = None, return_trace: bool = True):
    if observations.ndim != 3: raise ValueError("observations must be [B,T,D]")
    B,T,D=observations.shape
    if D != model.obs_dim: raise ValueError("observation dimension mismatch")
    patch_plan=patch_plan or PatchPlan();device=observations.device;dtype=observations.dtype
    prev_states=[torch.zeros(B,LOCAL_STATE_DIM,device=device,dtype=dtype) for _ in range(NUM_CELLS)]
    prev_outputs=[torch.zeros(B,MESSAGE_DIM,device=device,dtype=dtype) for _ in range(NUM_CELLS)]
    trace={k:[] for k in ("states","outputs","messages","global","local","input","gate","factor")}
    incoming=getattr(model,"_v837aj_incoming_edge_indices",None)
    if incoming is None:
        incoming=tuple(tuple(i for i,e in enumerate(model.graph.edges) if int(e.dst)==cell) for cell in range(NUM_CELLS))
    model._shared_projection_cache_key=None;model._shared_projection_cache_value=None
    for t in range(T):
        x_t=observations[:,t,:];snapshot=prev_states
        global_terms=model._global_terms(snapshot);matched_terms=model._matched_local_terms(snapshot)
        gflat=torch.stack(global_terms,dim=1);gflat=_masked_replace(gflat,patch_plan.global_term.get(t));global_terms=list(gflat.unbind(dim=1))
        gate=model._global_gate(snapshot,x_t);gate=_masked_replace(gate,patch_plan.gate.get(t))
        if hasattr(model,"global_v"):
            factor=torch.cat(snapshot,dim=1)@model.global_v
        else:
            factor=torch.zeros(B,4,device=device,dtype=dtype)
        active=(t<lengths).to(dtype).unsqueeze(1) if lengths is not None else torch.ones(B,1,device=device,dtype=dtype);inactive=1.0-active
        current_states=[];current_outputs=[];current_messages=[];current_local=[];current_input=[]
        state_patch=patch_plan.state.get(t);output_patch=patch_plan.output.get(t);message_patch=patch_plan.message.get(t)
        for cell in range(NUM_CELLS):
            message=torch.zeros(B,MESSAGE_DIM,device=device,dtype=dtype)
            for edge_index in incoming[cell]:
                edge=model.graph.edges[edge_index]
                source=prev_outputs[edge.src] if edge.recurrent or edge.src>=len(current_outputs) else current_outputs[edge.src]
                message=message+model.base.edge_weights[edge_index]*source
            if message_patch is not None:
                values=message_patch.values[:,cell,:] if message_patch.values.ndim==3 else message_patch.values
                mask=None
                if message_patch.mask is not None:
                    mask=message_patch.mask[cell] if message_patch.mask.ndim==2 else message_patch.mask[:,cell,:]
                message=_masked_replace(message,TensorPatch(values,mask))
            visible=model._visible_input(x_t,cell)
            local=snapshot[cell]@model.base.cell_ws[cell].T
            inp=visible@model.base.cell_wx[cell].T
            msg=message@model.base.cell_wm[cell].T
            candidate=torch.tanh(local+global_terms[cell]+matched_terms[cell]+msg+inp+model.base.cell_b[cell])
            proposed=gate*snapshot[cell]+(1.0-gate)*candidate
            state=active*proposed+inactive*snapshot[cell]
            if state_patch is not None:
                values=state_patch.values[:,cell,:] if state_patch.values.ndim==3 else state_patch.values
                mask=None
                if state_patch.mask is not None:
                    mask=state_patch.mask[cell] if state_patch.mask.ndim==2 else state_patch.mask[:,cell,:]
                patched=_masked_replace(state,TensorPatch(values,mask));state=active*patched+inactive*snapshot[cell]
            proposed_out=state@model.base.cell_wo[cell].T
            output=active*proposed_out+inactive*prev_outputs[cell]
            if output_patch is not None:
                values=output_patch.values[:,cell,:] if output_patch.values.ndim==3 else output_patch.values
                mask=None
                if output_patch.mask is not None:
                    mask=output_patch.mask[cell] if output_patch.mask.ndim==2 else output_patch.mask[:,cell,:]
                patched=_masked_replace(output,TensorPatch(values,mask));output=active*patched+inactive*prev_outputs[cell]
            current_states.append(state);current_outputs.append(output);current_messages.append(active*message);current_local.append(active*local);current_input.append(active*inp)
        prev_states=current_states;prev_outputs=current_outputs
        if return_trace:
            trace["states"].append(torch.stack(current_states,1));trace["outputs"].append(torch.stack(current_outputs,1));trace["messages"].append(torch.stack(current_messages,1));trace["global"].append(active.unsqueeze(1)*torch.stack(global_terms,1));trace["local"].append(torch.stack(current_local,1));trace["input"].append(torch.stack(current_input,1));trace["gate"].append(active*gate+inactive);trace["factor"].append(active*factor)
    prediction=torch.tanh(model.base.readout(torch.cat(prev_states,dim=1))).squeeze(-1)
    if not return_trace:return prediction
    return prediction,AF1DTrace(states=torch.stack(trace["states"],1),outputs=torch.stack(trace["outputs"],1),messages=torch.stack(trace["messages"],1),global_terms=torch.stack(trace["global"],1),local_terms=torch.stack(trace["local"],1),input_terms=torch.stack(trace["input"],1),gate=torch.stack(trace["gate"],1),coupling_factor4=torch.stack(trace["factor"],1))


def load_population_rows() -> list[dict]:
    return read_json(ROOT/SOURCE["v837ak_reconstruction"])["rows"]


def load_model_by_id(organism_id: str):
    row=next(r for r in load_population_rows() if r["organism_id"]==organism_id);model,source,checkpoint=load_reconstructed_model(row);return model,row,source,checkpoint


def verify_runtime_equivalence() -> dict:
    rows=load_population_rows();sample_seeds=[10000,10031,10127,10447]
    maxima={"prediction":0.0,"state":0.0,"output":0.0,"message":0.0,"global":0.0,"gate":0.0};checked=0
    for row in rows:
        model,_,_,_=load_model_by_id(row["organism_id"]);episodes=[instrument_episode(row["family"],s,"development").as_episode() for s in sample_seeds];obs,lengths,_=episodes_to_batch(episodes)
        with torch.no_grad():
            ref_pred,ref=model(obs,lengths,return_trace=True);got_pred,got=run_instrumented(model,obs,lengths,return_trace=True)
        pairs={"prediction":(got_pred,ref_pred),"state":(got.states,ref.states),"output":(got.outputs,ref.outputs),"message":(got.messages,ref.messages),"global":(got.global_terms,ref.global_recurrent_terms),"gate":(got.gate,ref.global_gates)}
        for key,(a,b) in pairs.items():
            delta=float(torch.max(torch.abs(a-b)).item());maxima[key]=max(maxima[key],delta)
            if delta>1e-6:raise RuntimeError(f"V837AN_INSTRUMENTED_RUNTIME_DRIFT:{row['organism_id']}:{key}:{delta}")
        checked+=len(sample_seeds)
    payload={"version":"V837an","pass":True,"organisms":len(rows),"episode_runs":checked,"max_abs":maxima,"intervention_hooks":["patch_state","patch_output","patch_message","patch_global_term","patch_gate"],"state_patch_semantics":"post-update state replacement -> patched output -> later same-step recipients -> recurrent next state","cross_organism_state_invertibility_required":False}
    write_json(HERE/"diagnostics/instrumented_runtime_equivalence.json",payload)
    write_json(HERE/"diagnostics/carrier_shapes.json",{"version":"V837an","STATE40":40,"OUTPUT40":40,"MESSAGE40":40,"GLOBAL40":40,"GATE1":1,"COUPLING_FACTOR4":4,"coupling_factor4_diagnostic_only":True})
    return payload


if __name__=="__main__":
    import json;print(json.dumps(verify_runtime_equivalence(),indent=2))
