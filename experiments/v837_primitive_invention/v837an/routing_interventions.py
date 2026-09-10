from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .global_contributions import decompose_global_source_contributions
from .instrumented_af1d import PatchPlan, TensorPatch, run_instrumented
from .message_contributions import decompose_message_contributions
from .phase_masks import routing_window

GATE_CHANNEL="GATE"


@dataclass
class RoutingNaturalDecomposition:
    message: torch.Tensor       # [N,T,E,4]
    global_source: torch.Tensor # [N,T,10,10,4]
    message_ids: list[str]
    windows: list[list[int]]


def channel_universe(model,message_ids:list[str]|None=None)->list[str]:
    mids=message_ids if message_ids is not None else [f"M:e{i}:{int(e.src)}->{int(e.dst)}:R{int(bool(e.recurrent))}" for i,e in enumerate(model.graph.edges)]
    return list(mids)+[f"G:{i}" for i in range(10)]+[GATE_CHANNEL]


def decompose_natural(model,trace,pairs)->RoutingNaturalDecomposition:
    lengths=torch.as_tensor([len(p.base_episode.observations) for p in pairs],dtype=torch.long,device=trace.outputs.device)
    m=decompose_message_contributions(model,trace,lengths=lengths);g=decompose_global_source_contributions(model,trace,lengths=lengths)
    return RoutingNaturalDecomposition(m.contributions,g.contributions,m.edge_ids,[routing_window(p.base_episode) for p in pairs])


def _candidate_replacements(base_trace,cf_trace,base_dec:RoutingNaturalDecomposition,cf_dec:RoutingNaturalDecomposition,sets:list[tuple[str,...]],*,reverse:bool)->dict[int,dict[str,torch.Tensor]]:
    # Results are per timestep and candidate: [C,N,...]. Only episodes whose
    # frozen task-specific propagation window contains t are later masked in.
    C=len(sets);N,T=base_trace.gate.shape[:2];out={}
    source_trace=cf_trace if reverse else base_trace;target_trace=base_trace if reverse else cf_trace
    source_dec=cf_dec if reverse else base_dec;target_dec=base_dec if reverse else cf_dec
    for t in range(T):
        message=[];global_term=[];gate=[]
        for channels in sets:
            m=source_trace.messages[:,t].clone();g=source_trace.global_terms[:,t].clone();ga=source_trace.gate[:,t].clone()
            for ch in channels:
                if ch.startswith("M:"):
                    ei=source_dec.message_ids.index(ch);dst=int(ch.split(":")[2].split("->")[1])
                    m[:,dst,:]+=target_dec.message[:,t,ei,:]-source_dec.message[:,t,ei,:]
                elif ch.startswith("G:"):
                    cell=int(ch.split(":")[1]);g+=target_dec.global_source[:,t,cell,:,:]-source_dec.global_source[:,t,cell,:,:]
                elif ch==GATE_CHANNEL:
                    ga=target_trace.gate[:,t].clone()
                else:raise KeyError(ch)
            message.append(m);global_term.append(g);gate.append(ga)
        out[t]={"message":torch.stack(message),"global":torch.stack(global_term),"gate":torch.stack(gate)}
    return out


def routing_predictions(model,observations:torch.Tensor,lengths:torch.Tensor,base_trace,cf_trace,base_dec:RoutingNaturalDecomposition,cf_dec:RoutingNaturalDecomposition,sets:list[tuple[str,...]],pairs,*,reverse:bool=False)->np.ndarray:
    C=len(sets);N=len(pairs);source_obs=observations.repeat((C,1,1));source_lengths=lengths.repeat(C);rep=_candidate_replacements(base_trace,cf_trace,base_dec,cf_dec,sets,reverse=reverse);plan=PatchPlan()
    windows=[routing_window(p.counterfactual_episode if reverse else p.base_episode) for p in pairs]
    for t in range(observations.shape[1]):
        episode_mask=torch.as_tensor([t in w for w in windows],dtype=torch.bool)
        if not bool(episode_mask.any()):continue
        mask_n=episode_mask.repeat(C);vals=rep[t]
        plan.message[t]=TensorPatch(vals["message"].reshape(C*N,10,4),mask_n[:,None,None].expand(C*N,10,4))
        plan.global_term[t]=TensorPatch(vals["global"].reshape(C*N,10,4),mask_n[:,None,None].expand(C*N,10,4))
        plan.gate[t]=TensorPatch(vals["gate"].reshape(C*N,1),mask_n[:,None])
    with torch.no_grad():pred=run_instrumented(model,source_obs,source_lengths,patch_plan=plan,return_trace=False)
    return pred.detach().cpu().numpy().reshape(C,N).astype(np.float64)


def routing_forward_reverse(data:dict,sets:list[tuple[str,...]],indices:np.ndarray|None=None)->tuple[np.ndarray,np.ndarray,RoutingNaturalDecomposition,RoutingNaturalDecomposition]:
    if indices is None:indices=np.arange(len(data["pairs"]));indices=np.asarray(indices,dtype=np.int64)
    # Decompose full natural traces once, then slice everything consistently.
    bd=decompose_natural(data["model"],data["base_trace"],data["pairs"]);cd=decompose_natural(data["model"],data["cf_trace"],data["pairs"])
    def trslice(tr):
        from .instrumented_af1d import AF1DTrace
        return AF1DTrace(tr.states[indices],tr.outputs[indices],tr.messages[indices],tr.global_terms[indices],tr.local_terms[indices],tr.input_terms[indices],tr.gate[indices],tr.coupling_factor4[indices])
    btr=trslice(data["base_trace"]);ctr=trslice(data["cf_trace"])
    bdec=RoutingNaturalDecomposition(bd.message[indices],bd.global_source[indices],bd.message_ids,[bd.windows[i] for i in indices]);cdec=RoutingNaturalDecomposition(cd.message[indices],cd.global_source[indices],cd.message_ids,[cd.windows[i] for i in indices])
    pairs=[data["pairs"][i] for i in indices];bo=data["base_obs"][indices];bl=data["base_lengths"][indices];co=data["cf_obs"][indices];cl=data["cf_lengths"][indices]
    f=routing_predictions(data["model"],bo,bl,btr,ctr,bdec,cdec,sets,pairs,reverse=False);r=routing_predictions(data["model"],co,cl,btr,ctr,bdec,cdec,sets,pairs,reverse=True);return f,r,bdec,cdec
