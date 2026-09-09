from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from experiments.v837_primitive_invention.v837ak.authorization import assert_v837ak_authorized
from experiments.v837_primitive_invention.v837ak.boundary_traces import boundary_streams, internal_raw_messages, run_full_probe
from experiments.v837_primitive_invention.v837ak.reconstruct_organisms import load_reconstructed_model
from experiments.v837_primitive_invention.v837ak.utils import DIRECTED, RANDOM, HERE, occurrence_id, read_json, write_json


@dataclass
class PrimitiveReplay:
    states: torch.Tensor
    candidates: torch.Tensor
    outputs: torch.Tensor
    internal_messages: torch.Tensor


class PortedPrimitive:
    """AF1D subsystem with full-organism coupling/controller exposed as boundary ports."""
    def __init__(self, model, nodes: Sequence[int]):
        self.nodes = tuple(sorted(int(n) for n in nodes))
        self.node_set = set(self.nodes)
        self.ws = [model.base.cell_ws[i].detach().cpu().clone() for i in self.nodes]
        self.wm = [model.base.cell_wm[i].detach().cpu().clone() for i in self.nodes]
        self.wx = [model.base.cell_wx[i].detach().cpu().clone() for i in self.nodes]
        self.b = [model.base.cell_b[i].detach().cpu().clone() for i in self.nodes]
        self.wo = [model.base.cell_wo[i].detach().cpu().clone() for i in self.nodes]
        if model.cell_candidate_projections is None:
            raise RuntimeError("V837ak port requires AF1D de-shared candidate projections")
        self.proj_w = [model.cell_candidate_projections[i].weight.detach().cpu().clone() for i in self.nodes]
        self.proj_b = [model.cell_candidate_projections[i].bias.detach().cpu().clone() for i in self.nodes]
        self.internal_edges = []
        for edge_index, edge in enumerate(model.graph.edges):
            if int(edge.src) in self.node_set and int(edge.dst) in self.node_set:
                self.internal_edges.append({
                    "src": int(edge.src), "dst": int(edge.dst), "recurrent": bool(edge.recurrent),
                    "weight": model.base.edge_weights[edge_index].detach().cpu().clone(),
                })
        self.node_to_local = {n:i for i,n in enumerate(self.nodes)}

    def replay(self, streams: dict[str, torch.Tensor]) -> PrimitiveReplay:
        observations = streams["observations"].detach().cpu(); lengths = streams["lengths"].detach().cpu()
        external = streams["external_messages"].detach().cpu(); global_terms = streams["global_terms"].detach().cpu(); gates = streams["gates"].detach().cpu()
        batch, steps = observations.shape[:2]; k = len(self.nodes); dtype = observations.dtype
        prev_states = [torch.zeros(batch, 4, dtype=dtype) for _ in range(k)]
        prev_outputs = [torch.zeros(batch, 4, dtype=dtype) for _ in range(k)]
        states=[]; candidates=[]; outputs=[]; internal_messages=[]
        for t in range(steps):
            x_t = observations[:, t, :]
            current_states=[]; current_outputs=[]; current_candidates=[]; current_internal=[]
            for li, original in enumerate(self.nodes):
                message = torch.zeros(batch, 4, dtype=dtype)
                for edge in self.internal_edges:
                    if edge["dst"] != original: continue
                    src_local = self.node_to_local[edge["src"]]
                    source = prev_outputs[src_local] if edge["recurrent"] or src_local >= len(current_outputs) else current_outputs[src_local]
                    message = message + edge["weight"] * source
                total_message = external[:, t, li, :] + message
                projected = F.linear(x_t, self.proj_w[li], self.proj_b[li])
                local_term = prev_states[li] @ self.ws[li].T
                message_term = total_message @ self.wm[li].T
                input_term = projected @ self.wx[li].T
                candidate = torch.tanh(local_term + global_terms[:, t, li, :] + message_term + input_term + self.b[li])
                gate = gates[:, t, :]
                proposed_state = gate * prev_states[li] + (1.0 - gate) * candidate
                proposed_output = proposed_state @ self.wo[li].T
                active = (t < lengths).to(dtype).unsqueeze(1)
                state = active * proposed_state + (1.0-active) * prev_states[li]
                output = active * proposed_output + (1.0-active) * prev_outputs[li]
                current_states.append(state); current_outputs.append(output); current_candidates.append(candidate); current_internal.append(active * message)
            prev_states=current_states; prev_outputs=current_outputs
            states.append(torch.stack(current_states,dim=1)); candidates.append(torch.stack(current_candidates,dim=1)); outputs.append(torch.stack(current_outputs,dim=1)); internal_messages.append(torch.stack(current_internal,dim=1))
        return PrimitiveReplay(states=torch.stack(states,dim=1), candidates=torch.stack(candidates,dim=1), outputs=torch.stack(outputs,dim=1), internal_messages=torch.stack(internal_messages,dim=1))

    def replay_teacher_forced(self, streams: dict[str, torch.Tensor], recipient_states: torch.Tensor) -> PrimitiveReplay:
        """Apply this occurrence as a one-step operator on recipient previous states."""
        observations=streams["observations"].detach().cpu(); lengths=streams["lengths"].detach().cpu(); external=streams["external_messages"].detach().cpu(); global_terms=streams["global_terms"].detach().cpu(); gates=streams["gates"].detach().cpu()
        recipient_states=recipient_states.detach().cpu(); batch,steps=observations.shape[:2]; k=len(self.nodes); dtype=observations.dtype
        prev_rec=torch.zeros_like(recipient_states); prev_rec[:,1:]=recipient_states[:,:-1]
        states=[]; candidates=[]; outputs=[]; internal_messages=[]
        for t in range(steps):
            x_t=observations[:,t,:]; current_states=[]; current_outputs=[]; current_candidates=[]; current_internal=[]
            prev_states=[prev_rec[:,t,li,:] for li in range(k)]; prev_outputs=[prev_states[li]@self.wo[li].T for li in range(k)]
            for li,original in enumerate(self.nodes):
                message=torch.zeros(batch,4,dtype=dtype)
                for edge in self.internal_edges:
                    if edge["dst"]!=original: continue
                    src_local=self.node_to_local[edge["src"]]
                    source=prev_outputs[src_local] if edge["recurrent"] or src_local>=len(current_outputs) else current_outputs[src_local]
                    message=message+edge["weight"]*source
                total=external[:,t,li,:]+message; projected=F.linear(x_t,self.proj_w[li],self.proj_b[li]); local=prev_states[li]@self.ws[li].T; msg=total@self.wm[li].T; inp=projected@self.wx[li].T
                candidate=torch.tanh(local+global_terms[:,t,li,:]+msg+inp+self.b[li]); gate=gates[:,t,:]; proposed=gate*prev_states[li]+(1.0-gate)*candidate; proposed_out=proposed@self.wo[li].T
                active=(t<lengths).to(dtype).unsqueeze(1); state=active*proposed+(1.0-active)*prev_states[li]; output=active*proposed_out+(1.0-active)*prev_outputs[li]
                current_states.append(state); current_outputs.append(output); current_candidates.append(candidate); current_internal.append(active*message)
            states.append(torch.stack(current_states,1)); outputs.append(torch.stack(current_outputs,1)); candidates.append(torch.stack(current_candidates,1)); internal_messages.append(torch.stack(current_internal,1))
        return PrimitiveReplay(states=torch.stack(states,1),candidates=torch.stack(candidates,1),outputs=torch.stack(outputs,1),internal_messages=torch.stack(internal_messages,1))


def _internal_edge_count(topology: dict, nodes: Sequence[int]) -> tuple[int,int]:
    s=set(nodes); edges=[e for e in topology["edges"] if int(e["src"]) in s and int(e["dst"]) in s]
    return len(edges), sum(bool(e["recurrent"]) for e in edges)


def calibration_occurrences() -> list[dict]:
    recon = read_json(HERE / "raw/reconstruction_results.json")
    competent=[r for r in recon["rows"] if r["competent"]]
    directed=[r for r in competent if r["engine"]==DIRECTED]; random=[r for r in competent if r["engine"]==RANDOM]
    selected=[]
    for k in range(1,11):
        for source, maximize in ((directed,True),(random,False)):
            row=source[(k-1)%len(source)]
            model, source_row, _ = load_reconstructed_model(row)
            combos=list(itertools.combinations(range(10),k))
            combos.sort(key=lambda nodes: (_internal_edge_count(source_row["topology"],nodes), nodes), reverse=maximize)
            nodes=combos[0]
            ec,rc=_internal_edge_count(source_row["topology"],nodes)
            selected.append({"organism_id":row["organism_id"],"engine":row["engine"],"family":row["family"],"run_index":row["run_index"],"nodes":list(nodes),"size":k,"internal_edge_count":ec,"recurrent_edge_count":rc,"message_connected":ec>0,"occurrence_id":occurrence_id(row["organism_id"],nodes)})
    return selected


def run_replay_gate() -> dict:
    assert_v837ak_authorized()
    recon=read_json(HERE / "raw/reconstruction_results.json")
    if recon.get("complete") is not True or recon.get("organisms_reconstructed") != 50:
        raise RuntimeError("AK2 blocked: AK0 reconstruction incomplete")
    rows={r["organism_id"]:r for r in recon["rows"]}
    occurrences=calibration_occurrences(); trace_cache={}; results=[]
    seeds=list(range(20000,20016))
    for occurrence in occurrences:
        oid=occurrence["organism_id"]; row=rows[oid]
        model,_,_=load_reconstructed_model(row)
        if oid not in trace_cache: trace_cache[oid]=run_full_probe(oid,seeds,cache_key="replay16")
        trace=trace_cache[oid]; nodes=occurrence["nodes"]
        streams=boundary_streams(model,trace,nodes); primitive=PortedPrimitive(model,nodes); replay=primitive.replay(streams)
        idx=list(nodes); internal=internal_raw_messages(model,trace,nodes)
        active=(torch.arange(trace.states.shape[1]).view(1,-1)<trace.lengths.view(-1,1))
        # Candidate values on padded timesteps are not part of the episode and
        # the historical trace masks its boundary terms there; compare the
        # exact operator only on active timesteps. State/output continuity is
        # still checked over the full padded tensor.
        errors={
            "candidate_max_abs_error":float(torch.max(torch.abs(replay.candidates[active]-trace.candidates[:,:,idx,:][active])).item()),
            "state_max_abs_error":float(torch.max(torch.abs(replay.states-trace.states[:,:,idx,:])).item()),
            "output_max_abs_error":float(torch.max(torch.abs(replay.outputs-trace.outputs[:,:,idx,:])).item()),
            "internal_message_max_abs_error":float(torch.max(torch.abs(replay.internal_messages-internal)).item()),
        }
        passed=max(errors.values()) <= 1e-6
        results.append({**occurrence,**errors,"pass":passed})
        if not passed:
            write_json(HERE/"diagnostics/ported_replay_gate.json",{"version":"V837ak","pass":False,"diagnosis":"PORTED_PRIMITIVE_REPLAY_INVALID","occurrences":results})
            raise RuntimeError("PORTED_PRIMITIVE_REPLAY_INVALID")
    payload={"version":"V837ak","stage":"AK2","pass":True,"tolerance":1e-6,"episodes_per_occurrence":16,"occurrences_tested":len(results),"sizes":sorted(set(r["size"] for r in results)),"occurrences":results,"max_error":max(max(r[k] for k in ("candidate_max_abs_error","state_max_abs_error","output_max_abs_error","internal_message_max_abs_error")) for r in results)}
    write_json(HERE/"diagnostics/ported_replay_gate.json",payload); return payload


def main()->int:
    payload=run_replay_gate(); print(json.dumps({"pass":payload["pass"],"occurrences":payload["occurrences_tested"],"max_error":payload["max_error"]},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
