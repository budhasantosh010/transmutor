from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from experiments.v837_primitive_invention.v837ak.utils import sha256_json


def structural_signature(topology: dict, nodes: Sequence[int]) -> dict:
    ordered=tuple(sorted(int(n) for n in nodes)); node_set=set(ordered); local={n:i for i,n in enumerate(ordered)}; k=len(ordered)
    internal=[]
    incoming_same=[0]*k; incoming_rec=[0]*k; outgoing_same=[0]*k; outgoing_rec=[0]*k
    for edge in topology["edges"]:
        src,dst,rec=int(edge["src"]),int(edge["dst"]),bool(edge["recurrent"])
        src_in,dst_in=src in node_set,dst in node_set
        if src_in and dst_in:
            if not rec and local[src] >= local[dst]:
                raise RuntimeError("V837AK_SAME_STEP_ORDER_VIOLATION")
            internal.append({"src":local[src],"dst":local[dst],"recurrent":rec})
        elif (not src_in) and dst_in:
            (incoming_rec if rec else incoming_same)[local[dst]] += 1
        elif src_in and (not dst_in):
            (outgoing_rec if rec else outgoing_same)[local[src]] += 1
    internal.sort(key=lambda e:(e["dst"],e["src"],e["recurrent"]))
    return {
        "size":k,
        "relative_ordered_nodes":list(range(k)),
        "internal_same_step_edges":[[e["src"],e["dst"]] for e in internal if not e["recurrent"]],
        "internal_recurrent_edges":[[e["src"],e["dst"]] for e in internal if e["recurrent"]],
        "external_incoming_same_step_per_cell":incoming_same,
        "external_incoming_recurrent_per_cell":incoming_rec,
        "external_outgoing_same_step_per_cell":outgoing_same,
        "external_outgoing_recurrent_per_cell":outgoing_rec,
        "message_connected":bool(internal),
    }


def structural_class_id(signature: dict) -> str:
    return sha256_json(signature)


def parameter_signature(model, nodes: Sequence[int]) -> dict:
    ordered=tuple(sorted(int(n) for n in nodes)); s=set(ordered)
    internal=[]
    for i,edge in enumerate(model.graph.edges):
        if int(edge.src) in s and int(edge.dst) in s:
            internal.append(float(model.base.edge_weights[i].detach().cpu().item()))
    effective=[]
    for i in ordered:
        W,b=model.effective_candidate_input_map(i)
        effective.append({"cell_position":len(effective),"weight_norm":float(W.detach().norm().item()),"bias_norm":float(b.detach().norm().item())})
    return {
        "local_recurrent_norms":[float(model.base.cell_ws[i].detach().norm().item()) for i in ordered],
        "local_message_norms":[float(model.base.cell_wm[i].detach().norm().item()) for i in ordered],
        "local_input_norms":[float(model.base.cell_wx[i].detach().norm().item()) for i in ordered],
        "local_output_norms":[float(model.base.cell_wo[i].detach().norm().item()) for i in ordered],
        "projection_norms":[float(model.cell_candidate_projections[i].weight.detach().norm().item()) for i in ordered],
        "internal_edge_weight_vector":internal,
        "effective_candidate_input_maps":effective,
    }
