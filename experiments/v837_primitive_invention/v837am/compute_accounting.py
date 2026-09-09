from __future__ import annotations
from experiments.v837_primitive_invention.v837ak.ported_primitive import PortedPrimitive
from .interface_data import load_model

def primitive_accounting(occ):
    m,_=load_model(occ["organism_id"]);p=PortedPrimitive(m,occ["nodes"]);params=0;macs=0
    for li in range(len(p.nodes)):
        for t in (p.ws[li],p.wm[li],p.wx[li],p.b[li],p.wo[li],p.proj_w[li],p.proj_b[li]):params+=t.numel()
        macs+=p.ws[li].numel()+p.wm[li].numel()+p.wx[li].numel()+p.wo[li].numel()+p.proj_w[li].numel()
    params+=len(p.internal_edges);macs+=4*len(p.internal_edges)
    return {"primitive_parameters":int(params),"primitive_macs_per_timestep":int(macs),"primitive_cells":len(p.nodes)}
