from __future__ import annotations

import time
from .adapter_families import FAMILIES, fit_vector_map, fit_gate_map, apply_np, apply_gate_np
from .interface_ports import collect_interface_trace, port_matrices, trace_integrity
from .utils import HERE, read_json, seeds, write_json

FIT=seeds(10000,10063);SELECT=seeds(10064,10127)


def _fit_bundle(rec,don,rec_nodes,don_nodes,family,rec_select=None,don_select=None):
    trace_integrity(rec,don);rp=port_matrices(rec,rec_nodes);dp=port_matrices(don,don_nodes);ports={}
    for port in ("state","external_messages","global_term","projected_input"):
        ports[port]=[fit_vector_map(x,y,family,require_invertible=(port=="state")) for x,y in zip(rp[port],dp[port])]
    # Outgoing communication direction is donor -> recipient.
    ports["output"]=[fit_vector_map(x,y,family) for x,y in zip(dp["output"],rp["output"])]
    ports["gate"]=fit_gate_map(rp["gate"][0],dp["gate"][0],family)
    if rec_select is not None and don_select is not None:
        trace_integrity(rec_select,don_select);rs=port_matrices(rec_select,rec_nodes);ds=port_matrices(don_select,don_nodes)
        for port in ("state","external_messages","global_term","projected_input"):
            for m,x,y in zip(ports[port],rs[port],ds[port]):
                m["select_mse"]=float(((apply_np(x,m)-y)**2).mean())
        for m,x,y in zip(ports["output"],ds["output"],rs["output"]):
            m["select_mse"]=float(((apply_np(x,m)-y)**2).mean())
        ports["gate"]["select_mse"]=float(((apply_gate_np(rs["gate"][0],ports["gate"])-ds["gate"][0])**2).mean())
    valid=all(m.get("valid",True) for p in ports.values() for m in (p if isinstance(p,list) else [p]))
    return {"family":family,"ports":ports,"valid":valid}


def fit_all_adapters():
    pairs=read_json(HERE/"raw/frozen_pairs.json")["pairs"]
    cached=HERE/"raw/adapter_fits.json"
    if cached.is_file():
        prior=read_json(cached)
        if len(prior.get("rows",[]))==len(pairs) and prior.get("gradient_steps_used_for_adapters")==0:return prior
    rows=[];t0=time.perf_counter();cpu0=time.process_time()
    for pi,pair in enumerate(pairs):
        rec=pair["recipient"];same=pair["same_class_donor"];diff=pair["different_class_donor"];rn=rec["nodes"]
        rt=collect_interface_trace(rec["organism_id"],rn,FIT,"development");rsel=collect_interface_trace(rec["organism_id"],rn,SELECT,"development")
        row={"pair_index":pi,"class_id":pair["class_id"],"primary_causal":pair["primary_causal"],"recipient":rec,"same_class_donor":same,"different_class_donor":diff,"families":{}}
        st=collect_interface_trace(same["organism_id"],same["nodes"],FIT,"development");dt=collect_interface_trace(diff["organism_id"],diff["nodes"],FIT,"development")
        ssel=collect_interface_trace(same["organism_id"],same["nodes"],SELECT,"development");dsel=collect_interface_trace(diff["organism_id"],diff["nodes"],SELECT,"development")
        for fam in FAMILIES:
            row["families"][fam]={"same":_fit_bundle(rt,st,rn,same["nodes"],fam,rsel,ssel),"different":_fit_bundle(rt,dt,rn,diff["nodes"],fam,rsel,dsel)}
        rows.append(row)
    payload={"version":"V837al","fit_seeds":[10000,10063],"fit_split":"development","gradient_steps_used_for_adapters":0,"rows":rows,"cpu_seconds":time.process_time()-cpu0,"wall_seconds":time.perf_counter()-t0}
    write_json(HERE/"raw/adapter_fits.json",payload)
    cond=[]
    for r in rows:
        for fam,x in r["families"].items():
            for role,b in x.items():
                for port,maps in b["ports"].items():
                    for li,m in enumerate(maps if isinstance(maps,list) else [maps]):cond.append({"pair_index":r["pair_index"],"family":fam,"role":role,"port":port,"position":li,"valid":m.get("valid",True),"fit_mse":m.get("fit_mse"),"select_mse":m.get("select_mse"),**m.get("diagnostics",{})})
    write_json(HERE/"diagnostics/adapter_conditioning.json",{"version":"V837al","rows":cond});return payload

if __name__=="__main__": fit_all_adapters()
