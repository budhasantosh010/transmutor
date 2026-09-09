from __future__ import annotations

import os,time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .adapter_families import FAMILIES,map_dof_macs
from .pairwise_replay import SELECT,evaluate_pair,aggregate
from .utils import HERE,read_json,write_json

PORTS=("state","external_messages","global_term","projected_input","gate","output")


def configurations():
    out=[{"config_id":"IDENTITY-000000","family":"IDENTITY","scope":"000000","ports":[]}]
    for n in range(1,64):
        scope=f"{n:06b}";ports=[p for p,b in zip(PORTS,scope) if b=="1"]
        for fam in FAMILIES:out.append({"config_id":f"{fam}-{scope}","family":fam,"scope":scope,"ports":ports})
    if len(out)!=253 or len({x["config_id"] for x in out})!=253:raise RuntimeError("V837AL_CONFIG_GRID_INVALID")
    return out


def complexity(family,scope,k):
    if family=="IDENTITY":return {"continuous_dof":0,"adapter_macs_per_timestep":0,"enabled_ports":0}
    dof=macs=0
    for p,b in zip(PORTS,scope):
        if b!="1":continue
        if p=="gate":dof+=(0 if family=="SIGNED_PERMUTATION" else 2);macs+=1
        else:
            d=6 if p=="projected_input" else 4;pd,pm=map_dof_macs(family,d);mult=k
            if p=="state":pm*=2
            dof+=pd*mult;macs+=pm*mult
    return {"continuous_dof":dof,"adapter_macs_per_timestep":macs,"enabled_ports":scope.count("1")}


def _valid(bundle,scope):
    if not bundle:return True
    for p,b in zip(PORTS,scope):
        if b!="1":continue
        maps=bundle["ports"][p];maps=maps if isinstance(maps,list) else [maps]
        if any(not m.get("valid",True) for m in maps):return False
    return True


def _evaluate_config(cfg,pairs,fits,cache):
    cp=cache/f"{cfg['config_id']}.json"
    if cp.is_file():return read_json(cp)
    rows=[]
    for pi,pair in enumerate(pairs):
        if cfg["family"]=="IDENTITY":bundles={}
        else:
            f=fits[pi]["families"][cfg["family"]];bundles={"same":f["same"],"different":f["different"]}
            if not _valid(bundles["same"],cfg["scope"]) or not _valid(bundles["different"],cfg["scope"]):continue
        rows.append(evaluate_pair(pair,bundles,cfg["scope"],SELECT,"development"))
    causal=[r for r in rows if r["primary_causal"]];global_gate=aggregate(rows,False);causal_gate=aggregate(causal,False);comp_global=complexity(cfg["family"],cfg["scope"],3)
    row={**cfg,"rows":rows,"global_track":global_gate,"causal_track":causal_gate,"complexity":comp_global,"valid_pair_count":len(rows),"causal_pair_count":len(causal)}
    write_json(cp,row);return row


def run_scope_localization():
    final=HERE/"raw/scope_selection_results.json"
    if final.is_file():
        prior=read_json(final)
        if prior.get("configs_evaluated")==253:return prior
    pairs=read_json(HERE/"raw/frozen_pairs.json")["pairs"];fits={r["pair_index"]:r for r in read_json(HERE/"raw/adapter_fits.json")["rows"]};configs=configurations();cache=HERE/"raw/cache/select_configs";cache.mkdir(parents=True,exist_ok=True);t0=time.perf_counter();cpu0=time.process_time();by_id={}
    pending=[]
    for cfg in configs:
        cp=cache/f"{cfg['config_id']}.json"
        if cp.is_file():by_id[cfg['config_id']]=read_json(cp)
        else:pending.append(cfg)
    workers=min(8,max(1,os.cpu_count() or 1),max(1,len(pending)))
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(_evaluate_config,cfg,pairs,fits,cache):cfg for cfg in pending}
            for future in as_completed(futures):
                row=future.result();by_id[row['config_id']]=row
    results=[by_id[cfg['config_id']] for cfg in configs]
    payload={"version":"V837al","stage":"AL5","configs_evaluated":len(results),"selection_seeds":[10064,10127],"parallel_workers":workers,"results":results,"cpu_seconds":time.process_time()-cpu0,"wall_seconds":time.perf_counter()-t0}
    write_json(HERE/"raw/scope_selection_results.json",payload);write_json(HERE/"diagnostics/scope_localization.json",{"version":"V837al","configs_evaluated":len(results),"global_pass_count":sum(r["global_track"]["pass"] for r in results),"causal_pass_count":sum(r["causal_track"]["pass"] for r in results)});write_json(HERE/"diagnostics/transform_family_localization.json",{"version":"V837al","families":{f:{"global_pass":sum(r["global_track"]["pass"] for r in results if r["family"]==f),"causal_pass":sum(r["causal_track"]["pass"] for r in results if r["family"]==f)} for f in ("IDENTITY",)+FAMILIES}});return payload

if __name__=="__main__":run_scope_localization()
