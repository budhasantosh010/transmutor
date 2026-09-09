from __future__ import annotations

import json,os,time
from concurrent.futures import ThreadPoolExecutor,as_completed

from .authorization import AM_B_BOUNDARIES,assert_authorized
from .boundary_influence import freeze_rankings
from .context_replay import fit_pair_roles,evaluate_pair,aggregate
from .expanded_primitive import expanded_pair,ranking_index
from .failure_ledger import add,make_entry
from .utils import HERE,read_json,seeds,sha256_file,write_json

FIT=seeds(10128,10191);SELECT=seeds(10192,10255);CACHE=HERE/"raw/cache/am_b"
def configs():
    x=[]
    for b in AM_B_BOUNDARIES:
        x.append({"config_id":f"B-{b}-IDENTITY","boundary":b,"adapter":"IDENTITY"});x.append({"config_id":f"B-{b}-FULL_AFFINE_ALL_PORTS","boundary":b,"adapter":"FULL_AFFINE_ALL_PORTS"})
    if len(x)!=20:raise RuntimeError("V837AM_B_CONFIG_COUNT")
    return x
def _dummy(pair):
    seed=int(pair["recipient"]["organism_id"][:8],16)%2_000_000_000;return {"random_seed":seed,"roles":{"same":{"ports":{},"valid":True,"parameters":0,"macs":0},"different_refit":{"ports":{},"valid":True,"parameters":0,"macs":0},"randomized_refit":{"ports":{},"valid":True,"parameters":0,"macs":0}}}
def _fit(pi,p,cfg):
    if cfg["adapter"]=="IDENTITY":return _dummy(p)
    cp=CACHE/"fits"/f"{cfg['boundary'].replace('+','p')}__p{pi:02d}.json";return fit_pair_roles(p,"STATIC_FULL_AFFINE","NONE",1,FIT,SELECT,cp)
def _one(cfg,pairs,idx):
    cp=CACHE/"select"/f"{cfg['config_id'].replace('+','p')}.json"
    if cp.is_file():return json.loads(cp.read_text(encoding="utf-8"))
    rows=[];maxparams=maxmacs=0
    for pi,p0 in enumerate(pairs):
        p=expanded_pair(p0,cfg["boundary"],idx);fit=_fit(pi,p,cfg);ports=set() if cfg["adapter"]=="IDENTITY" else {"state","external_messages","global_term","projected_input","gate","output"};rows.append(evaluate_pair(p,fit,ports,"NONE",1,SELECT));maxparams=max(maxparams,fit["roles"]["same"].get("parameters",0));maxmacs=max(maxmacs,fit["roles"]["same"].get("macs",0))
    agg=aggregate(rows);diagnostic=cfg["boundary"]=="WHOLE_SYSTEM";proper_pass=bool(agg.get("pass") and not diagnostic);added=10-len(pairs[0]["recipient"]["nodes"]) if diagnostic else int(cfg["boundary"].split('+')[1]);out={**cfg,"rows":rows,"aggregate":agg,"effect_pass":bool(agg.get("pass")),"pass":proper_pass,"diagnostic_only":diagnostic,"added_cells":added,"adapter_parameters":maxparams,"adapter_macs":maxmacs};cp.parent.mkdir(parents=True,exist_ok=True);cp.write_text(json.dumps(out,sort_keys=True,separators=(",",":")),encoding="utf-8");return out
def run():
    assert_authorized();freeze_rankings();pairs=read_json(HERE/"raw/frozen_pairs.json")["pairs"];idx=ranking_index();cfgs=configs();t0=time.perf_counter();cpu0=time.process_time();workers=min(8,os.cpu_count() or 1);by={}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut={ex.submit(_one,c,pairs,idx):c for c in cfgs}
        for f in as_completed(fut):r=f.result();by[r["config_id"]]=r
    rows=[by[c["config_id"]] for c in cfgs];passing=[r for r in rows if r["pass"]]
    tie={"MESSAGE":0,"GLOBAL":1,"COMBINED":2,"WHOLE_SYSTEM":3}
    def cost(r):typ=r["boundary"].split('+')[0];return (r["added_cells"],0 if r["adapter"]=="IDENTITY" else 1,r["adapter_macs"],r["adapter_parameters"],tie[typ],r["config_id"])
    w=min(passing,key=cost) if passing else None;payload={"version":"V837am","stage":"AM-B","configs_evaluated":20,"fit_seeds":[10128,10191],"selection_seeds":[10192,10255],"pass_count":len(passing),"whole_system_effect_pass_count":sum(r['diagnostic_only'] and r['effect_pass'] for r in rows),"winner":None if w is None else {k:w[k] for k in ("config_id","boundary","adapter","added_cells","adapter_parameters","adapter_macs","aggregate")},"results":[{k:r[k] for k in ("config_id","boundary","adapter","aggregate","effect_pass","pass","diagnostic_only","added_cells","adapter_parameters","adapter_macs")} for r in rows],"cpu_seconds":time.process_time()-cpu0,"wall_seconds":time.perf_counter()-t0};write_json(HERE/"raw/am_b_selection.json",payload);write_json(HERE/"diagnostics/boundary_expansion.json",payload)
    for r in rows:
        if r["pass"]:continue
        reason=["WHOLE_SYSTEM_DIAGNOSTIC_NOT_PROMOTABLE"] if r["diagnostic_only"] and r["effect_pass"] else (["AM_B_SELECT_GATE_FAIL"] if not r['aggregate'].get('pass') else ["NOT_PROMOTABLE"])
        add(make_entry(failure_id=f"V837am-{r['config_id']}",version="V837am",stage="AM-B",hypothesis="The V837ak motif boundary is too narrow and a minimal influence-ranked spatial expansion restores static interoperability.",why="Test H2 independently of context-conditioned adapters.",implementation={"boundary":r['boundary'],"adapter":r['adapter']},data="AM-B FIT 10128-10191; SELECT 10192-10255",fit_seeds=[10128,10191],selection_seeds=[10192,10255],controls=["SAME_CLASS","DIFFERENT_CLASS_REFIT","RANDOMIZED_REFIT","RANDOMIZED_FIXED_ADAPTER"],params=r['adapter_parameters'],macs=r['adapter_macs'],metrics=r['aggregate'],gate={"output_ratio":.75,"state_ratio":.85,"beat_both":.60,"whole_system_not_promotable":True},failed=reason,distance={"beat_both_shortfall":max(0,.60-r['aggregate'].get('beat_both_fraction',0))},status="definitive within tested scope",failure_type="SCIENTIFIC_FAILURE",ruled=["frozen influence ranking","equal expansion/control capacity"],remaining=["temporal history","context×boundary interaction"],meaning="This exact spatial boundary/static-interface configuration does not establish a reusable proper primitive.",uncertainty="AM-C remains predeclared.",next_action="Continue AM-C",reproduce="python scripts/reproduce_v837_recovery.py --variant v837am --stage boundary",artifacts=["experiments/v837_primitive_invention/v837am/raw/am_b_selection.json"]))
    return payload
if __name__=="__main__":print(json.dumps(run(),indent=2)[:4000])
