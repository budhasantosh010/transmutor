from __future__ import annotations

import json,os,time
from concurrent.futures import ThreadPoolExecutor,as_completed

from .authorization import assert_authorized
from .boundary_influence import freeze_rankings
from .context_replay import fit_pair_roles,evaluate_pair,aggregate
from .expanded_primitive import expanded_pair,ranking_index
from .failure_ledger import add,make_entry
from .utils import HERE,read_json,seeds,write_json

FIT=seeds(10256,10319);SELECT=seeds(10320,10383);CACHE=HERE/"raw/cache/am_c";PORTS={"state","external_messages","global_term","projected_input","gate","output"}
def configs():
    out=[{"config_id":f"C-TEMPORAL-H{h}","kind":"TEMPORAL","boundary":"ORIGINAL","history":h,"diagnostic_only":False} for h in (2,4,8)]
    out += [{"config_id":f"C-INTERACTION-COMBINED+{n}","kind":"INTERACTION","boundary":f"COMBINED+{n}","history":1,"diagnostic_only":False} for n in (1,2,4)]
    out += [{"config_id":"C-WHOLE_SYSTEM-INTERACTION","kind":"INTERACTION","boundary":"WHOLE_SYSTEM","history":1,"diagnostic_only":True}]
    if len(out)!=7:raise RuntimeError("V837AM_C_CONFIG_COUNT")
    return out
def _pair(p,cfg,idx):return p if cfg["boundary"]=="ORIGINAL" else expanded_pair(p,cfg["boundary"],idx)
def _one(cfg,pairs,idx):
    cp=CACHE/"select"/f"{cfg['config_id'].replace('+','p')}.json"
    if cp.is_file():return json.loads(cp.read_text(encoding="utf-8"))
    rows=[];invalid=[];maxparams=maxmacs=0
    for pi,p0 in enumerate(pairs):
        p=_pair(p0,cfg,idx);fp=CACHE/"fits"/f"{cfg['config_id'].replace('+','p')}__p{pi:02d}.json";fit=fit_pair_roles(p,"CONTEXT_MOD_R2","C5_FULL_LOCAL_CONTEXT",cfg["history"],FIT,SELECT,fp)
        if any(not b["valid"] for b in fit["roles"].values()):invalid.append(pi);continue
        rows.append(evaluate_pair(p,fit,PORTS,"C5_FULL_LOCAL_CONTEXT",cfg["history"],SELECT));maxparams=max(maxparams,fit["roles"]["same"]["parameters"]);maxmacs=max(maxmacs,fit["roles"]["same"]["macs"])
    agg=aggregate(rows);valid=not invalid and len(rows)==len(pairs);effect=valid and bool(agg.get("pass"));proper=effect and not cfg["diagnostic_only"];added=0 if cfg["boundary"]=="ORIGINAL" else (10-len(pairs[0]["recipient"]["nodes"]) if cfg["boundary"]=="WHOLE_SYSTEM" else int(cfg["boundary"].split('+')[1]));out={**cfg,"rows":rows,"aggregate":agg,"valid":valid,"invalid_pairs":invalid,"effect_pass":effect,"pass":proper,"added_cells":added,"adapter_parameters":maxparams,"adapter_macs":maxmacs};cp.parent.mkdir(parents=True,exist_ok=True);cp.write_text(json.dumps(out,sort_keys=True,separators=(",",":")),encoding="utf-8");return out
def run():
    assert_authorized();freeze_rankings();pairs=read_json(HERE/"raw/frozen_pairs.json")["pairs"];idx=ranking_index();cfgs=configs();t0=time.perf_counter();cpu0=time.process_time();workers=min(7,os.cpu_count() or 1);by={}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut={ex.submit(_one,c,pairs,idx):c for c in cfgs}
        for f in as_completed(fut):r=f.result();by[r["config_id"]]=r
    rows=[by[c["config_id"]] for c in cfgs];passing=[r for r in rows if r["pass"]]
    def cost(r):return (r["added_cells"],r["adapter_macs"],r["adapter_parameters"],r["history"],r["config_id"])
    w=min(passing,key=cost) if passing else None;payload={"version":"V837am","stage":"AM-C","configs_evaluated":7,"fit_seeds":[10256,10319],"selection_seeds":[10320,10383],"pass_count":len(passing),"whole_system_effect_pass_count":sum(r['diagnostic_only'] and r['effect_pass'] for r in rows),"winner":None if w is None else {k:w[k] for k in ("config_id","kind","boundary","history","added_cells","adapter_parameters","adapter_macs","aggregate")},"results":[{k:r[k] for k in ("config_id","kind","boundary","history","diagnostic_only","valid","effect_pass","pass","added_cells","adapter_parameters","adapter_macs","aggregate","invalid_pairs")} for r in rows],"cpu_seconds":time.process_time()-cpu0,"wall_seconds":time.perf_counter()-t0};write_json(HERE/"raw/am_c_selection.json",payload);write_json(HERE/"diagnostics/temporal_context.json",payload)
    for r in rows:
        if r["pass"]:continue
        failed=["DYNAMIC_STATE_MAP_NONINVERTIBLE"] if not r['valid'] else (["WHOLE_SYSTEM_DIAGNOSTIC_NOT_PROMOTABLE"] if r['diagnostic_only'] and r['effect_pass'] else ["AM_C_SELECT_GATE_FAIL"])
        add(make_entry(failure_id=f"V837am-{r['config_id']}",version="V837am",stage="AM-C",hypothesis="Temporal context or a predeclared boundary×context interaction makes the causal motif portable.",why="Test H4 and the interaction fallback before rejecting operator equivalence.",implementation={"kind":r['kind'],"boundary":r['boundary'],"history":r['history'],"adapter":"CONTEXT_MOD_R2","context":"C5_FULL_LOCAL_CONTEXT","ports":"P6_ALL"},data="AM-C FIT 10256-10319; SELECT 10320-10383",fit_seeds=[10256,10319],selection_seeds=[10320,10383],controls=["SAME_CLASS","DIFFERENT_CLASS_REFIT","RANDOMIZED_REFIT","RANDOMIZED_FIXED_ADAPTER"],params=r['adapter_parameters'],macs=r['adapter_macs'],metrics=r['aggregate'],gate={"output_ratio":.75,"state_ratio":.85,"beat_both":.60,"random_refit_ratio_min":1.20},failed=failed,distance={"beat_both_shortfall":max(0,.60-r['aggregate'].get('beat_both_fraction',0))},status="definitive within tested scope" if r['valid'] else "provisional configuration-invalid",failure_type="SCIENTIFIC_FAILURE",ruled=["current-only static linear maps","predeclared temporal/boundary interaction configuration"],remaining=["meta-confirmed surviving branch if any","operator nonequivalence if all proper branches fail"],meaning="This exact AM-C fallback does not establish a proper reusable primitive under the frozen gate.",uncertainty="Other AM-C configs or branch winners may remain.",next_action="Proceed to meta-confirmation after all branches close",reproduce="python scripts/reproduce_v837_recovery.py --variant v837am --stage temporal",artifacts=["experiments/v837_primitive_invention/v837am/raw/am_c_selection.json"]))
    return payload
if __name__=="__main__":print(json.dumps(run(),indent=2)[:4000])
