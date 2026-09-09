from __future__ import annotations

import json,os,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

from .authorization import CONTEXT_SETS,PORT_BUNDLES,assert_authorized
from .context_replay import fit_pair_roles,evaluate_pair,aggregate,enabled_bundle_cost
from .failure_ledger import add,make_entry
from .utils import HERE,read_json,seeds,sha256_file,write_json

FIT=seeds(10000,10063);SELECT=seeds(10064,10127);CACHE=HERE/"raw/cache/am_a"
FAMS=("CONTEXT_ADDITIVE","CONTEXT_MOD_R1","CONTEXT_MOD_R2","CONTEXT_MOD_R4")

def configs():
    out=[]
    for b,ports in PORT_BUNDLES.items():out.append({"config_id":f"A-STATIC-{b}","family":"STATIC_FULL_AFFINE","context_set":"NONE","bundle":b,"ports":ports,"history":1})
    for c in CONTEXT_SETS:
        for f in FAMS:
            for b,ports in PORT_BUNDLES.items():out.append({"config_id":f"A-{f}-{c}-{b}","family":f,"context_set":c,"bundle":b,"ports":ports,"history":1})
    if len(out)!=126 or len({x['config_id'] for x in out})!=126:raise RuntimeError("V837AM_A_CONFIG_COUNT")
    return out

def _fit_key(fam,ctx):return f"{fam}__{ctx}"
def run_fit():
    pairs=read_json(HERE/"raw/frozen_pairs.json")["pairs"];combos=[("STATIC_FULL_AFFINE","NONE")]+[(f,c) for c in CONTEXT_SETS for f in FAMS];summ=[];t0=time.perf_counter();cpu0=time.process_time()
    for pi,p in enumerate(pairs):
        for fam,ctx in combos:
            cp=CACHE/"fits"/f"p{pi:02d}__{_fit_key(fam,ctx)}.json";fit=fit_pair_roles(p,fam,ctx,1,FIT,SELECT,cp);roles={r:{"valid":b["valid"],"parameters":b["parameters"],"macs":b["macs"]} for r,b in fit["roles"].items()};summ.append({"pair_index":pi,"family":fam,"context_set":ctx,"cache_path":str(cp.relative_to(HERE)).replace('\\','/'),"cache_sha256":sha256_file(cp),"roles":roles})
    payload={"version":"V837am","stage":"AM-A-FIT","fit_seeds":[10000,10063],"select_conditioning_seeds":[10064,10127],"analytic_adapter_fits":len(summ)*3,"ridge_solves":"deterministic analytic","gradient_steps":0,"rows":summ,"cpu_seconds":time.process_time()-cpu0,"wall_seconds":time.perf_counter()-t0};write_json(HERE/"raw/am_a_adapter_fits.json",payload);return payload
def _load_fit(pi,cfg):return json.loads((CACHE/"fits"/f"p{pi:02d}__{_fit_key(cfg['family'],cfg['context_set'])}.json").read_text(encoding="utf-8"))
def _one(cfg,pairs):
    cp=CACHE/"select"/f"{cfg['config_id']}.json"
    if cp.is_file():
        prior=json.loads(cp.read_text(encoding="utf-8"))
        if prior.get("science_schema")==2:return prior
    rows=[];invalid=[]
    for pi,pair in enumerate(pairs):
        fit=_load_fit(pi,cfg);bad=[role for role,b in fit["roles"].items() if not b["valid"] and "state" in cfg["ports"]]
        if bad:invalid.append({"pair_index":pi,"roles":bad});continue
        rows.append(evaluate_pair(pair,fit,set(cfg["ports"]),cfg["context_set"],1,SELECT))
    agg=aggregate(rows);valid=(not invalid and len(rows)==len(pairs));passed=valid and agg.get("pass",False);costs=[enabled_bundle_cost(_load_fit(pi,cfg)["roles"]["same"],set(cfg["ports"])) for pi in range(len(pairs))];params=max([x["parameters"] for x in costs] or [0]);macs=max([x["macs"] for x in costs] or [0]);out={"science_schema":2,**cfg,"rows":rows,"aggregate":agg,"valid":valid,"invalid_state_maps":invalid,"pass":bool(passed),"adapter_parameters":params,"adapter_macs":macs};cp.parent.mkdir(parents=True,exist_ok=True);cp.write_text(json.dumps(out,sort_keys=True,separators=(",",":")),encoding="utf-8");return out
def _fail_conditions(r):
    a=r["aggregate"];f=[]
    if not r["valid"]:f.append("DYNAMIC_STATE_MAP_NONINVERTIBLE")
    if not a.get("effect_gate_pass",False):f.append("SELECT_EFFECT_GATE_FAIL")
    if not a.get("adapter_cheating_gate_pass",False):f.append("ADAPTER_LEARNS_COMPUTATION_NOT_INTERFACE")
    return f or ["SELECT_PASS_FALSE"]
def run_select():
    assert_authorized();pairs=read_json(HERE/"raw/frozen_pairs.json")["pairs"];cfgs=configs();final=HERE/"raw/am_a_selection.json"
    if final.is_file():
        prior=read_json(final)
        if prior.get("science_schema")==2 and prior.get("configs_evaluated")==126:return prior
    t0=time.perf_counter();cpu0=time.process_time();workers=min(8,os.cpu_count() or 1);by={}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut={ex.submit(_one,c,pairs):c for c in cfgs}
        for f in as_completed(fut):r=f.result();by[r["config_id"]]=r
    rows=[by[c["config_id"]] for c in cfgs];passing=[r for r in rows if r["pass"]]
    def cost(r):return (r["adapter_macs"],r["adapter_parameters"],len(r["ports"]),r["config_id"])
    winner=min(passing,key=cost) if passing else None
    payload={"version":"V837am","science_schema":2,"stage":"AM-A","configs_evaluated":126,"selection_seeds":[10064,10127],"pass_count":len(passing),"winner":None if winner is None else {k:winner[k] for k in ("config_id","family","context_set","bundle","ports","history","adapter_parameters","adapter_macs","aggregate")},"results":[{k:r[k] for k in ("config_id","family","context_set","bundle","ports","history","valid","pass","adapter_parameters","adapter_macs","aggregate","invalid_state_maps")} for r in rows],"cpu_seconds":time.process_time()-cpu0,"wall_seconds":time.perf_counter()-t0};write_json(final,payload);write_json(HERE/"diagnostics/context_adapter_conditioning.json",{"version":"V837am","invalid_config_count":sum(not r['valid'] for r in rows)});write_json(HERE/"diagnostics/adapter_capacity_controls.json",{"version":"V837am","adapter_cheating_fail_count":sum(not r['aggregate'].get('adapter_cheating_gate_pass',False) for r in rows),"configs":126})
    for r in rows:
        if r["pass"]:continue
        a=r["aggregate"];same=a.get("same_output");diff=a.get("different_output");rnd=a.get("random_refit_output");failed=_fail_conditions(r);dist={"output_ratio_to_different":None if not diff else same/diff,"output_ratio_to_random":None if not rnd else same/rnd,"beat_both_shortfall":max(0,.60-a.get('beat_both_fraction',0)),"random_refit_ratio_shortfall":None if not same else max(0,1.20-(rnd/same if rnd is not None else 0))}
        add(make_entry(failure_id=f"V837am-SCHEMA2-{r['config_id']}",version="V837am",stage="AM-A",hypothesis="Context-conditioned boundary translation makes the powered causal motif interoperable.",why="Test H1 without reopening static-linear scopes.",implementation={"science_schema":2,**{k:r[k] for k in ("family","context_set","bundle","ports","history")}},data="AM-A FIT 10000-10063; SELECT 10064-10127",fit_seeds=[10000,10063],selection_seeds=[10064,10127],controls=["SAME_CLASS","DIFFERENT_CLASS_REFIT","RANDOMIZED_REFIT","RANDOMIZED_FIXED_ADAPTER"],params=r["adapter_parameters"],macs=r["adapter_macs"],metrics=a,gate={"output_ratio":.75,"state_ratio":.85,"beat_both":.60,"random_refit_ratio_min":1.20},failed=failed,distance=dist,status="definitive within tested scope" if r['valid'] else "provisional configuration-invalid",failure_type="SCIENTIFIC_FAILURE",ruled=["same pairing/data budget","equal-capacity refit controls"],remaining=["wider spatial boundary","temporal history","boundary×context interaction"],meaning="This exact corrected schema-2 conditioned interface configuration does not satisfy the frozen AM-A selection standard.",uncertainty="Other predeclared V837am branches remain.",next_action="Continue AM-B/AM-C under frozen gate",reproduce="python scripts/reproduce_v837_recovery.py --variant v837am --stage context",artifacts=["experiments/v837_primitive_invention/v837am/raw/am_a_selection.json"]))
    return payload

def run():run_fit();return run_select()
if __name__=="__main__":print(json.dumps(run(),indent=2)[:4000])
