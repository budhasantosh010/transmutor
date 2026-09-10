from __future__ import annotations

import json
import math
import numpy as np

from .authorization import POWERED_FAMILIES
from .candidate_selection import candidate_order,family_gate
from .chart_data import pooled_phase_dataset,phase_dataset
from .chart_reader import read_chart,reader_metrics
from .data_roles import seeds
from .projected_causal_spaces import space_map
from .projected_charts import fit_projected_chart
from .scalar_charts import fit_scalar_chart
from .setpoint_grid import family_grid,freeze_target_grids
from .source_folds import freeze_source_folds
from .utils import HERE,sha256_json,write_json

_DATA_CACHE: dict[tuple[str,str,int,str],dict] = {}

def _cached_dataset(oid:str,family:str,k:int,partition:str,q:np.ndarray)->dict:
    key=(oid,family,int(k),partition)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key]=pooled_phase_dataset(oid,family,seeds(partition),q)
    return _DATA_CACHE[key]


def _fit_one(oid:str,family:str,cand:dict,q:np.ndarray)->dict:
    binary=family in {"conditional_routing","delayed_recall"};fit=_cached_dataset(oid,family,cand["k"],"AP_CHART_FIT",q);sel=_cached_dataset(oid,family,cand["k"],"AP_CHART_SELECT",q)
    if len(fit["semantic"])<8 or len(sel["semantic"])<8:return {**cand,"organism_id":oid,"family":family,"valid":False,"failure_code":"INSUFFICIENT_CHART_SAMPLES"}
    try:
        chart=fit_scalar_chart(fit["h"].reshape(-1),fit["semantic"],cand["chart_family"],binary) if cand["k"]==1 else fit_projected_chart(fit["h"],fit["semantic"],cand["chart_family"],binary)
    except Exception as exc:
        return {**cand,"organism_id":oid,"family":family,"valid":False,"failure_code":"CHART_FIT_INVALID","error":f"{type(exc).__name__}:{exc}"}
    if chart.get("derivative_sign_consistent") is False:return {**cand,"organism_id":oid,"family":family,"valid":False,"failure_code":"NONMONOTONIC_CANONICAL_CHART","chart":chart}
    pred=read_chart(chart,sel["h"] if cand["k"]>1 else sel["h"].reshape(-1));grid=family_grid(family);metrics=reader_metrics(pred,sel["semantic"],grid["semantic_range"],binary)
    phase_metrics=[]
    for phase in sel["chunks"]:
        ch=sel["chunks"][phase];pp=read_chart(chart,ch["h"] if cand["k"]>1 else ch["h"].reshape(-1)) if len(ch["semantic"]) else np.empty(0);pm=reader_metrics(pp,ch["semantic"],grid["semantic_range"],binary) if len(ch["semantic"]) else {"pass":False,"n":0};phase_metrics.append({"phase":phase,**pm})
    coverage=1.0
    if cand["k"]==1 and not binary:
        lo,hi=sorted([float(chart.get("fit_z_min",-np.inf)),float(chart.get("fit_z_max",np.inf))]);coverage=float(np.mean((sel["semantic"]>=lo)&(sel["semantic"]<=hi)))
    metrics["chart_coverage"]=coverage;metrics["coverage_pass"]=coverage>=.95;reader_pass=bool(metrics["pass"] and coverage>=.95 and all(pm.get("pass",False) for pm in phase_metrics))
    if binary and cand["k"]==1:
        h=fit["h"].reshape(-1);z=fit["semantic"];chart["prototype_minus"]=float(np.median(h[z<0])) if np.any(z<0) else None;chart["prototype_plus"]=float(np.median(h[z>0])) if np.any(z>0) else None;chart["prototype_separation"]=None if chart["prototype_minus"] is None or chart["prototype_plus"] is None else abs(chart["prototype_plus"]-chart["prototype_minus"])
    stored=len(json.dumps(chart,sort_keys=True,separators=(",",":")).encode("utf-8"));
    return {**cand,"organism_id":oid,"family":family,"valid":True,"chart":chart,"reader_metrics":metrics,"phase_reader_metrics":phase_metrics,"reader_pass":reader_pass,"fit_examples":len(fit["semantic"]),"select_examples":len(sel["semantic"]),"parameter_count":int(chart.get("parameter_count",0)),"stored_bytes":stored,"semantic_dimension":1,"cross_organism_alignment":False}

def _reader_gate_distance(row:dict)->float:
    m=row.get("reader_metrics",{})
    if not row.get("valid",False) or not m:
        return 1e6
    if row.get("family") in {"conditional_routing","delayed_recall"}:
        vals=[max(0.0,float(m.get("normalized_rmse",99.0))/.15-1.0),max(0.0,.95-float(m.get("classification_accuracy",0.0)))/.95,max(0.0,.95-float(m.get("balanced_accuracy",0.0)))/.95]
    else:
        vals=[max(0.0,float(m.get("normalized_rmse",99.0))/.10-1.0),max(0.0,.95-abs(float(m.get("pearson",0.0))))/.95,max(0.0,.95-float(m.get("spearman",0.0)))/.95]
    vals.append(max(0.0,.95-float(m.get("chart_coverage",0.0)))/.95)
    return float(sum(vals))

def _summarize_rows(rows:list[dict],family:str)->list[dict]:
    out=[]
    for cand in candidate_order(family):
        crows=[r for r in rows if r.get("family")==family and int(r.get("k",-1))==int(cand["k"]) and r.get("chart_family")==cand["chart_family"] and not r.get("phase_atlas",False)]
        gate=family_gate(crows,"reader_pass")
        out.append({**cand,"reader_family_gate":gate,"reader_family_pass":gate["pass"],"median_reader_nrmse":float(np.median([r.get("reader_metrics",{}).get("normalized_rmse",99.) for r in crows])) if crows else 99.0,"median_reader_gate_distance":float(np.median([_reader_gate_distance(r) for r in crows])) if crows else 1e6})
    return out

def run_discovery_reader_ladder()->dict:
    freeze_target_grids();folds=freeze_source_folds();spaces=space_map();rows=[];summaries={}
    for family in POWERED_FAMILIES:
        ids=folds["families"][family]["discovery"];summaries[family]=[]
        for cand in candidate_order(family):
            crows=[]
            for idx,oid in enumerate(ids,1):
                sr=spaces[(family,oid,cand["k"])];q=np.asarray(sr["q"],dtype=np.float64);r=_fit_one(oid,family,cand,q);r["engine"]=sr["engine"];rows.append(r);crows.append(r)
                print(f"V837ap reader {family} k{cand['k']} {cand['chart_family']} {idx}/{len(ids)} pass={r.get('reader_pass',False)}",flush=True)
            gate=family_gate(crows,"reader_pass");summaries[family].append({**cand,"reader_family_gate":gate,"reader_family_pass":gate["pass"],"median_reader_nrmse":float(np.median([r.get("reader_metrics",{}).get("normalized_rmse",99.) for r in crows]))})
    payload={"version":"V837ap","stage":"AP3_AP4_READER_DISCOVERY","fit_partition":"AP_CHART_FIT","selection_partition":"AP_CHART_SELECT","rows":rows,"family_candidate_summaries":summaries,"heldout_opened":False}
    write_json(HERE/"raw/nonlinear_k1_fit.json",{"version":"V837ap","rows":[r for r in rows if r["k"]==1]});write_json(HERE/"raw/projected_chart_fit.json",{"version":"V837ap","rows":[r for r in rows if r["k"]>1]});write_json(HERE/"diagnostics/chart_conditioning.json",payload)
    write_json(HERE/"diagnostics/chart_coverage.json",{"version":"V837ap","rows":[{"family":r["family"],"organism_id":r["organism_id"],"k":r["k"],"chart_family":r["chart_family"],"coverage":r.get("reader_metrics",{}).get("chart_coverage"),"pass":r.get("reader_metrics",{}).get("coverage_pass",False)} for r in rows]})
    write_json(HERE/"diagnostics/chart_monotonicity.json",{"version":"V837ap","rows":[{"family":r["family"],"organism_id":r["organism_id"],"k":r["k"],"chart_family":r["chart_family"],"derivative_sign_consistent":r.get("chart",{}).get("derivative_sign_consistent",True),"valid":r.get("valid",False)} for r in rows]})
    return payload


def repair_k1_rows()->dict:
    """Recompute only K1 rows after a deterministic implementation repair."""
    p=HERE/"diagnostics/chart_conditioning.json"
    if not p.is_file():
        raise RuntimeError("V837AP_REPAIR_REQUIRES_COMPLETED_READER_SWEEP")
    payload=json.loads(p.read_text(encoding="utf-8"));folds=freeze_source_folds();spaces=space_map()
    kept=[r for r in payload["rows"] if int(r.get("k",-1))!=1];fresh=[]
    for r in kept:
        if r.get("valid",False):
            r["reader_pass"]=bool(r.get("reader_metrics",{}).get("pass",False) and r.get("reader_metrics",{}).get("coverage_pass",False) and all(pm.get("pass",False) for pm in r.get("phase_reader_metrics",[])))
    for family in POWERED_FAMILIES:
        ids=folds["families"][family]["discovery"]
        for cand in [c for c in candidate_order(family) if int(c["k"])==1]:
            for idx,oid in enumerate(ids,1):
                sr=spaces[(family,oid,1)];q=np.asarray(sr["q"],dtype=np.float64);r=_fit_one(oid,family,cand,q);r["engine"]=sr["engine"];fresh.append(r)
                print(f"V837ap K1 repair {family} {cand['chart_family']} {idx}/{len(ids)} pass={r.get('reader_pass',False)}",flush=True)
    rows=kept+fresh
    summaries={family:_summarize_rows(rows,family) for family in POWERED_FAMILIES}
    out={**payload,"rows":rows,"family_candidate_summaries":summaries,"engineering_repair":{"scope":"K1 only","reason":"1D array orientation in polynomial/logistic chart fit/predict","k2_k4_k8_reused":True}}
    write_json(HERE/"raw/nonlinear_k1_fit.json",{"version":"V837ap","rows":[r for r in rows if int(r["k"])==1]});write_json(HERE/"raw/projected_chart_fit.json",{"version":"V837ap","rows":[r for r in rows if int(r["k"])>1]});write_json(HERE/"diagnostics/chart_conditioning.json",out)
    write_json(HERE/"diagnostics/chart_coverage.json",{"version":"V837ap","rows":[{"family":r["family"],"organism_id":r["organism_id"],"k":r["k"],"chart_family":r["chart_family"],"coverage":r.get("reader_metrics",{}).get("chart_coverage"),"pass":r.get("reader_metrics",{}).get("coverage_pass",False)} for r in rows]})
    write_json(HERE/"diagnostics/chart_monotonicity.json",{"version":"V837ap","rows":[{"family":r["family"],"organism_id":r["organism_id"],"k":r["k"],"chart_family":r["chart_family"],"derivative_sign_consistent":r.get("chart",{}).get("derivative_sign_consistent",True),"valid":r.get("valid",False)} for r in rows]})
    return out

if __name__=="__main__":print(json.dumps(run_discovery_reader_ladder(),indent=2,default=str))
