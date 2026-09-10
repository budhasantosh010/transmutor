from __future__ import annotations

import json
import math
from typing import Any

from experiments.v837_primitive_invention.v837ao.episode_partitions import seeds as ao_seeds
from experiments.v837_primitive_invention.v837ao.k1_backend import evaluate_reader_on_partition, fit_backend
from experiments.v837_primitive_invention.v837ao.setpoint_eval import evaluate_setpoints

from .failure_ledger import add, make_entry
from .source_folds import freeze_source_folds
from .utils import HERE, ROOT, read_json, sha256_json, write_json

TOL=1e-9

def _numeric_diffs(a:Any,b:Any,prefix:str="")->list[dict]:
    out=[]
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(set(a)&set(b)): out.extend(_numeric_diffs(a[k],b[k],f"{prefix}.{k}" if prefix else k))
    elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        for i,(x,y) in enumerate(zip(a,b)): out.extend(_numeric_diffs(x,y,f"{prefix}[{i}]"))
    elif isinstance(a,(int,float)) and isinstance(b,(int,float)) and not isinstance(a,bool) and not isinstance(b,bool):
        if math.isfinite(float(a)) and math.isfinite(float(b)):
            out.append({"path":prefix,"expected":float(a),"actual":float(b),"abs_diff":abs(float(a)-float(b))})
    return out

def _anchor_rows(path:str)->dict:
    rows=read_json(ROOT/path)["rows"]
    return {(r["family"],r["organism_id"],r["variant"]):r for r in rows}

def reproduce_v837ao_anchor()->dict:
    folds=freeze_source_folds()
    reader_anchor=_anchor_rows("experiments/v837_primitive_invention/v837ao/raw/discovery_reader_results.json")
    set_anchor=_anchor_rows("experiments/v837_primitive_invention/v837ao/raw/discovery_setpoint_results.json")
    fit_anchor=_anchor_rows("experiments/v837_primitive_invention/v837ao/raw/discovery_backend_fits.json")
    rows=[];max_diff=0.0;drifts=[]
    for family,frow in folds["families"].items():
        for index,oid in enumerate(frow["discovery"],1):
            key=(family,oid,"B0_GLOBAL_K1")
            backend=fit_backend(oid,family,"B0_GLOBAL_K1",ao_seeds("AO_BACKEND_FIT"))
            reader=evaluate_reader_on_partition(backend,ao_seeds("AO_BACKEND_SELECT")) if backend.get("valid") else {"reader_pass":False,"algebra_pass":False,"phases":[]}
            setres=evaluate_setpoints(backend,ao_seeds("AO_BACKEND_SELECT")) if backend.get("valid") else {"pass":False,"sample_count":0}
            expected_fit=fit_anchor[key]; expected_reader=reader_anchor[key]; expected_set=set_anchor[key]
            compare_actual={"fit_components":backend.get("components"),"reader":reader,"setpoint":setres}
            compare_expected={"fit_components":expected_fit.get("components"),"reader":expected_reader,"setpoint":expected_set}
            diffs=_numeric_diffs(compare_expected,compare_actual)
            local=max((d["abs_diff"] for d in diffs),default=0.0);max_diff=max(max_diff,local)
            bad=[d for d in diffs if d["abs_diff"]>TOL]
            if bad: drifts.extend({"family":family,"organism_id":oid,**d} for d in bad[:20])
            rows.append({"family":family,"organism_id":oid,"engine":backend.get("engine"),"variant":"B0_GLOBAL_K1","max_abs_metric_diff":local,"metric_tolerance":TOL,"metric_count":len(diffs),"reader_reproduced":not any(d["path"].startswith("reader") and d["abs_diff"]>TOL for d in diffs),"setpoint_reproduced":not any(d["path"].startswith("setpoint") and d["abs_diff"]>TOL for d in diffs),"algebra_pass":reader.get("algebra_pass"),"reader_phases":reader.get("phases",[]),"setpoint_metrics":setres.get("metrics",{}),"pass":len(bad)==0})
            print(f"V837ap AP1 anchor {family} {index}/{len(frow['discovery'])} maxdiff={local:.3e}",flush=True)
    payload={"version":"V837ap","stage":"AP1_V837AO_NEGATIVE_ANCHOR","source_version":"V837ao","metric_tolerance":TOL,"rows":rows,"max_abs_metric_diff":max_diff,"drift_count":len(drifts),"drifts":drifts[:100],"pass":len(drifts)==0,"v837ao_diagnosis":"CAUSAL_DIRECTION_NOT_GLOBAL_COORDINATE","v837an_k1_causal_steering_remains_valid":True}
    write_json(HERE/"raw/v837ao_baseline_reproduction.json",payload);write_json(HERE/"diagnostics/v837ao_baseline_reproduction.json",payload)
    write_json(HERE/"raw/v837ao_negative_anchor_reproduction.json",payload);write_json(HERE/"diagnostics/v837ao_reproduction.json",payload)
    if drifts:
        add(make_entry(failure_id="V837ap-AP1-V837AO-BASELINE-DRIFT",stage="AP1",branch="V837ao_negative_anchor",family=None,organism=None,phase=None,carrier_dimension=1,chart_family="AFFINE",chart_degree_rank=1,writer_family="V837ao",fit_partition="V837ao AO_BACKEND_FIT",selection_partition="V837ao AO_BACKEND_SELECT",parameter_count=None,stored_bytes=None,mac_estimate=None,metrics={"max_abs_metric_diff":max_diff,"drift_count":len(drifts)},acceptance_gate={"metric_difference_max":TOL},failed_conditions=["metric difference >1e-9"],distance_from_threshold={"excess":max(0.0,max_diff-TOL)},scientific_interpretation="The frozen V837ao negative anchor did not reproduce exactly enough for V837ap science.",confounds_remaining=["environment/numerical drift"],result_status="ENGINEERING_BLOCKER",failure_type="ENGINEERING_FAILURE",next_justified_experiment="Resolve baseline drift before any V837ap nonlinear science.",artifact_paths=["experiments/v837_primitive_invention/v837ap/raw/v837ao_baseline_reproduction.json"]))
        raise RuntimeError("V837AP_V837AO_BASELINE_DRIFT")
    return payload

if __name__=="__main__": print(json.dumps(reproduce_v837ao_anchor(),indent=2,default=str))
