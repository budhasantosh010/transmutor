from __future__ import annotations

import json
import numpy as np

from experiments.v837_primitive_invention.v837ao.phase_backends import PHASES

from .chart_data import phase_dataset, pooled_phase_dataset
from .projected_causal_spaces import reconstruct_space
from .projected_charts import fit_projected_chart
from .scalar_charts import fit_scalar_chart
from .tangent_field import fit_tangent_field

BINARY={"conditional_routing","delayed_recall"}


def _fit_chart(h,z,chart_family,k,binary):
    if int(k)==1:
        chart=fit_scalar_chart(np.asarray(h).reshape(-1),np.asarray(z),chart_family,binary)
    else:
        chart=fit_projected_chart(np.asarray(h),np.asarray(z),chart_family,binary)
    if binary and int(k)==1:
        x=np.asarray(h).reshape(-1);zz=np.asarray(z);chart["prototype_minus"]=float(np.median(x[zz<0])) if np.any(zz<0) else None;chart["prototype_plus"]=float(np.median(x[zz>0])) if np.any(zz>0) else None;chart["prototype_separation"]=None if chart["prototype_minus"] is None or chart["prototype_plus"] is None else abs(chart["prototype_plus"]-chart["prototype_minus"])
    return chart


def _fit_tangent(oid,family,q,writer_seeds,kind):
    hb=[];hc=[];zb=[];zc=[]
    for phase in PHASES[family]:
        ch=phase_dataset(oid,family,writer_seeds,q,phase)
        if len(ch["base_states"]):
            hb.append(ch["base_states"]@q);hc.append(ch["cf_states"]@q);zb.append(ch["base_semantic"]);zc.append(ch["cf_semantic"])
    if not hb:return {"valid":False,"failure_code":"NO_TANGENT_OBSERVATIONS","kind":kind}
    return fit_tangent_field(np.concatenate(hb),np.concatenate(hc),np.concatenate(zb),np.concatenate(zc),kind)


def compile_holdout_geometry(organism_id:str,family:str,spec:dict,chart_seeds:list[int],writer_seeds:list[int])->dict:
    k=int(spec["k"]);binary=family in BINARY;space=reconstruct_space(organism_id,family,chart_seeds,k)
    if not space.get("valid"):
        return {"version":"V837ap","organism_id":organism_id,"family":family,"valid":False,"failure_code":"HELDOUT_SUBSPACE_FIT_INVALID","k":k,"chart_family":spec["chart_family"],"calibration_n":len(chart_seeds)}
    q=np.asarray(space["q"],dtype=np.float64);geometry={"version":"V837ap","organism_id":organism_id,"family":family,"engine":space["engine"],"k":k,"chart_family":spec["chart_family"],"writer_family":spec["writer_family"],"phase_atlas":bool(spec["phase_atlas"]),"semantic_dimension":1,"valid":True,"historical_v837an_geometry_loaded":False,"historical_v837ao_geometry_loaded":False,"cross_organism_alignment":False,"calibration_n":len(chart_seeds),"chart_fit_seed_range":[min(chart_seeds),max(chart_seeds)],"writer_fit_seed_range":[min(writer_seeds),max(writer_seeds)],"q":q.tolist(),"subspace_sha256":space["subspace_sha256"]}
    try:
        if geometry["phase_atlas"]:
            charts={}
            for phase in PHASES[family]:
                ch=phase_dataset(organism_id,family,chart_seeds,q,phase)
                if len(ch["semantic"])<2:raise RuntimeError(f"INSUFFICIENT_PHASE_CALIBRATION:{phase}")
                charts[phase]=_fit_chart(ch["h"],ch["semantic"],spec["chart_family"],k,binary)
            geometry["charts_by_phase"]=charts
        else:
            ch=pooled_phase_dataset(organism_id,family,chart_seeds,q)
            if len(ch["semantic"])<2:raise RuntimeError("INSUFFICIENT_GLOBAL_CALIBRATION")
            geometry["chart"]=_fit_chart(ch["h"],ch["semantic"],spec["chart_family"],k,binary)
        if str(spec["writer_family"]).startswith("TANGENT_"):
            kind=str(spec["writer_family"])[len("TANGENT_"):]
            field=_fit_tangent(organism_id,family,q,writer_seeds,kind)
            geometry["tangent_field"]=field
            if not field.get("valid"):raise RuntimeError(field.get("failure_code","TANGENT_FIELD_INVALID"))
    except Exception as exc:
        geometry["valid"]=False;geometry["failure_code"]="HELDOUT_GEOMETRY_FIT_INVALID";geometry["error"]=f"{type(exc).__name__}:{exc}"
    geometry["parameter_count"]=int(sum(int(c.get("parameter_count",0)) for c in geometry.get("charts_by_phase",{}).values()) if geometry.get("phase_atlas") else geometry.get("chart",{}).get("parameter_count",0))+int(geometry.get("tangent_field",{}).get("parameter_count",0))
    geometry["stored_bytes"]=len(json.dumps({k:v for k,v in geometry.items() if k!="q"},sort_keys=True,separators=(",",":"),default=str).encode("utf-8"))
    geometry["backend_gradient_steps"]=0
    return geometry
