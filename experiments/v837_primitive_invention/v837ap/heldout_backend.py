from __future__ import annotations

import json
import numpy as np

from experiments.v837_primitive_invention.v837ao.phase_backends import PHASES

from .chart_data import phase_dataset, pooled_phase_dataset
from .chart_features import feature_count
from .data_roles import seeds
from .projected_causal_spaces import reconstruct_space
from .projected_charts import fit_projected_chart
from .scalar_charts import fit_scalar_chart
from .tangent_field import fit_tangent_field
from .utils import sha256_json

BINARY={"conditional_routing","delayed_recall"}


def chart_coefficient_count(chart_family:str,k:int)->int:
    if k==1:
        return {
            "AFFINE":2,
            "POLYNOMIAL_2":3,
            "POLYNOMIAL_3":4,
            "MONOTONE_PWL_4":8,
            "MONOTONE_PWL_8":16,
            "MONOTONE_PCHIP_6":12,
            "LOGISTIC_MONOTONE":2,
            "ISOTONIC_4":8,
            "ISOTONIC_8":16,
        }[chart_family]
    degree={"LINEAR":1,"QUADRATIC":2,"CUBIC":3}[chart_family]
    return feature_count(int(k),degree)


def tangent_coefficient_count(writer_family:str,k:int)->int:
    if not str(writer_family).startswith("TANGENT_"):
        return 0
    kind=str(writer_family).replace("TANGENT_","")
    degree={"CONSTANT":0,"AFFINE_STATE_FIELD":1,"QUADRATIC_STATE_FIELD":2}[kind]
    return int(k)*feature_count(int(k),degree)


def _add_binary_prototypes(chart:dict,h:np.ndarray,z:np.ndarray)->dict:
    if int(chart.get("k",1))==1:
        x=np.asarray(h,dtype=np.float64).reshape(-1);zz=np.asarray(z,dtype=np.float64).reshape(-1)
        chart["prototype_minus"]=float(np.median(x[zz<0])) if np.any(zz<0) else None
        chart["prototype_plus"]=float(np.median(x[zz>0])) if np.any(zz>0) else None
        chart["prototype_separation"]=None if chart["prototype_minus"] is None or chart["prototype_plus"] is None else abs(chart["prototype_plus"]-chart["prototype_minus"])
    return chart


def _fit_chart(h,z,chart_family,k,binary):
    chart=fit_scalar_chart(np.asarray(h).reshape(-1),z,chart_family,binary) if int(k)==1 else fit_projected_chart(h,z,chart_family,binary)
    return _add_binary_prototypes(chart,h,z) if binary else chart


def compile_heldout_backend(organism_id:str,family:str,frozen_family:dict,calibration_n:int)->dict:
    """Compile exactly the frozen geometry on a held-out organism.

    No historical V837an/V837ao per-organism q/backend metadata is read here.
    Qk is reconstructed from checkpoint + permitted calibration traces only.
    """
    spec=frozen_family["candidate"];k=int(spec["k"]);chart_family=spec["chart_family"];writer_family=spec.get("writer_family","AUTO");atlas=bool(spec.get("phase_atlas",False));binary=family in BINARY
    chart_fit=seeds("AP_CHART_FIT")[:int(calibration_n)];writer_fit=seeds("AP_WRITER_FIT")[:int(calibration_n)]
    space=reconstruct_space(organism_id,family,chart_fit,k)
    if not space.get("valid"):
        return {"version":"V837ap","organism_id":organism_id,"family":family,"calibration_n":calibration_n,"valid":False,"underdetermined":False,"failure_code":"HELDOUT_SUBSPACE_RECONSTRUCTION_FAIL","historical_backend_loaded":False}
    q=np.asarray(space["q"],dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != k or int(np.linalg.matrix_rank(q)) < k:
        return {"version":"V837ap","organism_id":organism_id,"family":family,"engine":space.get("engine"),"calibration_n":calibration_n,"valid":False,"underdetermined":True,"failure_code":"UNDERDETERMINED_SUBSPACE_AT_N","required_independent_subspace_directions":k,"recovered_subspace_directions":0 if q.ndim!=2 else int(q.shape[1]),"historical_backend_loaded":False,"q_reconstructed_from_calibration":True}
    coeff_per_chart=chart_coefficient_count(chart_family,k)
    if atlas:
        phase_samples={}
        for phase in PHASES[family]:
            d=phase_dataset(organism_id,family,chart_fit,q,phase);phase_samples[phase]=int(len(d["semantic"]))
        if any(n < 2*coeff_per_chart for n in phase_samples.values()):
            return {"version":"V837ap","organism_id":organism_id,"family":family,"engine":space.get("engine"),"calibration_n":calibration_n,"valid":False,"underdetermined":True,"failure_code":"UNDERDETERMINED_AT_N","required_per_phase":2*coeff_per_chart,"phase_samples":phase_samples,"historical_backend_loaded":False,"q_reconstructed_from_calibration":True}
        charts={}
        for phase in PHASES[family]:
            d=phase_dataset(organism_id,family,chart_fit,q,phase);charts[phase]=_fit_chart(d["h"],d["semantic"],chart_family,k,binary)
        geometry={"version":"V837ap","branch":"AP_D_PHASE_ATLAS","organism_id":organism_id,"family":family,"engine":space["engine"],"k":k,"chart_family":chart_family,"writer_family":writer_family,"phase_atlas":True,"charts_by_phase":charts,"semantic_dimension":1,"parameter_count":sum(int(c.get("parameter_count",0)) for c in charts.values()),"valid":True}
        effective_samples=sum(phase_samples.values())
    else:
        d=pooled_phase_dataset(organism_id,family,chart_fit,q);effective_samples=int(len(d["semantic"]))
        if effective_samples < 2*coeff_per_chart:
            return {"version":"V837ap","organism_id":organism_id,"family":family,"engine":space.get("engine"),"calibration_n":calibration_n,"valid":False,"underdetermined":True,"failure_code":"UNDERDETERMINED_AT_N","required_samples":2*coeff_per_chart,"effective_samples":effective_samples,"historical_backend_loaded":False,"q_reconstructed_from_calibration":True}
        chart=_fit_chart(d["h"],d["semantic"],chart_family,k,binary)
        geometry={"version":"V837ap","branch":"AP_A_K1" if k==1 else "AP_B_PROJECTED","organism_id":organism_id,"family":family,"engine":space["engine"],"k":k,"chart_family":chart_family,"writer_family":writer_family,"phase_atlas":False,"chart":chart,"semantic_dimension":1,"parameter_count":int(chart.get("parameter_count",0)),"valid":True}
    tangent_count=tangent_coefficient_count(writer_family,k)
    if tangent_count:
        hb=[];hc=[];zb=[];zc=[]
        for phase in PHASES[family]:
            d=phase_dataset(organism_id,family,writer_fit,q,phase)
            if len(d["base_states"]):
                hb.append(d["base_states"]@q);hc.append(d["cf_states"]@q);zb.append(d["base_semantic"]);zc.append(d["cf_semantic"])
        tangent_samples=sum(len(x) for x in zb)
        if tangent_samples < 2*tangent_count:
            return {"version":"V837ap","organism_id":organism_id,"family":family,"engine":space.get("engine"),"calibration_n":calibration_n,"valid":False,"underdetermined":True,"failure_code":"UNDERDETERMINED_AT_N","required_tangent_samples":2*tangent_count,"effective_tangent_samples":tangent_samples,"historical_backend_loaded":False,"q_reconstructed_from_calibration":True}
        kind=writer_family.replace("TANGENT_","");field=fit_tangent_field(np.concatenate(hb),np.concatenate(hc),np.concatenate(zb),np.concatenate(zc),kind);geometry["tangent_field"]=field;geometry["parameter_count"]+=int(field.get("parameter_count",0))
    geometry["stored_bytes"]=len(json.dumps(geometry,sort_keys=True,separators=(",",":"),default=str).encode("utf-8"))
    return {"version":"V837ap","organism_id":organism_id,"family":family,"engine":space["engine"],"calibration_n":int(calibration_n),"valid":True,"underdetermined":False,"geometry":geometry,"q":q.tolist(),"q_sha256":sha256_json(q.tolist()),"chart_fit_seeds":chart_fit,"writer_fit_seeds":writer_fit,"effective_chart_samples":effective_samples,"deployed_continuous_coefficients":int(geometry["parameter_count"]),"historical_v837an_q_loaded":False,"historical_v837an_backend_loaded":False,"historical_v837ao_backend_loaded":False,"geometry_search_on_holdout":False,"degree_search_on_holdout":False,"k_search_on_holdout":False,"writer_search_on_holdout":False,"gradient_steps":0}
