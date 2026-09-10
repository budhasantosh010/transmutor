from __future__ import annotations

import numpy as np

from .operator_metrics import RANGE
from .utils import deterministic_seed


def _nrmse(y, pred) -> float:
    y=np.asarray(y,dtype=float);pred=np.asarray(pred,dtype=float)
    return float(np.sqrt(np.mean((y-pred)**2))/RANGE) if len(y) else float("inf")


def _features(rows: list[dict]) -> np.ndarray:
    out=[]
    for r in rows:
        req=float(r.get("actual_delta") or 0.0); h=float(r.get("horizon") or 1); t=float(r.get("time_index") or 0)
        out.append([1.0,req,t/10.0,1.0/max(h,1.0),req/max(h,1.0)])
    return np.asarray(out,dtype=float)


def _fit_generic(fit_rows: list[dict]) -> np.ndarray:
    X=_features(fit_rows);y=np.asarray([r["predicted_response"] for r in fit_rows],dtype=float)
    if len(y)<X.shape[1]:return np.zeros(X.shape[1])
    return np.linalg.lstsq(X,y,rcond=None)[0]


def _endpoint_profile(fit_rows: list[dict]) -> dict[str,float]:
    groups={}
    for r in fit_rows:groups.setdefault(r["intervention"],[]).append(float(r["predicted_response"]))
    return {k:float(np.mean(v)) for k,v in groups.items()}


def wrong_family_template(family: str, row: dict) -> float:
    d=float(row.get("actual_delta") or 0.0);h=max(int(row.get("horizon") or 1),1);name=row.get("intervention","")
    if family=="conditional_routing":
        return 0.35*(0.65**(h-1))*d
    if family=="delayed_recall":
        return d
    if family=="iterative_state":
        return d
    if family=="variable_composition":
        return 0.35*(0.65**(h-1))*d
    return 0.0


def evaluate_controls(family: str, fit_rows: list[dict], select_rows: list[dict], organism_id: str) -> dict:
    y=np.asarray([r["predicted_response"] for r in select_rows],dtype=float);oracle=np.asarray([r["oracle_response"] for r in select_rows],dtype=float)
    true=_nrmse(y,oracle);rng=np.random.default_rng(deterministic_seed("v837aq-controls",organism_id,family))
    # Time shuffle preserves intervention/magnitude but rotates phase/horizon labels by permuting oracle values within intervention classes.
    time_labels=oracle.copy()
    for name in sorted({r["intervention"] for r in select_rows}):
        idx=np.asarray([i for i,r in enumerate(select_rows) if r["intervention"]==name],dtype=int)
        if len(idx)>1:time_labels[idx]=oracle[idx][np.roll(np.arange(len(idx)),1)]
    # Magnitude shuffle preserves intervention/phase/horizon where possible.
    mag_labels=oracle.copy()
    keys=sorted({(r["intervention"],r["phase"],r["horizon"]) for r in select_rows})
    for key in keys:
        idx=np.asarray([i for i,r in enumerate(select_rows) if (r["intervention"],r["phase"],r["horizon"])==key],dtype=int)
        if len(idx)>1:mag_labels[idx]=oracle[idx][::-1]
    wrong=np.asarray([wrong_family_template(family,r) for r in select_rows],dtype=float)
    profile=_endpoint_profile(fit_rows);endpoint=np.asarray([profile.get(r["intervention"],0.0) for r in select_rows],dtype=float)
    beta=_fit_generic(fit_rows);generic=_features(select_rows)@beta
    errors={
      "true_oracle_nrmse":true,
      "time_shuffled_nrmse":_nrmse(y,time_labels),
      "magnitude_shuffled_nrmse":_nrmse(y,mag_labels),
      "wrong_family_template_nrmse":_nrmse(y,wrong),
      "endpoint_only_nrmse":_nrmse(y,endpoint),
      "matched_low_order_generic_regression_nrmse":_nrmse(y,generic),
    }
    margins={
      "time_shuffle":errors["time_shuffled_nrmse"]-true,
      "magnitude_shuffle":errors["magnitude_shuffled_nrmse"]-true,
      "wrong_family":errors["wrong_family_template_nrmse"]-true,
      "endpoint_only":errors["endpoint_only_nrmse"]-true,
      "generic_regression":errors["matched_low_order_generic_regression_nrmse"]-true,
    }
    failed=[]
    for k in ("time_shuffle","magnitude_shuffle","wrong_family"):
        if margins[k]<0.03:failed.append(k)
    for k in ("endpoint_only","generic_regression"):
        if margins[k]<0.0:failed.append(k)
    return {"errors":errors,"margins":margins,"pass":not failed,"failed_conditions":failed,"wrong_phase_zero_effect_in_primary_gate":True,"seed":deterministic_seed("v837aq-controls",organism_id,family)}


def build_frozen_control_baseline(family: str, pooled_discovery_fit_rows: list[dict]) -> dict:
    """Freeze control-only comparators on discovery evidence before heldout."""
    beta=_fit_generic(pooled_discovery_fit_rows)
    return {"family":family,"endpoint_profile":_endpoint_profile(pooled_discovery_fit_rows),"generic_beta":beta.tolist(),"feature_schema":["intercept","actual_delta","time_index/10","1/horizon","actual_delta/horizon"],"fit_source":"pooled discovery organisms / AQ_OPERATOR_FIT","heldout_rows_used":0}


def evaluate_frozen_controls(family: str, select_rows: list[dict], baseline: dict) -> dict:
    y=np.asarray([r["predicted_response"] for r in select_rows],dtype=float);oracle=np.asarray([r["oracle_response"] for r in select_rows],dtype=float);true=_nrmse(y,oracle)
    time_labels=oracle.copy()
    for name in sorted({r["intervention"] for r in select_rows}):
        idx=np.asarray([i for i,r in enumerate(select_rows) if r["intervention"]==name],dtype=int)
        if len(idx)>1:time_labels[idx]=oracle[idx][np.roll(np.arange(len(idx)),1)]
    mag_labels=oracle.copy()
    for key in sorted({(r["intervention"],r["phase"],r["horizon"]) for r in select_rows}):
        idx=np.asarray([i for i,r in enumerate(select_rows) if (r["intervention"],r["phase"],r["horizon"])==key],dtype=int)
        if len(idx)>1:mag_labels[idx]=oracle[idx][::-1]
    wrong=np.asarray([wrong_family_template(family,r) for r in select_rows],dtype=float);profile=baseline["endpoint_profile"];endpoint=np.asarray([float(profile.get(r["intervention"],0.0)) for r in select_rows],dtype=float);generic=_features(select_rows)@np.asarray(baseline["generic_beta"],dtype=float)
    errors={"true_oracle_nrmse":true,"time_shuffled_nrmse":_nrmse(y,time_labels),"magnitude_shuffled_nrmse":_nrmse(y,mag_labels),"wrong_family_template_nrmse":_nrmse(y,wrong),"endpoint_only_nrmse":_nrmse(y,endpoint),"matched_low_order_generic_regression_nrmse":_nrmse(y,generic)}
    margins={"time_shuffle":errors["time_shuffled_nrmse"]-true,"magnitude_shuffle":errors["magnitude_shuffled_nrmse"]-true,"wrong_family":errors["wrong_family_template_nrmse"]-true,"endpoint_only":errors["endpoint_only_nrmse"]-true,"generic_regression":errors["matched_low_order_generic_regression_nrmse"]-true}
    failed=[]
    for k in ("time_shuffle","magnitude_shuffle","wrong_family"):
        if margins[k]<0.03:failed.append(k)
    for k in ("endpoint_only","generic_regression"):
        if margins[k]<0.0:failed.append(k)
    return {"errors":errors,"margins":margins,"pass":not failed,"failed_conditions":failed,"baseline_frozen_before_heldout":True}
