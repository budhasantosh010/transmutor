from __future__ import annotations

import numpy as np

RANGE = 2.0


def pearson(a, b) -> float:
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12:
        return 1.0 if np.sqrt(np.mean((a-b)**2)) < 1e-12 else 0.0
    return float(np.corrcoef(a,b)[0,1])


def direction_agreement(oracle, pred, eps: float = 1e-8) -> float:
    o=np.asarray(oracle,dtype=float);p=np.asarray(pred,dtype=float);m=np.abs(o)>eps
    if not np.any(m): return 1.0
    return float(np.mean(np.sign(o[m])==np.sign(p[m])))


def response_metrics(oracle, pred, task_success, *, normalization: float = RANGE) -> dict:
    o=np.asarray(oracle,dtype=float);p=np.asarray(pred,dtype=float);err=p-o
    nz=np.abs(o)>1e-8
    gain=float(np.dot(o[nz],p[nz])/(np.dot(o[nz],o[nz])+1e-12)) if np.any(nz) else 1.0
    zero=~nz
    return {
        "n": int(len(o)),
        "response_nrmse": float(np.sqrt(np.mean(err**2))/normalization) if len(o) else float("inf"),
        "response_mae_normalized": float(np.mean(np.abs(err))/normalization) if len(o) else float("inf"),
        "pearson": pearson(o,p),
        "direction_agreement": direction_agreement(o,p),
        "gain_ratio": gain,
        "perturbed_task_success": float(np.mean(task_success)) if len(task_success) else 0.0,
        "zero_effect_median_abs": float(np.median(np.abs(p[zero]))/normalization) if np.any(zero) else 0.0,
        "oracle_response_rms": float(np.sqrt(np.mean(o**2))) if len(o) else 0.0,
        "predicted_response_rms": float(np.sqrt(np.mean(p**2))) if len(p) else 0.0,
    }


def family_gate(rows: list[dict], *, fraction: float = 0.60) -> dict:
    n=len(rows);required=int(np.ceil(fraction*n)) if n else 0;passing=[r for r in rows if r.get("pass")];engines=sorted({r.get("engine") for r in passing if r.get("engine")})
    return {"organisms":n,"required":required,"passing":len(passing),"pass_fraction":len(passing)/n if n else 0.0,"passing_engines":engines,"both_engines":len(engines)>=2,"pass":bool(n and len(passing)>=required and len(engines)>=2)}
