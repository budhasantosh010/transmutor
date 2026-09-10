from __future__ import annotations

import math

CONTINUOUS_K1_ORDER=("AFFINE","POLYNOMIAL_2","POLYNOMIAL_3","MONOTONE_PWL_4","MONOTONE_PWL_8","MONOTONE_PCHIP_6")
BINARY_K1_ORDER=("AFFINE","LOGISTIC_MONOTONE","ISOTONIC_4","ISOTONIC_8")
PROJECTED_ORDER=("LINEAR","QUADRATIC","CUBIC")
K_ORDER=(2,4,8)

def candidate_order(family:str)->list[dict]:
    binary=family in {"conditional_routing","delayed_recall"};out=[]
    for kind in (BINARY_K1_ORDER if binary else CONTINUOUS_K1_ORDER): out.append({"branch":"AP_A_K1","k":1,"chart_family":kind,"writer_family":"DIRECT_INVERSE_OR_CLASS_PROTOTYPE","phase_atlas":False})
    for k in K_ORDER:
        for kind in PROJECTED_ORDER:out.append({"branch":"AP_B_PROJECTED","k":k,"chart_family":kind,"writer_family":"GRADIENT_NEWTON","phase_atlas":False})
    return out

def first_passing_candidate(candidates:list[dict])->dict|None:
    for c in candidates:
        if c.get("family_pass"):return c
    return None

def family_gate(rows:list[dict],field:str="full_pass")->dict:
    passing=[r for r in rows if bool(r.get(field))];n=len(rows);required=max(1,math.ceil(.60*n));engines=sorted({r.get("engine") for r in passing if r.get("engine")})
    return {"organisms":n,"passing":len(passing),"required":required,"pass_fraction":0.0 if n==0 else len(passing)/n,"passing_engines":engines,"pass":bool(n and len(passing)>=required and len(engines)>=2)}
