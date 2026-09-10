from __future__ import annotations

import math

CALIBRATION_LADDER=(2,4,8,16,32,64)


def heldout_family_gate(rows:list[dict])->dict:
    valid=[r for r in rows if not r.get("underdetermined",False)]
    passing=[r for r in valid if r.get("pass")]
    n=len(rows);required=math.ceil(.75*n) if n else 0
    engines=sorted({r.get("engine") for r in passing if r.get("engine")})
    return {"organisms":n,"eligible":len(valid),"passing":len(passing),"required":required,"pass_fraction":0.0 if not n else len(passing)/n,"passing_engines":engines,"pass":bool(n and len(passing)>=required and len(engines)>=2)}


def qualifier(n:int|None)->str|None:
    if n is None:return None
    if n<=8:return "LOW_DATA_CAUSAL_BACKEND_COMPILATION"
    if n==16:return "MODERATE_CAUSAL_BACKEND_COMPILATION"
    return "HIGH_CALIBRATION_COST"
