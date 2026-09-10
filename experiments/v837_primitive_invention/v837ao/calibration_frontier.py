from __future__ import annotations

import math

CALIBRATION_LADDER=(1,2,4,8,16,32,64)


def family_gate(rows:list[dict])->dict:
    n=len(rows);passing=[r for r in rows if r.get('pass')];required=math.ceil(.75*n) if n else 0;engines=sorted({r['engine'] for r in passing})
    return {'holdout_organisms':n,'passing':len(passing),'required_passes':required,'pass_engines':engines,'pass':bool(n and len(passing)>=required and len(engines)>=2)}


def qualifier(n:int|None)->str|None:
    if n is None:return None
    if n<=8:return 'LOW_DATA_BACKEND_COMPILATION'
    if n==16:return 'MODERATE_BACKEND_CALIBRATION'
    return 'BACKEND_CALIBRATION_EXPENSIVE'
