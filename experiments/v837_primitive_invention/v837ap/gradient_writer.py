from __future__ import annotations
import numpy as np
from .chart_reader import chart_gradient,read_chart

EPS=1e-8

def gradient_step(chart,h,target,trust_radius:float,epsilon:float=EPS):
    x=np.asarray(h,dtype=np.float64).reshape(-1);g=np.asarray(chart_gradient(chart,x),dtype=np.float64).reshape(-1);gn=float(np.linalg.norm(g))
    err=float(target)-float(read_chart(chart,x.reshape(1,-1))[0])
    if gn<1e-6 and abs(err)>1e-12:return {"valid":False,"failure_code":"NONLINEAR_READER_GRADIENT_DEGENERATE","h":x,"gradient_norm":gn,"semantic_error":err}
    dh=g/(gn*gn+epsilon)*err;dn=float(np.linalg.norm(dh));limit=2.0*float(trust_radius)
    if dn>limit>0:dh*=limit/dn
    return {"valid":True,"h":x+dh,"gradient_norm":gn,"semantic_error":err,"delta_h":dh,"clipped":bool(dn>limit>0)}
