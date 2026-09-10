from __future__ import annotations
import numpy as np
from .chart_features import features,robust_normalization,normalize
from .chart_reader import chart_gradient,read_chart

RIDGE=1e-6
TANGENT_FAMILIES=("CONSTANT","AFFINE_STATE_FIELD","QUADRATIC_STATE_FIELD")

def fit_tangent_field(h_base,h_cf,z_base,z_cf,kind:str)->dict:
    hb=np.asarray(h_base,dtype=np.float64);hc=np.asarray(h_cf,dtype=np.float64);zb=np.asarray(z_base,dtype=np.float64);zc=np.asarray(z_cf,dtype=np.float64);dz=zc-zb;mask=np.abs(dz)>=1e-8;h=hb[mask];v=(hc[mask]-hb[mask])/dz[mask,None]
    if not len(h):return {"valid":False,"kind":kind,"failure_code":"NO_TANGENT_OBSERVATIONS"}
    degree={"CONSTANT":0,"AFFINE_STATE_FIELD":1,"QUADRATIC_STATE_FIELD":2}[kind];norm=robust_normalization(h);X=features(normalize(h,norm),degree);reg=RIDGE*np.eye(X.shape[1]);reg[0,0]=0;coef=np.linalg.solve(X.T@X+reg,X.T@v)
    return {"valid":True,"kind":kind,"degree":degree,"k":h.shape[1],"coef":coef.tolist(),"normalization":norm,"ridge_lambda":RIDGE,"parameter_count":int(coef.size),"gradient_steps":0}
def tangent_vector(field,h):
    a=np.asarray(h,dtype=np.float64).reshape(1,-1);return (features(normalize(a,field["normalization"]),int(field["degree"]))@np.asarray(field["coef"]))[0]
def tangent_step(chart,field,h,target,d90):
    x=np.asarray(h,dtype=np.float64).reshape(-1);g=np.asarray(chart_gradient(chart,x),dtype=np.float64).reshape(-1);v=tangent_vector(field,x);gamma=float(g@v);gn=float(np.linalg.norm(g));err=float(target)-float(read_chart(chart,x.reshape(1,-1))[0])
    if abs(gamma)<1e-4:return {"valid":False,"failure_code":"TANGENT_SEMANTIC_GAIN_DEGENERATE","h":x,"gradient_norm":gn,"gamma":gamma}
    tv=v/gamma;dh=tv*err;dn=float(np.linalg.norm(dh));limit=2*float(d90)
    if dn>limit>0:dh*=limit/dn
    return {"valid":True,"h":x+dh,"gradient_norm":gn,"gamma":gamma,"delta_h":dh,"clipped":bool(dn>limit>0)}
