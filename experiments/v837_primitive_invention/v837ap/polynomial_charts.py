from __future__ import annotations
import numpy as np
from .chart_features import features,feature_gradient,robust_normalization,normalize

RIDGE=1e-6

def fit_polynomial(x,z,degree:int,lam:float=RIDGE)->dict:
    x=np.asarray(x,dtype=np.float64);x=x.reshape(-1,1) if x.ndim==1 else np.atleast_2d(x);z=np.asarray(z,dtype=np.float64).reshape(-1);norm=robust_normalization(x);xn=normalize(x,norm);X=features(xn,degree);reg=lam*np.eye(X.shape[1]);reg[0,0]=0.0;coef=np.linalg.solve(X.T@X+reg,X.T@z)
    return {"kind":"POLYNOMIAL","degree":degree,"k":x.shape[1],"coef":coef.tolist(),"normalization":norm,"ridge_lambda":lam,"parameter_count":int(len(coef)),"fit_h_min":np.min(x,axis=0).tolist(),"fit_h_max":np.max(x,axis=0).tolist(),"fit_z_min":float(np.min(z)),"fit_z_max":float(np.max(z)),"gradient_steps":0}
def predict_polynomial(chart,x):
    a=np.asarray(x,dtype=np.float64);a=a.reshape(-1,1) if a.ndim==1 and int(chart["k"])==1 else np.atleast_2d(a);return features(normalize(a,chart["normalization"]),int(chart["degree"]))@np.asarray(chart["coef"],dtype=np.float64)
def gradient_polynomial(chart,x):
    a=np.asarray(x,dtype=np.float64).reshape(-1);xn=normalize(a,chart["normalization"]);fg=feature_gradient(xn,int(chart["degree"]));g_norm=np.asarray(chart["coef"])@fg;return g_norm/np.asarray(chart["normalization"]["scale"],dtype=np.float64)
def inverse_polynomial_1d(chart,target,current_h):
    if int(chart["k"])!=1:raise ValueError("1D only")
    coef=np.asarray(chart["coef"],dtype=np.float64);degree=int(chart["degree"]);med=float(chart["normalization"]["median"][0]);scale=float(chart["normalization"]["scale"][0]);poly=np.zeros(degree+1)
    for power in range(degree+1):poly[degree-power]=coef[power]
    poly[-1]-=float(target);roots=np.roots(poly);lo,hi=float(chart["fit_h_min"][0]),float(chart["fit_h_max"][0]);valid=[float(r.real*scale+med) for r in roots if abs(r.imag)<=1e-8 and lo-1e-12<=r.real*scale+med<=hi+1e-12]
    if not valid:return None
    return min(valid,key=lambda v:abs(v-float(current_h)))
