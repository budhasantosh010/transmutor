from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression
from .chart_features import features,feature_gradient,robust_normalization,normalize

def _sigmoid(x):
    x=np.clip(np.asarray(x,dtype=np.float64),-40,40);return 1/(1+np.exp(-x))
def fit_logistic(x,z,degree:int=1,l2:float=1e-6,max_iter:int=64)->dict:
    a=np.asarray(x,dtype=np.float64);a=a.reshape(-1,1) if a.ndim==1 else np.atleast_2d(a);y=(np.asarray(z,dtype=np.float64).reshape(-1)>0).astype(np.float64);norm=robust_normalization(a);xn=normalize(a,norm);X=features(xn,degree);b=np.zeros(X.shape[1]);reg=l2*np.eye(X.shape[1]);reg[0,0]=0
    iterations=0
    for i in range(max_iter):
        p=_sigmoid(X@b);w=np.clip(p*(1-p),1e-9,None);H=X.T@(w[:,None]*X)+reg;g=X.T@(p-y)+reg@b;step=np.linalg.solve(H,g);b-=step;iterations=i+1
        if np.linalg.norm(step)<=1e-10:break
    return {"kind":"LOGISTIC","degree":degree,"k":a.shape[1],"coef":b.tolist(),"normalization":norm,"l2":l2,"irls_iterations":iterations,"parameter_count":len(b),"fit_h_min":np.min(a,axis=0).tolist(),"fit_h_max":np.max(a,axis=0).tolist(),"fit_z_min":-1.0,"fit_z_max":1.0,"gradient_steps":0}
def predict_logistic(chart,x):
    a=np.asarray(x,dtype=np.float64);a=a.reshape(-1,1) if a.ndim==1 and int(chart["k"])==1 else np.atleast_2d(a);X=features(normalize(a,chart["normalization"]),int(chart["degree"]));return 2*_sigmoid(X@np.asarray(chart["coef"]))-1
def gradient_logistic(chart,x):
    a=np.asarray(x,dtype=np.float64).reshape(-1);xn=normalize(a,chart["normalization"]);X=features(xn.reshape(1,-1),int(chart["degree"]))[0];p=float(_sigmoid(X@np.asarray(chart["coef"])));fg=feature_gradient(xn,int(chart["degree"]));dlogit=np.asarray(chart["coef"])@fg;return (2*p*(1-p)*dlogit)/np.asarray(chart["normalization"]["scale"])
def fit_binary_isotonic(h,z,knots:int)->dict:
    h=np.asarray(h,dtype=np.float64).reshape(-1);z=np.asarray(z,dtype=np.float64).reshape(-1);corr=float(np.corrcoef(h,z)[0,1]) if np.std(h)>1e-12 else 1.;inc=corr>=0;order=np.argsort(h,kind="mergesort");hs=h[order];ys=(z[order]>0).astype(float);ir=IsotonicRegression(increasing=inc,out_of_bounds="clip");prob=ir.fit_transform(hs,ys);kh=np.quantile(hs,np.linspace(0,1,knots));kp=[]
    for x in kh:kp.append(float(prob[int(np.argmin(np.abs(hs-x)))]))
    return {"kind":f"ISOTONIC_{knots}","k":1,"knots_h":[float(x) for x in kh],"knots_p":kp,"direction":1 if inc else -1,"parameter_count":2*knots,"fit_h_min":[float(np.min(h))],"fit_h_max":[float(np.max(h))],"fit_z_min":-1.,"fit_z_max":1.,"gradient_steps":0}
def predict_binary_isotonic(chart,h):
    x=np.asarray(h,dtype=np.float64).reshape(-1);p=np.interp(x,np.asarray(chart["knots_h"]),np.asarray(chart["knots_p"]));return 2*p-1
