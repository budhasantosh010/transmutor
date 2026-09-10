from __future__ import annotations

from itertools import combinations_with_replacement
import numpy as np

def robust_normalization(x:np.ndarray)->dict:
    a=np.asarray(x,dtype=np.float64); med=np.median(a,axis=0); q25,q75=np.quantile(a,[.25,.75],axis=0); scale=np.asarray(q75-q25,dtype=np.float64); std=np.std(a,axis=0); scale=np.where(scale>1e-8,scale,np.where(std>1e-8,std,1.0)); return {"median":np.atleast_1d(med).tolist(),"scale":np.atleast_1d(scale).tolist()}

def normalize(x,norm):
    a=np.asarray(x,dtype=np.float64);return (a-np.asarray(norm["median"]))/np.asarray(norm["scale"])

def exponent_basis(k:int,degree:int)->list[tuple[int,...]]:
    if degree<0 or degree>3: raise ValueError("degree must be 0..3")
    exps=[(0,)*k]
    for d in range(1,degree+1):
        for combo in combinations_with_replacement(range(k),d):
            e=[0]*k
            for j in combo:e[j]+=1
            exps.append(tuple(e))
    return exps

def feature_count(k:int,degree:int)->int: return len(exponent_basis(k,degree))
def features(x:np.ndarray,degree:int)->np.ndarray:
    a=np.asarray(x,dtype=np.float64);a=np.atleast_2d(a);exps=exponent_basis(a.shape[1],degree);out=np.ones((len(a),len(exps)),dtype=np.float64)
    for j,e in enumerate(exps[1:],1):
        v=np.ones(len(a))
        for d,p in enumerate(e):
            if p:v*=a[:,d]**p
        out[:,j]=v
    return out

def feature_gradient(x:np.ndarray,degree:int)->np.ndarray:
    a=np.asarray(x,dtype=np.float64).reshape(-1);exps=exponent_basis(len(a),degree);g=np.zeros((len(exps),len(a)),dtype=np.float64)
    for j,e in enumerate(exps):
        for d,p in enumerate(e):
            if p==0:continue
            v=float(p)
            for m,pm in enumerate(e):
                power=pm-(1 if m==d else 0)
                if power:v*=a[m]**power
            g[j,d]=v
    return g
