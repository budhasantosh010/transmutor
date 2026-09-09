from __future__ import annotations

import numpy as np
import torch

from .bilinear_compression import compress_interaction
from .context_features import fit_normalizer,normalize,normalize_tensor

RIDGE=1e-6

def _ridge(X,Y):return np.linalg.solve(X.T@X+RIDGE*np.eye(X.shape[1]),X.T@Y)

def _interaction_np(z,c,context_factors,delta_matrices):
    if context_factors.size==0:return np.zeros_like(z)
    alpha=c@context_factors
    return np.einsum("nr,rij,ni->nj",alpha,delta_matrices,z)

def fit_map(z,y,c,family:str,state=False):
    z=np.asarray(z,dtype=np.float64);y=np.asarray(y,dtype=np.float64);c=np.asarray(c,dtype=np.float64);norm=fit_normalizer(c);cn=normalize(c,norm);N,d=z.shape;q=cn.shape[1]
    context_factors=np.zeros((q,0),dtype=np.float64);delta_matrices=np.zeros((0,d,d),dtype=np.float64)
    if family=="STATIC_FULL_AFFINE":
        X=np.concatenate([z,np.ones((N,1))],1);coef=_ridge(X,y);A=coef[:d];b=coef[d];D=np.zeros((q,d));comp={"requested_rank":0,"effective_rank":0,"singular_values":[],"retained_fraction":1.0,"reconstruction_mse":0.0}
    elif family=="CONTEXT_ADDITIVE":
        X=np.concatenate([z,np.ones((N,1)),cn],1);coef=_ridge(X,y);A=coef[:d];b=coef[d];D=coef[d+1:];comp={"requested_rank":0,"effective_rank":0,"singular_values":[],"retained_fraction":1.0,"reconstruction_mse":0.0}
    else:
        rank=int(family.rsplit("R",1)[1]);inter=np.einsum("nq,nd->nqd",cn,z).reshape(N,q*d);X=np.concatenate([z,np.ones((N,1)),cn,inter],1);coef=_ridge(X,y);A=coef[:d];b=coef[d];D=coef[d+1:d+1+q];raw=coef[d+1+q:].reshape(q,d,d);context_factors,delta_matrices,comp=compress_interaction(raw,rank)
    pred=z@A+b+(cn@D if q else 0)+_interaction_np(z,cn,context_factors,delta_matrices);mse=float(np.mean((pred-y)**2));r=int(comp["effective_rank"])
    params=d*d+d+q*d+q*r+r*d*d
    macs=d*d+q*d+q*r+r*d*d+r*d
    out={"family":family,"dimension":d,"context_dim":q,"normalizer":norm,"A0":A.tolist(),"b0":b.tolist(),"D":D.tolist(),"context_factors":context_factors.tolist(),"delta_matrices":delta_matrices.tolist(),"compression":comp,"fit_mse":mse,"parameters":int(params),"macs":int(macs),"state":bool(state),"valid":True,"deployment_representation":"FACTORED_LOW_RANK" if r else "BASE_AFFINE"}
    return out

def fit_gate_map(z,y,c,family):
    eps=1e-5;z=np.clip(np.asarray(z).reshape(-1,1),eps,1-eps);y=np.clip(np.asarray(y).reshape(-1,1),eps,1-eps);return {**fit_map(np.log(z/(1-z)),np.log(y/(1-y)),c,family,False),"gate_logit":True}

def apply_np(z,c,m):
    z=np.asarray(z,dtype=np.float64);cn=normalize(c,m["normalizer"]);A=np.asarray(m["A0"]);b=np.asarray(m["b0"]);D=np.asarray(m["D"]);cf=np.asarray(m["context_factors"]);dm=np.asarray(m["delta_matrices"])
    return z@A+b+(cn@D if cn.shape[1] else 0)+_interaction_np(z,cn,cf,dm)

def matrices_np(c,m):
    cn=normalize(c,m["normalizer"]);A=np.asarray(m["A0"]);cf=np.asarray(m["context_factors"]);dm=np.asarray(m["delta_matrices"])
    if cf.size==0:return np.repeat(A[None,:,:],len(cn),0)
    alpha=cn@cf
    return A[None,:,:]+np.einsum("nr,rij->nij",alpha,dm)

def inverse_np(y,c,m):
    y=np.asarray(y,dtype=np.float64);cn=normalize(c,m["normalizer"]);M=matrices_np(c,m);off=np.asarray(m["b0"])+(cn@np.asarray(m["D"]) if cn.shape[1] else 0);return np.linalg.solve(M,np.expand_dims(y-off,-1)).squeeze(-1)

def apply_gate_np(z,c,m):
    eps=1e-5;z=np.clip(np.asarray(z).reshape(-1,1),eps,1-eps);a=apply_np(np.log(z/(1-z)),c,m);return (1/(1+np.exp(-a))).reshape(-1,1)

def apply_tensor(z,c,m):
    cn=normalize_tensor(c,m["normalizer"]);A=torch.tensor(m["A0"],dtype=z.dtype);b=torch.tensor(m["b0"],dtype=z.dtype);D=torch.tensor(m["D"],dtype=z.dtype);cf=torch.tensor(m["context_factors"],dtype=z.dtype);dm=torch.tensor(m["delta_matrices"],dtype=z.dtype);out=z@A+b
    if cn.shape[-1]:out=out+cn@D
    if cf.numel():
        alpha=cn@cf;out=out+torch.einsum("br,rij,bi->bj",alpha,dm,z)
    return out

def inverse_tensor(y,c,m):
    cn=normalize_tensor(c,m["normalizer"]);A=torch.tensor(m["A0"],dtype=y.dtype);cf=torch.tensor(m["context_factors"],dtype=y.dtype);dm=torch.tensor(m["delta_matrices"],dtype=y.dtype);M=A.unsqueeze(0).expand(y.shape[0],-1,-1)
    if cf.numel():M=M+torch.einsum("br,rij->bij",cn@cf,dm)
    off=torch.tensor(m["b0"],dtype=y.dtype)
    if cn.shape[-1]:off=off+cn@torch.tensor(m["D"],dtype=y.dtype)
    return torch.linalg.solve(M,(y-off).unsqueeze(-1)).squeeze(-1)

def apply_gate_tensor(z,c,m):
    z=torch.clamp(z,1e-5,1-1e-5);return torch.sigmoid(apply_tensor(torch.logit(z),c,m))

def conditioning(c,m):
    M=matrices_np(c,m);s=np.linalg.svd(M,compute_uv=False);mins=s[:,-1];conds=s[:,0]/np.maximum(mins,1e-30);return {"minimum_singular_value":float(np.min(mins)),"maximum_condition_number":float(np.max(conds)),"valid":bool(np.min(mins)>=0.05 and np.max(conds)<=1000.0)}
