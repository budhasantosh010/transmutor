from __future__ import annotations

import itertools, math
import numpy as np
import torch

FAMILIES=("SIGNED_PERMUTATION","DIAGONAL_AFFINE","RIGID_AFFINE","FULL_AFFINE")


def _arr(x): return np.asarray(x,dtype=np.float64)


def fit_vector_map(x,y,family:str,ridge:float=1e-6,require_invertible:bool=False)->dict:
    x=_arr(x);y=_arr(y)
    if x.ndim!=2 or y.ndim!=2 or x.shape!=y.shape: raise ValueError("adapter matrices must have equal [N,D] shapes")
    n,d=x.shape
    if family=="SIGNED_PERMUTATION":
        best=None
        for perm in itertools.permutations(range(d)):
            xp=x[:,perm]; score=np.sum(xp*y,axis=0); signs=np.where(score<0,-1.0,1.0); pred=xp*signs; mse=float(np.mean((pred-y)**2)); key=(mse,perm,tuple(signs.tolist()))
            if best is None or key<best[0]: best=(key,{"kind":family,"perm":list(perm),"signs":signs.tolist()})
        m=best[1]
    elif family=="DIAGONAL_AFFINE":
        mx=x.mean(0);my=y.mean(0);xc=x-mx;den=np.sum(xc*xc,axis=0);a=np.divide(np.sum(xc*(y-my),axis=0),den,out=np.zeros(d),where=den>1e-18);b=my-a*mx
        m={"kind":family,"a":a.tolist(),"b":b.tolist()}
    elif family=="RIGID_AFFINE":
        mx=x.mean(0);my=y.mean(0);u,_,vt=np.linalg.svd((x-mx).T@(y-my),full_matrices=False);q=u@vt
        m={"kind":family,"q":q.tolist(),"mu_x":mx.tolist(),"mu_y":my.tolist()}
    elif family=="FULL_AFFINE":
        mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my;A=np.linalg.solve(xc.T@xc+ridge*np.eye(d),xc.T@yc);b=my-mx@A
        m={"kind":family,"A":A.tolist(),"b":b.tolist(),"ridge":ridge}
    else: raise ValueError(family)
    pred=apply_np(x,m); mse=float(np.mean((pred-y)**2)); diag=diagnostics(m)
    m["fit_mse"]=mse;m["dimension"]=d;m["diagnostics"]=diag
    if require_invertible:
        if family=="DIAGONAL_AFFINE" and min(abs(v) for v in m["a"])<1e-4: m["valid"]=False;m["invalid_reason"]="DIAGONAL_STATE_SLOPE_TOO_SMALL"
        elif family=="FULL_AFFINE" and (diag["rank"]<d or diag["condition_number"]>1000.0): m["valid"]=False;m["invalid_reason"]="FULL_AFFINE_STATE_CONDITION_GUARD"
        else:m["valid"]=True
    else:m["valid"]=True
    return m


def fit_gate_map(x,y,family:str)->dict:
    eps=1e-5;x=np.clip(_arr(x).reshape(-1),eps,1-eps);y=np.clip(_arr(y).reshape(-1),eps,1-eps);lx=np.log(x/(1-x));ly=np.log(y/(1-y))
    if family=="SIGNED_PERMUTATION":
        e0=float(np.mean((lx-ly)**2));e1=float(np.mean((-lx-ly)**2));m={"kind":"GATE_SIGN","sign":-1.0 if e1<e0 else 1.0,"a":-1.0 if e1<e0 else 1.0,"b":0.0}
    else:
        vx=float(np.sum((lx-lx.mean())**2));a=float(np.sum((lx-lx.mean())*(ly-ly.mean()))/vx) if vx>1e-18 else 0.0;b=float(ly.mean()-a*lx.mean());m={"kind":"GATE_LOGIT_AFFINE","a":a,"b":b}
    pred=apply_gate_np(x,m);m["fit_mse"]=float(np.mean((pred-y)**2));m["valid"]=abs(float(m["a"]))>=1e-4;return m


def apply_np(x,m):
    x=_arr(x);k=m["kind"]
    if k=="IDENTITY":return x
    if k=="SIGNED_PERMUTATION":return x[:,m["perm"]]*np.asarray(m["signs"])
    if k=="DIAGONAL_AFFINE":return x*np.asarray(m["a"])+np.asarray(m["b"])
    if k=="RIGID_AFFINE":return (x-np.asarray(m["mu_x"]))@np.asarray(m["q"])+np.asarray(m["mu_y"])
    if k=="FULL_AFFINE":return x@np.asarray(m["A"])+np.asarray(m["b"])
    raise ValueError(k)


def inverse_np(y,m):
    y=_arr(y);k=m["kind"]
    if k=="IDENTITY":return y
    if k=="SIGNED_PERMUTATION":
        out=np.empty_like(y);out[:,m["perm"]]=y*np.asarray(m["signs"]);return out
    if k=="DIAGONAL_AFFINE":return (y-np.asarray(m["b"]))/np.asarray(m["a"])
    if k=="RIGID_AFFINE":return (y-np.asarray(m["mu_y"]))@np.asarray(m["q"]).T+np.asarray(m["mu_x"])
    if k=="FULL_AFFINE":return (y-np.asarray(m["b"]))@np.linalg.inv(np.asarray(m["A"]))
    raise ValueError(k)


def apply_gate_np(x,m):
    eps=1e-5;x=np.clip(_arr(x),eps,1-eps);l=np.log(x/(1-x));z=float(m["a"])*l+float(m["b"]);return 1/(1+np.exp(-z))


def inverse_gate_np(y,m):
    eps=1e-5;y=np.clip(_arr(y),eps,1-eps);l=np.log(y/(1-y));z=(l-float(m["b"]))/float(m["a"]);return 1/(1+np.exp(-z))


def apply_tensor(x:torch.Tensor,m:dict)->torch.Tensor:
    k=m["kind"]
    if k=="IDENTITY":return x
    if k=="SIGNED_PERMUTATION":return x[...,m["perm"]]*torch.tensor(m["signs"],dtype=x.dtype)
    if k=="DIAGONAL_AFFINE":return x*torch.tensor(m["a"],dtype=x.dtype)+torch.tensor(m["b"],dtype=x.dtype)
    if k=="RIGID_AFFINE":return (x-torch.tensor(m["mu_x"],dtype=x.dtype))@torch.tensor(m["q"],dtype=x.dtype)+torch.tensor(m["mu_y"],dtype=x.dtype)
    if k=="FULL_AFFINE":return x@torch.tensor(m["A"],dtype=x.dtype)+torch.tensor(m["b"],dtype=x.dtype)
    raise ValueError(k)


def inverse_tensor(y:torch.Tensor,m:dict)->torch.Tensor:
    k=m["kind"]
    if k=="IDENTITY":return y
    if k=="SIGNED_PERMUTATION":
        out=torch.empty_like(y);out[...,m["perm"]]=y*torch.tensor(m["signs"],dtype=y.dtype);return out
    if k=="DIAGONAL_AFFINE":return (y-torch.tensor(m["b"],dtype=y.dtype))/torch.tensor(m["a"],dtype=y.dtype)
    if k=="RIGID_AFFINE":return (y-torch.tensor(m["mu_y"],dtype=y.dtype))@torch.tensor(m["q"],dtype=y.dtype).T+torch.tensor(m["mu_x"],dtype=y.dtype)
    if k=="FULL_AFFINE":return (y-torch.tensor(m["b"],dtype=y.dtype))@torch.linalg.inv(torch.tensor(m["A"],dtype=y.dtype))
    raise ValueError(k)


def apply_gate_tensor(x:torch.Tensor,m:dict)->torch.Tensor:
    z=torch.clamp(x,1e-5,1-1e-5);l=torch.logit(z);return torch.sigmoid(float(m["a"])*l+float(m["b"]))


def diagnostics(m:dict)->dict:
    k=m["kind"]
    if k=="SIGNED_PERMUTATION":return {"condition_number":1.0,"rank":len(m["perm"]),"determinant_magnitude":1.0,"translation_norm":0.0,"orthogonality_error":0.0,"inverse_consistency_error":0.0}
    if k=="DIAGONAL_AFFINE":
        a=np.asarray(m["a"]);b=np.asarray(m["b"]);cond=float(np.max(abs(a))/max(np.min(abs(a)),1e-30));return {"condition_number":cond,"rank":int(np.sum(abs(a)>1e-12)),"determinant_magnitude":float(abs(np.prod(a))),"translation_norm":float(np.linalg.norm(b)),"orthogonality_error":None,"inverse_consistency_error":0.0 if np.all(abs(a)>1e-12) else float("inf")}
    if k=="RIGID_AFFINE":
        q=np.asarray(m["q"]);return {"condition_number":1.0,"rank":q.shape[0],"determinant_magnitude":float(abs(np.linalg.det(q))),"translation_norm":float(np.linalg.norm(np.asarray(m["mu_y"])-np.asarray(m["mu_x"]))),"orthogonality_error":float(np.linalg.norm(q.T@q-np.eye(q.shape[0]))),"inverse_consistency_error":0.0}
    if k=="FULL_AFFINE":
        A=np.asarray(m["A"]);s=np.linalg.svd(A,compute_uv=False);return {"condition_number":float(s.max()/max(s.min(),1e-30)),"rank":int(np.linalg.matrix_rank(A)),"determinant_magnitude":float(abs(np.linalg.det(A))),"translation_norm":float(np.linalg.norm(m["b"])),"orthogonality_error":None,"inverse_consistency_error":0.0 if np.linalg.matrix_rank(A)==A.shape[0] else float("inf"),"singular_values":s.tolist()}
    return {}


def map_dof_macs(family:str,d:int)->tuple[int,int]:
    if family=="SIGNED_PERMUTATION":return 0,d
    if family=="DIAGONAL_AFFINE":return 2*d,d
    if family=="RIGID_AFFINE":return d*(d-1)//2+d,d*d
    if family=="FULL_AFFINE":return d*d+d,d*d
    return 0,0
