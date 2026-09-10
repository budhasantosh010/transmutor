from __future__ import annotations

import numpy as np


def _orient_columns(q:np.ndarray)->np.ndarray:
    q=np.asarray(q,dtype=np.float64).copy()
    for j in range(q.shape[1]):
        i=int(np.argmax(np.abs(q[:,j])))
        if q[i,j]<0:q[:,j]*=-1.0
    return q


def fit_difference_subspace(base:np.ndarray,counterfactual:np.ndarray,k:int)->dict:
    base=np.asarray(base,dtype=np.float64);cf=np.asarray(counterfactual,dtype=np.float64)
    if base.shape!=cf.shape or base.ndim!=2:raise ValueError("base/counterfactual must be same [N,D]")
    if not (1<=int(k)<=base.shape[1]):raise ValueError("invalid k")
    delta=cf-base
    _,s,vt=np.linalg.svd(delta,full_matrices=False)
    q=_orient_columns(vt[:int(k)].T)
    return {"q":q,"singular_values":s,"k":int(k),"carrier_dimension":base.shape[1],"rank":int(np.linalg.matrix_rank(q)),"invertibility_required":False}


def projected_counterfactual(base:np.ndarray,counterfactual:np.ndarray,q:np.ndarray)->np.ndarray:
    b=np.asarray(base,dtype=np.float64);d=np.asarray(counterfactual,dtype=np.float64)-b;q=np.asarray(q,dtype=np.float64)
    return b+(d@q)@q.T


def projected_delta(delta:np.ndarray,q:np.ndarray)->np.ndarray:
    d=np.asarray(delta,dtype=np.float64);q=np.asarray(q,dtype=np.float64);return (d@q)@q.T


def shuffled_pair_subspace(base:np.ndarray,counterfactual:np.ndarray,k:int,seed:int)->dict:
    b=np.asarray(base);cf=np.asarray(counterfactual);rng=np.random.default_rng(int(seed));perm=rng.permutation(len(cf))
    if len(cf)>1 and np.array_equal(perm,np.arange(len(cf))):perm=np.roll(perm,1)
    return fit_difference_subspace(b,cf[perm],k)
