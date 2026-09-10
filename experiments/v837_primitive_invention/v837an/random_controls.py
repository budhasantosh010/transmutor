from __future__ import annotations

import numpy as np

from .counterfactual_subspace import _orient_columns


def haar_subspace(dimension:int,k:int,seed:int)->np.ndarray:
    rng=np.random.default_rng(int(seed));a=rng.normal(size=(int(dimension),int(k)));q,_=np.linalg.qr(a,mode="reduced");return _orient_columns(q[:,:int(k)])


def random_subspaces(dimension:int,k:int,count:int,seed:int)->list[np.ndarray]:
    master=np.random.default_rng(int(seed));return [haar_subspace(dimension,k,int(master.integers(0,2**63-1))) for _ in range(int(count))]


def same_norm_noise(delta:np.ndarray,seed:int)->np.ndarray:
    d=np.asarray(delta,dtype=np.float64);rng=np.random.default_rng(int(seed));noise=rng.normal(size=d.shape);dn=np.linalg.norm(d,axis=1,keepdims=True);nn=np.linalg.norm(noise,axis=1,keepdims=True);return noise*(dn/np.maximum(nn,1e-12))


def orthogonal_residual(delta:np.ndarray,q:np.ndarray)->np.ndarray:
    d=np.asarray(delta,dtype=np.float64);q=np.asarray(q,dtype=np.float64);res=d-(d@q)@q.T;dn=np.linalg.norm((d@q)@q.T,axis=1,keepdims=True);rn=np.linalg.norm(res,axis=1,keepdims=True);return res*(dn/np.maximum(rn,1e-12))
