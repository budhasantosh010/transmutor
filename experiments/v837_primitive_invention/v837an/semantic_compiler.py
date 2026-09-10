from __future__ import annotations

import numpy as np

RIDGE_LAMBDA=1e-6


def fit_semantic_compiler(semantic_delta:np.ndarray,carrier_delta:np.ndarray,q:np.ndarray,lam:float=RIDGE_LAMBDA)->dict:
    z=np.asarray(semantic_delta,dtype=np.float64).reshape(-1,1);d=np.asarray(carrier_delta,dtype=np.float64);q=np.asarray(q,dtype=np.float64)
    a=d@q
    gram=z.T@z+float(lam)*np.eye(1)
    coef=np.linalg.solve(gram,z.T@a) # [1,k], zero intercept
    return {"coef":coef,"ridge_lambda":float(lam),"intercept":0.0,"q":q,"semantic_dimension":1,"latent_dimension":q.shape[1],"gradient_steps":0}


def compile_delta(semantic_delta:np.ndarray,compiler:dict)->np.ndarray:
    z=np.asarray(semantic_delta,dtype=np.float64).reshape(-1,1);coef=np.asarray(compiler["coef"],dtype=np.float64);q=np.asarray(compiler["q"],dtype=np.float64)
    return (z@coef)@q.T


def zero_delta_identity(compiler:dict,n:int=3)->bool:
    return bool(np.array_equal(compile_delta(np.zeros(n),compiler),np.zeros((n,np.asarray(compiler["q"]).shape[0]))))
