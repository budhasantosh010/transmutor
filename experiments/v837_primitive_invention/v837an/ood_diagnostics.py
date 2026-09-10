from __future__ import annotations

import numpy as np


def fit_reference(points:np.ndarray)->dict:
    x=np.asarray(points,dtype=np.float64);mu=x.mean(axis=0);centered=x-mu
    cov=(centered.T@centered)/max(1,len(x)-1);scale=float(np.trace(cov)/max(1,cov.shape[0]));cov=cov+max(scale,1e-8)*1e-4*np.eye(cov.shape[0]);inv=np.linalg.pinv(cov,rcond=1e-10)
    return {"points":x,"mean":mu,"inverse_covariance":inv}


def nearest_distance(x:np.ndarray,reference:dict)->np.ndarray:
    q=np.asarray(x,dtype=np.float64);p=np.asarray(reference["points"],dtype=np.float64);out=[]
    for row in q:
        out.append(float(np.sqrt(np.min(np.sum((p-row)**2,axis=1)))))
    return np.asarray(out)


def mahalanobis_distance(x:np.ndarray,reference:dict)->np.ndarray:
    q=np.asarray(x,dtype=np.float64)-np.asarray(reference["mean"]);inv=np.asarray(reference["inverse_covariance"]);return np.sqrt(np.maximum(0.0,np.einsum("ni,ij,nj->n",q,inv,q)))


def compare(patched:np.ndarray,natural_cf:np.ndarray,reference:dict)->dict:
    pn=nearest_distance(patched,reference);cn=nearest_distance(natural_cf,reference);pm=mahalanobis_distance(patched,reference);cm=mahalanobis_distance(natural_cf,reference)
    nr=pn/np.maximum(cn,1e-8);mr=pm/np.maximum(cm,1e-8)
    return {"nearest_neighbor_ratio":nr,"mahalanobis_ratio":mr,"ood_ratio":mr,"median_nearest_neighbor_ratio":float(np.median(nr)),"median_mahalanobis_ratio":float(np.median(mr)),"median_ood_ratio":float(np.median(mr))}
