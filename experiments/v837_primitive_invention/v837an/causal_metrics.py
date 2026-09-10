from __future__ import annotations

import numpy as np


def recovery(base_prediction,counterfactual_prediction,patched_prediction,epsilon:float=1e-8)->np.ndarray:
    b=np.asarray(base_prediction,dtype=np.float64);c=np.asarray(counterfactual_prediction,dtype=np.float64);p=np.asarray(patched_prediction,dtype=np.float64)
    return 1.0-np.abs(p-c)/(np.abs(b-c)+float(epsilon))


def clipped_recovery(values)->np.ndarray:return np.clip(np.asarray(values,dtype=np.float64),-1.0,1.0)


def direction_agreement(base_prediction,counterfactual_prediction,patched_prediction)->np.ndarray:
    b=np.asarray(base_prediction);c=np.asarray(counterfactual_prediction);p=np.asarray(patched_prediction)
    return (np.sign(p-b)==np.sign(c-b)).astype(np.float64)


def effect_faithfulness(base_prediction,counterfactual_prediction,patched_prediction,epsilon:float=1e-8)->np.ndarray:
    b=np.asarray(base_prediction,dtype=np.float64);c=np.asarray(counterfactual_prediction,dtype=np.float64);p=np.asarray(patched_prediction,dtype=np.float64)
    den=c-b;safe=np.where(np.abs(den)<epsilon,np.where(den>=0,epsilon,-epsilon),den);return (p-b)/safe


def success_vector(task,predictions,targets)->np.ndarray:
    return np.asarray([bool(task.success(float(p),float(t))) for p,t in zip(np.asarray(predictions),np.asarray(targets))],dtype=np.float64)


def paired_sign_flip_p(differences,seed:int,samples:int=16384)->float:
    d=np.asarray(differences,dtype=np.float64);d=d[np.isfinite(d)]
    if d.size==0:return 1.0
    observed=float(np.mean(d))
    if observed<=0:return 1.0
    # Exhaustive exact enumeration for small N; deterministic Monte Carlo otherwise.
    if d.size<=18:
        count=0;total=1<<d.size
        for mask in range(total):
            signs=np.fromiter((1.0 if (mask>>i)&1 else -1.0 for i in range(d.size)),dtype=np.float64,count=d.size)
            if float(np.mean(signs*d))>=observed-1e-15:count+=1
        return count/total
    rng=np.random.default_rng(int(seed));count=0
    batch=1024;done=0
    while done<int(samples):
        n=min(batch,int(samples)-done);signs=rng.choice(np.asarray([-1.0,1.0]),size=(n,d.size));means=(signs*d).mean(axis=1);count+=int(np.sum(means>=observed-1e-15));done+=n
    return (count+1.0)/(int(samples)+1.0)


def summarize_effect(base,cf,patched,task,targets)->dict:
    r=recovery(base,cf,patched);d=direction_agreement(base,cf,patched);f=effect_faithfulness(base,cf,patched);s=success_vector(task,patched,targets)
    return {"median_recovery":float(np.median(r)),"mean_recovery":float(np.mean(r)),"direction_agreement":float(np.mean(d)),"counterfactual_success":float(np.mean(s)),"effect_ratio_median":float(np.median(f)),"effect_ratio_iqr":[float(np.quantile(f,.25)),float(np.quantile(f,.75))],"overshoot_fraction":float(np.mean(np.abs(f)>1.0)),"recovery":r,"direction":d,"success":s,"effect_ratio":f}
