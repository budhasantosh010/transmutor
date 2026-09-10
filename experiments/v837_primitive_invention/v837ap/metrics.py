from __future__ import annotations
import numpy as np

def causal_recovery(base,expected,patched,eps=1e-8):
    b=np.asarray(base,dtype=np.float64);t=np.asarray(expected,dtype=np.float64);p=np.asarray(patched,dtype=np.float64);return 1-np.abs(t-p)/(np.abs(t-b)+eps)
def direction_agreement(base,expected,patched):
    b=np.asarray(base);t=np.asarray(expected);p=np.asarray(patched);desired=np.sign(t-b);got=np.sign(p-b);neutral=np.abs(t-b)<1e-10;return np.where(neutral,np.abs(p-t)<1e-6,desired==got).astype(float)
def nrmse(pred,truth,scale):return float(np.sqrt(np.mean((np.asarray(pred)-np.asarray(truth))**2))/max(float(scale),1e-12))
