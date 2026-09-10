from __future__ import annotations
import numpy as np
from experiments.v837_primitive_invention.v837an.causal_metrics import paired_sign_flip_p
from .binary_charts import fit_binary_isotonic,fit_logistic
from .polynomial_charts import fit_polynomial
from .monotone_charts import fit_monotone
from .projected_charts import fit_projected_chart
from .scalar_charts import fit_scalar_chart
from .utils import deterministic_seed

def random_subspaces(dim:int,k:int,count:int,seed:int)->list[np.ndarray]:
    rng=np.random.default_rng(seed);out=[]
    for _ in range(count):
        q,_=np.linalg.qr(rng.normal(size=(dim,k)));out.append(q[:,:k])
    return out

def random_feature_rotation(k:int,seed:int)->np.ndarray:
    rng=np.random.default_rng(seed);q,_=np.linalg.qr(rng.normal(size=(k,k)));return q

def shuffled_chart(h,z,chart_family,k,binary,seed):
    rng=np.random.default_rng(seed);zz=np.asarray(z)[rng.permutation(len(z))]
    return fit_scalar_chart(np.asarray(h).reshape(-1),zz,chart_family,binary) if k==1 else fit_projected_chart(h,zz,chart_family,binary)
def fit_same_capacity(h,z,chart_family,k,binary):
    return fit_scalar_chart(np.asarray(h).reshape(-1),z,chart_family,binary) if k==1 else fit_projected_chart(h,z,chart_family,binary)
def matched_p(canonical_recovery,control_recovery,seed):
    return float(paired_sign_flip_p(np.asarray(canonical_recovery)-np.asarray(control_recovery),seed))
