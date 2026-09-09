from __future__ import annotations
import numpy as np

def compress_interaction(tensor:np.ndarray,rank:int):
    q,din,dout=tensor.shape
    flat=tensor.reshape(q,din*dout)
    u,s,vt=np.linalg.svd(flat,full_matrices=False)
    r=min(int(rank),len(s))
    context_factors=u[:,:r]*s[:r]
    delta_matrices=vt[:r].reshape(r,din,dout)
    reconstructed=(context_factors@vt[:r]).reshape(q,din,dout)
    diagnostics={
      "requested_rank":int(rank),"effective_rank":int(r),"singular_values":s.tolist(),
      "retained_fraction":float(np.sum(s[:r]**2)/max(np.sum(s**2),1e-30)),
      "reconstruction_mse":float(np.mean((reconstructed-tensor)**2)),
    }
    return context_factors,delta_matrices,diagnostics
