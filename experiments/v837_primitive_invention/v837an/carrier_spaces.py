from __future__ import annotations

import torch

from .authorization import CARRIERS, DIAGNOSTIC_CARRIERS, LATENT_DIMS
from .instrumented_af1d import AF1DTrace

CARRIER_DIM={"STATE40":40,"OUTPUT40":40,"MESSAGE40":40,"GLOBAL40":40,"GATE1":1,"COUPLING_FACTOR4":4}
CARRIER_DEPLOYMENT_COST={"GATE1":1,"STATE40":40,"OUTPUT40":40,"MESSAGE40":40,"GLOBAL40":40,"COUPLING_FACTOR4":4}


def carrier_at(trace:AF1DTrace,name:str,timesteps:torch.Tensor|list[int]|int)->torch.Tensor:
    x=trace.carrier(name)
    if isinstance(timesteps,int):return x[:,timesteps,:]
    idx=torch.as_tensor(timesteps,dtype=torch.long,device=x.device)
    if idx.ndim==1 and idx.numel()==x.shape[0]:return x[torch.arange(x.shape[0],device=x.device),idx]
    return x[:,idx,:]


def validate_carrier_schema()->dict:
    return {"carriers":CARRIERS,"diagnostic":DIAGNOSTIC_CARRIERS,"dimensions":CARRIER_DIM,"latent_dimensions":LATENT_DIMS,"coupling_factor4_diagnostic_only":True,"cross_organism_state_invertibility_required":False}
