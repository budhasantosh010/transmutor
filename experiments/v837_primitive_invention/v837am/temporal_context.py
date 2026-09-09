from __future__ import annotations

from .context_features import context_tensor

HISTORIES=(2,4,8)

def temporal_context_tensor(trace,nodes,li,target_port,history):
    if history not in HISTORIES:raise ValueError(history)
    return context_tensor(trace,nodes,li,"C5_FULL_LOCAL_CONTEXT",target_port,history)
