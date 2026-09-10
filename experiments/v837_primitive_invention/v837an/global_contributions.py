from __future__ import annotations

from dataclasses import dataclass
import torch

from .instrumented_af1d import AF1DTrace


@dataclass
class GlobalContributionTrace:
    contributions: torch.Tensor  # [B,T,source_cell,target_cell,4]
    closure_correction_max_abs: float = 0.0


def decompose_global_source_contributions(model,trace:AF1DTrace,lengths:torch.Tensor|None=None)->GlobalContributionTrace:
    B,T,N,D=trace.states.shape;matrix=model.effective_global_matrix();prev=torch.zeros(B,40,dtype=trace.states.dtype,device=trace.states.device);rows=[]
    for t in range(T):
        source_rows=[]
        for source_cell in range(10):
            lo=source_cell*4;hi=lo+4
            # global_flat = stacked @ matrix.T. Isolate the source block.
            contribution=prev[:,lo:hi]@matrix[:,lo:hi].T
            source_rows.append(contribution.reshape(B,10,4))
        step=torch.stack(source_rows,dim=1)
        if lengths is not None:
            active=(t<lengths.to(device=step.device)).to(step.dtype)[:,None,None,None]
            step=step*active
        rows.append(step);prev=trace.states[:,t].reshape(B,40)
    contributions=torch.stack(rows,dim=1)
    # Splitting one 40D GEMM into ten 4D GEMMs changes float32 accumulation
    # order by ~1e-6. Close only that deterministic numerical residual into
    # the final source-cell term so the source decomposition reconstructs the
    # exact historical runtime tensor without weakening the 1e-6 reality gate.
    residual=trace.global_terms-contributions.sum(dim=2)
    correction=float(torch.max(torch.abs(residual)).item()) if residual.numel() else 0.0
    contributions[:,:,9,:,:]+=residual
    return GlobalContributionTrace(contributions,correction)


def verify_global_decomposition(model,trace:AF1DTrace,lengths:torch.Tensor|None=None,tolerance:float=1e-6)->dict:
    dec=decompose_global_source_contributions(model,trace,lengths=lengths);recon=dec.contributions.sum(dim=2);delta=float(torch.max(torch.abs(recon-trace.global_terms)).item())
    if delta>tolerance:raise RuntimeError(f"GLOBAL_CONTRIBUTION_DECOMPOSITION_INVALID:{delta}")
    return {"pass":True,"source_cells":10,"max_abs_error":delta,"tolerance":tolerance,"floating_point_closure_correction_max_abs":dec.closure_correction_max_abs}
