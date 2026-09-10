from __future__ import annotations

import numpy as np

from .canonical_writer import set_state
from .gauge_fix import biorthogonal_projector


def projector(component: dict) -> np.ndarray:
    return biorthogonal_projector(component["reader"], np.asarray(component["writer"], dtype=np.float64))


def residual(state: np.ndarray, component: dict) -> np.ndarray:
    s=np.asarray(state,dtype=np.float64);P=projector(component);return s @ (np.eye(40)-P).T


def residual_difference(a: np.ndarray,b: np.ndarray,component:dict)->np.ndarray:
    return residual(np.asarray(a)-np.asarray(b),component)


def projector_diagnostics(component:dict)->dict:
    P=projector(component);I=np.eye(40);w=np.asarray(component["writer"],dtype=np.float64);idem=float(np.max(np.abs(P@P-P)));ann=float(np.max(np.abs((I-P)@w)));return {"projector_idempotence_max_abs":idem,"residual_writer_annihilation_max_abs":ann,"pass":bool(idem<=1e-10 and ann<=1e-10)}


def residual_invariance_under_set(state:np.ndarray,target:float,component:dict)->float:
    before=residual(state,component);after=residual(set_state(state,target,component["reader"],np.asarray(component["writer"])),component);return float(np.max(np.abs(after-before)))
