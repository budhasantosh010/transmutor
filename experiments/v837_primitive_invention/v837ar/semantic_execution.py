from __future__ import annotations

import numpy as np
import torch

from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.v837an.instrumented_af1d import load_model_by_id, run_instrumented
from .utils import HERE, read_json


def folds() -> dict:
    p = HERE / "raw/frozen_source_folds.json"
    if not p.is_file():
        from .source_contracts import load_aq_contracts
        return load_aq_contracts()["folds"]
    return read_json(p)


def discovery_ids(family: str) -> list[str]: return list(folds()["families"][family]["discovery"])
def reused_heldout_ids(family: str) -> list[str]: return list(folds()["families"][family]["holdout"])
def engine_for(family: str, organism_id: str) -> str:
    f=folds()["families"][family]["engines"]
    for eng,d in f.items():
        if organism_id in d.get("discovery",[]) or organism_id in d.get("holdout",[]): return eng
    raise KeyError(organism_id)


def predict_episodes(organism_id: str, episodes, chunk_size: int = 1024) -> np.ndarray:
    model,row,_,_=load_model_by_id(organism_id)
    values=[]; model.eval()
    for i in range(0,len(episodes),chunk_size):
        obs,lengths,_=episodes_to_batch(episodes[i:i+chunk_size])
        with torch.no_grad(): pred=run_instrumented(model,obs,lengths,return_trace=False)
        values.extend(pred.detach().cpu().numpy().astype(np.float64).tolist())
    return np.asarray(values,dtype=np.float64)
