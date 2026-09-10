from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch

from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name

from .counterfactual_tasks import make_counterfactual
from .instrumented_af1d import AF1DTrace, load_model_by_id, run_instrumented
from .utils import HERE, deterministic_seed, sha256_file, write_json

CACHE_DIR=HERE/"cache"
_MEM:dict[tuple,dict]={}


def pair_traces(organism_id:str,family:str,seeds:list[int])->dict:
    key=(organism_id,family,tuple(seeds))
    if key in _MEM:return _MEM[key]
    model,row,source,checkpoint=load_model_by_id(organism_id);split="validation" if seeds and min(seeds)>=20000 else "development";pairs=[make_counterfactual(family,s,split=split) for s in seeds]
    base_eps=[p.base_episode.as_episode() for p in pairs];cf_eps=[p.counterfactual_episode.as_episode() for p in pairs]
    bo,bl,bt=episodes_to_batch(base_eps);co,cl,ct=episodes_to_batch(cf_eps)
    with torch.no_grad():bp,btr=run_instrumented(model,bo,bl,return_trace=True);cp,ctr=run_instrumented(model,co,cl,return_trace=True)
    payload={"model":model,"row":row,"pairs":pairs,"base_obs":bo,"base_lengths":bl,"base_targets":bt,"cf_obs":co,"cf_lengths":cl,"cf_targets":ct,"base_prediction":bp,"cf_prediction":cp,"base_trace":btr,"cf_trace":ctr}
    _MEM[key]=payload;return payload


def clear_memory()->None:_MEM.clear()


def cache_manifest()->dict:
    files=[]
    if CACHE_DIR.is_dir():
        for p in CACHE_DIR.rglob("*"):
            if p.is_file():files.append({"path":p.relative_to(HERE).as_posix(),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    payload={"version":"V837an","cache_reconstructable":True,"gitignored":True,"files":files,"total_bytes":sum(x["bytes"] for x in files)};write_json(HERE/"raw/trace_manifest.json",payload);return payload
