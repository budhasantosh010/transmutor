from __future__ import annotations

import numpy as np
from experiments.v837_primitive_invention.v837an.an_a_causal_macrovariables import _eligibility
from experiments.v837_primitive_invention.v837an.trace_cache import pair_traces
from experiments.v837_primitive_invention.v837ao.k1_backend import _phase_arrays
from experiments.v837_primitive_invention.v837ao.phase_backends import PHASES


def phase_dataset(organism_id:str,family:str,seeds:list[int],q:np.ndarray,phase:str)->dict:
    data=pair_traces(organism_id,family,seeds);mask,power=_eligibility(data,family);b,c,z0,z1=_phase_arrays(data,mask,family,phase)
    q=np.asarray(q,dtype=np.float64);states=np.concatenate([b,c],axis=0) if len(b) else np.empty((0,40));semantic=np.concatenate([z0,z1],axis=0) if len(b) else np.empty(0);h=states@q if len(states) else np.empty((0,q.shape[1]));return {"states":states,"semantic":semantic,"h":h,"base_states":b,"cf_states":c,"base_semantic":z0,"cf_semantic":z1,"data":data,"mask":mask,"power":power}

def pooled_phase_dataset(organism_id:str,family:str,seeds:list[int],q:np.ndarray)->dict:
    chunks=[phase_dataset(organism_id,family,seeds,q,p) for p in PHASES[family]];states=[];semantic=[];h=[];phases=[]
    for p,ch in zip(PHASES[family],chunks):
        states.append(ch["states"]);semantic.append(ch["semantic"]);h.append(ch["h"]);phases.extend([p]*len(ch["semantic"]))
    return {"states":np.concatenate(states) if states else np.empty((0,40)),"semantic":np.concatenate(semantic) if semantic else np.empty(0),"h":np.concatenate(h) if h else np.empty((0,np.asarray(q).shape[1])),"phases":phases,"chunks":dict(zip(PHASES[family],chunks))}
