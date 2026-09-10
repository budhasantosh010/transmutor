from __future__ import annotations

import json
import numpy as np

from experiments.v837_primitive_invention.v837an.an_a_causal_macrovariables import _eligibility
from experiments.v837_primitive_invention.v837an.counterfactual_subspace import fit_difference_subspace
from experiments.v837_primitive_invention.v837an.trace_cache import pair_traces
from experiments.v837_primitive_invention.v837ao.k1_backend import _primary_arrays

from .authorization import POWERED_FAMILIES
from .data_roles import seeds
from .source_folds import freeze_source_folds
from .utils import HERE, sha256_json, write_json

K_VALUES=(1,2,4,8)

def reconstruct_space(organism_id:str,family:str,fit_seeds:list[int],k:int)->dict:
    data=pair_traces(organism_id,family,fit_seeds); mask,power=_eligibility(data,family)
    if not np.any(mask): return {"valid":False,"organism_id":organism_id,"family":family,"k":k,"power":power}
    base,cf,dz=_primary_arrays(data,mask); sub=fit_difference_subspace(base,cf,k)
    q=np.asarray(sub["q"],dtype=np.float64)
    return {"valid":True,"organism_id":organism_id,"family":family,"engine":data["row"]["engine"],"k":k,"q":q.tolist(),"singular_values":np.asarray(sub["singular_values"]).tolist(),"eligible_pairs":int(np.sum(mask)),"source":"V837an counterfactual-difference SVD","invertibility_required":False,"cross_organism_alignment":False,"subspace_sha256":sha256_json(q.tolist())}

def run_projected_spaces()->dict:
    folds=freeze_source_folds(); rows=[]; nesting=[]
    for family in POWERED_FAMILIES:
        for idx,oid in enumerate(folds["families"][family]["discovery"],1):
            spaces={k:reconstruct_space(oid,family,seeds("AP_CHART_FIT"),k) for k in K_VALUES}
            rows.extend(spaces.values())
            residuals={}
            for a,b in ((1,2),(2,4),(4,8)):
                qa=np.asarray(spaces[a]["q"]);qb=np.asarray(spaces[b]["q"]);r=float(np.linalg.norm(qa-qb@(qb.T@qa),ord=2));residuals[f"K{a}_in_K{b}"]=r
            passed=max(residuals.values())<=1e-8
            if not passed: raise RuntimeError(f"PROJECTED_SUBSPACE_RECONSTRUCTION_DRIFT:{family}:{oid}:{residuals}")
            nesting.append({"family":family,"organism_id":oid,"engine":spaces[1]["engine"],"residuals":residuals,"pass":True})
            print(f"V837ap subspaces {family} {idx}/{len(folds['families'][family]['discovery'])} maxnest={max(residuals.values()):.3e}",flush=True)
    payload={"version":"V837ap","fit_partition":"AP_CHART_FIT","k_values":list(K_VALUES),"rows":rows}
    write_json(HERE/"raw/projected_subspace_hashes.json",payload);write_json(HERE/"diagnostics/subspace_nesting.json",{"version":"V837ap","threshold":1e-8,"rows":nesting,"pass":all(r["pass"] for r in nesting)})
    return payload

def space_map()->dict:
    import pathlib
    p=HERE/"raw/projected_subspace_hashes.json"
    if not p.is_file(): run_projected_spaces()
    rows=json.loads(p.read_text(encoding="utf-8"))["rows"]
    return {(r["family"],r["organism_id"],int(r["k"])):r for r in rows}

if __name__=="__main__": print(json.dumps(run_projected_spaces(),indent=2))
