from __future__ import annotations

from collections import defaultdict
import numpy as np
from .utils import AQ,HERE,read_json,write_json

FAMILIES=("conditional_routing","delayed_recall","iterative_state")

def rank_at_energy(matrix,energy):
    a=np.asarray(matrix,float)
    if a.size==0:return 0,[],[]
    s=np.linalg.svd(a,full_matrices=False,compute_uv=False);e=s*s;tot=float(e.sum())
    if tot<=1e-15:return 0,s.tolist(),[0.0 for _ in s]
    cum=np.cumsum(e)/tot;return int(np.searchsorted(cum,float(energy))+1),s.tolist(),(e/tot).tolist()
def _key(r):
    mag=None if r.get("actual_delta") is None else round(abs(float(r["actual_delta"])),6)
    return (r.get("intervention"),r.get("phase"),int(r.get("horizon",1)),mag,int(r.get("order",1)))
def build_hankel():
    out={"version":"V837ar","state_definition":"semantic history x future intervention tests; no neural coordinates","families":{}}
    ranks={}
    for fam in FAMILIES:
        src=read_json(AQ/f"raw/response_tensor_{fam}.json");by_seed=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for r in src["rows"]:by_seed[int(r["seed"])][_key(r)][r["organism_id"]].append(float(r["predicted_response"]))
        keys=None
        for s,d in by_seed.items():keys=set(d) if keys is None else keys&set(d)
        cols=sorted(keys or set(),key=str);seeds=sorted(by_seed)
        mat=[]
        for s in seeds:
            row=[]
            for k in cols:
                orgmeans=[float(np.mean(v)) for v in by_seed[s][k].values()];row.append(float(np.mean(orgmeans)))
            mat.append(row)
        M=np.asarray(mat,float);rr={}
        for name,e in (("rank90",.90),("rank95",.95),("rank99",.99),("rank999",.999)):
            r,sv,ef=rank_at_energy(M,e);rr[name]=r
        r99,sv,ef=rank_at_energy(M,.99);out["families"][fam]={"histories":len(seeds),"future_tests":len(cols),"columns":[list(k) for k in cols],"matrix":M.tolist(),"singular_values":sv,"energy_fractions":ef,**rr};ranks[fam]=rr
    write_json(HERE/"raw/predictive_hankel.json",out);write_json(HERE/"raw/predictive_ranks.json",{"version":"V837ar","families":ranks});write_json(HERE/"diagnostics/hankel_rank.json",{"version":"V837ar","families":ranks});return out
