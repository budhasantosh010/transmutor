from __future__ import annotations

from collections import defaultdict
import numpy as np
import torch

from .interface_data import collect_trace,load_model,mask
from .utils import HERE,read_json,seeds,write_json

FIT=seeds(10128,10191)

def _robust(x):
    vals=np.asarray(list(x.values()),dtype=float);med=np.median(vals);mad=np.median(np.abs(vals-med));scale=mad if mad>1e-12 else (np.std(vals) if np.std(vals)>1e-12 else 1.0);return {k:(v-med)/scale for k,v in x.items()}
def _ranking(occ):
    nodes=set(occ["nodes"]);outside=[j for j in range(10) if j not in nodes];tr=collect_trace(occ["organism_id"],occ["nodes"],FIT);model,_=load_model(occ["organism_id"]);full=tr.states;prev=torch.zeros_like(full);prev[:,1:]=full[:,:-1];m=mask(tr);msg={j:0.0 for j in outside};glob={j:0.0 for j in outside}
    # Message influence: actual weighted source-output magnitudes on graph edges crossing the motif boundary.
    for ei,e in enumerate(model.graph.edges):
        s,d=int(e.src),int(e.dst)
        if not ((s in nodes)^(d in nodes)):continue
        j=s if s in outside else d
        src=tr.outputs[:,:,s,:] if not bool(e.recurrent) and s<d else torch.cat([torch.zeros_like(tr.outputs[:,:1,s,:]),tr.outputs[:,:-1,s,:]],1)
        msg[j]+=float(torch.linalg.vector_norm(model.base.edge_weights[ei].detach().cpu()*src[m],dim=-1).mean())
    M=model.effective_global_matrix().detach().cpu();
    for j in outside:
        scores=[]
        for target in nodes:
            block=M[target*4:(target+1)*4,j*4:(j+1)*4];contrib=prev[:,:,j,:]@block.T;scores.append(torch.linalg.vector_norm(contrib[m],dim=-1))
        for source in nodes:
            block=M[j*4:(j+1)*4,source*4:(source+1)*4];contrib=prev[:,:,source,:]@block.T;scores.append(torch.linalg.vector_norm(contrib[m],dim=-1))
        glob[j]=float(torch.cat(scores).mean()) if scores else 0.0
    mz=_robust(msg);gz=_robust(glob);combined={j:mz[j]+gz[j] for j in outside}
    def rank(x):return [j for j,_ in sorted(x.items(),key=lambda kv:(-kv[1],kv[0]))]
    return {"organism_id":occ["organism_id"],"occurrence_id":occ["occurrence_id"],"original_nodes":occ["nodes"],"message_scores":msg,"global_scores":glob,"combined_scores":combined,"message_ranking":rank(msg),"global_ranking":rank(glob),"combined_ranking":rank(combined)}
def freeze_rankings():
    pairs=read_json(HERE/"raw/frozen_pairs.json")["pairs"];uniq={}
    for p in pairs:
        for key in ("recipient","same_class_donor","different_class_donor"):
            o=p[key];uniq[o["occurrence_id"]]=o
    rows=[_ranking(uniq[k]) for k in sorted(uniq)];payload={"version":"V837am","stage":"AM-B-RANKING","fit_seeds":[10128,10191],"ranking_frozen_before_selection":True,"occurrence_count":len(rows),"rows":rows};write_json(HERE/"raw/frozen_context_cell_rankings.json",payload);write_json(HERE/"diagnostics/boundary_influence.json",payload);return payload
if __name__=="__main__":freeze_rankings()
