from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import numpy as np
from sklearn.neighbors import NearestNeighbors

from experiments.v837_primitive_invention.v837ak.dynamic_fingerprint import load_size_matrix
from experiments.v837_primitive_invention.v837ak.fingerprint_reliability import calibrate_reliability
from experiments.v837_primitive_invention.v837ak.structural_signatures import structural_class_id, structural_signature
from experiments.v837_primitive_invention.v837ak.reconstruct_organisms import _load_rows
from experiments.v837_primitive_invention.v837ak.utils import DIRECTED,RANDOM,HERE,normalize,read_json,sha256_json,write_json


class DSU:
    def __init__(self,n:int):self.p=list(range(n))
    def find(self,x):
        while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def union(self,a,b):
        a=self.find(a);b=self.find(b)
        if a!=b:self.p[b]=a


def _refine_component(indices:list[int],matrix:np.ndarray,tau:float)->tuple[list[int],int]:
    active=sorted(indices)
    while active:
        sub=matrix[active]; center=np.mean(sub,axis=0); local_medoid=int(np.argmin(np.linalg.norm(sub-center,axis=1))); medoid=active[local_medoid]
        dist=np.linalg.norm(sub-matrix[medoid],axis=1)/math.sqrt(matrix.shape[1]); kept=[idx for idx,d in zip(active,dist) if float(d)<=tau+1e-12]
        if kept==active:return kept,medoid
        active=kept
    return [],-1


def cluster_dynamic_classes()->dict:
    rel_path=HERE/"diagnostics/fingerprint_reliability.json"
    if not rel_path.is_file():calibrate_reliability()
    reliability=read_json(rel_path); thresholds=read_json(HERE/"diagnostics/fingerprint_thresholds.json")["sizes"]
    source={r["organism_id"]:r for r in _load_rows()}; classes=[]
    for k in reliability["eligible_sizes"]:
        records,raw=load_size_matrix(k,"mean",competent_only=True); th=thresholds[str(k)]; matrix=normalize(raw,np.asarray(th["median"]),np.asarray(th["iqr"])); tau=float(th["tau"])
        n=len(records); neighbors=min(65,n); nn=NearestNeighbors(n_neighbors=neighbors,metric="euclidean",algorithm="auto").fit(matrix); distances,indices=nn.kneighbors(matrix)
        dsu=DSU(n); scale=math.sqrt(matrix.shape[1])
        for i in range(n):
            for d,j in zip(distances[i,1:],indices[i,1:]):
                if float(d)/scale<=tau+1e-12:dsu.union(i,int(j))
        components=defaultdict(list)
        for i in range(n):components[dsu.find(i)].append(i)
        for comp in components.values():
            refined,medoid=_refine_component(comp,matrix,tau)
            if not refined:continue
            by_org=defaultdict(list)
            for idx in refined:by_org[records[idx]["organism_id"]].append(idx)
            selected=[]
            for oid,idxs in by_org.items():
                idx=min(idxs,key=lambda q:(float(np.linalg.norm(matrix[q]-matrix[medoid])),records[q]["occurrence_id"]));selected.append(idx)
            members=[records[i] for i in sorted(selected,key=lambda i:records[i]["organism_id"])]
            if len(members)<3:continue
            engine=defaultdict(int); fam=defaultdict(int); structures=defaultdict(int)
            for m in members:
                engine[m["engine"]]+=1;fam[m["family"]]+=1; sr=source[m["organism_id"]]; structures[structural_class_id(structural_signature(sr["topology"],m["nodes"]))]+=1
            dominant=max(structures.values())/len(members); topology_transcending=len(structures)>=2 and dominant<=0.80
            cid=sha256_json({"stream":"D","size":k,"medoid":records[medoid]["occurrence_id"],"support":[m["organism_id"] for m in members]})
            classes.append({
                "class_id":cid,"stream":"D","size":k,"tau":tau,"medoid_occurrence_id":records[medoid]["occurrence_id"],"medoid_organism_id":records[medoid]["organism_id"],"medoid_nodes":records[medoid]["nodes"],
                "support":len(members),"support_organisms":[m["organism_id"] for m in members],"member_occurrences":members,"engine_counts":dict(engine),"family_counts":dict(fam),
                "structural_signature_counts":dict(structures),"topology_transcending":bool(topology_transcending),"dominant_structural_fraction":float(dominant),
                "general_eligible":len(members)>=6 and engine[DIRECTED]>=2 and engine[RANDOM]>=2 and len(fam)>=3,
                "family_specific_eligible":any(count>=3 and count>=0.60*sum(1 for r in read_json(HERE/"raw/reconstruction_results.json")["rows"] if r["competent"] and r["family"]==family) for family,count in fam.items()),
                "discovery_stability":float(max(0.0,1.0-np.median([np.linalg.norm(matrix[i]-matrix[medoid])/scale for i in selected])/max(tau,1e-12))),
            })
    classes.sort(key=lambda c:(-c["support"],-min(c["engine_counts"].get(DIRECTED,0),c["engine_counts"].get(RANDOM,0)),-len(c["family_counts"]),-c["discovery_stability"],c["size"],c["class_id"]))
    payload={"version":"V837ak","stage":"AK5-D","classes":classes,"eligible_classes":[c["class_id"] for c in classes if c["general_eligible"] or c["family_specific_eligible"]],"topology_transcending_classes":[c["class_id"] for c in classes if c["topology_transcending"]]}
    write_json(HERE/"raw/dynamic_classes_discovery.json",payload);write_json(HERE/"diagnostics/dynamic_recurrence.json",{"version":"V837ak","class_count":len(classes),"eligible_count":len(payload["eligible_classes"]),"top_classes":classes[:50]});write_json(HERE/"diagnostics/topology_transcendence.json",{"version":"V837ak","count":len(payload["topology_transcending_classes"]),"class_ids":payload["topology_transcending_classes"]})
    return payload


def main()->int:
    p=cluster_dynamic_classes();print(json.dumps({"classes":len(p["classes"]),"eligible":len(p["eligible_classes"]),"topology_transcending":len(p["topology_transcending_classes"])},indent=2));return 0

if __name__=="__main__":raise SystemExit(main())
