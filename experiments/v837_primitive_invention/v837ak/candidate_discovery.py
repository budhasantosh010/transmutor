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

from experiments.v837_primitive_invention.v837ak.dynamic_clustering import cluster_dynamic_classes
from experiments.v837_primitive_invention.v837ak.dynamic_fingerprint import load_size_matrix
from experiments.v837_primitive_invention.v837ak.subset_census import run_census
from experiments.v837_primitive_invention.v837ak.utils import DIRECTED,RANDOM,HERE,jaccard,normalize,read_json,sha256_json,write_json


def _eligible_structural_classes()->list[dict]:
    path=HERE/"raw/structural_classes.json"
    if not path.is_file():run_census()
    classes=read_json(path)["classes"]; recon=read_json(HERE/"raw/reconstruction_results.json")["rows"]
    family_total=defaultdict(int)
    for r in recon:
        if r["competent"]:family_total[r["family"]]+=1
    out=[]
    for c in classes:
        eng=c["engine_counts"]; fam=c["family_counts"]; support=int(c["competent_support"])
        general=support>=6 and eng.get(DIRECTED,0)>=2 and eng.get(RANDOM,0)>=2 and len(fam)>=3
        family_specific=any(count>=3 and count>=0.60*family_total[family] for family,count in fam.items())
        if not (general or family_specific):continue
        members=[x for x in c["representative_occurrences"] if x["competent"]]
        out.append({
            "class_id":c["class_id"],"stream":"S","size":c["size"],"support":support,"support_organisms":sorted(c["competent_organisms"]),"member_occurrences":members,
            "engine_counts":eng,"family_counts":fam,"general_eligible":general,"family_specific_eligible":family_specific,"global_promotion_eligible":support>=6,
            "topology_transcending":False,"structural_signatures":[c["class_id"]],"discovery_stability":1.0,
        })
    return out


def _rank_key(c:dict):
    balance=min(c["engine_counts"].get(DIRECTED,0),c["engine_counts"].get(RANDOM,0))
    return (-int(c["support"]),-balance,-len(c["family_counts"]),-float(c.get("discovery_stability",0.0)),int(c["size"]),c["class_id"])


def _dedup_ranked(classes:list[dict],limit:int)->list[dict]:
    chosen=[]
    for c in sorted(classes,key=_rank_key):
        if any(jaccard(c["support_organisms"],x["support_organisms"])>=0.80 for x in chosen):continue
        chosen.append(c)
        if len(chosen)>=limit:break
    if len(chosen)<limit:
        for c in sorted(classes,key=_rank_key):
            if c not in chosen:
                chosen.append(c)
                if len(chosen)>=limit:break
    return chosen


def _fingerprint_context(candidates:list[dict])->None:
    thresholds=read_json(HERE/"diagnostics/fingerprint_thresholds.json")["sizes"]
    by_size=defaultdict(list)
    for c in candidates:by_size[int(c["size"])].append(c)
    for k,items in by_size.items():
        th=thresholds[str(k)]
        if not th["eligible"]:
            for c in items:c.update({"fingerprint_size_eligible":False,"tau":None,"discovery_medoid":None,"discovery_medoid_occurrence_id":None})
            continue
        records,raw=load_size_matrix(k,"mean",competent_only=True); mat=normalize(raw,np.asarray(th["median"]),np.asarray(th["iqr"])); idx={r["occurrence_id"]:i for i,r in enumerate(records)}
        for c in items:
            member_ids=[m["occurrence_id"] for m in c["member_occurrences"] if m["occurrence_id"] in idx]
            if not member_ids:
                c.update({"fingerprint_size_eligible":True,"tau":float(th["tau"]),"discovery_medoid":None,"discovery_medoid_occurrence_id":None});continue
            member_idx=[idx[x] for x in member_ids]; sub=mat[member_idx]; center=sub.mean(axis=0); local=int(np.argmin(np.linalg.norm(sub-center,axis=1))); medoid_idx=member_idx[local]
            c.update({"fingerprint_size_eligible":True,"tau":float(th["tau"]),"discovery_medoid":mat[medoid_idx].tolist(),"discovery_medoid_occurrence_id":records[medoid_idx]["occurrence_id"],"discovery_member_median_distance":float(np.median(np.linalg.norm(sub-mat[medoid_idx],axis=1)/math.sqrt(mat.shape[1])))})


def freeze_candidate_classes()->dict:
    dynamic_path=HERE/"raw/dynamic_classes_discovery.json"
    if not dynamic_path.is_file():cluster_dynamic_classes()
    s=_eligible_structural_classes(); d=[dict(c,global_promotion_eligible=int(c["support"])>=6,structural_signatures=list(c["structural_signature_counts"])) for c in read_json(dynamic_path)["classes"] if c["general_eligible"] or c["family_specific_eligible"]]
    # Freeze at most six classes from each independent stream. Deduplication is
    # global across already chosen support sets; an empty slot may be filled
    # only by the next-ranked class from the same stream, never by exceeding
    # the other stream's six-class cap.
    final=[]
    pools={"S":sorted(s,key=_rank_key),"D":sorted(d,key=_rank_key)}
    for stream in ("S","D"):
        count=0
        for c in pools[stream]:
            if any(jaccard(c["support_organisms"],x["support_organisms"])>=0.80 for x in final):
                continue
            final.append(c);count+=1
            if count>=6:
                break
    _fingerprint_context(final)
    for rank,c in enumerate(final,1):c["candidate_rank"]=rank;c["whole_system_control"]=int(c["size"])==10;c["reusable_subsystem_eligible"]=int(c["size"])<10 and bool(c["global_promotion_eligible"])
    core={"version":"V837ak","stage":"AK5","classes":final,"structural_selected":sum(c["stream"]=="S" for c in final),"dynamic_selected":sum(c["stream"]=="D" for c in final),"candidate_cap":12,"selection_uses_task_performance":False}
    payload=dict(core);payload["frozen_candidate_classes_sha256"]=sha256_json(core)
    write_json(HERE/"raw/frozen_candidate_classes.json",payload)
    write_json(HERE/"diagnostics/candidate_freeze.json",{"version":"V837ak","frozen":True,"sha256":payload["frozen_candidate_classes_sha256"],"class_ids":[c["class_id"] for c in final],"confirmation_probe_used":False})
    return payload


def main()->int:
    p=freeze_candidate_classes();print(json.dumps({"classes":len(p["classes"]),"structural":p["structural_selected"],"dynamic":p["dynamic_selected"],"sha256":p["frozen_candidate_classes_sha256"]},indent=2));return 0

if __name__=="__main__":raise SystemExit(main())
