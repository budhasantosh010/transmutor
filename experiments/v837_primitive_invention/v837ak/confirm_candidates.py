from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np

from experiments.v837_primitive_invention.v837ak.boundary_traces import FullProbeTrace, run_full_probe
from experiments.v837_primitive_invention.v837ak.candidate_discovery import freeze_candidate_classes
from experiments.v837_primitive_invention.v837ak.dynamic_fingerprint import _edge_contributions, fingerprint_occurrence
from experiments.v837_primitive_invention.v837ak.reconstruct_organisms import load_reconstructed_model
from experiments.v837_primitive_invention.v837ak.utils import DIRECTED,RANDOM,HERE,normalize,read_json,sha256_json,write_json

CONFIRMATION_SEEDS=list(range(20064,20128))


def _slice(trace:FullProbeTrace,a:int,b:int)->FullProbeTrace:
    return FullProbeTrace(**{f.name:getattr(trace,f.name)[a:b] for f in fields(FullProbeTrace)})


def confirm_candidates()->dict:
    frozen_path=HERE/"raw/frozen_candidate_classes.json"
    if not frozen_path.is_file():freeze_candidate_classes()
    frozen=read_json(frozen_path); expected_hash=frozen["frozen_candidate_classes_sha256"]; core={k:v for k,v in frozen.items() if k!="frozen_candidate_classes_sha256"}
    if sha256_json(core)!=expected_hash:raise RuntimeError("V837AK_FROZEN_CANDIDATE_HASH_MISMATCH")
    recon={r["organism_id"]:r for r in read_json(HERE/"raw/reconstruction_results.json")["rows"]}; thresholds=read_json(HERE/"diagnostics/fingerprint_thresholds.json")["sizes"]
    trace_cache={}; model_cache={}; split_cache={}; occurrence_results=defaultdict(list); self_by_size=defaultdict(list)

    def vectors_for(oid:str,nodes:list[int],k:int):
        if oid not in trace_cache:trace_cache[oid]=run_full_probe(oid,CONFIRMATION_SEEDS,cache_key="confirmation64")
        if oid not in model_cache:model_cache[oid]=load_reconstructed_model(recon[oid])[0]
        model=model_cache[oid]
        if oid not in split_cache:
            c1=_slice(trace_cache[oid],0,32);c2=_slice(trace_cache[oid],32,64)
            split_cache[oid]=(c1,c2,_edge_contributions(model,c1),_edge_contributions(model,c2))
        c1,c2,e1,e2=split_cache[oid]
        th=thresholds[str(k)];median=np.asarray(th["median"]);iqr=np.asarray(th["iqr"])
        v1=normalize(fingerprint_occurrence(model,c1,nodes,e1)[None,:],median,iqr)[0]
        v2=normalize(fingerprint_occurrence(model,c2,nodes,e2)[None,:],median,iqr)[0]
        return v1,v2

    # Confirmation self-stability thresholds are class-independent. Build a
    # deterministic round-robin calibration sample from the full competent
    # census for each frozen candidate size; no class IDs or confirmation
    # compatibility labels participate in this threshold.
    census_rows=read_json(HERE/"raw/cache/subset_occurrences.json")["rows"]
    candidate_sizes=sorted({int(c["size"]) for c in frozen["classes"] if c.get("fingerprint_size_eligible")})
    for k in candidate_sizes:
        by_org=defaultdict(list)
        for row in census_rows:
            if row["competent"] and int(row["size"])==k:
                by_org[row["organism_id"]].append(row)
        for rows in by_org.values():rows.sort(key=lambda r:r["occurrence_id"])
        sampled=[];depth=0
        while len(sampled)<256:
            added=False
            for oid in sorted(by_org):
                rows=by_org[oid]
                if depth<len(rows):
                    sampled.append(rows[depth]);added=True
                    if len(sampled)>=256:break
            if not added:break
            depth+=1
        for row in sampled:
            v1,v2=vectors_for(row["organism_id"],row["nodes"],k)
            self_by_size[k].append(float(np.linalg.norm(v1-v2)/math.sqrt(len(v1))))
    confirmation_thresholds={k:float(np.percentile(vals,95)) for k,vals in self_by_size.items() if vals}

    for c in frozen["classes"]:
        if not c.get("fingerprint_size_eligible") or c.get("discovery_medoid") is None:continue
        k=int(c["size"])
        for member in c["member_occurrences"]:
            v1,v2=vectors_for(member["organism_id"],member["nodes"],k)
            self_dist=float(np.linalg.norm(v1-v2)/math.sqrt(len(v1)));avg=(v1+v2)/2.0;medoid=np.asarray(c["discovery_medoid"]);class_dist=float(np.linalg.norm(avg-medoid)/math.sqrt(len(avg)))
            row={**member,"c1_c2_self_distance":self_dist,"confirmation_class_distance":class_dist,"compatible":class_dist<=float(c["tau"])+1e-12}
            occurrence_results[c["class_id"]].append(row)
    classes=[]
    for c in frozen["classes"]:
        rows=occurrence_results.get(c["class_id"],[]);retained=[r for r in rows if r["compatible"]];eng=defaultdict(int)
        for r in retained:eng[r["engine"]]+=1
        retention=len(retained)/max(1,len(c["member_occurrences"]));engine_ok=(eng[DIRECTED]>=2 and eng[RANDOM]>=2) if c.get("general_eligible") else (eng[DIRECTED]>=1 and eng[RANDOM]>=1)
        dynamic_extra=True
        median_distance=float(np.median([r["confirmation_class_distance"] for r in rows])) if rows else float("inf")
        if c["stream"]=="D":dynamic_extra=bool(rows) and median_distance<=confirmation_thresholds.get(int(c["size"]),-1)+1e-12
        confirmed=bool(c.get("fingerprint_size_eligible")) and bool(rows) and retention>=0.70 and engine_ok and dynamic_extra
        classes.append({**c,"confirmation_rows":rows,"retained_organisms":sorted(r["organism_id"] for r in retained),"retention_fraction":retention,"retained_engine_counts":dict(eng),"engine_coverage_ok":engine_ok,"median_confirmation_distance":median_distance if np.isfinite(median_distance) else None,"confirmation_self_stability_threshold":confirmation_thresholds.get(int(c["size"])),"dynamic_confirmation_ok":dynamic_extra,"confirmed":confirmed,"confirmation_status":"CONFIRMED" if confirmed else "DISCOVERY_ONLY_MOTIF"})
    payload={"version":"V837ak","stage":"AK6","frozen_candidate_classes_sha256":expected_hash,"class_count":len(classes),"confirmed_count":sum(c["confirmed"] for c in classes),"confirmation_self_stability_thresholds":{str(k):v for k,v in confirmation_thresholds.items()},"classes":classes,"candidate_creation_from_confirmation":False}
    write_json(HERE/"raw/confirmed_candidate_classes.json",payload);write_json(HERE/"diagnostics/heldout_confirmation.json",{"version":"V837ak","confirmed_count":payload["confirmed_count"],"class_count":len(classes),"thresholds":payload["confirmation_self_stability_thresholds"],"classes":[{"class_id":c["class_id"],"stream":c["stream"],"size":c["size"],"retention_fraction":c["retention_fraction"],"confirmed":c["confirmed"],"status":c["confirmation_status"]} for c in classes]})
    # Re-read frozen file after confirmation to prove it was not mutated.
    after=read_json(frozen_path);after_core={k:v for k,v in after.items() if k!="frozen_candidate_classes_sha256"}
    if after["frozen_candidate_classes_sha256"]!=expected_hash or sha256_json(after_core)!=expected_hash:raise RuntimeError("V837AK_CONFIRMATION_MUTATED_FROZEN_CLASSES")
    return payload


def main()->int:
    p=confirm_candidates();print(json.dumps({"class_count":p["class_count"],"confirmed_count":p["confirmed_count"]},indent=2));return 0

if __name__=="__main__":raise SystemExit(main())
