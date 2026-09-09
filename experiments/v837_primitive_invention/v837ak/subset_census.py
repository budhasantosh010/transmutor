from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from experiments.v837_primitive_invention.v837ak.authorization import assert_v837ak_authorized
from experiments.v837_primitive_invention.v837ak.reconstruct_organisms import _load_rows
from experiments.v837_primitive_invention.v837ak.structural_signatures import structural_class_id, structural_signature
from experiments.v837_primitive_invention.v837ak.utils import HERE, occurrence_id, read_json, size_category, write_json


def enumerate_occurrences() -> list[dict]:
    recon=read_json(HERE/"raw/reconstruction_results.json")
    if recon.get("complete") is not True or recon.get("organisms_reconstructed") != 50:
        raise RuntimeError("AK3 blocked: reconstruction incomplete")
    source={r["organism_id"]:r for r in _load_rows()}; rec={r["organism_id"]:r for r in recon["rows"]}
    rows=[]
    for oid in sorted(source):
        sr=source[oid]; rr=rec[oid]
        for k in range(1,11):
            for nodes in itertools.combinations(range(10),k):
                sig=structural_signature(sr["topology"],nodes); cid=structural_class_id(sig)
                rows.append({
                    "occurrence_id":occurrence_id(oid,nodes),"organism_id":oid,"engine":rr["engine"],"family":rr["family"],"run_index":rr["run_index"],
                    "competent":bool(rr["competent"]),"nodes":list(nodes),"size":k,"size_category":size_category(k),
                    "structural_class_id":cid,"message_connected":bool(sig["message_connected"]),
                    "internal_edge_count":len(sig["internal_same_step_edges"])+len(sig["internal_recurrent_edges"]),
                    "internal_recurrent_edge_count":len(sig["internal_recurrent_edges"]),
                })
    if len(rows)!=50*1023: raise RuntimeError(f"V837AK_SUBSET_CENSUS_COUNT_MISMATCH {len(rows)}")
    return rows


def build_structural_classes(occurrences:list[dict]) -> dict:
    source={r["organism_id"]:r for r in _load_rows()}; signatures={}
    per_class=defaultdict(lambda: defaultdict(list))
    for occ in occurrences:
        sr=source[occ["organism_id"]]
        cid=occ["structural_class_id"]
        if cid not in signatures: signatures[cid]=structural_signature(sr["topology"],occ["nodes"])
        per_class[cid][occ["organism_id"]].append(occ)
    classes=[]
    all_class_summaries=[]
    for cid,by_org in per_class.items():
        representatives=[]
        for oid,items in by_org.items(): representatives.append(sorted(items,key=lambda x:x["occurrence_id"])[0])
        competent=[x for x in representatives if x["competent"]]; incompetent=[x for x in representatives if not x["competent"]]
        family_counts=defaultdict(int); engine_counts=defaultdict(int)
        for x in competent: family_counts[x["family"]]+=1; engine_counts[x["engine"]]+=1
        summary={"class_id":cid,"size":signatures[cid]["size"],"message_connected":signatures[cid]["message_connected"],"competent_support":len(competent),"incompetent_support":len(incompetent)}
        all_class_summaries.append(summary)
        # Classes with <3 competent organisms can satisfy neither the global
        # support gate (>=6) nor the family-specific gate (>=3). Keep their
        # exhaustive census identity/counts compact here; full memberships are
        # deterministically reconstructable from raw/cache/subset_occurrences.json.
        if len(competent)<3:
            continue
        classes.append({
            "class_id":cid,"signature":signatures[cid],"size":signatures[cid]["size"],"message_connected":signatures[cid]["message_connected"],
            "competent_support":len(competent),"incompetent_support":len(incompetent),"competent_organisms":sorted(x["organism_id"] for x in competent),
            "engine_counts":dict(sorted(engine_counts.items())),"family_counts":dict(sorted(family_counts.items())),
            "representative_occurrences":sorted(representatives,key=lambda x:x["organism_id"]),
        })
    classes.sort(key=lambda c:(-c["competent_support"],c["size"],c["class_id"]))
    all_class_summaries.sort(key=lambda c:(-c["competent_support"],c["size"],c["class_id"]))
    return {"version":"V837ak","stage":"AK3","class_count_total":len(all_class_summaries),"candidate_detail_min_competent_support":3,"classes":classes,"all_class_summaries":all_class_summaries,"full_occurrence_membership_cache":"raw/cache/subset_occurrences.json"}


def run_census() -> dict:
    assert_v837ak_authorized()
    occurrences=enumerate_occurrences()
    size_distribution={str(k):sum(1 for x in occurrences if x["size"]==k) for k in range(1,11)}
    connected=sum(x["message_connected"] for x in occurrences); disconnected=len(occurrences)-connected
    expected={str(k):50*math.comb(10,k) for k in range(1,11)}
    if size_distribution!=expected: raise RuntimeError("V837AK_SUBSET_SIZE_DISTRIBUTION_MISMATCH")
    summary={"version":"V837ak","stage":"AK3","organisms":50,"subsets_per_organism":1023,"total_occurrences":len(occurrences),"size_distribution":size_distribution,"connected_occurrences":connected,"disconnected_occurrences":disconnected,"whole_system_occurrences":50,"all_nonempty_subsets":True}
    write_json(HERE/"raw/subset_census_summary.json",summary)
    write_json(HERE/"diagnostics/subset_size_distribution.json",summary)
    classes=build_structural_classes(occurrences); write_json(HERE/"raw/structural_classes.json",classes)
    recurring=[c for c in classes["classes"] if c["competent_support"]>=6]
    write_json(HERE/"diagnostics/structural_recurrence.json",{"version":"V837ak","classes_total":classes["class_count_total"],"classes_support_ge_6":len(recurring),"candidate_detail_classes":len(classes["classes"]),"top_classes":recurring[:50]})
    cache=HERE/"raw/cache"; cache.mkdir(parents=True,exist_ok=True); write_json(cache/"subset_occurrences.json",{"version":"V837ak","rows":occurrences})
    return summary


def main()->int:
    p=run_census(); print(json.dumps(p,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
