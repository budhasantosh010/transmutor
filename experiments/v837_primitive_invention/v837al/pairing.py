from __future__ import annotations

from collections import Counter, defaultdict

from .authorization import assert_authorized
from .utils import HERE, ROOT, read_json, sha256_json, write_json

AK=ROOT/"experiments/v837_primitive_invention/v837ak"


def _dedup_recipients(rows:list[dict])->list[dict]:
    by={}
    for r in rows:
        oid=r["organism_id"]
        key=(float(r.get("confirmation_class_distance",1e30)),r["occurrence_id"])
        if oid not in by or key < by[oid][0]: by[oid]=(key,r)
    return [by[k][1] for k in sorted(by)]


def _same_donor(rows:list[dict], recipient:dict):
    cand=[r for r in rows if r["organism_id"]!=recipient["organism_id"] and r["family"]==recipient["family"]]
    cand.sort(key=lambda r:(r["engine"]==recipient["engine"],float(r.get("confirmation_class_distance",1e30)),r["organism_id"],r["occurrence_id"]))
    return cand[0] if cand else None


def _different_control(classes:list[dict], structural:list[dict], cls:dict, recipient:dict):
    pool=[]
    for other in classes:
        if other["class_id"]==cls["class_id"] or int(other["size"])!=int(cls["size"]): continue
        for r in other["compatible_occurrences"]:
            if r["competent"] and r["family"]==recipient["family"] and r["organism_id"]!=recipient["organism_id"]:
                pool.append((0, other["class_id"], r))
    if not pool:
        for sc in structural:
            if sc["class_id"]==cls["class_id"] or int(sc["size"])!=int(cls["size"]): continue
            for r in sc.get("representative_occurrences",[]):
                if r.get("competent") and r["family"]==recipient["family"] and r["organism_id"]!=recipient["organism_id"]:
                    pool.append((1, sc["class_id"], r))
    pool.sort(key=lambda x:(x[2]["engine"]==recipient["engine"],x[0],x[1],x[2]["organism_id"],x[2]["occurrence_id"]))
    if not pool:return None
    src,cid,r=pool[0]
    return {**r,"control_class_id":cid,"control_source":"confirmed_class" if src==0 else "frozen_structural_census"}


def freeze_pairs():
    assert_authorized()
    source=read_json(HERE/"raw/source_classes.json")
    structural=read_json(AK/"raw/structural_classes.json")["classes"]
    pairs=[]; support=[]
    for cls in source["classes"]:
        recipients=_dedup_recipients(cls["compatible_occurrences"])
        class_pairs=[]
        for recipient in recipients:
            donor=_same_donor(recipients,recipient)
            if donor is None: continue
            control=_different_control(source["classes"],structural,cls,recipient)
            if control is None: continue
            row={"class_id":cls["class_id"],"stream":cls["stream"],"size":int(cls["size"]),"primary_causal":bool(cls["primary_causal"]),"recipient":recipient,"same_class_donor":donor,"different_class_donor":control}
            class_pairs.append(row); pairs.append(row)
        fam=Counter(p["recipient"]["family"] for p in class_pairs); eng=Counter(p["recipient"]["engine"] for p in class_pairs)
        n=len(class_pairs)
        support.append({"class_id":cls["class_id"],"size":int(cls["size"]),"primary_causal":bool(cls["primary_causal"]),"independent_recipients":n,"same_family_donor_count":n,"engine_balance":dict(eng),"family_balance":dict(fam),"minimum_possible_p":2.0**(-n),"strong_claim_powered":n>=7,"power_status":"POWERED" if n>=7 else "STATISTICALLY_UNDERPOWERED_FOR_STRONG_INTERCHANGEABILITY_CLAIM"})
    payload={"version":"V837al","pairing_frozen_before_alignment_results":True,"pair_count":len(pairs),"pairs":pairs}
    payload["frozen_pairs_sha256"]=sha256_json(payload)
    write_json(HERE/"raw/frozen_pairs.json",payload)
    write_json(HERE/"diagnostics/pair_support.json",{"version":"V837al","classes":support,"pair_count":len(pairs)})
    write_json(HERE/"diagnostics/statistical_power.json",{"version":"V837al","rule":"p_min=2^-N; strong p<=0.01 requires N>=7","classes":support,"primary_causal":next(x for x in support if x["primary_causal"])})
    return payload,support


def main()->int:
    p,s=freeze_pairs(); print({"pairs":p["pair_count"],"support":[(x["class_id"][:8],x["independent_recipients"],x["strong_claim_powered"]) for x in s]}); return 0

if __name__=="__main__": raise SystemExit(main())
