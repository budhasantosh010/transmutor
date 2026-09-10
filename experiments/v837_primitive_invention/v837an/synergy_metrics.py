from __future__ import annotations


def synergy_gain(row:dict,singletons:dict[int,dict])->float:
    best=max((singletons[i]["median_recovery"] for i in row["nodes"]),default=0.0);return float(row["median_recovery"]-best)


def pairwise_additive_interaction(pair_row:dict,singletons:dict[int,dict])->float:
    i,j=pair_row["nodes"];return float(pair_row["median_recovery"]-singletons[i]["median_recovery"]-singletons[j]["median_recovery"])


def characterize(rows:list[dict],winner:dict|None)->dict:
    singletons={r["nodes"][0]:r for r in rows if r["cardinality"]==1};pairs=[]
    for r in rows:
        if r["cardinality"]==2:pairs.append({"nodes":r["nodes"],"interaction":pairwise_additive_interaction(r,singletons)})
    if winner is None:return {"minimum_sufficient":None,"strong_synergy":False,"pairwise_interactions":pairs}
    gain=synergy_gain(winner,singletons);best=max(singletons[i]["median_recovery"] for i in winner["nodes"]);strong=winner["median_recovery"]>=.60 and best<.30 and gain>=.30 and winner["cardinality"]<=4
    qualifier="COMPACT_SYNERGISTIC_CAUSAL_COALITION" if strong else ("WHOLE_ORGANISM_CAUSAL_STATE_REQUIRED" if winner["cardinality"]==10 else ("ORGANISM_SCALE_DISTRIBUTED_COMPUTATION" if winner["cardinality"]>=7 else "NON_SYNERGISTIC_OR_MODERATE_COALITION"))
    return {"minimum_sufficient":winner,"best_member_recovery":float(best),"synergy_gain":float(gain),"strong_synergy":bool(strong),"qualifier":qualifier,"pairwise_interactions":pairs}
