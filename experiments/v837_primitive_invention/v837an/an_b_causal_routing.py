from __future__ import annotations

from collections import defaultdict
import json
import math
import time

import numpy as np

from experiments.v837_primitive_invention.tasks import task_by_name

from .an_a_causal_macrovariables import _eligibility
from .authorization import FAMILY_SUPPORT, MIN_ELIGIBLE, PARTITIONS, ROUTING_PREFIXES, ROUTING_THRESHOLDS
from .causal_metrics import direction_agreement, paired_sign_flip_p, recovery, success_vector
from .failure_ledger import add, make_entry
from .instrumented_af1d import load_population_rows
from .message_contributions import verify_message_decomposition
from .global_contributions import verify_global_decomposition
from .routing_interventions import GATE_CHANNEL, channel_universe, routing_forward_reverse
from .trace_cache import pair_traces
from .utils import HERE, deterministic_seed, range_list, write_json


def _paired_metrics(data,indices,forward,reverse,random_forward,random_reverse,family,seed)->dict:
    task=task_by_name(family);idx=np.asarray(indices,dtype=np.int64)
    bp=data["base_prediction"].detach().cpu().numpy()[idx].astype(np.float64);cp=data["cf_prediction"].detach().cpu().numpy()[idx].astype(np.float64);bt=data["base_targets"].detach().cpu().numpy()[idx].astype(np.float64);ct=data["cf_targets"].detach().cpu().numpy()[idx].astype(np.float64)
    rec=np.concatenate([recovery(bp,cp,forward),recovery(cp,bp,reverse)]);direction=np.concatenate([direction_agreement(bp,cp,forward),direction_agreement(cp,bp,reverse)]);success=np.concatenate([success_vector(task,forward,ct),success_vector(task,reverse,bt)])
    rr=[]
    for f,r in zip(random_forward,random_reverse):rr.append(np.concatenate([recovery(bp,cp,f),recovery(cp,bp,r)]))
    random=np.stack(rr) if rr else np.full((1,len(rec)),-np.inf);control=np.median(random,axis=0);margin=float(np.median(rec)-np.median(random));p=paired_sign_flip_p(rec-control,seed)
    passed=float(np.median(rec))>=ROUTING_THRESHOLDS["median_recovery_min"] and float(np.mean(success))>=ROUTING_THRESHOLDS["counterfactual_success_min"] and float(np.mean(direction))>=ROUTING_THRESHOLDS["direction_agreement_min"] and margin>=ROUTING_THRESHOLDS["random_margin_min"] and p<=ROUTING_THRESHOLDS["paired_p_max"]
    return {"median_recovery":float(np.median(rec)),"counterfactual_success":float(np.mean(success)),"direction_agreement":float(np.mean(direction)),"random_subset_median_recovery":float(np.median(random)),"random_margin":margin,"paired_p":float(p),"pass":bool(passed)}


def _random_sets(universe:list[str],cardinality:int,count:int,seed:int)->list[tuple[str,...]]:
    if cardinality<=0:return [tuple() for _ in range(count)]
    cardinality=min(cardinality,len(universe));rng=np.random.default_rng(int(seed));seen=set();rows=[];attempt=0
    while len(rows)<count and attempt<count*200:
        choice=tuple(sorted(rng.choice(np.asarray(universe,dtype=object),size=cardinality,replace=False).tolist()));attempt+=1
        if choice not in seen:seen.add(choice);rows.append(choice)
    while len(rows)<count:rows.append(rows[len(rows)%max(1,len(rows))] if rows else tuple(sorted(universe[:cardinality])))
    return rows


def _rank_channels(data,indices,universe,family,organism_id)->list[dict]:
    sets=[(ch,) for ch in universe];f,r,_,_=routing_forward_reverse(data,sets,indices);idx=np.asarray(indices);bp=data["base_prediction"].detach().cpu().numpy()[idx];cp=data["cf_prediction"].detach().cpu().numpy()[idx]
    rows=[]
    for i,ch in enumerate(universe):
        rec=np.concatenate([recovery(bp,cp,f[i]),recovery(cp,bp,r[i])]);rows.append({"channel":ch,"median_recovery":float(np.median(rec)),"mean_recovery":float(np.mean(rec))})
    rows.sort(key=lambda x:(-x["median_recovery"],-x["mean_recovery"],x["channel"]));return rows


def _config_sets(ranking:list[dict])->dict[str,tuple[str,...]]:
    ordered=[r["channel"] for r in ranking];message=[x for x in ordered if x.startswith("M:")];glob=[x for x in ordered if x.startswith("G:")];combined=ordered
    configs={}
    for n in ROUTING_PREFIXES["message"]:
        k=min(n,len(message));configs[f"MESSAGE_PREFIX_{n}"]=tuple(message[:k])
    for n in ROUTING_PREFIXES["global"]:
        k=min(n,len(glob));configs[f"GLOBAL_PREFIX_{n}"]=tuple(glob[:k])
    for n in ROUTING_PREFIXES["combined"]:
        k=min(n,len(combined));configs[f"COMBINED_PREFIX_{n}"]=tuple(combined[:k])
    # Frozen diagnostic bus comparisons, not an extra free search.
    configs["BUS_MESSAGE_ONLY"]=tuple(message);configs["BUS_GLOBAL_ONLY"]=tuple(glob);configs["BUS_MESSAGE_GLOBAL"]=tuple(message+glob);configs["BUS_GATE_ONLY"]=(GATE_CHANNEL,);configs["BUS_MESSAGE_GATE"]=tuple(message+[GATE_CHANNEL]);configs["BUS_GLOBAL_GATE"]=tuple(glob+[GATE_CHANNEL]);configs["BUS_ALL"]=tuple(message+glob+[GATE_CHANNEL])
    return configs


def _dof(channels:tuple[str,...])->int:
    return sum(4 if ch.startswith("M:") or ch.startswith("G:") else 1 for ch in channels)


def run_an_b()->dict:
    start=time.perf_counter();cpu_start=time.process_time();population=load_population_rows();competent=[r for r in population if bool(r["competent"])];fit_seeds=range_list(PARTITIONS["AN_ROUTING_FIT"]);select_seeds=range_list(PARTITIONS["AN_ROUTING_SELECT"]);scores=[];selection=[];decomposition_rows=[]
    for oi,organism in enumerate(competent,1):
        family=organism["family"];fd=pair_traces(organism["organism_id"],family,fit_seeds);sd=pair_traces(organism["organism_id"],family,select_seeds);fm,_=_eligibility(fd,family);sm,_=_eligibility(sd,family);fi=np.flatnonzero(fm);si=np.flatnonzero(sm)
        mb=verify_message_decomposition(fd["model"],fd["base_trace"],lengths=fd["base_lengths"]);mc=verify_message_decomposition(fd["model"],fd["cf_trace"],lengths=fd["cf_lengths"]);gb=verify_global_decomposition(fd["model"],fd["base_trace"],lengths=fd["base_lengths"]);gc=verify_global_decomposition(fd["model"],fd["cf_trace"],lengths=fd["cf_lengths"]);decomposition_rows.append({"organism_id":organism["organism_id"],"family":family,"message_base":mb,"message_counterfactual":mc,"global_base":gb,"global_counterfactual":gc})
        if len(fi)<32 or len(si)<32:
            selection.append({"organism_id":organism["organism_id"],"family":family,"engine":organism["engine"],"powered":False,"fit_eligible":len(fi),"select_eligible":len(si),"configs":{}});continue
        universe=channel_universe(fd["model"]);ranking=_rank_channels(fd,fi,universe,family,organism["organism_id"]);scores.append({"organism_id":organism["organism_id"],"family":family,"engine":organism["engine"],"ranking":ranking,"channel_universe":universe});configs=_config_sets(ranking);sets=list(configs.values());names=list(configs);f,r,_,_=routing_forward_reverse(sd,sets,si);config_rows={}
        message_u=[x for x in universe if x.startswith("M:")];global_u=[x for x in universe if x.startswith("G:")]
        for ci,(name,channels) in enumerate(zip(names,sets)):
            if name.startswith("MESSAGE_"):control_u=message_u
            elif name.startswith("GLOBAL_"):control_u=global_u
            elif name=="BUS_GATE_ONLY":control_u=[GATE_CHANNEL]
            else:control_u=universe
            random_sets=_random_sets(control_u,len(channels),64,deterministic_seed("v837an-routing-control",organism["organism_id"],name));rf,rr,_,_=routing_forward_reverse(sd,random_sets,si);metrics=_paired_metrics(sd,si,f[ci],r[ci],rf,rr,family,deterministic_seed("v837an-routing-p",organism["organism_id"],name));config_rows[name]={"channels":list(channels),"channel_count":len(channels),"intervention_dof":_dof(channels),"metrics":metrics}
        selection.append({"organism_id":organism["organism_id"],"family":family,"engine":organism["engine"],"powered":True,"fit_eligible":len(fi),"select_eligible":len(si),"configs":config_rows});print(f"V837an AN-B {oi}/{len(competent)} {organism['organism_id']} complete",flush=True)
    # Family-level branch candidates are config semantics (e.g. COMBINED_PREFIX_4),
    # while physical channels remain organism-specific and were frozen by FIT ranking.
    family_summaries=[];winners={};families=sorted({r["family"] for r in competent});candidate_names=sorted({name for row in selection for name in row.get("configs",{}) if not name.startswith("BUS_")})
    for family in families:
        rows=[r for r in selection if r["family"]==family and r["powered"]];n=len(rows);required=max(1,math.ceil(.60*n));source_engines={r["engine"] for r in competent if r["family"]==family};configs=[]
        for name in candidate_names:
            eligible=[r for r in rows if name in r["configs"]];passed=[r for r in eligible if r["configs"][name]["metrics"]["pass"]];engines={r["engine"] for r in passed};family_pass=n>=5 and len(passed)>=required and (len(source_engines)<2 or len(engines)>=2);dofs=[r["configs"][name]["intervention_dof"] for r in eligible];counts=[r["configs"][name]["channel_count"] for r in eligible]
            summary={"family":family,"config_id":name,"powered_organisms":n,"required_passes":required,"organisms_passing":len(passed),"pass_fraction":0 if n==0 else len(passed)/n,"engines":sorted(engines),"family_pass":family_pass,"median_intervention_dof":float(np.median(dofs)) if dofs else math.inf,"median_channel_count":float(np.median(counts)) if counts else math.inf};configs.append(summary);family_summaries.append(summary)
            if not family_pass:
                add(make_entry(failure_id=f"V837an-AN-B-{family}-{name}",stage="AN-B",hypothesis=f"{family} causal effect is mediated by frozen FIT-ranked routing configuration {name}.",why="Localize causal communication without assuming shared microstate coordinates.",implementation={"routing_config":name,"organism_specific_channels_ranked_on_fit_only":True},data={"fit":PARTITIONS["AN_ROUTING_FIT"],"select":PARTITIONS["AN_ROUTING_SELECT"]},fit_seeds=PARTITIONS["AN_ROUTING_FIT"],selection_seeds=PARTITIONS["AN_ROUTING_SELECT"],controls=["64 same-cardinality random channel sets"],parameter_count=0,mac_cost=summary["median_intervention_dof"],metrics=summary,gate=ROUTING_THRESHOLDS,failed=["family routing support gate not met"],distance={"pass_fraction_shortfall":max(0,.60-summary["pass_fraction"])},result_status="underpowered" if n<5 else "definitive within tested development scope",failure_type="UNDERPOWERED" if n<5 else "SCIENTIFIC_FAILURE",confounds_ruled_out=["channel-count advantage","free arbitrary-window search"],confounds_remaining=["other causal granularity","coalition synergy"],meaning="This routing configuration did not establish a shared family-level causal routing mechanism.",uncertainty="Other frozen routing configurations or cell coalitions may still pass.",next_experiment="Continue V837an frozen routing/coalition program.",reproduce="python scripts/reproduce_v837_recovery.py --variant v837an --stage routing --execute",artifacts=["experiments/v837_primitive_invention/v837an/raw/routing_selection.json"]))
        passing=[c for c in configs if c["family_pass"]];winners[family]=min(passing,key=lambda c:(c["median_intervention_dof"],c["median_channel_count"],c["config_id"])) if passing else None
    # Rank-4/global bus is a qualifier based on intervention evidence, not configured rank.
    rank4_families=[];bus_rows=[]
    for family in families:
        fam=[r for r in selection if r["family"]==family and r["powered"]];n=len(fam);required=max(1,math.ceil(.60*n));stats={}
        for name in ["BUS_MESSAGE_ONLY","BUS_GLOBAL_ONLY","BUS_MESSAGE_GLOBAL","BUS_GATE_ONLY","BUS_MESSAGE_GATE","BUS_GLOBAL_GATE","BUS_ALL"]:
            vals=[r["configs"][name]["metrics"]["median_recovery"] for r in fam if name in r["configs"]];passes=sum(r["configs"][name]["metrics"]["pass"] for r in fam if name in r["configs"]);stats[name]={"median_recovery":float(np.median(vals)) if vals else None,"passes":int(passes),"required":required}
        global_dominant=n>=5 and stats["BUS_GLOBAL_ONLY"]["passes"]>=required
        if global_dominant:rank4_families.append(family)
        bus_rows.append({"family":family,"powered_organisms":n,"conditions":stats,"global_dominant":global_dominant})
    rank4_supported=len(rank4_families)>=3
    payload={"version":"V837an","stage":"AN-B","fit_seeds":PARTITIONS["AN_ROUTING_FIT"],"select_seeds":PARTITIONS["AN_ROUTING_SELECT"],"organism_results":selection,"family_summaries":family_summaries,"family_winners":winners,"causal_routing_families":sum(v is not None for v in winners.values()),"rank4_global_bus_supported":rank4_supported,"rank4_support_families":rank4_families,"wall_seconds":time.perf_counter()-start,"cpu_seconds":time.process_time()-cpu_start}
    max_message=max(max(r["message_base"]["max_abs_error"],r["message_counterfactual"]["max_abs_error"]) for r in decomposition_rows) if decomposition_rows else 0.0;max_global=max(max(r["global_base"]["max_abs_error"],r["global_counterfactual"]["max_abs_error"]) for r in decomposition_rows) if decomposition_rows else 0.0
    write_json(HERE/"raw/routing_channel_scores.json",{"version":"V837an","rows":scores});write_json(HERE/"raw/routing_selection.json",payload);write_json(HERE/"diagnostics/routing_localization.json",{"version":"V837an","family_winners":winners});write_json(HERE/"diagnostics/rank4_bus.json",{"version":"V837an","supported":rank4_supported,"support_families":rank4_families,"rows":bus_rows,"intervention_based":True});write_json(HERE/"diagnostics/message_decomposition.json",{"version":"V837an","pass":max_message<=1e-6,"max_abs_error":max_message,"rows":decomposition_rows});write_json(HERE/"diagnostics/global_decomposition.json",{"version":"V837an","pass":max_global<=1e-6,"max_abs_error":max_global,"rows":decomposition_rows});return payload


if __name__=="__main__":print(json.dumps(run_an_b(),indent=2))
