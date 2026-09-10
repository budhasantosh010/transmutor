from __future__ import annotations

from collections import defaultdict
import json
import math
import time

import numpy as np

from .an_a_causal_macrovariables import _eligibility, _carrier_values
from .authorization import MIN_ELIGIBLE, PARTITIONS
from .coalition_scan import minimum_sufficient, scan_organism_coalitions
from .failure_ledger import add, make_entry
from .instrumented_af1d import load_population_rows
from .synergy_metrics import characterize
from .trace_cache import pair_traces
from .utils import HERE, range_list, read_json, write_json


def _trigger_families()->dict[str,dict]:
    a=read_json(HERE/"raw/an_a_selection.json");b=read_json(HERE/"raw/routing_selection.json")
    out={}
    for family in sorted(a["family_winners"]):
        aw=a["family_winners"].get(family);bw=b["family_winners"].get(family)
        compact_a=bool(aw is not None and int(aw.get("k",99))<=4)
        broad_b=bool(bw is None or float(bw.get("median_channel_count",999))>8)
        trigger=(not compact_a) or broad_b
        out[family]={"trigger":trigger,"an_a_compact":compact_a,"an_b_broad_or_absent":broad_b,"an_a_winner":None if aw is None else aw["config_id"],"an_b_winner":None if bw is None else bw["config_id"]}
    return out


CACHE_DIR=HERE/"cache/an_c_scans"


def _load_scan_checkpoint(organism:dict)->dict|None:
    path=CACHE_DIR/f"{organism['organism_id']}.json"
    if not path.is_file():return None
    try:d=read_json(path)
    except Exception:return None
    rows=d.get("rows",[])
    valid=(d.get("version")=="V837an" and d.get("organism_id")==organism["organism_id"] and d.get("family")==organism["family"] and d.get("fit_seeds")==PARTITIONS["AN_FIT"] and d.get("select_seeds")==PARTITIONS["AN_SELECT"] and len(rows)==1023 and len({tuple(r.get("nodes",[])) for r in rows})==1023)
    return d if valid else None


def _save_scan_checkpoint(organism:dict,rows:list[dict],winner:dict|None,syn:dict,fit_eligible:int,select_eligible:int)->None:
    CACHE_DIR.mkdir(parents=True,exist_ok=True)
    write_json(CACHE_DIR/f"{organism['organism_id']}.json",{"version":"V837an","organism_id":organism["organism_id"],"family":organism["family"],"engine":organism["engine"],"fit_seeds":PARTITIONS["AN_FIT"],"select_seeds":PARTITIONS["AN_SELECT"],"fit_eligible":int(fit_eligible),"select_eligible":int(select_eligible),"subset_count":1023,"rows":rows,"winner":winner,"synergy":{k:v for k,v in syn.items() if k!="pairwise_interactions"}})


def run_an_c()->dict:
    start=time.perf_counter();cpu_start=time.process_time();triggers=_trigger_families();population=load_population_rows();competent=[r for r in population if bool(r["competent"])];fit_seeds=range_list(PARTITIONS["AN_FIT"]);select_seeds=range_list(PARTITIONS["AN_SELECT"]);organism_rows=[];raw_scans=[]
    for oi,organism in enumerate(competent,1):
        family=organism["family"]
        if not triggers[family]["trigger"]:continue
        cached=_load_scan_checkpoint(organism)
        if cached is not None:
            rows=cached["rows"];winner=cached.get("winner");syn=cached.get("synergy") or {}
            organism_rows.append({"organism_id":organism["organism_id"],"family":family,"engine":organism["engine"],"powered":True,"fit_eligible":int(cached["fit_eligible"]),"select_eligible":int(cached["select_eligible"]),"winner":winner,"synergy":syn})
            raw_scans.append({"organism_id":organism["organism_id"],"family":family,"engine":organism["engine"],"rows":rows});print(f"V837an AN-C {oi}/{len(competent)} {organism['organism_id']} cached",flush=True);continue
        fd=pair_traces(organism["organism_id"],family,fit_seeds);sd=pair_traces(organism["organism_id"],family,select_seeds);fm,_=_eligibility(fd,family);sm,_=_eligibility(sd,family);fi=np.flatnonzero(fm);si=np.flatnonzero(sm)
        if len(fi)<MIN_ELIGIBLE["AN_FIT"] or len(si)<MIN_ELIGIBLE["AN_SELECT"]:
            organism_rows.append({"organism_id":organism["organism_id"],"family":family,"engine":organism["engine"],"powered":False,"fit_eligible":len(fi),"select_eligible":len(si),"winner":None,"synergy":None});continue
        ft=np.asarray([p.primary_phase for p in fd["pairs"]],dtype=np.int64);fbase=_carrier_values(fd["base_trace"],"STATE40",ft)[fi];fcf=_carrier_values(fd["cf_trace"],"STATE40",ft)[fi];reference=np.concatenate([fbase,fcf],axis=0)
        rows=scan_organism_coalitions(sd,si,reference);winner=minimum_sufficient(rows);syn=characterize(rows,winner);_save_scan_checkpoint(organism,rows,winner,syn,len(fi),len(si))
        organism_rows.append({"organism_id":organism["organism_id"],"family":family,"engine":organism["engine"],"powered":True,"fit_eligible":len(fi),"select_eligible":len(si),"winner":winner,"synergy":{k:v for k,v in syn.items() if k!="pairwise_interactions"}})
        raw_scans.append({"organism_id":organism["organism_id"],"family":family,"engine":organism["engine"],"rows":rows});print(f"V837an AN-C {oi}/{len(competent)} {organism['organism_id']} complete",flush=True)
    family_results={};synergistic_families=[]
    for family in sorted(triggers):
        fam=[r for r in organism_rows if r["family"]==family and r["powered"]];n=len(fam);required=max(1,math.ceil(.60*n));passed=[r for r in fam if r["winner"] is not None];engines={r["engine"] for r in passed};source_engines={r["engine"] for r in competent if r["family"]==family};family_pass=n>=5 and len(passed)>=required and (len(source_engines)<2 or len(engines)>=2);strong=[r for r in passed if r["synergy"] and r["synergy"].get("strong_synergy")];strong_family=family_pass and len(strong)>=required
        if strong_family:synergistic_families.append(family)
        cards=[r["winner"]["cardinality"] for r in passed];family_results[family]={"trigger":triggers[family],"powered_organisms":n,"required_passes":required,"organisms_passing":len(passed),"pass_fraction":0 if n==0 else len(passed)/n,"engines":sorted(engines),"family_pass":family_pass,"strong_synergy_family":strong_family,"median_minimum_coalition_cardinality":None if not cards else float(np.median(cards)),"organism_winners":[{"organism_id":r["organism_id"],"engine":r["engine"],"winner":r["winner"],"synergy":r["synergy"]} for r in fam]}
        if triggers[family]["trigger"] and not family_pass:
            add(make_entry(failure_id=f"V837an-AN-C-{family}-COALITION",stage="AN-C",hypothesis=f"{family} is implemented by a sufficient distributed cell coalition under within-organism natural counterfactual state patching.",why="Test synergistic or organism-scale causal computation after compact macrovariable/routing explanation was absent or broad.",implementation={"all_nonempty_cell_subsets":1023,"cross_organism_mapping":False},data={"fit_reference":PARTITIONS["AN_FIT"],"select":PARTITIONS["AN_SELECT"]},fit_seeds=PARTITIONS["AN_FIT"],selection_seeds=PARTITIONS["AN_SELECT"],controls=["singletons","all pair/triple/.../full coalitions","OOD manifold gate"],parameter_count=0,mac_cost="4 * coalition cardinality",metrics={k:v for k,v in family_results[family].items() if k!="organism_winners"},gate={"minimum_competent_organisms":5,"pass_fraction":.60,"recovery":.60,"success":.70,"direction":.75,"p":.01,"ood":2.0},failed=["family coalition support gate not met"],distance={"pass_fraction_shortfall":max(0,.60-family_results[family]["pass_fraction"])},result_status="underpowered" if n<5 else "definitive within tested development scope",failure_type="UNDERPOWERED" if n<5 else "SCIENTIFIC_FAILURE",confounds_ruled_out=["single-cell-only interpretation","cross-organism state invertibility"],confounds_remaining=["behavioral/algorithm-level primitive","different temporal granularity"],meaning="No sufficiently supported coalition abstraction was established for this family under the frozen V837an cell-subset intervention scope.",uncertainty="The common abstraction may exist outside static cell coalitions.",next_experiment="Proceed through V837an meta/final gates; unresolved families remain explicit.",reproduce="python scripts/reproduce_v837_recovery.py --variant v837an --stage synergy --execute",artifacts=["experiments/v837_primitive_invention/v837an/raw/coalition_scan.json"]))
    # Communication-channel synergy from the seven predeclared MESSAGE/GLOBAL/GATE combinations already measured in AN-B.
    routing=read_json(HERE/"raw/routing_selection.json");channel_synergy=[]
    for row in routing["organism_results"]:
        if not row.get("powered"):continue
        c=row["configs"]
        if all(k in c for k in ("BUS_MESSAGE_ONLY","BUS_GLOBAL_ONLY","BUS_GATE_ONLY","BUS_MESSAGE_GLOBAL","BUS_MESSAGE_GATE","BUS_GLOBAL_GATE","BUS_ALL")):
            vals={k:c[k]["metrics"]["median_recovery"] for k in ("BUS_MESSAGE_ONLY","BUS_GLOBAL_ONLY","BUS_GATE_ONLY","BUS_MESSAGE_GLOBAL","BUS_MESSAGE_GATE","BUS_GLOBAL_GATE","BUS_ALL")}
            qualifiers=[]
            if vals["BUS_MESSAGE_ONLY"]<.60 and vals["BUS_GLOBAL_ONLY"]<.60 and vals["BUS_MESSAGE_GLOBAL"]>=.60:qualifiers.append("COMMUNICATION_CHANNEL_SYNERGY_MESSAGE_GLOBAL")
            if vals["BUS_MESSAGE_ONLY"]<.60 and vals["BUS_GATE_ONLY"]<.60 and vals["BUS_MESSAGE_GATE"]>=.60:qualifiers.append("COMMUNICATION_CHANNEL_SYNERGY_MESSAGE_GATE")
            if vals["BUS_GLOBAL_ONLY"]<.60 and vals["BUS_GATE_ONLY"]<.60 and vals["BUS_GLOBAL_GATE"]>=.60:qualifiers.append("COMMUNICATION_CHANNEL_SYNERGY_GLOBAL_GATE")
            channel_synergy.append({"organism_id":row["organism_id"],"family":row["family"],"values":vals,"qualifiers":qualifiers})
    payload={"version":"V837an","stage":"AN-C","triggers":triggers,"organism_results":organism_rows,"family_results":family_results,"synergistic_families":synergistic_families,"synergistic_family_count":len(synergistic_families),"communication_channel_synergy":channel_synergy,"wall_seconds":time.perf_counter()-start,"cpu_seconds":time.process_time()-cpu_start}
    write_json(HERE/"raw/coalition_scan.json",{"version":"V837an","scans":raw_scans});write_json(HERE/"raw/synergy_results.json",payload);write_json(HERE/"diagnostics/coalition_scan.json",{"version":"V837an","family_results":family_results});write_json(HERE/"diagnostics/synergy.json",{"version":"V837an","synergistic_families":synergistic_families,"communication_channel_synergy":channel_synergy});return payload


if __name__=="__main__":print(json.dumps(run_an_c(),indent=2))
