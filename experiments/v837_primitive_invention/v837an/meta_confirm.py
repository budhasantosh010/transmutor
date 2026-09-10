from __future__ import annotations

import json, math, time
from pathlib import Path
import numpy as np

from .authorization import PARTITIONS
from .branch_evaluation import evaluate_a, evaluate_b, evaluate_c, gate_result
from .instrumented_af1d import load_population_rows
from .failure_ledger import add, make_entry
from .utils import HERE, read_json, write_json


def _branch_candidates()->dict[str,dict[str,dict|None]]:
    a=read_json(HERE/"raw/an_a_selection.json")
    b=read_json(HERE/"raw/routing_selection.json")
    c=read_json(HERE/"raw/synergy_results.json")
    out={}
    families=sorted(set(a["family_winners"])|set(b["family_winners"])|set(c["family_results"]))
    for family in families:
        source=a["family_winners"].get(family); compiler=a["semantic_compiler_winners"].get(family)
        # One AN-A candidate only. Prefer the independently passing semantic compiler;
        # otherwise retain the minimal source-swap causal subspace candidate.
        aw=compiler if compiler is not None else source
        out[family]={
            "AN-A":None if aw is None else {"branch":"AN-A","config_id":aw["config_id"],"carrier":aw["carrier"],"k":int(aw["k"]),"semantic_compiler":compiler is not None},
            "AN-B":None if b["family_winners"].get(family) is None else {"branch":"AN-B","config_id":b["family_winners"][family]["config_id"]},
            "AN-C":None if not c["family_results"].get(family,{}).get("family_pass") else {"branch":"AN-C","config_id":"COALITION","strong_synergy_family":bool(c["family_results"][family].get("strong_synergy_family"))},
        }
    write_json(HERE/"raw/branch_winners.json",{"version":"V837an","frozen_before_meta":True,"families":out})
    return out


def _evaluate_family(family:str,candidate:dict,partition:str,p_max:float)->dict:
    population=[r for r in load_population_rows() if r["family"]==family and bool(r["competent"])]
    rows=[]
    synergy=read_json(HERE/"raw/synergy_results.json") if (HERE/"raw/synergy_results.json").is_file() else None
    c_nodes={}
    if candidate["branch"]=="AN-C" and synergy is not None:
        fr=synergy["family_results"].get(family,{})
        c_nodes={x["organism_id"]:x["winner"]["nodes"] for x in fr.get("organism_winners",[]) if x.get("winner") is not None}
    for org in population:
        if candidate["branch"]=="AN-A": m=evaluate_a(org,candidate["config_id"],partition,use_compiler=bool(candidate.get("semantic_compiler")))
        elif candidate["branch"]=="AN-B": m=evaluate_b(org,candidate["config_id"],partition)
        else:
            nodes=c_nodes.get(org["organism_id"])
            m={"eligible":0,"powered":False,"pass":False,"reason":"NO_FROZEN_ORGANISM_COALITION"} if nodes is None else evaluate_c(org,nodes,partition)
        m=dict(m);m["organism_id"]=org["organism_id"];m["engine"]=org["engine"]
        m["pass"]=gate_result(m,p_max=p_max,require_control_margin=True)
        rows.append(m)
    powered=[r for r in rows if r.get("powered")];passed=[r for r in powered if r.get("pass")];required=max(1,math.ceil(.60*len(powered)));source_engines={o["engine"] for o in population};pass_engines={r["engine"] for r in passed}
    family_pass=len(powered)>=5 and len(passed)>=required and (len(source_engines)<2 or len(pass_engines)>=2)
    if candidate["branch"]=="AN-A": dof=int(candidate["k"]);components=1;footprint=dof
    elif candidate["branch"]=="AN-B":
        dofs=[r.get("intervention_dof",math.inf) for r in powered];comps=[r.get("channel_count",math.inf) for r in powered];dof=float(np.median(dofs)) if dofs else math.inf;components=float(np.median(comps)) if comps else math.inf;footprint=dof
    else:
        dofs=[r.get("intervention_dof",math.inf) for r in powered];cards=[r.get("cardinality",math.inf) for r in powered];dof=float(np.median(dofs)) if dofs else math.inf;components=float(np.median(cards)) if cards else math.inf;footprint=dof
    return {"family":family,"candidate":candidate,"partition":partition,"organism_results":rows,"powered_organisms":len(powered),"organisms_passing":len(passed),"required_passes":required,"pass_fraction":0.0 if not powered else len(passed)/len(powered),"pass_engines":sorted(pass_engines),"family_pass":bool(family_pass),"intervention_dof":dof,"physical_components":components,"intervention_mac_footprint":footprint}


def run_meta()->dict:
    start=time.perf_counter();cpu_start=time.process_time();candidates=_branch_candidates();results=[];winners={}
    for family,branches in candidates.items():
        evaluated=[]
        for branch in ("AN-A","AN-B","AN-C"):
            c=branches.get(branch)
            if c is None:continue
            r=_evaluate_family(family,c,"AN_META_CONFIRM",.05);evaluated.append(r);results.append(r);print(f"V837an META {family} {branch} pass={r['family_pass']} powered={r['powered_organisms']} passing={r['organisms_passing']}",flush=True)
            if not r["family_pass"]:
                add(make_entry(failure_id=f"V837an-META-{family}-{branch}-{c['config_id']}",stage="META_CONFIRM",hypothesis=f"The frozen {branch} candidate for {family} generalizes without refitting to META_CONFIRM.",why="Predeclared independent development confirmation before any final family freeze.",implementation={"branch":branch,"candidate":c,"no_refit":True},data={"meta":PARTITIONS["AN_META_CONFIRM"]},fit_seeds=[],selection_seeds=PARTITIONS["AN_META_CONFIRM"],controls=["same frozen branch controls as selection"],parameter_count=0,mac_cost=r.get("intervention_dof"),metrics={k:v for k,v in r.items() if k!="organism_results"},gate={"minimum_competent_organisms":5,"minimum_pass_fraction":.60,"paired_p_max":.05},failed=["META_CONFIRM family gate not met"],distance={"pass_fraction_shortfall":max(0.0,.60-float(r.get("pass_fraction",0.0)))},result_status="underpowered" if int(r.get("powered_organisms",0))<5 else "definitive within META_CONFIRM scope",failure_type="UNDERPOWERED" if int(r.get("powered_organisms",0))<5 else "SCIENTIFIC_FAILURE",confounds_ruled_out=["selection-set overfit","refitting on meta data"],confounds_remaining=["other primitive level or family-specific implementation"],meaning="The frozen branch candidate did not survive the independent META_CONFIRM block.",uncertainty="Other already-frozen branch candidates may still survive; no new candidate is created here.",next_experiment="Continue the predeclared V837an meta/final chain without replacing this candidate.",reproduce="python scripts/reproduce_v837_recovery.py --variant v837an --stage meta --execute",artifacts=["experiments/v837_primitive_invention/v837an/raw/meta_confirmation.json"]))
        passing=[r for r in evaluated if r["family_pass"]]
        winners[family]=min(passing,key=lambda r:(r["intervention_dof"],r["physical_components"],r["intervention_mac_footprint"],r["candidate"]["branch"],r["candidate"]["config_id"])) if passing else None
    payload={"version":"V837an","stage":"META_CONFIRM","seeds":PARTITIONS["AN_META_CONFIRM"],"no_refit":True,"results":results,"family_winners":winners,"wall_seconds":time.perf_counter()-start,"cpu_seconds":time.process_time()-cpu_start}
    write_json(HERE/"raw/meta_confirmation.json",payload);write_json(HERE/"raw/meta_confirmed_family_abstractions.json",{"version":"V837an","families":winners});write_json(HERE/"diagnostics/meta_confirmation.json",payload);return payload


if __name__=="__main__":print(json.dumps(run_meta(),indent=2))
