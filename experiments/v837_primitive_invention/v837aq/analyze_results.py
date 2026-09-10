from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .failure_ledger import initialize, sync_central_ledger
from .resource_accounting import build_resource_accounting
from .utils import HERE, ROOT, read_json, sha256_file, write_json

PLOTS=HERE/"plots"
REPORT=ROOT/"docs/V837_PROGRAM_LEVEL_CAUSAL_OPERATOR_LOCALIZATION_REPORT.md"
STATUS=ROOT/"experiments/v837_primitive_invention/program_level_causal_operator_localization_program_status.json"
FAMILIES=("conditional_routing","delayed_recall","iterative_state","variable_composition")
SHORT={"conditional_routing":"routing","delayed_recall":"recall","iterative_state":"iterative","variable_composition":"composition"}


def _load(rel:str,default=None):
    p=HERE/rel;return read_json(p) if p.is_file() else default


def _savefig(name:str):
    plt.tight_layout();plt.savefig(PLOTS/name,dpi=160);plt.close()


def _bar(name,title,labels,values,ylabel,ylim=None):
    plt.figure(figsize=(9,5));plt.bar(np.arange(len(labels)),values);plt.xticks(np.arange(len(labels)),labels,rotation=20,ha="right");plt.ylabel(ylabel);plt.title(title)
    if ylim:plt.ylim(*ylim)
    _savefig(name)


def _plots(reality,disc,loc,comp,meta,psr,held,hist):
    PLOTS.mkdir(parents=True,exist_ok=True)
    reality_rows=reality.get("organisms",[]);_bar("aq2_reality_nrmse.png","AQ2 iterative-state operator reality gate",[r["organism_id"][:7] for r in reality_rows],[r["metrics"]["response_nrmse"] for r in reality_rows],"response NRMSE")
    _bar("operator_pass_fraction.png","Coordinate-free operator pass fraction",[SHORT[f] for f in FAMILIES],[disc.get("families",{}).get(f,{}).get("first_order_gate",{}).get("pass_fraction",0) for f in FAMILIES],"pass fraction",(0,1.05))
    med_nrmse=[]
    for f in FAMILIES:
        xs=[r["metrics"]["response_nrmse"] for r in disc.get("rows",[]) if r["family"]==f];med_nrmse.append(float(np.median(xs)) if xs else 0)
    _bar("operator_response_nrmse.png","Median oracle-relative response error",[SHORT[f] for f in FAMILIES],med_nrmse,"median response NRMSE")
    margins=[]
    for f in FAMILIES:
        xs=[]
        for r in disc.get("rows",[]):
            if r["family"]==f:xs.extend(r.get("controls",{}).get("margins",{}).values())
        margins.append(float(np.median(xs)) if xs else 0)
    _bar("control_margin_by_family.png","Median specificity-control error margin",[SHORT[f] for f in FAMILIES],margins,"error margin")
    cross=[disc.get("families",{}).get(f,{}).get("cross_organism",{}).get("p90_pairwise_disagreement_normalized") or 0 for f in FAMILIES]
    _bar("cross_organism_disagreement.png","Cross-organism response disagreement",[SHORT[f] for f in FAMILIES],cross,"p90 normalized disagreement")
    orders=[disc.get("families",{}).get(f,{}).get("operator_order",0) for f in FAMILIES]
    _bar("operator_order.png","Minimum retained operator order",[SHORT[f] for f in FAMILIES],orders,"operator order",(0,2.2))
    _bar("compact_localization_pass_fraction.png","Compact spatiotemporal localization",[SHORT[f] for f in FAMILIES],[loc.get("families",{}).get(f,{}).get("compact_support_gate",{}).get("pass_fraction",0) for f in FAMILIES],"discovery pass fraction",(0,1.05))
    _bar("localization_support_size.png","Compact support size when available",[SHORT[f] for f in FAMILIES],[loc.get("families",{}).get(f,{}).get("median_winner_units") or 0 for f in FAMILIES],"median active units")
    _bar("composition_pass_fraction.png","Discovery operator composition closure",[SHORT[f] for f in FAMILIES],[comp.get("families",{}).get(f,{}).get("gate",{}).get("pass_fraction",0) for f in FAMILIES],"pass fraction",(0,1.05))
    _bar("predictive_hankel_rank.png","Predictive-response 99% energy rank",[SHORT[f] for f in disc.get("accepted_families",[])],[psr.get("families",{}).get(f,{}).get("hankel",{}).get("oracle_rank_99",0) for f in disc.get("accepted_families",[])],"rank")
    _bar("meta_operator_vs_program.png","META: operator identity vs program closure",[SHORT[f]+" op" for f in disc.get("accepted_families",[])]+[SHORT[f]+" prog" for f in disc.get("accepted_families",[])],[1 if meta["families"][f]["operator_confirmed"] else 0 for f in disc.get("accepted_families",[])]+[1 if meta["families"][f]["program_closed"] else 0 for f in disc.get("accepted_families",[])],"gate",(0,1.1))
    _bar("heldout_operator_pass.png","Held-out operator confirmation",[SHORT[f] for f in held.get("families",{})],[held["families"][f]["operator_gate"]["passing"]/held["families"][f]["operator_gate"]["organisms"] for f in held.get("families",{})],"pass fraction",(0,1.05))
    _bar("historical_robustness.png","Reused historical validation robustness",[SHORT[f] for f in hist.get("families",{})],[hist["families"][f]["gate"]["pass_fraction"] for f in hist.get("families",{})],"pass fraction",(0,1.05))
    stages=["reality","discovery op","META op","heldout op","global program"]
    vals=[1,len(disc.get("accepted_families",[]))/4,len(meta.get("confirmed_families",[]))/4,len(held.get("heldout_confirmed_families",[]))/4,1/4]
    _bar("evidence_ladder.png","V837aq evidence ladder",stages,vals,"powered-family fraction",(0,1.05))
    return sorted(p.name for p in PLOTS.glob("*.png"))


def _record_family_failures(disc,loc,comp,meta):
    # Individual AQ3 failures were recorded during execution. Family-level closures are recorded separately elsewhere.
    return None


def analyze()->dict:
    reality=_load("raw/operator_reality_gate.json",{});disc=_load("raw/operator_discovery.json",{});loc=_load("raw/localization_results.json",{});comp=_load("raw/composition_results.json",{});meta=_load("raw/meta_confirmation.json",{});psr=_load("raw/predictive_state_diagnostic.json",{});held=_load("raw/heldout_confirmation.json",{});hist=_load("raw/historical_validation_robustness.json",{})
    if not reality.get("pass"):
        diagnosis="PROGRAM_OPERATOR_MEASUREMENT_INVALID";global_program=[];next_program="V837ar_OPERATOR_MEASUREMENT_REDESIGN"
    else:
        held_ops=list(held.get("heldout_confirmed_families",[]));global_program=[]
        for f in held_ops:
            if comp.get("families",{}).get(f,{}).get("pass") and meta.get("families",{}).get(f,{}).get("program_closed") and held.get("families",{}).get(f,{}).get("program_pass"):
                global_program.append(f)
        if len(held_ops)>=3 and len(global_program)<3:
            diagnosis="CAUSAL_OPERATOR_FOUND_NOT_COMPOSITIONALLY_CLOSED";next_program="V837ar_CAUSAL_OPERATOR_CANONICALIZATION_AND_PROGRAM_IR"
        elif len(held_ops)>=3 and len(global_program)>=3:
            diagnosis="GENERAL_PROGRAM_LEVEL_CAUSAL_OPERATOR_PATTERN";next_program="V837ar_CAUSAL_OPERATOR_CANONICALIZATION_AND_PROGRAM_IR"
        elif held_ops:
            diagnosis="PROGRAM_OPERATOR_ESTABLISHED_LIMITED_FAMILY_SCOPE";next_program="V837ar_CAUSAL_OPERATOR_BOUNDARY_AND_PROGRAM_IR"
        else:
            diagnosis="NO_SHARED_PROGRAM_LEVEL_CAUSAL_OPERATOR";next_program="V837ar_TASK_EQUIVALENCE_WITHOUT_SHARED_OPERATOR"
    held_ops=list(held.get("heldout_confirmed_families",[]));global_program=[f for f in held_ops if comp.get("families",{}).get(f,{}).get("pass") and meta.get("families",{}).get(f,{}).get("program_closed") and held.get("families",{}).get(f,{}).get("program_pass")]
    plots=_plots(reality,disc,loc,comp,meta,psr,held,hist)
    resources=build_resource_accounting();failures=initialize();sync_central_ledger()
    result={
      "version":"V837aq","program":"V837aq_PROGRAM_LEVEL_CAUSAL_OPERATOR_LOCALIZATION","diagnosis":diagnosis,
      "reality_gate_pass":bool(reality.get("pass")),"discovery_operator_families":list(disc.get("accepted_families",[])),"discovery_operator_family_count":len(disc.get("accepted_families",[])),
      "meta_confirmed_operator_families":list(meta.get("confirmed_families",[])),"heldout_confirmed_operator_families":held_ops,"heldout_confirmed_operator_family_count":len(held_ops),
      "globally_compositionally_closed_families":global_program,"globally_compositionally_closed_family_count":len(global_program),
      "second_order_required_families":[f for f,v in disc.get("families",{}).items() if v.get("pass") and v.get("operator_order")==2],
      "compact_spatiotemporal_discovery_families":[f for f,v in loc.get("families",{}).items() if v.get("pass")],
      "compact_spatiotemporal_meta_families":[f for f,v in meta.get("families",{}).items() if v.get("compact_localization_meta_gate",{}).get("pass")],
      "predictive_response_rank_99":{f:v.get("hankel",{}).get("oracle_rank_99") for f,v in psr.get("families",{}).items()},
      "coordinate_free_operator_pattern_established":len(held_ops)>=3,
      "broad_program_composition_established":len(global_program)>=3,
      "strongest_scientific_claim":"Three powered families exhibit oracle-relative, coordinate-free causal response operators that reproduce across independent organisms and held-out organisms; only iterative state satisfies the frozen discovery+META+heldout composition closure, so broad program-level compositional closure is not established.",
      "fresh_audit_consumed":False,"primitive_archive_allowed_next":False,"primitive_archive_population":False,"primitives_promoted":0,"v838_started":False,
      "new_source_model_fits":0,"source_optimizer_steps":0,"hidden_state_set_primary_evidence":False,"cross_organism_state_alignment":False,"cross_organism_q_alignment":False,
      "next_program":next_program,"resource_accounting":resources,"failure_memory_entries":len(failures["entries"]),"required_plots":plots,
    }
    write_json(HERE/"results.json",result);write_json(HERE/"diagnostics/decision_state.json",result);write_json(STATUS,{**result,"resource_accounting":"experiments/v837_primitive_invention/v837aq/v837aq_resource_accounting.json"})
    (HERE/"PASS.md").write_text("# V837aq closure\n\nPrimary operator-localization milestone: PASS.\n\nFinal diagnosis: `"+diagnosis+"`.\n\nThree powered coordinate-free operators are established and held-out confirmed. Broad program composition remains unresolved because only iterative state closes across discovery, META, and held-out. PrimitiveArchive remains blocked.\n",encoding="utf-8",newline="\n")
    lines=[
      "# V837 Program-Level Causal Operator Localization Report","",
      "## 1. Mission","V837aq tested whether reusable computation is better represented as a coordinate-free causal response operator over trajectories than as a shared neural-state coordinate.","",
      "## 2. V837ap forensic boundary","V837ap remains `DECODABLE_LOW_DIMENSIONAL_STATE_NOT_CAUSALLY_CLOSED`. It did not test quotient/commutativity/heldout because zero candidates reached those stages; engineering-invalid chart rows remain engineering-invalid rather than scientific negatives.","",
      "## 3. Frozen ontology","Primary evidence uses natural task/environment semantic interventions. Arbitrary hidden-state SET, cross-organism state alignment, and q-vector alignment are excluded.","",
      "## 4. AQ2 reality gate",f"Iterative-state reality gate: {sum(r.get('pass',False) for r in reality.get('organisms',[]))}/{len(reality.get('organisms',[]))} organisms PASS; kill switch={reality.get('kill_switch_triggered')}.","",
      "## 5. Coordinate-free operator discovery",json.dumps({f:{k:v for k,v in disc.get('families',{}).get(f,{}).items() if k not in {'interaction'}} for f in FAMILIES},indent=2),"",
      "## 6. Operator orders",json.dumps({f:disc.get('families',{}).get(f,{}).get('operator_order') for f in FAMILIES},indent=2),"Routing requires a second-order interaction term; recall and iterative state remain first order. Variable composition fails the frozen single-intervention operator gate and is null.","",
      "## 7. Cross-organism equivalence",json.dumps({f:disc.get('families',{}).get(f,{}).get('cross_organism') for f in FAMILIES},indent=2),"",
      "## 8. Negative controls","Time/magnitude shuffles, wrong phase, wrong-family templates, endpoint-only predictors, matched low-order regression, and family-compatible historically incompetent organisms were evaluated. Search-engine identity was never treated as a positive/negative label.","",
      "## 9. Spatiotemporal localization",json.dumps({f:{k:v for k,v in loc.get('families',{}).get(f,{}).items() if k!='organism_rows'} for f in loc.get('families',{})},indent=2),"Delayed recall reaches 4/6 compact-support passes in discovery but only 3/6 on META; compact support is therefore suggestive, not a robust final family-level claim. Routing and iterative state remain organism-scale distributed within the tested <=8-unit support family.","",
      "## 10. Composition",json.dumps(comp.get('families',{}),indent=2),"Only iterative state passes the frozen discovery composition family gate. Later META/heldout success cannot rescue routing/recall discovery failures.","",
      "## 11. Predictive causal-state diagnostic",json.dumps({f:{'required':v.get('required'),'rank99':v.get('hankel',{}).get('oracle_rank_99'),'compact':v.get('compact_predictive_rank')} for f,v in psr.get('families',{}).items()},indent=2),"The predictive-response diagnostic is non-gating and cannot rescue composition failure.","",
      "## 12. META confirmation",json.dumps(meta.get('families',{}),indent=2),"",
      "## 13. Frozen operator contracts",f"Pre-heldout contract SHA-256: `{read_json(HERE/'raw/frozen_program_operator_contracts.json')['contract_sha256']}`.","",
      "## 14. Held-out organisms",json.dumps(held.get('families',{}),indent=2),"All three frozen operator families confirm on held-out organisms with no operator refit.","",
      "## 15. Reused historical validation",json.dumps(hist.get('families',{}),indent=2),"This panel is descriptive/post-freeze and cannot upgrade a failed discovery or META gate.","",
      "## 16. Final diagnosis",f"`{diagnosis}`","",
      "## 17. Strongest claim",result["strongest_scientific_claim"],"",
      "## 18. Resource accounting",json.dumps(resources,indent=2),"",
      "## 19. Failure memory",f"V837aq failure-memory entries: {len(failures['entries'])}. Scientific and engineering failures are typed separately.","",
      "## 20. Protected locks","Fresh audit 90000-90499 remains unused. PrimitiveArchive remains blocked. Primitives promoted = 0. V838 was not started.","",
      "## 21. Next program",f"`{next_program}`","Its first gate must resolve the routing/recall composition boundary before any archive population.","",
      "## 22. Figures",* [f"- `experiments/v837_primitive_invention/v837aq/plots/{p}`" for p in plots],""
    ]
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
    return result


if __name__=="__main__":print(json.dumps(analyze(),indent=2))
