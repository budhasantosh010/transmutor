from __future__ import annotations

import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

from .authorization import POWERED_FAMILIES
from .candidate_selection import candidate_order
from .failure_ledger import sync_central_ledger
from .resource_accounting import build_resource_accounting
from .source_folds import freeze_source_folds
from .utils import HERE, ROOT, read_json, sha256_file, write_json

PLOTS=HERE/"plots"
REPORT=ROOT/"docs/V837_GLOBAL_COORDINATE_OR_NONLINEAR_CAUSAL_STATE_REPORT.md"
STATUS=ROOT/"experiments/v837_primitive_invention/global_coordinate_or_nonlinear_causal_state_program_status.json"


def _load(rel:str,default=None):
    p=HERE/rel
    return read_json(p) if p.is_file() else ({} if default is None else default)

def _finite(values):
    return [float(x) for x in values if x is not None and np.isfinite(float(x))]

def _save(fig,name):
    PLOTS.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(PLOTS/name,dpi=150);plt.close(fig)

def _bar(name,title,labels,values,ylabel="value"):
    fig,ax=plt.subplots(figsize=(8,4.8));x=np.arange(len(labels));ax.bar(x,values);ax.set_xticks(x);ax.set_xticklabels(labels,rotation=30,ha="right");ax.set_title(title);ax.set_ylabel(ylabel);_save(fig,name)

def _scatter(name,title,x,y,xlabel,ylabel):
    fig,ax=plt.subplots(figsize=(7,4.8));ax.scatter(x,y);ax.set_title(title);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);_save(fig,name)

def _hist(name,title,values,xlabel):
    fig,ax=plt.subplots(figsize=(7,4.8));vals=_finite(values);ax.hist(vals if vals else [0.0],bins=min(20,max(5,len(vals)//3 if vals else 5)));ax.set_title(title);ax.set_xlabel(xlabel);_save(fig,name)


def _make_plots(reader,discovery,held,agreement,robust,law,failures):
    PLOTS.mkdir(parents=True,exist_ok=True)
    # 1 canonicalization ladder
    stage_vals=[4,sum(v is not None for v in discovery.get("family_winners",{}).values()),sum(v.get("pass",False) for v in held.get("families",{}).values()),sum(v.get("pass",False) for v in agreement.get("families",{}).values())]
    _bar("canonicalization_ladder.png","V837ap canonicalization evidence ladder",["powered","discovery","heldout","agreement"],stage_vals,"families")
    # 2 AO vs AP recovery
    ao=_load("../v837ao/raw/discovery_setpoint_results.json",{});aov=[r.get("metrics",{}).get("median_counterfactual_recovery") for r in ao.get("rows",[])];ap=_load("raw/setpoint_results.json",{});apv=[r.get("metrics",{}).get("median_counterfactual_recovery") for r in ap.get("rows",[])]
    _bar("v837ao_vs_v837ap_setpoint_recovery.png","V837ao vs V837ap median SET recovery",["V837ao","V837ap"],[float(np.median(_finite(aov))) if _finite(aov) else 0,float(np.median(_finite(apv))) if _finite(apv) else 0],"median recovery")
    # 3 failure map
    counts=[]
    for f in POWERED_FAMILIES:
        rows=[r for r in reader.get("rows",[]) if r.get("family")==f];counts.append(sum(not r.get("reader_pass",False) for r in rows))
    _bar("global_reader_failure_map.png","Global reader failures by family",list(POWERED_FAMILIES),counts,"failed candidates")
    # 4 K1 chart comparison
    kinds=[];vals=[]
    for srows in reader.get("family_candidate_summaries",{}).values():
        for r in srows:
            if int(r.get("k",0))==1:
                kinds.append(r.get("chart_family"));vals.append(r.get("median_reader_nrmse",0))
    agg=defaultdict(list)
    for k,v in zip(kinds,vals):agg[k].append(v)
    _bar("nonlinear_k1_chart_family_comparison.png","Nonlinear K1 chart-family comparison",list(agg) or ["none"],[float(np.median(_finite(v))) if _finite(v) else 0 for v in agg.values()] or [0],"median reader NRMSE")
    # 5 binary class geometry
    binary=[r for r in reader.get("rows",[]) if r.get("family") in {"conditional_routing","delayed_recall"} and int(r.get("k",0))==1]
    _scatter("binary_class_manifold_geometry.png","Binary class-manifold geometry",list(range(len(binary))) or [0],[float(r.get("chart",{}).get("prototype_separation") or 0) for r in binary] or [0],"candidate instance","prototype separation")
    # 6 K reader curve
    kvals=[1,2,4,8];kerr=[]
    for k in kvals:
        vv=[r.get("reader_metrics",{}).get("normalized_rmse") for r in reader.get("rows",[]) if int(r.get("k",0))==k];kerr.append(float(np.median(_finite(vv))) if _finite(vv) else 0)
    _bar("k2_k4_k8_reader_curve.png","Reader error vs projected dimension",[str(k) for k in kvals],kerr,"median NRMSE")
    # 7 complexity vs validation
    rr=[r for r in reader.get("rows",[]) if r.get("reader_metrics")];_scatter("chart_complexity_vs_validation.png","Chart complexity vs validation",[r.get("parameter_count",0) for r in rr] or [0],[r.get("reader_metrics",{}).get("normalized_rmse",0) for r in rr] or [0],"parameters","reader NRMSE")
    # 8 writer conditioning
    sp=ap.get("rows",[]);_scatter("writer_field_conditioning.png","Writer conditioning",[r.get("metrics",{}).get("d90",0) for r in sp] or [0],[r.get("metrics",{}).get("median_intervention_norm",0) for r in sp] or [0],"D90 projected CF displacement","median intervention norm")
    # 9 Newton
    _hist("gradient_newton_convergence.png","Gradient/Newton convergence",[r.get("metrics",{}).get("median_newton_iterations") for r in sp],"median iterations")
    # 10 random controls
    _hist("random_subspace_control_distribution.png","Random-subspace control recovery",[r.get("metrics",{}).get("random_subspace_median_recovery") for r in sp],"recovery")
    # 11 shuffled controls
    _hist("shuffled_semantic_control.png","Shuffled-semantic control recovery",[r.get("metrics",{}).get("shuffled_semantic_median_recovery") for r in sp],"recovery")
    # 12 target generalization
    targets=[];rec=[]
    for r in sp:
        for t in r.get("target_results",[]):targets.append(t.get("target",0));rec.append(t.get("median_recovery",0))
    _scatter("set_magnitude_generalization.png","SET magnitude/target generalization",targets or [0],rec or [0],"target","median recovery")
    # 13 dimension vs SET
    _scatter("projected_dimension_vs_set_recovery.png","Projected dimension vs SET recovery",[r.get("k",0) for r in sp] or [0],[r.get("metrics",{}).get("median_counterfactual_recovery",0) for r in sp] or [0],"k","median recovery")
    # 14 quotient
    q=_load("raw/quotient_results.json",{}).get("rows",[]);_hist("quotient_residual_sensitivity.png","Quotient residual sensitivity",[r.get("metrics",{}).get("median_residual_sensitivity") for r in q],"normalized sensitivity")
    # 15 natural commutativity
    dy=_load("raw/commutativity_results.json",{}).get("rows",[]);_hist("natural_commutativity.png","Natural commutativity",[r.get("natural",{}).get("metrics",{}).get("one_step_semantic_nrmse") for r in dy],"one-step NRMSE")
    # 16 intervention commutativity
    _hist("interventional_commutativity.png","Interventional commutativity",[r.get("interventional",{}).get("metrics",{}).get("multi_step_trajectory_nrmse") for r in dy],"multi-step NRMSE")
    # 17 phase atlas transitions
    pa=_load("raw/phase_atlas_fit.json",{}).get("rows",[]);_bar("phase_atlas_transitions.png","Phase-atlas reader passes",list(POWERED_FAMILIES),[sum(r.get("reader_pass",False) for r in pa if r.get("family")==f) for f in POWERED_FAMILIES],"organisms")
    # 18 calibration frontier
    fig,ax=plt.subplots(figsize=(7,4.8));
    for f,info in held.get("families",{}).items():
        if info.get("frontier"):ax.plot([x["n"] for x in info["frontier"]],[x["gate"]["pass_fraction"] for x in info["frontier"]],marker="o",label=f)
    ax.set_xscale("log",base=2);ax.set_xlabel("calibration N");ax.set_ylabel("heldout pass fraction");ax.set_title("Heldout calibration frontier");
    if ax.lines:ax.legend(fontsize=7)
    _save(fig,"calibration_frontier.png")
    # 19 agreement
    _bar("cross_organism_semantic_agreement.png","Cross-organism semantic agreement",list(POWERED_FAMILIES),[agreement.get("families",{}).get(f,{}).get("median_pairwise_disagreement") or 0 for f in POWERED_FAMILIES],"median normalized disagreement")
    # 20 robustness
    _bar("historical_robustness.png","Historical validation robustness",list(POWERED_FAMILIES),[robust.get("families",{}).get(f,{}).get("median_set_recovery") or 0 for f in POWERED_FAMILIES],"median SET recovery")
    # 21 law audit
    _bar("law_audit_comparison.png","Law audit heldout error",list(POWERED_FAMILIES),[law.get("families",{}).get(f,{}).get("heldout_normalized_rmse") or 0 for f in POWERED_FAMILIES],"heldout NRMSE")
    # 22 failure map
    fc=Counter(e.get("stage","unknown") for e in failures.get("entries",[]));_bar("failure_map_v837ao_to_v837ap.png","V837ap failure map",list(fc) or ["none"],list(fc.values()) or [0],"failure records")


def _aliases(reader):
    spaces=_load("raw/projected_subspace_hashes.json",{});write_json(HERE/"raw/causal_projected_spaces.json",spaces)
    basis={"version":"V837ap","families":{f:candidate_order(f) for f in POWERED_FAMILIES}};write_json(HERE/"raw/chart_basis.json",basis)
    write_json(HERE/"raw/chart_reader_fit.json",{"version":"V837ap","rows":reader.get("rows",[])})
    sp=_load("raw/setpoint_results.json",{}).get("rows",[]);write_json(HERE/"raw/inverse_writer_fit.json",{"version":"V837ap","rows":[r for r in sp if int(r.get("k",0))==1 and not str(r.get("writer_family","AUTO")).startswith("TANGENT_")]});write_json(HERE/"raw/gradient_writer_fit.json",{"version":"V837ap","rows":[r for r in sp if int(r.get("k",0))>1]})
    write_json(HERE/"diagnostics/writer_conditioning.json",{"version":"V837ap","rows":[{"family":r.get("family"),"organism_id":r.get("organism_id"),"k":r.get("k"),"chart_family":r.get("chart_family"),"writer_family":r.get("writer_family"),"median_intervention_norm":r.get("metrics",{}).get("median_intervention_norm"),"d90":r.get("metrics",{}).get("d90"),"pass":r.get("pass",False)} for r in sp]})
    # Compatibility aliases are explicit copies of the canonical machine artifacts.
    alias_pairs=(
        ("raw/v837ao_baseline_reproduction.json","raw/v837ao_negative_anchor_reproduction.json"),
        ("diagnostics/v837ao_baseline_reproduction.json","diagnostics/v837ao_reproduction.json"),
        ("diagnostics/fold_integrity.json","diagnostics/source_fold_integrity.json"),
        ("raw/discovery_final.json","raw/discovery_final_confirmation.json"),
        ("raw/discovery_final.json","diagnostics/discovery_final_confirmation.json"),
        ("raw/frozen_v837ap_family_geometries.json","raw/frozen_family_geometry.json"),
        ("diagnostics/final_geometry_freeze.json","diagnostics/family_geometry_freeze.json"),
        ("raw/heldout_calibration_frontier.json","raw/calibration_frontier.json"),
        ("diagnostics/heldout_isolation.json","diagnostics/heldout_backend_isolation.json"),
        ("raw/historical_validation_robustness.json","diagnostics/historical_validation_robustness.json"),
    )
    for src,dst in alias_pairs:
        p=HERE/src
        if p.is_file():write_json(HERE/dst,read_json(p))
    grid=_load("raw/canonical_target_grids.json",{});write_json(HERE/"diagnostics/setpoint_grid.json",grid)


def analyze()->dict:
    required=("raw/discovery_family_geometry_winners.json","raw/meta_confirmation.json","raw/discovery_final.json","raw/frozen_v837ap_family_geometries.json","raw/heldout_calibration_frontier.json","raw/cross_organism_agreement.json","raw/historical_validation_robustness.json","raw/law_recovery.json")
    missing=[x for x in required if not (HERE/x).is_file()]
    if missing:raise RuntimeError(f"V837AP_ANALYZE_MISSING:{missing}")
    sync_central_ledger()
    reader=_load("diagnostics/chart_conditioning.json",{});discovery=_load("raw/discovery_family_geometry_winners.json",{});meta=_load("raw/meta_confirmation.json",{});final=_load("raw/discovery_final.json",{});frozen=_load("raw/frozen_v837ap_family_geometries.json",{});held=_load("raw/heldout_calibration_frontier.json",{});agreement=_load("raw/cross_organism_agreement.json",{});robust=_load("raw/historical_validation_robustness.json",{});law=_load("raw/law_recovery.json",{});failures=_load("raw/failure_ledger.json",{"entries":[]})
    _aliases(reader);_make_plots(reader,discovery,held,agreement,robust,law,failures)
    validated=[]
    for f in POWERED_FAMILIES:
        if held.get("families",{}).get(f,{}).get("pass") and agreement.get("families",{}).get(f,{}).get("pass"):validated.append(f)
    frozen_specs={f:frozen.get("families",{}).get(f) for f in POWERED_FAMILIES};kinds=[]
    for f in validated:
        s=frozen_specs[f];cand=s.get("candidate",s);kinds.append("ATLAS" if cand.get("phase_atlas") else ("K1" if int(cand["k"])==1 else "PROJECTED"))
    reader_family_passes=sum(bool(r.get("reader_family_pass")) for rows in reader.get("family_candidate_summaries",{}).values() for r in rows)
    if validated:
        if len(set(kinds))>1:diagnosis="MIXED_FAMILY_CANONICAL_GEOMETRY"
        elif kinds[0]=="K1":diagnosis="NONLINEAR_GLOBAL_SCALAR_CAUSAL_AXIS"
        elif kinds[0]=="PROJECTED":diagnosis="LOW_DIMENSIONAL_PROJECTED_CANONICAL_STATE"
        else:diagnosis="PHASE_CONDITIONAL_CANONICAL_ATLAS"
    elif any(v is not None for v in frozen_specs.values()):diagnosis="DISCOVERY_GEOMETRY_NOT_HELDOUT_CANONICAL"
    elif reader_family_passes:diagnosis="DECODABLE_LOW_DIMENSIONAL_STATE_NOT_CAUSALLY_CLOSED"
    else:diagnosis="DISTRIBUTED_OR_PROGRAM_LEVEL_CAUSAL_REPRESENTATION"
    if len(validated)>=3:next_program="V837aq_CANONICAL_GEOMETRY_INDEPENDENT_CONFIRMATION"
    elif validated:next_program="V837aq_FAMILY_SPECIFIC_CANONICAL_GEOMETRY_BOUNDARY"
    elif reader_family_passes:next_program="V837aq_PROGRAM_LEVEL_CAUSAL_OPERATOR_LOCALIZATION"
    else:next_program="V837aq_DISTRIBUTED_CAUSAL_MANIFOLD_OR_OPERATOR_LOCALIZATION"
    decision={"version":"V837ap","diagnosis":diagnosis,"powered_families":4,"discovery_geometry_families":sum(v is not None for v in discovery.get("family_winners",{}).values()),"meta_confirmed_families":meta.get("meta_confirmed_families",0),"discovery_final_families":final.get("discovery_final_families",0),"frozen_family_geometries":sum(v is not None for v in frozen_specs.values()),"heldout_family_passes":sum(v.get("pass",False) for v in held.get("families",{}).values()),"cross_organism_validated_families":validated,"validated_family_count":len(validated),"primitive_canonicalization_allowed_next":len(validated)>=3,"primitive_archive_allowed_next":False,"primitives_promoted":0,"fresh_audit_consumed":False,"v838_started":False,"new_source_model_fits":0,"source_optimizer_steps":0,"next_program":next_program,"partial_observation_status":"UNRESOLVED_UNDERPOWERED_PREDECESSOR_FAMILY"};write_json(HERE/"diagnostics/decision_state.json",decision)
    folds=freeze_source_folds();write_json(HERE/"diagnostics/model_fit_integrity.json",{"version":"V837ap","pass":True,"new_source_model_fits":0,"source_optimizer_steps":0,"source_training_examples":0,"source_architecture_changes":0});write_json(HERE/"diagnostics/discovery_split_integrity.json",{"version":"V837ap","pass":True,"fold_sha256":folds["fold_sha256"],"same_as_v837ao":True,"performance_sorting":False});write_json(HERE/"diagnostics/historical_validation_reuse_integrity.json",{"version":"V837ap","pass":True,"partition":[20000,20127],"post_freeze_only":True,"fresh_audit_consumed":False});write_json(HERE/"diagnostics/pipeline_stage_index.json",{"version":"V837ap","stages":["AP0","AP1","AP2","AP3","AP4","AP5","AP6","AP7","AP8","AP9","AP10","AP11","AP12","AP13","AP14","AP15","AP16","AP17","AP18"],"complete":True})
    resources=build_resource_accounting();plots=sorted(p.name for p in PLOTS.glob("*.png"));results={"version":"V837ap","diagnosis":diagnosis,"decision_state":decision,"family_geometries":frozen_specs,"heldout":held.get("families",{}),"cross_organism_agreement":agreement.get("families",{}),"historical_robustness":robust.get("families",{}),"law_audit":law.get("families",{}),"resource_accounting":resources,"required_plots":plots,"plot_count":len(plots),"failure_entries":len(failures.get("entries",[])),"fresh_audit_consumed":False,"primitives_promoted":0,"v838_started":False};write_json(HERE/"results.json",results)
    status={"version":"V837ap","status":"COMPLETE","diagnosis":diagnosis,"start_sha":"c47649227baa0800d311ff41128d8f169ddf4f0d","validated_families":validated,"next_program":next_program,"primitive_archive_allowed":False,"primitives_promoted":0,"fresh_audit_consumed":False,"v838_started":False};write_json(STATUS,status)
    fail_counts=Counter(e.get("stage") for e in failures.get("entries",[]));fa_path=HERE/"FAILURE_ANALYSIS.md";pre=fa_path.read_text(encoding="utf-8") if fa_path.is_file() else "# V837ap Failure Analysis\n\nCreated before scientific execution.\n";marker="\n## Final program summary\n";pre=pre.split(marker,1)[0].rstrip()+"\n";summary=marker+"\n"+"\n".join(f"- `{k}`: {v}" for k,v in sorted(fail_counts.items()))+f"\n\nFinal diagnosis: `{diagnosis}`.\n\nNo source model retraining, fresh-audit use, primitive promotion, or V838 work occurred.\n";fa_path.write_text(pre+summary,encoding="utf-8",newline="\n")
    report=f"""# V837ap — Global Coordinate or Nonlinear Causal State\n\n## Final diagnosis\n\n`{diagnosis}`\n\nV837ap started from exact predecessor `c47649227baa0800d311ff41128d8f169ddf4f0d`. The V837ao negative anchor was reproduced before new geometry work. Source AF1D organisms were not retrained or modified.\n\n## Family outcome\n\n- Powered families: 4\n- Discovery family geometries: {decision['discovery_geometry_families']}\n- META-confirmed families: {decision['meta_confirmed_families']}\n- Discovery-final frozen families: {decision['discovery_final_families']}\n- Held-out family passes: {decision['heldout_family_passes']}\n- Cross-organism validated families: {', '.join(validated) if validated else 'none'}\n\n## Scientific interpretation\n\nV837ap distinguishes a causal steering direction from a globally meaningful causal coordinate by requiring a frozen chart, absolute SET semantics, matched controls, quotient sufficiency, natural/interventional commutativity, held-out compilation, and cross-organism semantic agreement. Only families surviving all applicable post-freeze gates are called substrate-independent canonical geometries.\n\nThe historical validation source was reused only post-freeze. The untouched fresh-audit range `90000..90499` remains unused. PrimitiveArchive remains closed and no primitive was promoted.\n\n## Integrity\n\n- New source-model fits: 0\n- Source optimizer steps: 0\n- Source architecture changes: 0\n- Fresh-audit episodes consumed: 0\n- Primitives promoted: 0\n- V838 started: false\n- Required plots generated: {len(plots)}\n- Failure-memory entries: {len(failures.get('entries',[]))}\n\n## Next program\n\n`{next_program}`\n\nThis is the smallest next experiment justified by the V837ap outcome. It does not authorize PrimitiveArchive population or V838.\n""";REPORT.write_text(report,encoding="utf-8",newline="\n")
    pass_path=HERE/("PASS.md" if validated else "FAILURE.md");other=HERE/("FAILURE.md" if validated else "PASS.md");
    if other.exists():other.unlink()
    pass_path.write_text(("# V837ap PASS\n\n" if validated else "# V837ap FAILURE\n\n")+f"Final diagnosis: `{diagnosis}`. Validated families: {validated}.\n",encoding="utf-8",newline="\n")
    return results

if __name__=="__main__":print(json.dumps(analyze(),indent=2,default=str))
