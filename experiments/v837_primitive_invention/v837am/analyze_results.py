from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .compute_accounting import primitive_accounting
from .expanded_primitive import expanded_pair, ranking_index
from .utils import HERE, ROOT, read_json, write_json

RAW=HERE/"raw";DIAG=HERE/"diagnostics";PLOTS=HERE/"plots"


def _r(rel, default=None):
    p=HERE/rel
    return read_json(p) if p.is_file() else default


def _branch_payload(letter): return _r(f"raw/am_{letter}_selection.json",{})

def _selection_rows():
    out=[]
    for branch,letter in (("AM-A","a"),("AM-B","b"),("AM-C","c")):
        for r in _branch_payload(letter).get("results",[]):out.append((branch,r))
    return out


def _failure_counts():
    led=_r("raw/failure_ledger.json",{"entries":[]});entries=led.get("entries",[])
    correction=next((e for e in entries if e.get("failure_id")=="V837am-CORRECTION-AM-A-SCHEMA1-INVALIDATED"),{})
    invalidated=set((correction.get("exact_implementation") or {}).get("invalidated_failure_ids",[]))
    accepted_scientific=sum(e.get("failure_type")=="SCIENTIFIC_FAILURE" and e.get("failure_id") not in invalidated for e in entries)
    return {"total":len(entries),"engineering":sum(e.get("failure_type")=="ENGINEERING_FAILURE" for e in entries),"scientific":accepted_scientific,"invalidated_measurements":len(invalidated)}


def _decision():
    src=_r("diagnostics/source_integrity.json",{});a=_branch_payload("a");b=_branch_payload("b");c=_branch_payload("c");meta=_r("raw/meta_confirmation.json",{});freeze=_r("raw/selected_final_hypothesis.json",{});dev=_r("raw/final_dev_confirmation.json",{});val=_r("raw/final_validation.json",{});closed=_r("raw/closed_loop_results.json",{})
    selected=freeze.get("selected");branch=None if selected is None else selected.get("branch");cfg=None if selected is None else selected.get("config",{})
    cheating=bool((val.get("metrics") or {}).get("effect_gate_pass") and not (val.get("metrics") or {}).get("adapter_cheating_gate_pass",True)) or bool(closed.get("adapter_cheating_detected"))
    whole=bool(b.get("whole_system_effect_pass_count",0) or c.get("whole_system_effect_pass_count",0))
    canonical=False
    if cheating:
        diagnosis="ADAPTER_LEARNS_COMPUTATION_NOT_INTERFACE";next_program="V837an_PRIMITIVE_REDEFINITION_CAUSAL_ROUTING_DISTRIBUTED"
    elif selected is None:
        if whole:
            diagnosis="COMPUTATION_DISTRIBUTED_AT_ORGANISM_SCALE";next_program="V837an_DISTRIBUTED_PRIMITIVE_REDEFINITION"
        else:
            diagnosis="DYNAMICAL_RECURRENCE_NOT_OPERATOR_EQUIVALENCE";next_program="V837an_PRIMITIVE_REDEFINITION_CAUSAL_ROUTING_DISTRIBUTED"
    elif not dev.get("pass"):
        diagnosis="V837AM_DEV_CONFIRMATION_FAILURE";next_program="V837an_PRIMITIVE_REDEFINITION_CAUSAL_ROUTING_DISTRIBUTED"
    elif not val.get("pass"):
        diagnosis="V837AM_FINAL_VALIDATION_FAILURE";next_program="V837an_PRIMITIVE_REDEFINITION_CAUSAL_ROUTING_DISTRIBUTED"
    elif branch=="AM-A":
        if closed.get("run") and closed.get("pass"):
            diagnosis="CONTEXT_CONDITIONED_INTERFACE_SUFFICIENT";next_program="V837an_CANONICAL_CONTEXT_CONDITIONED_INTERFACE";canonical=True
        else:
            diagnosis="CONTEXT_ALIGNMENT_NOT_CLOSED_LOOP_STABLE";next_program="V837an_CONTEXT_FEEDBACK_INTERFACE_LOCALIZATION"
    elif branch=="AM-B":
        diagnosis="PRIMITIVE_BOUNDARY_TOO_NARROW";next_program="V837an_EXPANDED_PRIMITIVE_BOUNDARY_STANDARDIZATION"
    elif branch=="AM-C" and cfg.get("kind")=="TEMPORAL":
        diagnosis="TEMPORAL_PRIMITIVE_STATE_REQUIRED";next_program="V837an_TEMPORAL_PRIMITIVE_INTERFACE"
    else:
        diagnosis="BOUNDARY_CONTEXT_INTERACTION_REQUIRED";next_program="V837an_CONTEXTUAL_EXPANDED_PRIMITIVE_STANDARDIZATION"
    f=_failure_counts()
    return {"version":"V837am","source_integrity":bool(src.get("source_integrity")),"primary_causal_class":src.get("primary_causal_class"),"causal_recipient_count":int(src.get("causal_recipient_count",0)),"am_a_configs":int(a.get("configs_evaluated",0)),"am_a_pass_count":int(a.get("pass_count",0)),"am_b_configs":int(b.get("configs_evaluated",0)),"am_b_pass_count":int(b.get("pass_count",0)),"am_c_configs":int(c.get("configs_evaluated",0)),"am_c_pass_count":int(c.get("pass_count",0)),"meta_confirm_passes":int(meta.get("pass_count",0)),"selected_branch":branch,"selected_config":cfg if selected else None,"final_dev_confirmed":bool(dev.get("pass")),"final_validation_run":bool(val.get("run")),"final_validation_pass":bool(val.get("pass")),"closed_loop_run":bool(closed.get("run")),"closed_loop_pass":bool(closed.get("pass")),"adapter_cheating_detected":cheating,"minimum_context_rank":None,"minimum_added_cells":None,"minimum_history":None,"diagnosis":diagnosis,"canonicalization_allowed_next":canonical,"primitive_archive_allowed_next":False,"primitives_promoted":0,"new_model_fits":0,"optimizer_steps":0,"failure_entries":f["total"],"engineering_failures":f["engineering"],"scientific_failures":f["scientific"],"invalidated_measurement_records":f["invalidated_measurements"],"fresh_audit_consumed":False,"large_persistent_storage_tested":False,"v838_started":False,"next_program":next_program}


def _selected_economics(decision):
    selected=_r("raw/selected_final_hypothesis.json",{}).get("selected")
    if selected is None:return {"primitive_parameters":None,"adapter_parameters":None,"primitive_macs_per_timestep":None,"adapter_macs_per_timestep":None,"adapter_primitive_mac_ratio":None,"adapter_primitive_parameter_ratio":None,"added_context_cells":None,"history_state_bytes":0,"total_reusable_footprint":None}
    e=selected.get("economics",{});prim_p=e.get("primitive_parameters");prim_m=e.get("primitive_macs_per_timestep");ad_p=e.get("adapter_parameters");ad_m=e.get("adapter_macs");hist=e.get("history",1);k=(selected.get("config") or {}).get("added_cells",0)
    return {"primitive_parameters":prim_p,"adapter_parameters":ad_p,"primitive_macs_per_timestep":prim_m,"adapter_macs_per_timestep":ad_m,"adapter_primitive_mac_ratio":None if not prim_m or ad_m is None else float(ad_m)/float(prim_m),"adapter_primitive_parameter_ratio":None if not prim_p or ad_p is None else float(ad_p)/float(prim_p),"added_context_cells":k,"history_state_bytes":max(0,int(hist)-1)*((selected.get("config") or {}).get("added_cells",0)+3)*4*4,"total_reusable_footprint":None if prim_m is None else prim_m+(ad_m or 0)}


def _resource(decision):
    a=_branch_payload("a");b=_branch_payload("b");c=_branch_payload("c");meta=_r("raw/meta_confirmation.json",{});dev=_r("raw/final_dev_confirmation.json",{});val=_r("raw/final_validation.json",{});closed=_r("raw/closed_loop_results.json",{})
    rows_a=sum((r.get("aggregate",{}).get("row_count") or 0) for r in a.get("results",[]));rows_b=sum((r.get("aggregate",{}).get("row_count") or 0) for r in b.get("results",[]));rows_c=sum((r.get("aggregate",{}).get("row_count") or 0) for r in c.get("results",[]));meta_rows=sum((r.get("metrics",{}).get("row_count") or 0) for r in meta.get("results",[]) if r.get("run"));dev_rows=(dev.get("metrics",{}).get("row_count") or 0) if dev.get("run") else 0;val_rows=(val.get("metrics",{}).get("row_count") or 0) if val.get("run") else 0
    fit_files=list((RAW/"cache").glob("am_*/fits/*.json")) if (RAW/"cache").exists() else [];trace_files=list((RAW/"cache/traces").glob("*.pt")) if (RAW/"cache/traces").exists() else []
    svds=0
    for p in fit_files:
        try:
            d=json.loads(p.read_text(encoding="utf-8"))
            for role in d.get("roles",{}).values():
                for maps in role.get("ports",{}).values():
                    for m in (maps if isinstance(maps,list) else [maps]):
                        if (m.get("compression") or {}).get("requested_rank",0):svds+=1
        except Exception:pass
    econ=_selected_economics(decision)
    return {"version":"V837am","new_organism_fits":0,"organism_optimizer_steps":0,"processed_training_examples":0,"adapter_gradient_steps":0,"analytic_adapter_fit_files":len(fit_files),"svds":svds,"full_organism_trace_calls":len(trace_files),"pairwise_replay_calls":4*(rows_a+rows_b+rows_c+meta_rows+dev_rows+val_rows),"different_control_replay_calls":rows_a+rows_b+rows_c+meta_rows+dev_rows+val_rows,"random_refit_replay_calls":rows_a+rows_b+rows_c+meta_rows+dev_rows+val_rows,"boundary_influence_computations":int(_r("raw/frozen_context_cell_rankings.json",{}).get("occurrence_count",0)),"temporal_replay_calls":4*sum((r.get("aggregate",{}).get("row_count") or 0) for r in c.get("results",[]) if r.get("kind")=="TEMPORAL"),"final_validation_calls":4*val_rows,"closed_loop_calls":int(closed.get("forward_calls",0) or 0),"cpu_seconds":sum(float(x.get("cpu_seconds",0) or 0) for x in (a,b,c)),"wall_seconds":sum(float(x.get("wall_seconds",0) or 0) for x in (a,b,c)),"gpu_seconds":0.0,**econ}


def _safe(v):return np.nan if v is None else float(v)
def _best(rows,key="same_output"):
    vals=[_safe(r.get("aggregate",{}).get(key)) for r in rows];vals=[v for v in vals if np.isfinite(v)];return min(vals) if vals else np.nan


def _plots(decision):
    PLOTS.mkdir(parents=True,exist_ok=True);a=_branch_payload("a");b=_branch_payload("b");c=_branch_payload("c");meta=_r("raw/meta_confirmation.json",{});val=_r("raw/final_validation.json",{});econ=_selected_economics(decision)
    ar=a.get("results",[]);br=b.get("results",[]);cr=c.get("results",[])
    fams=["STATIC_FULL_AFFINE","CONTEXT_ADDITIVE","CONTEXT_MOD_R1","CONTEXT_MOD_R2","CONTEXT_MOD_R4"]
    plt.figure(figsize=(8,4));plt.bar(fams,[_best([r for r in ar if r.get("family")==f]) for f in fams]);plt.xticks(rotation=25);plt.ylabel("best output NRMSE");plt.tight_layout();plt.savefig(PLOTS/"context_family_selection.png");plt.close()
    contexts=["NONE","C1_CONTROL","C2_INPUT","C3_LOCAL_STATE","C4_COMMUNICATION","C5_FULL_LOCAL_CONTEXT"];bundles=["P1_OUTPUT","P2_STATE_OUTPUT","P3_COMMUNICATION","P4_GLOBAL_CONTROL","P5_INPUT_LOCAL","P6_ALL"];mat=np.full((len(contexts),len(bundles)),np.nan)
    for i,x in enumerate(contexts):
        for j,y in enumerate(bundles):mat[i,j]=_best([r for r in ar if r.get("context_set")==x and r.get("bundle")==y])
    plt.figure(figsize=(8,5));plt.imshow(mat,aspect="auto");plt.xticks(range(len(bundles)),bundles,rotation=35);plt.yticks(range(len(contexts)),contexts);plt.colorbar(label="best output NRMSE");plt.tight_layout();plt.savefig(PLOTS/"context_port_bundle_heatmap.png");plt.close()
    best_a=min(ar,key=lambda r:_safe(r.get("aggregate",{}).get("same_output")) if np.isfinite(_safe(r.get("aggregate",{}).get("same_output"))) else 1e99) if ar else None;m=(best_a or {}).get("aggregate",{});plt.figure(figsize=(6,4));plt.bar(["same","different refit","random refit"],[_safe(m.get("same_output")),_safe(m.get("different_output")),_safe(m.get("random_refit_output"))]);plt.ylabel("output NRMSE");plt.tight_layout();plt.savefig(PLOTS/"same_vs_random_refit.png");plt.close()
    xs=[r.get("adapter_parameters",0) for r in ar];ys=[_safe(r.get("aggregate",{}).get("same_output")) for r in ar];plt.figure(figsize=(6,4));plt.scatter(xs,ys,s=10);plt.xlabel("adapter parameters");plt.ylabel("output NRMSE");plt.tight_layout();plt.savefig(PLOTS/"adapter_capacity_vs_replay.png");plt.close()
    infl=_r("raw/frozen_context_cell_rankings.json",{}).get("rows",[]);msg=[max((r.get("message_scores") or {0:0}).values()) for r in infl] if infl else [];glob=[max((r.get("global_scores") or {0:0}).values()) for r in infl] if infl else [];plt.figure(figsize=(6,4));plt.scatter(msg,glob,s=12);plt.xlabel("max message influence");plt.ylabel("max global influence");plt.tight_layout();plt.savefig(PLOTS/"message_global_boundary_influence.png");plt.close()
    plt.figure(figsize=(8,4));plt.bar([r.get("config_id") for r in br],[_safe(r.get("aggregate",{}).get("same_output")) for r in br]);plt.xticks(rotation=90,fontsize=6);plt.ylabel("output NRMSE");plt.tight_layout();plt.savefig(PLOTS/"boundary_expansion_vs_error.png");plt.close()
    plt.figure(figsize=(6,4));plt.scatter([3+r.get("added_cells",0) for r in br],[_safe(r.get("aggregate",{}).get("same_output")) for r in br]);plt.xlabel("primitive cells");plt.ylabel("output NRMSE");plt.tight_layout();plt.savefig(PLOTS/"primitive_size_vs_replay.png");plt.close()
    temporal=[r for r in cr if r.get("kind")=="TEMPORAL"];plt.figure(figsize=(6,4));plt.plot([r.get("history") for r in temporal],[_safe(r.get("aggregate",{}).get("same_output")) for r in temporal],marker="o");plt.xlabel("history length");plt.ylabel("output NRMSE");plt.tight_layout();plt.savefig(PLOTS/"temporal_history_vs_error.png");plt.close()
    plt.figure(figsize=(6,4));plt.bar([r.get("branch") for r in meta.get("results",[])],[1 if r.get("pass") else 0 for r in meta.get("results",[])]);plt.ylabel("meta-confirm pass");plt.tight_layout();plt.savefig(PLOTS/"branch_meta_confirmation.png");plt.close()
    labels=["AM-A","AM-B","AM-C","meta","dev","validation","closed-loop"];vals=[a.get("pass_count",0),b.get("pass_count",0),c.get("pass_count",0),meta.get("pass_count",0),int(_r("raw/final_dev_confirmation.json",{}).get("pass",False)),int(val.get("pass",False)),int(_r("raw/closed_loop_results.json",{}).get("pass",False))];plt.figure(figsize=(8,4));plt.bar(labels,vals);plt.xticks(rotation=25);plt.ylabel("passing candidates / gate");plt.tight_layout();plt.savefig(PLOTS/"final_hypothesis_evidence.png");plt.close()
    plt.figure(figsize=(6,4));plt.bar(["primitive MACs","adapter MACs"],[econ.get("primitive_macs_per_timestep") or 0,econ.get("adapter_macs_per_timestep") or 0]);plt.tight_layout();plt.savefig(PLOTS/"adapter_vs_primitive_compute.png");plt.close()
    failure_counts=[1,1,a.get("configs_evaluated",0)-a.get("pass_count",0),b.get("configs_evaluated",0)-b.get("pass_count",0),c.get("configs_evaluated",0)-c.get("pass_count",0)];plt.figure(figsize=(8,4));plt.bar(["V837ak orthogonal","V837al static","V837am-A","V837am-B","V837am-C"],failure_counts);plt.xticks(rotation=25);plt.ylabel("failed hypotheses/configs");plt.tight_layout();plt.savefig(PLOTS/"failure_map.png");plt.close()


def _failure_doc(decision):
    ledger=_r("raw/failure_ledger.json",{"entries":[]});rows=[];invalidated_rows=[]
    for branch,r in _selection_rows():
        a=r.get("aggregate",{});rows.append(f"| {r.get('config_id')} | {branch} | {r.get('context_set','-')} | {r.get('bundle','-')} | {r.get('family',r.get('adapter','-'))} | {r.get('boundary','ORIGINAL')} | {r.get('history',1)} | {a.get('same_output')} | {a.get('different_output')} | {a.get('random_refit_output')} | {'PASS' if r.get('pass') else 'FAIL'} |")
    invalid=_r("raw/invalidated_am_a_selection_schema1.json",{})
    for r in invalid.get("results",[]):
        a=r.get("aggregate",{});invalidated_rows.append(f"| {r.get('config_id')} | AM-A schema-1 | {r.get('context_set','-')} | {r.get('bundle','-')} | {r.get('family','-')} | {a.get('same_output')} | {a.get('different_output')} | {a.get('random_refit_output')} | INVALIDATED | engineering correction supersedes measurement |")
    sections=["# V837am Failure Analysis","","## 1. Executive failure map",f"Final diagnosis: `{decision['diagnosis']}`. Failure records retained: **{decision['failure_entries']}** total = **{decision['scientific_failures']} accepted scientific failures + {decision['engineering_failures']} engineering failures + {decision['invalidated_measurement_records']} preserved invalidated measurements**.","","## 2. Historical failures inherited from V837ak","`CONTEXT_BOUND_COMPUTATIONAL_MOTIFS`: 6 confirmed, 1 causal, 0 boundary-interchangeable, 0 orthogonal rescues.","","## 3. Historical failures inherited from V837al","`LINEAR_INTERFACE_ALIGNMENT_INSUFFICIENT`: 253/253 static interface configurations evaluated; 0 GLOBAL and 0 CAUSAL SELECT_PASS.","","## 4. V837am hypotheses attempted","AM-A context-conditioned interfaces, AM-B influence-ranked spatial boundaries, and AM-C temporal/boundary-context fallbacks were all predeclared before final validation.","","## 5. Context-conditioned interface failures",f"Corrected AM-A failures: {decision['am_a_configs']-decision['am_a_pass_count']} / {decision['am_a_configs']}. The earlier schema-1 126-config batch remains below as invalidated engineering history and is excluded from accepted scientific counts.","","## 6. Boundary-expansion failures",f"AM-B failures/nonpromotable diagnostics: {decision['am_b_configs']-decision['am_b_pass_count']} / {decision['am_b_configs']}.","","## 7. Temporal-interface failures",f"AM-C failures/nonpromotable diagnostics: {decision['am_c_configs']-decision['am_c_pass_count']} / {decision['am_c_configs']}.","","## 8. Boundary/context interaction failures","See the complete configuration table and machine ledger.","","## 9. Control failures / adapter-cheating cases",f"Adapter cheating detected in final decision: {decision['adapter_cheating_detected']}.","","## 10. Implementation problems encountered",f"Engineering failures recorded: {decision['engineering_failures']}; invalidated measurements preserved: {decision['invalidated_measurement_records']}.","","## 11. Statistical failures","Every ledger entry stores the exact failed gate, distance from gate, and power status.","","## 12. Reality-gate failures","Source/checkpoint/data/context reality gates are recorded separately from scientific failures.","","## 13. Approaches permanently ruled out","V837ak orthogonal boundary rescue and V837al context-independent linear interfaces must not be repeated unchanged; every definitive V837am failure is also marked `do_not_repeat_unchanged`.","","## 14. Approaches only partially ruled out","Only the explicitly frozen context ranks, boundary expansions, histories, controls, and costs are ruled out.","","## 15. What remains unknown","The final diagnosis and next program delimit remaining hypotheses without erasing negative evidence.","","## 16. Exact next justified experiments",f"`{decision['next_program']}`.","","## Complete accepted tested-configuration table","| config ID | branch | context | ports/bundle | family/adapter | boundary | history | same output | different refit | random refit | result |","|---|---|---|---|---|---|---:|---:|---:|---:|---|",*rows,"","## Preserved invalidated schema-1 AM-A configuration table","These measurements are retained for audit but are scientifically superseded by the corrected schema-2 run.","","| config ID | branch | context | bundle | family | same output | different refit | random refit | status | reason |","|---|---|---|---|---|---:|---:|---:|---|---|",*invalidated_rows,"",f"Machine ledger entries: `{len(ledger.get('entries',[]))}` in `raw/failure_ledger.json` and `diagnostics/failure_ledger.json`."]
    (HERE/"FAILURE_ANALYSIS.md").write_text("\n".join(sections)+"\n",encoding="utf-8")


def _claim(d):
    if d["diagnosis"]=="DYNAMICAL_RECURRENCE_NOT_OPERATOR_EQUIVALENCE":return "The tested V837ak dynamical class does not behave as a portable operator under the tested static, contextual, spatial, temporal, or interaction redefinitions."
    if d["diagnosis"]=="COMPUTATION_DISTRIBUTED_AT_ORGANISM_SCALE":return "Proper local redefinitions failed while a whole-system diagnostic succeeded, supporting computation distributed at organism scale rather than a reusable local primitive."
    if d["diagnosis"]=="CONTEXT_CONDITIONED_INTERFACE_SUFFICIENT":return "A causally important recurring computation becomes closed-loop reusable through a context-conditioned interface with zero organism retraining; canonicalization remains future work."
    if d["diagnosis"]=="PRIMITIVE_BOUNDARY_TOO_NARROW":return "A wider influence-ranked spatial boundary restores teacher-forced interoperability, supporting a broader primitive boundary but not yet a standardized transplantable primitive."
    if d["diagnosis"]=="TEMPORAL_PRIMITIVE_STATE_REQUIRED":return "The reusable object requires temporal boundary history rather than a memoryless spatial interface."
    if d["diagnosis"]=="BOUNDARY_CONTEXT_INTERACTION_REQUIRED":return "Reuse requires the predeclared interaction of a wider primitive boundary and context-conditioned interface."
    if d["diagnosis"]=="ADAPTER_LEARNS_COMPUTATION_NOT_INTERFACE":return "Equal-capacity randomized-refit controls show that adapter capacity can reproduce the effect without meaningful donor computation; primitive reuse is rejected."
    return "No frozen V837am explanation survived all required confirmation gates."


def _report(decision,resource):
    a=_branch_payload("a");b=_branch_payload("b");c=_branch_payload("c");meta=_r("raw/meta_confirmation.json",{});freeze=_r("raw/selected_final_hypothesis.json",{});dev=_r("raw/final_dev_confirmation.json",{});val=_r("raw/final_validation.json",{});closed=_r("raw/closed_loop_results.json",{});source=_r("diagnostics/source_integrity.json",{})
    lines=["# V837 Context-Conditioned Interface or Primitive Redefinition Report","","## 1. V837al starting state","V837al closed `LINEAR_INTERFACE_ALIGNMENT_INSUFFICIENT` after 253/253 static configurations and authorized V837am.","","## 2. Historical failure map","V837ak and V837al failures are backfilled into the permanent failure ledger without reinterpretation.","","## 3. Why static linear alignment is closed","Identity and legacy baselines reproduced in V837al; N=38 causal support was powered; no static interface selected.","","## 4. Primary causal class",f"`{source.get('primary_causal_class')}`, recipients={source.get('causal_recipient_count')}, p_min={source.get('minimum_possible_p')}.","","## 5. Anti-adapter-cheating controls","Every serious replay uses SAME_CLASS, DIFFERENT_CLASS_REFIT, RANDOMIZED_REFIT, and RANDOMIZED_FIXED_ADAPTER.","","## 6. Context-conditioned interface family","Analytic static/additive and rank-1/2/4 bilinear context maps; zero gradient steps.","","## 7. AM-A results",f"126 configs; passes={a.get('pass_count')}; winner=`{a.get('winner')}`.","","## 8. Boundary influence definition","Message and rank-4 global contributions are ranked on AM-B FIT only; rankings are frozen before AM-B SELECT.","","## 9. AM-B results",f"20 configs; passes={b.get('pass_count')}; whole-system effect passes={b.get('whole_system_effect_pass_count')}; winner=`{b.get('winner')}`.","","## 10. Temporal/context-interaction results",f"7 configs; passes={c.get('pass_count')}; whole-system interaction effect passes={c.get('whole_system_effect_pass_count')}; winner=`{c.get('winner')}`.","","## 11. Meta-confirmation",f"`{meta}`", "","## 12. Selected explanation",f"`{freeze.get('selected')}`", "","## 13. Final development confirmation",f"`{dev}`", "","## 14. Final validation",f"`{val}`", "","## 15. Closed-loop result",f"`{closed}`", "","## 16. Complexity / efficiency",f"`{resource}`", "","## 17. Failure analysis",f"Failure entries={decision['failure_entries']} (engineering={decision['engineering_failures']}, scientific={decision['scientific_failures']}). See `experiments/v837_primitive_invention/v837am/FAILURE_ANALYSIS.md` and `docs/V837_FAILURE_LEDGER.md`.","","## 18. Hypotheses now ruled out",f"Diagnosis: `{decision['diagnosis']}`.","","## 19. Hypotheses still alive",f"Next program: `{decision['next_program']}`.","","## 20. Primitive interpretation",_claim(decision),"","## 21. Archive/canonicalization authorization",f"Canonicalization allowed next: **{decision['canonicalization_allowed_next']}**. Primitive archive: **BLOCKED**. Primitives promoted: **0**.","","## 22. Strongest scientific claim",_claim(decision),"","## 23. Next single program",f"`{decision['next_program']}`",""]
    (ROOT/"docs/V837_CONTEXT_INTERFACE_OR_PRIMITIVE_REDEFINITION_REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def analyze():
    decision=_decision();resource=_resource(decision);write_json(DIAG/"compute_overhead.json",resource);write_json(DIAG/"decision_state.json",decision);write_json(HERE/"v837am_resource_accounting.json",resource);write_json(HERE.parent/"v837am_resource_accounting.json",resource)
    result={"version":"V837am","question":"Why can the causally important recurring V837ak computation not be reused across independently trained organisms?","decision_state":decision,"diagnosis":decision["diagnosis"],"next_program":decision["next_program"],"resource_accounting":resource,"strongest_scientific_claim":_claim(decision)};write_json(HERE/"results.json",result);write_json(HERE.parent/"context_interface_or_primitive_redefinition_program_status.json",{"version":"V837am","diagnosis":decision["diagnosis"],"next_program":decision["next_program"],"primitive_archive_allowed_next":False,"primitives_promoted":0,"fresh_audit_episodes_consumed":0,"v838_started":False})
    marker=HERE/("PASS.md" if decision["canonicalization_allowed_next"] else "FAILURE.md");other=HERE/("FAILURE.md" if decision["canonicalization_allowed_next"] else "PASS.md");
    if other.exists():other.unlink()
    marker.write_text(f"# V837am {decision['diagnosis']}\n\n{_claim(decision)}\n",encoding="utf-8");_failure_doc(decision);_plots(decision);_report(decision,resource);return result


if __name__=="__main__":print(json.dumps(analyze(),indent=2))
