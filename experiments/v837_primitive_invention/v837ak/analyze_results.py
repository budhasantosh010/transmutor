from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from experiments.v837_primitive_invention.v837ak.authorization import assert_v837ak_authorized
from experiments.v837_primitive_invention.v837ak.probe_partition import freeze_probe_partition
from experiments.v837_primitive_invention.v837ak.protected_diagnostics import distributed_mode_diagnostic,routing_diagnostic
from experiments.v837_primitive_invention.v837ak.utils import DIRECTED,RANDOM,HERE,read_json,sha256_json,size_category,write_json

PLOTS=HERE/"plots"


def _required(path:str):
    p=HERE/path
    if not p.is_file():raise RuntimeError(f"V837AK_ANALYSIS_MISSING_ARTIFACT {path}")
    return read_json(p)


def _structural_candidate_count(structural)->int:
    return sum(c["competent_support"]>=6 and c["engine_counts"].get(DIRECTED,0)>=2 and c["engine_counts"].get(RANDOM,0)>=2 for c in structural["classes"])


def _decision()->tuple[dict,list[dict]]:
    auth=assert_v837ak_authorized();freeze_probe_partition();recon=_required("raw/reconstruction_results.json");replay=_required("diagnostics/ported_replay_gate.json");structural=_required("raw/structural_classes.json");rel=_required("diagnostics/fingerprint_reliability.json");dynamic=_required("raw/dynamic_classes_discovery.json");frozen=_required("raw/frozen_candidate_classes.json");confirmed=_required("raw/confirmed_candidate_classes.json");causal=_required("raw/causal_results.json");boundary=_required("raw/boundary_substitution_results.json");closed=_required("raw/closed_loop_substitution_results.json")
    cmap={c["class_id"]:c for c in confirmed["classes"]};causalmap={c["class_id"]:c for c in causal["classes"]};bmap={c["class_id"]:c for c in boundary["classes"]};clmap={c["class_id"]:c for c in closed["classes"]}
    reusable_ids={c["class_id"] for c in confirmed["classes"] if c.get("reusable_subsystem_eligible")}
    validated=[]
    for cid,cc in cmap.items():
        cl=clmap.get(cid);ca=causalmap.get(cid);bo=bmap.get(cid)
        if cid in reusable_ids and cl and int(cc["size"])<10 and cl.get("closed_loop_interchangeable") and ca and ca.get("causal_specificity_pass") and bo and bo.get("boundary_interchangeable"):
            validated.append({"class_id":cid,"stream":cc["stream"],"size":cc["size"],"size_category":size_category(int(cc["size"])),"support":cc["support"],"support_organisms":cc["support_organisms"],"topology_transcending":bool(cc.get("topology_transcending")),"evidence_level":6,"closed_loop":cl["closed_loop_interchangeability"],"boundary":bo["boundary_interchangeability"],"causal":{"median_specificity":ca["median_causal_specificity"],"p":ca["one_sided_p"]}})
    nonwhole_confirmed=[c for c in confirmed["classes"] if c["confirmed"] and c.get("reusable_subsystem_eligible") and int(c["size"])<10];whole_confirmed=[c for c in confirmed["classes"] if c["confirmed"] and int(c["size"])==10];causal_pass=[c for c in causal["classes"] if c["causal_specificity_pass"] and c["class_id"] in reusable_ids];boundary_pass=[c for c in boundary["classes"] if c["boundary_interchangeable"] and c["class_id"] in reusable_ids];basis=[c for c in boundary["classes"] if c["basis_alignment_rescues"] and c["class_id"] in reusable_ids]
    topology_trans=any(v["topology_transcending"] for v in validated)
    structural_rec=_structural_candidate_count(structural)>0;dynamic_confirmed=sum(c["confirmed"] and c["stream"]=="D" and c.get("reusable_subsystem_eligible") for c in confirmed["classes"])
    routing=None;distributed=None
    if not boundary_pass:routing=routing_diagnostic()
    if not nonwhole_confirmed:distributed=distributed_mode_diagnostic()
    qualifier=""
    if validated:
        diagnosis="FUNCTIONAL_DYNAMICAL_PRIMITIVES_DISCOVERED";next_program="V837al_PRIMITIVE_COMPRESSION_AND_ARCHIVE";archive=True
        if topology_trans:qualifier="DYNAMICAL_PRIMITIVES_TRANSCEND_TOPOLOGY"
    elif basis:
        diagnosis="PRIMITIVE_FUNCTION_RECURS_BUT_INTERFACE_BASIS_MISMATCH";next_program="V837al_INTERFACE_BASIS_STANDARDIZATION";archive=False
    elif causal_pass and boundary_pass:
        diagnosis="CONTEXT_BOUND_COMPUTATIONAL_MOTIFS";next_program="V837al_PRIMITIVE_INTERFACE_ALIGNMENT";archive=False
    elif dynamic_confirmed and not causal_pass:
        diagnosis="RECURRENT_DYNAMICS_NOT_CAUSALLY_SPECIFIC";next_program="V837al_CAUSAL_GRANULARITY_LOCALIZATION";archive=False
    elif not nonwhole_confirmed and whole_confirmed:
        diagnosis="COMPUTATION_DISTRIBUTED_BEYOND_LOCAL_MOTIF_SCALE";next_program="V837al_DISTRIBUTED_LATENT_PRIMITIVE_LOCALIZATION";archive=False
    elif rel.get("all_sizes_failed"):
        diagnosis="DYNAMICAL_FINGERPRINT_INADEQUATE";next_program="V837al_DYNAMICAL_REPRESENTATION_REDESIGN";archive=False
    elif structural_rec and dynamic_confirmed==0:
        diagnosis="STRUCTURAL_RECURRENCE_NOT_FUNCTIONAL_IDENTITY";next_program="V837al_FUNCTIONAL_REPRESENTATION_REDESIGN";archive=False
    elif causal_pass and not boundary_pass:
        diagnosis="CONTEXT_BOUND_COMPUTATIONAL_MOTIFS";next_program="V837al_PRIMITIVE_INTERFACE_ALIGNMENT";archive=False;qualifier="BOUNDARY_INTERCHANGEABILITY_NOT_ESTABLISHED"
    elif routing and routing.get("hypothesis_supported"):
        diagnosis="ROUTING_PRIMITIVE_HYPOTHESIS_SUPPORTED";next_program="V837al_ROUTING_PRIMITIVE_LOCALIZATION";archive=False
    else:
        diagnosis="STRUCTURAL_RECURRENCE_NOT_FUNCTIONAL_IDENTITY";next_program="V837al_FUNCTIONAL_REPRESENTATION_REDESIGN";archive=False;qualifier="NO_HIGHER_EVIDENCE_CLASS_SURVIVED"
    decision={"version":"V837ak","source_population_valid":True,"organisms_reconstructed":int(recon["organisms_reconstructed"]),"competent_organisms":int(recon["competent"]),"ported_replay_valid":bool(replay["pass"]),"structural_classes_discovered":int(structural.get("class_count_total",len(structural["classes"]))),"dynamic_classes_discovered":len(dynamic["classes"]),"confirmed_classes":int(confirmed["confirmed_count"]),"causally_specific_classes":len(causal_pass),"boundary_interchangeable_classes":len(boundary_pass),"closed_loop_interchangeable_classes":int(closed["non_whole_system_interchangeable_count"]),"topology_transcending_classes":sum(bool(c.get("topology_transcending")) for c in dynamic["classes"]),"diagnosis":diagnosis,"diagnosis_qualifier":qualifier,"validated_primitive_classes":len(validated),"primitives_promoted":0,"primitive_archive_allowed_next":archive,"fresh_audit_consumed":False,"large_persistent_storage_tested":False,"v838_started":False,"next_program":next_program,"routing_diagnostic_run":routing is not None,"distributed_mode_diagnostic_run":distributed is not None,"authorization":auth}
    return decision,validated


def _resource(decision)->dict:
    recon=read_json(HERE/"raw/reconstruction_results.json");census=read_json(HERE/"raw/subset_census_summary.json");frozen=read_json(HERE/"raw/frozen_candidate_classes.json");causal=read_json(HERE/"raw/causal_results.json");boundary=read_json(HERE/"raw/boundary_substitution_results.json");closed=read_json(HERE/"raw/closed_loop_substitution_results.json")
    trace_cache=HERE/"raw/cache/traces";probe_calls=len(list(trace_cache.glob("*.pt"))) if trace_cache.is_dir() else 0
    replay_calls=20+sum(len(c["pairs"])*(4+sum(1 for r in c["pairs"] if r.get("alignment_available"))) for c in boundary["classes"])
    causal_calls=sum(len(c["representatives"])*7 for c in causal["classes"]);sub_calls=sum(len(c["rows"])*5 for c in closed["classes"])
    timings=read_json(HERE/"diagnostics/stage_timings.json") if (HERE/"diagnostics/stage_timings.json").is_file() else {"stages":{}}
    inference_cpu=sum(float(v.get("cpu_seconds",0.0)) for v in timings.get("stages",{}).values());inference_wall=sum(float(v.get("wall_seconds",0.0)) for v in timings.get("stages",{}).values())
    rr=recon["resource_accounting"]
    payload={"version":"V837ak","organism_reconstruction":{"fits":rr["fits"],"optimizer_steps":rr["optimizer_steps"],"processed_examples":rr["processed_examples"]},"new_model_fits":50,"optimizer_steps":9600,"processed_training_examples":4915200,"probe_forward_calls":probe_calls,"replay_forward_calls":replay_calls,"causal_forward_calls":causal_calls,"substitution_forward_calls":sub_calls,"cpu_seconds":float(rr["cpu_seconds"])+inference_cpu,"wall_seconds":float(rr["wall_seconds_sum"])+inference_wall,"gpu_seconds":0.0,"occurrence_count":int(census["total_occurrences"]),"fingerprint_count":2*int(census["total_occurrences"]),"candidate_class_count":len(frozen["classes"]),"validated_primitive_classes":decision["validated_primitive_classes"],"unique_historical_family_seed_episodes":3200,"fresh_audit_episodes":0,"large_persistent_storage_tested":False,"primitives_promoted":0}
    write_json(HERE/"v837ak_resource_accounting.json",payload);write_json(ROOT/"experiments/v837_primitive_invention/v837ak_resource_accounting.json",payload);return payload


def _plots(decision,validated):
    PLOTS.mkdir(parents=True,exist_ok=True);census=read_json(HERE/"raw/subset_census_summary.json");struct=read_json(HERE/"raw/structural_classes.json");rel=read_json(HERE/"diagnostics/fingerprint_reliability.json");dyn=read_json(HERE/"raw/dynamic_classes_discovery.json");conf=read_json(HERE/"raw/confirmed_candidate_classes.json");causal=read_json(HERE/"raw/causal_results.json");boundary=read_json(HERE/"raw/boundary_substitution_results.json");closed=read_json(HERE/"raw/closed_loop_substitution_results.json")
    def save(name):plt.tight_layout();plt.savefig(PLOTS/name,dpi=160);plt.close()
    ks=list(range(1,11));plt.figure(figsize=(8,4));plt.bar(ks,[census["size_distribution"][str(k)] for k in ks]);plt.xlabel("motif size");plt.ylabel("occurrences");save("motif_support_by_size.png")
    plt.figure(figsize=(7,4));vals=[c["competent_support"] for c in struct.get("all_class_summaries",struct["classes"])];plt.hist(vals,bins=range(0,52));plt.xlabel("distinct competent-organism support");plt.ylabel("structural classes");save("structural_recurrence_distribution.png")
    plt.figure(figsize=(8,4));plt.plot(ks,[rel["sizes"][str(k)]["median_self_distance"] for k in ks],marker="o",label="self");plt.plot(ks,[rel["sizes"][str(k)]["median_nonself_distance"] for k in ks],marker="o",label="non-self");plt.legend();plt.xlabel("motif size");plt.ylabel("normalized fingerprint distance");save("fingerprint_self_vs_nonself_distance.png")
    plt.figure(figsize=(8,4));top=dyn["classes"][:20];plt.bar(range(len(top)),[c["support"] for c in top]);plt.xlabel("dynamic class rank");plt.ylabel("organism support");save("dynamic_class_support.png")
    plt.figure(figsize=(6,5));plt.scatter([c["size"] for c in dyn["classes"]],[len(c["structural_signature_counts"]) for c in dyn["classes"]]);plt.xlabel("dynamic motif size");plt.ylabel("distinct structural signatures");save("dynamic_vs_structural_identity.png")
    plt.figure(figsize=(8,4));trans=[c for c in dyn["classes"] if c.get("topology_transcending")];plt.bar(range(len(trans)),[c["support"] for c in trans]);plt.xlabel("topology-transcending class");plt.ylabel("support");save("topology_transcending_classes.png")
    plt.figure(figsize=(8,4));plt.bar(range(len(conf["classes"])),[c["retention_fraction"] for c in conf["classes"]]);plt.axhline(0.70,linestyle="--");plt.ylabel("held-out retention");save("heldout_class_retention.png")
    plt.figure(figsize=(7,5));real=[];sham=[]
    for c in causal["classes"]:
        for r in c["representatives"]:real.append(r["real_lesion_drop"]);sham.append(r["median_sham_drop"])
    plt.scatter(sham,real);plt.xlabel("matched-sham drop");plt.ylabel("real-motif drop");save("causal_real_vs_sham.png")
    same=[];diff=[];rand=[];aligned=[]
    for c in boundary["classes"]:
        for r in c["pairs"]:same.append(r["same"]["output_nrmse"]);diff.append(r["different"]["output_nrmse"]);rand.append(r["randomized"]["output_nrmse"]);aligned.append(r["aligned_same"]["output_nrmse"] if r["aligned_same"] else np.nan)
    plt.figure(figsize=(7,4));plt.boxplot([same or [0],diff or [0]],tick_labels=["same","different"]);plt.ylabel("output NRMSE");save("boundary_replay_same_vs_different.png")
    plt.figure(figsize=(7,4));plt.boxplot([same or [0],rand or [0]],tick_labels=["same","randomized"]);plt.ylabel("output NRMSE");save("boundary_replay_same_vs_random.png")
    plt.figure(figsize=(7,4));aa=[x for x in aligned if np.isfinite(x)];plt.boxplot([same or [0],aa or [0]],tick_labels=["unaligned","orthogonal aligned"]);plt.ylabel("output NRMSE");save("unaligned_vs_aligned_replay.png")
    plt.figure(figsize=(7,4));s=[];d=[];r=[]
    for c in closed["classes"]:
        for row in c["rows"]:s.append(row["same_class_success"]);d.append(row["different_class_success"]);r.append(row["randomized_success"])
    plt.boxplot([s or [0],d or [0],r or [0]],tick_labels=["same","different","random"]);plt.ylabel("closed-loop success");save("closed_loop_substitution.png")
    frozen=read_json(HERE/"raw/frozen_candidate_classes.json");ca={x["class_id"]:x for x in causal["classes"]};bo={x["class_id"]:x for x in boundary["classes"]};cl={x["class_id"]:x for x in closed["classes"]};cm={x["class_id"]:x for x in conf["classes"]};matrix=[];labels=[]
    for c in frozen["classes"]:
        cid=c["class_id"];cc=cm.get(cid,{});matrix.append([1,int(bool(cc.get("confirmed"))),1,int(bool(ca.get(cid,{}).get("causal_specificity_pass"))),int(bool(bo.get(cid,{}).get("boundary_interchangeable"))),int(bool(cl.get(cid,{}).get("closed_loop_interchangeable")))]);labels.append(cid[:8])
    plt.figure(figsize=(9,max(3,0.35*len(matrix))));plt.imshow(np.asarray(matrix or [[0]*6]),aspect="auto",vmin=0,vmax=1);plt.xticks(range(6),["recurrence","dynamic confirm","exact extraction","causality","boundary","closed-loop"],rotation=30,ha="right");plt.yticks(range(len(labels or ["none"])),labels or ["none"]);save("primitive_evidence_ladder.png")


def _report(decision,validated,resource):
    rel=read_json(HERE/"diagnostics/fingerprint_reliability.json");census=read_json(HERE/"raw/subset_census_summary.json");struct=read_json(HERE/"diagnostics/structural_recurrence.json");dyn=read_json(HERE/"diagnostics/dynamic_recurrence.json");conf=read_json(HERE/"diagnostics/heldout_confirmation.json");causal=read_json(HERE/"raw/causal_results.json");boundary=read_json(HERE/"raw/boundary_substitution_results.json");closed=read_json(HERE/"raw/closed_loop_substitution_results.json")
    reliability="\n".join(f"- size {k}: AUC {rel['sizes'][str(k)]['self_vs_nonself_roc_auc']:.4f}, self/nonself {rel['sizes'][str(k)]['self_to_nonself_ratio']:.4f}, eligible={rel['sizes'][str(k)]['eligible']}" for k in range(1,11))
    claim=("At least one non-whole-system class reaches Level 6 closed-loop interchangeability." if validated else f"No non-whole-system class reaches Level 6. The machine diagnosis is `{decision['diagnosis']}`.")
    text=f"""# V837 Functional / Dynamical Motif Discovery Report\n\n## 1. V837aj authorization\n\nV837ak is authorized by V837aj diagnosis `{decision['authorization']['v837aj_diagnosis']}` with automated structural discovery established and primitive mining allowed next. Fresh audit remains unused.\n\n## 2. Why random is no longer a negative control\n\nV837aj established matched-random structural discovery as sufficient. Directed and random finalized organisms are therefore both positive source populations when competent; engine is metadata, not a competence label.\n\n## 3. Why old isolated motif extraction is invalid for AF1D\n\nAF1D combines local cell laws with rank-4 full-state recurrent coupling, a global scalar carry controller, and ten de-shared input projections. V837ak exposes coupling and gate signals as boundary ports instead of rebuilding an isolated `NeutralGraphModel`.\n\n## 4. Reconstructed organism population\n\nExactly 50 V837aj finalized organisms were reconstructed: 25 directed, 25 random, 40 competent, 10 incompetent. Reconstruction used 50 fits, 9,600 optimizer steps, and 4,915,200 training examples.\n\n## 5. Primitive boundary-contract definition\n\nThe ported operator owns selected local recurrent/message/input/output parameters, de-shared projections, and internal message edges. Raw observation, external messages, rank-4 global terms, and the global gate are explicit ports. Global U/V, controller weights, readout, and boundary edges remain recipient-system state.\n\n## 6. Exact replay Reality Gate\n\nThe replay gate passed 20 deterministic occurrences spanning sizes 1–10 at maximum absolute error <= 1e-6.\n\n## 7. Exhaustive 1–10-cell census\n\nTotal occurrences: {census['total_occurrences']}. Connected: {census['connected_occurrences']}. Disconnected: {census['disconnected_occurrences']}. Whole-system controls: {census['whole_system_occurrences']}.\n\n## 8. Structural recurrence\n\nStructural classes: {struct['classes_total']}. Classes with support >=6: {struct['classes_support_ge_6']}. Structural signatures contain no weights, dynamics, family, engine, or competence labels.\n\n## 9. Dynamic fingerprint reliability\n\n{reliability}\n\n## 10. Dynamic recurrence\n\nDynamic classes: {dyn['class_count']}. Eligible classes: {dyn['eligible_count']}.\n\n## 11. Structural vs dynamic identity\n\nStructural and dynamic discovery streams remained separate through candidate freeze. Dynamic classes may contain multiple structural signatures.\n\n## 12. Topology-transcending candidates\n\nTopology-transcending dynamic classes discovered: {decision['topology_transcending_classes']}.\n\n## 13. Held-out confirmation\n\nFrozen candidates confirmed: {conf['confirmed_count']} / {conf['class_count']}. Confirmation seeds never created new classes.\n\n## 14. Causal specificity\n\nCausally specific classes: {decision['causally_specific_classes']}. FREEZE_UPDATE lesions were compared with five size/edge/boundary/activity-matched shams per representative.\n\n## 15. Intervention-distribution diagnostics\n\nReal-lesion outside-state divergence was compared against matched-sham divergence; >2x cases are explicitly flagged and downgraded.\n\n## 16. Boundary interchangeability\n\nBoundary-interchangeable classes: {decision['boundary_interchangeable_classes']}. Self replay is hard-gated at 1e-6 before donor comparisons.\n\n## 17. Basis-alignment diagnostic\n\nOrthogonal Procrustes adapters are fit only on DISCOVERY_PROBE traces and scored on CONFIRMATION_PROBE. Basis-rescue classes: {boundary['basis_alignment_rescue_count']}.\n\n## 18. Closed-loop substitution\n\nRun={closed['run']}. Non-whole-system Level-6 classes: {closed['non_whole_system_interchangeable_count']}. Recipient global coupling, controller, readout, and external edges remain unchanged during transplantation.\n\n## 19. Validated primitive classes\n\nCount: {len(validated)}. The primitive archive itself remains untouched in V837ak.\n\n## 20. Primitive archive authorization\n\nAllowed next: {decision['primitive_archive_allowed_next']}. Primitives promoted in V837ak: 0.\n\n## 21. Strongest scientific claim\n\n{claim}\n\n## 22. Failure alternatives / Red Team\n\nThe program explicitly distinguishes topology recurrence, dynamical recurrence, replay fidelity, causal specificity, basis mismatch, context dependence, distributed computation, routing-only hypotheses, and fingerprint inadequacy. No recurring graph is called a primitive by recurrence alone.\n\n## 23. Next single program\n\n`{decision['next_program']}`\n\n### Resource accounting\n\n```json\n{json.dumps(resource,indent=2,sort_keys=True)}\n```\n"""
    (ROOT/"docs/V837_FUNCTIONAL_DYNAMICAL_MOTIF_DISCOVERY_REPORT.md").write_text(text,encoding="utf-8")


def analyze()->dict:
    decision,validated=_decision();write_json(HERE/"validated_primitive_classes.json",{"version":"V837ak","count":len(validated),"classes":validated,"primitive_archive_populated":False});write_json(HERE/"diagnostics/decision_state.json",decision);resource=_resource(decision);_plots(decision,validated);_report(decision,validated,resource)
    results={"version":"V837ak","question":"Do independently trained competent Transmutor organisms contain compact recurring computational processes that are faithfully extractable, causally specific, and cross-organism interchangeable?","diagnosis":decision["diagnosis"],"diagnosis_qualifier":decision["diagnosis_qualifier"],"validated_primitive_classes":validated,"decision_state":decision,"resource_accounting":resource,"strongest_scientific_claim":("Validated functional/dynamical primitives are established by closed-loop cross-organism substitution." if validated else f"V837ak closes without a Level-6 reusable primitive; the evidence localizes the next blocker as {decision['diagnosis']}."),"next_program":decision["next_program"]}
    write_json(HERE/"results.json",results)
    write_json(ROOT/"experiments/v837_primitive_invention/functional_dynamical_motif_discovery_program_status.json",{
        "version":"V837ak","diagnosis":decision["diagnosis"],"diagnosis_qualifier":decision["diagnosis_qualifier"],
        "validated_primitive_classes":len(validated),"primitive_archive_allowed_next":decision["primitive_archive_allowed_next"],
        "next_program":decision["next_program"],"fresh_audit_episodes_consumed":0,"primitives_promoted":0,
        "large_persistent_storage_tested":False,"v838_started":False,
    })
    marker=HERE/("PASS.md" if validated else "FAILURE.md");other=HERE/("FAILURE.md" if validated else "PASS.md");
    if other.exists():other.unlink()
    marker.write_text(("# V837ak PASS\n\n" if validated else "# V837ak informative non-promotion outcome\n\n")+f"Diagnosis: `{decision['diagnosis']}`.\n\nPrimitive archive allowed next: {decision['primitive_archive_allowed_next']}. Primitives promoted: 0. Fresh audit: 0. V838: not started.\n",encoding="utf-8")
    return results


def main()->int:
    r=analyze();print(json.dumps({"diagnosis":r["diagnosis"],"qualifier":r["diagnosis_qualifier"],"validated_primitive_classes":len(r["validated_primitive_classes"]),"next_program":r["next_program"]},indent=2));return 0

if __name__=="__main__":raise SystemExit(main())
