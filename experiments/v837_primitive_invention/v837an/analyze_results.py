from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .authorization import COUNTERFACTUALS, PRIMARY_PROBES
from .resource_accounting import compute_resource_accounting
from .utils import HERE, ROOT, read_json, write_json

RAW = HERE / "raw"
DIAG = HERE / "diagnostics"
PLOTS = HERE / "plots"

UPDATE_LAWS = {
    "conditional_routing": "SELECT(control, A, B)",
    "delayed_recall": "WRITE / MAINTAIN / READ",
    "iterative_state": "z_next = 0.65*z + 0.35*x",
    "partial_observation": "latent AR(1) state; candidate abstraction is physical z_t, not a proven belief state",
    "variable_composition": "z_next = tanh(gain*z + drive)",
}


def _r(rel: str, default=None):
    p = HERE / rel
    return read_json(p) if p.is_file() else default


def _validated_rows() -> list[dict]:
    val = _r("raw/final_validation.json", {}) or {}
    return [r for r in val.get("results", []) if r.get("family_pass")]


def _branch_counts(rows: list[dict]) -> dict[str, int]:
    out = {"AN-A": 0, "AN-A-COMPILER": 0, "AN-B": 0, "AN-C": 0}
    for row in rows:
        candidate = row.get("candidate") or {}
        branch = candidate.get("branch")
        if branch == "AN-A":
            out["AN-A"] += 1
            if candidate.get("semantic_compiler"):
                out["AN-A-COMPILER"] += 1
        elif branch in out:
            out[branch] += 1
    return out


def _whole_system_or_broad_coalition() -> bool:
    c = _r("raw/synergy_results.json", {}) or {}
    for fr in c.get("family_results", {}).values():
        med = fr.get("median_minimum_coalition_cardinality")
        if fr.get("family_pass") and med is not None and float(med) >= 7:
            return True
        for row in fr.get("organism_winners", []):
            winner = row.get("winner")
            if winner and int(winner.get("cardinality", 0)) == 10:
                return True
    return False


def _strong_synergy_validated_families(validated: list[dict]) -> list[str]:
    c = _r("raw/synergy_results.json", {}) or {}
    allowed = {r["family"] for r in validated if (r.get("candidate") or {}).get("branch") == "AN-C"}
    return sorted(
        family
        for family, fr in c.get("family_results", {}).items()
        if family in allowed and fr.get("strong_synergy_family")
    )


def _decision() -> dict:
    src = _r("diagnostics/source_integrity.json", {}) or {}
    a = _r("raw/an_a_selection.json", {}) or {}
    b = _r("raw/routing_selection.json", {}) or {}
    c = _r("raw/synergy_results.json", {}) or {}
    val = _r("raw/final_validation.json", {}) or {}
    ledger = _r("raw/failure_ledger.json", {"entries": []}) or {"entries": []}
    phase = _r("diagnostics/phase_stability.json", {}) or {}
    validated = _validated_rows()
    counts = _branch_counts(validated)
    compiler_families = sorted(r["family"] for r in validated if (r.get("candidate") or {}).get("branch") == "AN-A" and (r.get("candidate") or {}).get("semantic_compiler"))
    subspace_families = sorted(r["family"] for r in validated if (r.get("candidate") or {}).get("branch") == "AN-A")
    routing_families = sorted(r["family"] for r in validated if (r.get("candidate") or {}).get("branch") == "AN-B")
    coalition_families = sorted(r["family"] for r in validated if (r.get("candidate") or {}).get("branch") == "AN-C")
    synergy_families = _strong_synergy_validated_families(validated)
    rank4 = bool(b.get("rank4_global_bus_supported"))
    whole = _whole_system_or_broad_coalition()

    qualifier = None
    canonicalization = False
    if len(compiler_families) >= 3:
        diagnosis = "GENERAL_CAUSAL_LATENT_PRIMITIVE_PATTERN"
        next_program = "V837ao_LATENT_PRIMITIVE_CANONICALIZATION"
        canonicalization = True
    elif compiler_families:
        diagnosis = "CAUSAL_LATENT_PRIMITIVE_ESTABLISHED"
        next_program = "V837ao_LATENT_PRIMITIVE_CANONICALIZATION"
        canonicalization = True
    elif subspace_families:
        diagnosis = "CAUSAL_MACROVARIABLE_ESTABLISHED"
        qualifier = "SEMANTIC_COMPILER_NOT_ESTABLISHED"
        next_program = "V837ao_CAUSAL_COMPILER_LOCALIZATION"
        canonicalization = True
    elif routing_families:
        diagnosis = "CAUSAL_ROUTING_PRIMITIVE_ESTABLISHED"
        next_program = "V837ao_CAUSAL_ROUTING_CANONICALIZATION"
        canonicalization = True
    elif synergy_families:
        diagnosis = "DISTRIBUTED_SYNERGISTIC_PRIMITIVE_ESTABLISHED"
        next_program = "V837ao_DISTRIBUTED_CAUSAL_HYPEREDGE_CANONICALIZATION"
        canonicalization = True
    elif coalition_families and whole:
        diagnosis = "COMPUTATION_DISTRIBUTED_AT_ORGANISM_SCALE"
        next_program = "V837ao_GLOBAL_CAUSAL_ABSTRACTION"
    elif coalition_families:
        diagnosis = "DISTRIBUTED_CAUSAL_COALITION_ESTABLISHED"
        next_program = "V837ao_DISTRIBUTED_CAUSAL_HYPEREDGE_CANONICALIZATION"
        canonicalization = True
    else:
        diagnosis = "NO_SHARED_INTERNAL_CAUSAL_PRIMITIVE_AT_TESTED_GRANULARITY"
        next_program = "V837ao_BEHAVIORAL_PROGRAM_PRIMITIVE_REDEFINITION"

    powered_families = sum(bool(v.get("strong_cross_organism_powered")) for v in src.get("families", {}).values())
    unresolved = sorted(set(src.get("families", {})) - set(val.get("validated_families", [])))
    return {
        "version": "V837an",
        "source_integrity": bool(src.get("valid")),
        "historical_refinement_recorded": all(x in {e.get("failure_id") for e in ledger.get("entries", [])} for x in ("REF-AN-001", "REF-AN-002", "REF-AN-003")),
        "oracle_instrumentation_valid": bool((_r("diagnostics/oracle_equivalence.json", {}) or {}).get("pass")),
        "source_organisms": int(src.get("source_organisms", 0)),
        "competent_organisms": int(src.get("competent", 0)),
        "powered_families": powered_families,
        "development_causal_subspace_families": int(a.get("causal_subspace_families", 0)),
        "development_semantic_compiler_families": int(a.get("semantic_compiler_families", 0)),
        "development_causal_routing_families": int(b.get("causal_routing_families", 0)),
        "development_synergistic_families": int(c.get("synergistic_family_count", 0)),
        "causal_subspace_families": len(subspace_families),
        "semantic_compiler_families": len(compiler_families),
        "causal_routing_families": len(routing_families),
        "synergistic_families": len(synergy_families),
        "validated_coalition_families": len(coalition_families),
        "causal_subspace_family_names": subspace_families,
        "semantic_compiler_family_names": compiler_families,
        "causal_routing_family_names": routing_families,
        "synergistic_family_names": synergy_families,
        "unresolved_families": unresolved,
        "rank4_global_bus_supported": rank4,
        "rank4_support_families": b.get("rank4_support_families", []),
        "phase_conditional_representation": bool(phase.get("phase_conditional_representation")),
        "final_validation_run": bool(val.get("run")),
        "validated_family_abstractions": int(val.get("validated_family_count", 0)),
        "diagnosis": diagnosis,
        "qualifier": qualifier,
        "canonicalization_allowed_next": canonicalization,
        "primitive_archive_allowed_next": False,
        "primitives_promoted": 0,
        "new_model_fits": 0,
        "optimizer_steps": 0,
        "adapter_gradient_steps": 0,
        "failure_entries": len(ledger.get("entries", [])),
        "fresh_audit_consumed": False,
        "large_persistent_storage_tested": False,
        "v838_started": False,
        "next_program": next_program,
    }


def _claim(d: dict) -> str:
    diagnosis = d["diagnosis"]
    if diagnosis in {"GENERAL_CAUSAL_LATENT_PRIMITIVE_PATTERN", "CAUSAL_LATENT_PRIMITIVE_ESTABLISHED"}:
        return "At least one independently trained AF1D family contains a low-dimensional, interventionally faithful causal macrostate whose semantic delta can be compiled into held-out causal interventions without organism retraining or cross-organism state inversion."
    if diagnosis == "CAUSAL_MACROVARIABLE_ESTABLISHED":
        return "A shared low-dimensional causal macrovariable is supported by held-out source-swap interventions, but a canonical semantic compiler from high-level causal delta to local intervention has not been established."
    if diagnosis == "CAUSAL_ROUTING_PRIMITIVE_ESTABLISHED":
        return "A compact communication mechanism, rather than a transferable full microstate, reliably mediates the task-level counterfactual effect across independently trained organisms."
    if diagnosis == "DISTRIBUTED_SYNERGISTIC_PRIMITIVE_ESTABLISHED":
        return "The reusable causal object is a compact synergistic coalition: joint intervention succeeds while its individual members do not explain the effect alone."
    if diagnosis == "DISTRIBUTED_CAUSAL_COALITION_ESTABLISHED":
        return "A reproducible causal computation is supported only as a multi-cell coalition under the tested intervention granularity."
    if diagnosis == "COMPUTATION_DISTRIBUTED_AT_ORGANISM_SCALE":
        return "The tested causal effect is recoverable only through broad or whole-organism intervention, arguing against a small internal primitive at the tested granularity."
    return "Across the frozen causal-subspace, routing, and coalition scopes, no adequately supported shared internal causal primitive survived development confirmation and held-out validation."


def _contracts(decision: dict) -> None:
    out_dir = RAW / "candidate_abstraction_contracts"
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze = _r("raw/frozen_family_abstractions.json", {}) or {}
    val = _r("raw/final_validation.json", {}) or {}
    val_by_family = {r["family"]: r for r in val.get("results", [])}
    src = _r("raw/source_population.json", {}) or {}
    a = _r("raw/an_a_selection.json", {}) or {}
    b = _r("raw/routing_selection.json", {}) or {}
    c = _r("raw/synergy_results.json", {}) or {}
    for family, winner in freeze.get("families", {}).items():
        if winner is None:
            continue
        candidate = winner.get("candidate", {})
        branch = candidate.get("branch")
        carrier_type = None
        carrier_dimension = 0
        k = None
        routing_channels: list[str] = []
        coalition_cells: list[int] = []
        semantic_compiler = False
        support: list[str] = []
        engines: set[str] = set()
        metrics = val_by_family.get(family, {})
        if branch == "AN-A":
            carrier_type = candidate.get("carrier")
            k = int(candidate.get("k", 0))
            carrier_dimension = 40 if carrier_type != "GATE1" else 1
            semantic_compiler = bool(candidate.get("semantic_compiler"))
            row = next((s for s in a.get("family_config_summaries", []) if s.get("family") == family and s.get("config_id") == candidate.get("config_id")), None)
            if row:
                support = list(row.get("semantic_compiler_support_organisms" if semantic_compiler else "source_swap_support_organisms", []))
                pop_rows = src.get("rows", [])
                engines = {x.get("engine") for x in pop_rows if x.get("organism_id") in support}
        elif branch == "AN-B":
            carrier_type = "COMMUNICATION_ROUTING"
            fam_rows = [x for x in b.get("organism_results", []) if x.get("family") == family and x.get("powered")]
            for x in fam_rows:
                cfg = x.get("configs", {}).get(candidate.get("config_id"), {})
                if cfg.get("metrics", {}).get("pass"):
                    support.append(x["organism_id"]); engines.add(x.get("engine")); routing_channels.extend(cfg.get("channels", []))
            routing_channels = sorted(set(routing_channels))
            carrier_dimension = max((len(routing_channels) * 4), 1)
        elif branch == "AN-C":
            carrier_type = "CELL_COALITION_STATE"
            fr = c.get("family_results", {}).get(family, {})
            for row in fr.get("organism_winners", []):
                if row.get("winner"):
                    support.append(row["organism_id"]); engines.add(row.get("engine")); coalition_cells.extend(row["winner"].get("nodes", []))
            coalition_cells = sorted(set(coalition_cells))
            carrier_dimension = 4 * len(coalition_cells)
        contract = {
            "family": family,
            "semantic_variable": PRIMARY_PROBES.get(family, ""),
            "semantic_dimension": 1,
            "phase": "PRIMARY_SEMANTIC_PHASE",
            "counterfactual_intervention": COUNTERFACTUALS.get(family, ""),
            "carrier_type": carrier_type,
            "carrier_dimension": carrier_dimension,
            "causal_subspace_dimension": k,
            "routing_channels": routing_channels,
            "coalition_cells": coalition_cells,
            "semantic_compiler": semantic_compiler,
            "update_law": UPDATE_LAWS.get(family, ""),
            "support_organisms": sorted(set(x for x in support if x)),
            "engines": sorted(x for x in engines if x),
            "topology_ids": sorted({r.get("topology_id") for r in src.get("rows", []) if r.get("organism_id") in support and r.get("topology_id")}),
            "causal_metrics": metrics,
            "ood_metrics": {"family_validation_pass": bool(metrics.get("family_pass"))},
            "primitive_archive_record": False,
        }
        write_json(out_dir / f"{family}.json", contract)


def _failure_doc(decision: dict) -> None:
    ledger = _r("raw/failure_ledger.json", {"entries": []}) or {"entries": []}
    entries = ledger.get("entries", [])
    scientific = [e for e in entries if e.get("failure_type") == "SCIENTIFIC_FAILURE"]
    engineering = [e for e in entries if e.get("failure_type") == "ENGINEERING_FAILURE"]
    underpowered = [e for e in entries if e.get("failure_type") == "UNDERPOWERED" or e.get("result_status") == "underpowered"]
    refinements = [e for e in entries if e.get("failure_type") == "INTERPRETATION_REFINEMENT"]
    def table(rows: list[dict]) -> list[str]:
        out = ["| ID | Stage | Result | Failed gate | Meaning |", "|---|---|---|---|---|"]
        for e in rows:
            failed = "; ".join(str(x) for x in e.get("which_exact_gate_failed", [])) or "—"
            meaning = str(e.get("scientific_interpretation", "")).replace("|", "/")
            out.append(f"| {e.get('failure_id')} | {e.get('stage')} | {e.get('result_status')} | {failed.replace('|','/')} | {meaning} |")
        return out
    sections = [
        "# V837an Failure Analysis", "",
        "## 1. V837an question", "What is the lowest-dimensional interventionally faithful causal computation implemented across independently trained AF1D organisms, and how is it routed?", "",
        "## 2. Historical interpretation refinements", *table(refinements), "",
        "## 3. Ground-truth macrostate assumptions", "Five benchmark-defined semantic variables are instrumented without rewriting historical task generators; partial-observation physical `z_t` remains a candidate abstraction, not a claimed network belief state.", "",
        "## 4. Counterfactual-construction failures", *table([e for e in entries if "COUNTERFACTUAL" in e.get("stage", "")]), "",
        "## 5. Instrumentation failures", *table([e for e in entries if "INSTRUMENT" in e.get("stage", "") or "ORACLE" in e.get("stage", "")]), "",
        "## 6. Causal-subspace failures", *table([e for e in entries if e.get("stage") == "AN-A"]), "",
        "## 7. Semantic-compiler failures", "Semantic compiler status is stored separately from source-swap status for every AN-A configuration and preserved in the machine ledger/results.", "",
        "## 8. Random/sham-control failures", "Every AN-A candidate uses 32 Haar-random subspaces plus shuffled-semantic, same-norm-noise, and orthogonal-residual diagnostics.", "",
        "## 9. OOD intervention failures", "OOD failures remain explicit in per-organism metrics and are never converted into a causal pass.", "",
        "## 10. Routing failures", *table([e for e in entries if e.get("stage") == "AN-B"]), "",
        "## 11. Global-bus failures", "Rank-4/global-bus support is intervention-based and reported independently from configured rank.", "",
        "## 12. Coalition/synergy failures", *table([e for e in entries if e.get("stage") == "AN-C"]), "",
        "## 13. Statistical underpower", *table(underpowered), "",
        "## 14. Engineering failures", *table(engineering), "",
        "## 15. Definitively ruled-out hypotheses", "Only exact frozen configurations that were adequately powered and failed their predeclared gates are ruled out unchanged.", "",
        "## 16. Provisionally ruled-out hypotheses", "Underpowered families and diagnostics with missing exact intervention implementations remain provisional.", "",
        "## 17. What remains unknown", f"Final diagnosis: `{decision['diagnosis']}`. Unresolved families: {decision['unresolved_families']}.", "",
        "## 18. Exact next justified experiments", f"`{decision['next_program']}`", "",
        f"Machine-searchable entries: **{len(entries)}** in `raw/failure_ledger.json` and `diagnostics/failure_ledger.json`.", "",
    ]
    (HERE / "FAILURE_ANALYSIS.md").write_text("\n".join(sections), encoding="utf-8")


def _safe_median(xs):
    vals = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return float(np.median(vals)) if vals else np.nan


def _plots(decision: dict) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    a = _r("raw/an_a_selection.json", {}) or {}
    b = _r("raw/routing_selection.json", {}) or {}
    c = _r("raw/synergy_results.json", {}) or {}
    val = _r("raw/final_validation.json", {}) or {}
    phase = _r("diagnostics/phase_stability.json", {}) or {}

    families = sorted((_r("diagnostics/source_integrity.json", {}) or {}).get("families", {}))
    ladder = []
    for family in families:
        ladder.append([
            1,
            int(any(r.get("family") == family and (r.get("decodability", {}).get("r2", -9) > 0) for r in a.get("organism_results", []))),
            int(a.get("family_winners", {}).get(family) is not None),
            int(a.get("family_winners", {}).get(family) is not None),
            int(a.get("semantic_compiler_winners", {}).get(family) is not None),
            int(b.get("family_winners", {}).get(family) is not None),
            int(bool((c.get("family_results", {}).get(family) or {}).get("strong_synergy_family"))),
            int(family in val.get("validated_families", [])),
            int(family in val.get("validated_families", [])),
        ])
    plt.figure(figsize=(10, 4)); plt.imshow(np.asarray(ladder, dtype=float) if ladder else np.zeros((1, 9)), aspect="auto", vmin=0, vmax=1); plt.xticks(range(9), ["semantic", "decodable", "low-d causal", "source-swap", "compiler", "routing", "synergy", "held-out", "cross-org"], rotation=35, ha="right"); plt.yticks(range(max(1, len(families))), families or ["no data"]); plt.colorbar(label="evidence present"); plt.tight_layout(); plt.savefig(PLOTS / "causal_abstraction_evidence_ladder.png"); plt.close()

    dec = []; caus = []
    for r in a.get("organism_results", []):
        dec.append(r.get("decodability", {}).get("r2", np.nan)); caus.append(r.get("source_swap", {}).get("median_recovery", np.nan))
    plt.figure(figsize=(6,4)); plt.scatter(dec, caus, s=10); plt.axhline(.60, linestyle="--"); plt.xlabel("decoder R²"); plt.ylabel("causal recovery"); plt.tight_layout(); plt.savefig(PLOTS / "decodability_vs_causality.png"); plt.close()

    car = defaultdict(list)
    for r in a.get("organism_results", []): car[r.get("carrier")].append(r.get("source_swap", {}).get("median_recovery"))
    labels = sorted(car); plt.figure(figsize=(7,4)); plt.bar(labels, [_safe_median(car[x]) for x in labels]); plt.axhline(.60, linestyle="--"); plt.ylabel("median source-swap recovery"); plt.xticks(rotation=25); plt.tight_layout(); plt.savefig(PLOTS / "carrier_recovery_by_family.png"); plt.close()

    dims = defaultdict(list)
    for r in a.get("organism_results", []): dims[int(r.get("k", 0))].append(r.get("source_swap", {}).get("median_recovery"))
    ks = sorted(dims); plt.figure(figsize=(6,4)); plt.plot(ks, [_safe_median(dims[k]) for k in ks], marker="o"); plt.axhline(.60, linestyle="--"); plt.xlabel("latent dimension k"); plt.ylabel("median recovery"); plt.tight_layout(); plt.savefig(PLOTS / "recovery_vs_latent_dimension.png"); plt.close()

    real=[]; rand=[]
    for r in a.get("organism_results", []): real.append(r.get("source_swap", {}).get("median_recovery")); rand.append(r.get("source_swap", {}).get("random_subspace_median_recovery"))
    plt.figure(figsize=(6,4)); plt.hist([x for x in rand if x is not None], alpha=.6, label="random subspace"); plt.hist([x for x in real if x is not None], alpha=.6, label="candidate"); plt.legend(); plt.xlabel("recovery"); plt.tight_layout(); plt.savefig(PLOTS / "random_subspace_control_distribution.png"); plt.close()

    ss=[]; sc=[]
    for r in a.get("organism_results", []): ss.append(r.get("source_swap", {}).get("median_recovery")); sc.append(r.get("semantic_compiler", {}).get("median_recovery"))
    plt.figure(figsize=(6,4)); plt.scatter(ss, sc, s=10); plt.axhline(.60, linestyle="--"); plt.axvline(.60, linestyle="--"); plt.xlabel("source-swap recovery"); plt.ylabel("semantic-compiler recovery"); plt.tight_layout(); plt.savefig(PLOTS / "semantic_compiler_vs_source_swap.png"); plt.close()

    phase_rows = phase.get("phase_rows", []); plt.figure(figsize=(7,4));
    if phase_rows:
        names=[r.get("phase") for r in phase_rows]; vals=[r.get("median_recovery", np.nan) for r in phase_rows]; plt.bar(names, vals)
    else: plt.text(.5,.5,"Phase diagnostics unavailable or not triggered",ha="center",va="center"); plt.xticks([]); plt.yticks([])
    plt.tight_layout(); plt.savefig(PLOTS / "phase_specific_causal_carriers.png"); plt.close()

    scores = _r("raw/routing_channel_scores.json", {"rows": []}) or {"rows": []}
    message_vals=[]; global_vals=[]
    for row in scores.get("rows", []):
        for x in row.get("ranking", []):
            if str(x.get("channel","")).startswith("M:"): message_vals.append(x.get("median_recovery"))
            elif str(x.get("channel","")).startswith("G:"): global_vals.append(x.get("median_recovery"))
    plt.figure(figsize=(7,4)); plt.plot(sorted([x for x in message_vals if x is not None], reverse=True)); plt.ylabel("single-edge recovery"); plt.xlabel("ranked message edge"); plt.tight_layout(); plt.savefig(PLOTS / "message_edge_mediation.png"); plt.close()
    plt.figure(figsize=(7,4)); plt.plot(sorted([x for x in global_vals if x is not None], reverse=True)); plt.ylabel("single-global-source recovery"); plt.xlabel("ranked global source"); plt.tight_layout(); plt.savefig(PLOTS / "global_source_mediation.png"); plt.close()

    bus = (_r("diagnostics/rank4_bus.json", {}) or {}).get("rows", []); conds=["BUS_MESSAGE_ONLY","BUS_GLOBAL_ONLY","BUS_GATE_ONLY","BUS_MESSAGE_GLOBAL","BUS_ALL"]; plt.figure(figsize=(8,4)); vals=[]
    for name in conds: vals.append(_safe_median([(r.get("conditions", {}).get(name) or {}).get("median_recovery") for r in bus]))
    plt.bar(conds, vals); plt.xticks(rotation=30,ha="right"); plt.ylabel("median recovery"); plt.tight_layout(); plt.savefig(PLOTS / "message_vs_global_vs_gate.png"); plt.close()
    plt.figure(figsize=(6,4)); plt.bar([r.get("family") for r in bus], [1 if r.get("global_dominant") else 0 for r in bus]); plt.xticks(rotation=30,ha="right"); plt.ylabel("global-only family support"); plt.tight_layout(); plt.savefig(PLOTS / "rank4_bus_evidence.png"); plt.close()

    subset=defaultdict(list)
    for row in b.get("organism_results", []):
        for cfg in row.get("configs", {}).values(): subset[int(cfg.get("channel_count",0))].append(cfg.get("metrics",{}).get("median_recovery"))
    xs=sorted(subset); plt.figure(figsize=(6,4)); plt.plot(xs,[_safe_median(subset[x]) for x in xs],marker="o"); plt.axhline(.60,linestyle="--"); plt.xlabel("routing subset channels"); plt.ylabel("median recovery"); plt.tight_layout(); plt.savefig(PLOTS / "routing_subset_size_vs_recovery.png"); plt.close()

    coalition = _r("raw/coalition_scan.json", {"scans": []}) or {"scans": []}; cards=defaultdict(list)
    for scan in coalition.get("scans", []):
        for r in scan.get("rows", []): cards[int(r.get("cardinality",0))].append(r.get("median_recovery"))
    xs=sorted(cards); plt.figure(figsize=(6,4)); plt.plot(xs,[_safe_median(cards[x]) for x in xs],marker="o"); plt.axhline(.60,linestyle="--"); plt.xlabel("coalition size"); plt.ylabel("median recovery"); plt.tight_layout(); plt.savefig(PLOTS / "coalition_size_vs_recovery.png"); plt.close()

    pair_mat=np.full((10,10),np.nan)
    pair_values=defaultdict(list)
    for scan in coalition.get("scans", []):
        for r in scan.get("rows", []):
            nodes=r.get("nodes",[])
            if len(nodes)==2: pair_values[tuple(nodes)].append(r.get("median_recovery"))
    for (i,j),vals in pair_values.items(): pair_mat[i,j]=pair_mat[j,i]=_safe_median(vals)
    plt.figure(figsize=(6,5)); plt.imshow(pair_mat,vmin=np.nanmin(pair_mat) if np.isfinite(pair_mat).any() else 0,vmax=np.nanmax(pair_mat) if np.isfinite(pair_mat).any() else 1); plt.colorbar(label="pair recovery"); plt.xlabel("cell"); plt.ylabel("cell"); plt.tight_layout(); plt.savefig(PLOTS / "cell_pair_synergy_heatmap.png"); plt.close()

    gran=[]; names=[]
    for family in families:
        row=next((r for r in _validated_rows() if r.get("family")==family),None); names.append(family)
        if not row: gran.append(0); continue
        cnd=row.get("candidate",{}); branch=cnd.get("branch")
        if branch=="AN-A":gran.append(int(cnd.get("k",0)))
        elif branch=="AN-B":gran.append(int(round(row.get("intervention_dof",0))))
        elif branch=="AN-C":gran.append(int(round(row.get("intervention_dof",0))))
        else:gran.append(0)
    plt.figure(figsize=(8,4)); plt.bar(names,gran); plt.xticks(rotation=30,ha="right"); plt.ylabel("minimum validated intervention DOF"); plt.tight_layout(); plt.savefig(PLOTS / "minimum_causal_granularity.png"); plt.close()

    src = _r("raw/source_population.json", {}) or {}; diversity=[]; shared=[]
    for family in families:
        rows=[r for r in src.get("rows",[]) if r.get("family")==family and r.get("competent")]; diversity.append(len({r.get("topology_id") for r in rows})); shared.append(1 if family in val.get("validated_families",[]) else 0)
    plt.figure(figsize=(6,4)); plt.scatter(diversity,shared); plt.xlabel("distinct competent topology IDs"); plt.ylabel("validated shared causal abstraction"); plt.yticks([0,1],["no","yes"]); plt.tight_layout(); plt.savefig(PLOTS / "implementation_diversity_vs_shared_semantics.png"); plt.close()

    hist=[1,253,126,20,7,len((_r("raw/failure_ledger.json",{"entries":[]}) or {}).get("entries",[]))]
    plt.figure(figsize=(9,4)); plt.bar(["V837ak boundary","V837al linear","V837am-A","V837am-B","V837am-C","V837an ledger"],hist); plt.xticks(rotation=30,ha="right"); plt.ylabel("failed/recorded hypotheses"); plt.tight_layout(); plt.savefig(PLOTS / "failure_map_v837ak_to_v837an.png"); plt.close()


def _report(decision: dict, resource: dict) -> None:
    src = _r("diagnostics/source_integrity.json", {}) or {}
    oracle = _r("diagnostics/oracle_equivalence.json", {}) or {}
    a = _r("raw/an_a_selection.json", {}) or {}
    b = _r("raw/routing_selection.json", {}) or {}
    c = _r("raw/synergy_results.json", {}) or {}
    meta = _r("raw/meta_confirmation.json", {}) or {}
    dev = _r("raw/final_dev_confirmation.json", {}) or {}
    freeze = _r("raw/frozen_family_abstractions.json", {}) or {}
    val = _r("raw/final_validation.json", {}) or {}
    lines = [
        "# V837 Causal / Routing / Distributed Primitive Redefinition Report", "",
        "## 1. Starting state", "V837am closed `DYNAMICAL_RECURRENCE_NOT_OPERATOR_EQUIVALENCE` and machine-selected V837an.", "",
        "## 2. V837am interpretation refinement", "V837an preserves V837am's diagnosis while recording that 105/126 AM-A configs were invalid under the state-map gate and all seven AM-C configs had zero valid pairs.", "",
        "## 3. Why microstate invertibility was abandoned", "V837an tests many-to-one causal carriers `s_i -> z`; no cross-organism state bijection is required anywhere in the decision path.", "",
        "## 4. Source population", f"50 reconstructed organisms: {src.get('competent')} competent, {src.get('incompetent')} incompetent. Per-family support: `{src.get('families')}`.", "",
        "## 5. Ground-truth causal semantics", f"Primary probes: `{PRIMARY_PROBES}`.", "",
        "## 6. Counterfactual construction", "Each intervention is derived from an existing historical seed and preserves all nonintervened task randomness by construction.", "",
        "## 7. Instrumented-runtime reality gate", f"Oracle episodes verified: {oracle.get('episodes_verified')}; runtime equivalence: `{_r('diagnostics/instrumented_runtime_equivalence.json',{})}`.", "",
        "## 8. Causal carrier spaces", "STATE40, OUTPUT40, MESSAGE40, GLOBAL40, and GATE1 are primary causal carriers; COUPLING_FACTOR4 is diagnostic-only.", "",
        "## 9. Low-dimensional subspace discovery", f"Development causal-subspace families: {a.get('causal_subspace_families',0)}.", "",
        "## 10. Decodability versus causality", "Decoder quality is reported but is never a pass gate; intervention faithfulness and matched controls determine causal support.", "",
        "## 11. Semantic compiler", f"Development semantic-compiler families: {a.get('semantic_compiler_families',0)}.", "",
        "## 12. Family-by-family macrovariable results", f"`{a.get('family_winners',{})}`", "",
        "## 13. Phase-specific representations", f"`{_r('diagnostics/phase_stability.json',{})}`", "",
        "## 14. Message routing", f"Development routing winners: `{b.get('family_winners',{})}`", "",
        "## 15. Global/rank-4 routing", f"Rank-4/global bus supported: **{b.get('rank4_global_bus_supported',False)}**; support families: `{b.get('rank4_support_families',[])}`.", "",
        "## 16. Communication-channel synergy", f"`{c.get('communication_channel_synergy',[])}`", "",
        "## 17. Cell coalition scan", f"Triggered families: `{c.get('triggers',{})}`", "",
        "## 18. Minimum causal granularity", "See `minimum_causal_granularity.png` and frozen family abstractions.", "",
        "## 19. Cross-organism implementation diversity", "Topology diversity is reported separately from shared semantic causal support.", "",
        "## 20. Held-out validation", f"`{val}`", "",
        "## 21. Validated causal abstractions", f"Validated families: `{val.get('validated_families',[])}`.", "",
        "## 22. Failure analysis", f"Failure entries: **{decision['failure_entries']}**. See `experiments/v837_primitive_invention/v837an/FAILURE_ANALYSIS.md` and the append-only project ledger.", "",
        "## 23. Primitive-definition revision", f"Final diagnosis: `{decision['diagnosis']}`" + (f" / `{decision['qualifier']}`" if decision.get('qualifier') else "") + ".", "",
        "## 24. Canonicalization/archive status", f"Canonicalization allowed next: **{decision['canonicalization_allowed_next']}**. Primitive archive: **BLOCKED**. Primitives promoted: **0**.", "",
        "## 25. Strongest claim", _claim(decision), "",
        "## 26. Next single program", f"`{decision['next_program']}`", "",
        "### Resource accounting", f"`{resource}`", "",
        "### Meta confirmation", f"`{meta}`", "",
        "### Final development confirmation", f"`{dev}`", "",
        "### Frozen family abstractions", f"Freeze hash: `{freeze.get('frozen_sha256')}`", "",
    ]
    (ROOT / "docs/V837_CAUSAL_ROUTING_DISTRIBUTED_PRIMITIVE_REDEFINITION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def analyze() -> dict:
    decision = _decision()
    resource = compute_resource_accounting()
    _contracts(decision)
    _failure_doc(decision)
    _plots(decision)
    _report(decision, resource)
    write_json(DIAG / "decision_state.json", decision)
    result = {
        "version": "V837an",
        "question": "What is the lowest-dimensional interventionally faithful causal computation implemented across independently trained AF1D organisms, and how is that computation routed through their distributed state and communication systems?",
        "decision_state": decision,
        "diagnosis": decision["diagnosis"],
        "qualifier": decision.get("qualifier"),
        "next_program": decision["next_program"],
        "resource_accounting": resource,
        "strongest_scientific_claim": _claim(decision),
    }
    write_json(HERE / "results.json", result)
    write_json(HERE.parent / "causal_routing_distributed_primitive_redefinition_program_status.json", {
        "version": "V837an",
        "diagnosis": decision["diagnosis"],
        "qualifier": decision.get("qualifier"),
        "next_program": decision["next_program"],
        "canonicalization_allowed_next": decision["canonicalization_allowed_next"],
        "primitive_archive_allowed_next": False,
        "primitives_promoted": 0,
        "fresh_audit_episodes_consumed": 0,
        "v838_started": False,
    })
    marker = HERE / ("PASS.md" if decision["validated_family_abstractions"] > 0 else "FAILURE.md")
    other = HERE / ("FAILURE.md" if marker.name == "PASS.md" else "PASS.md")
    if other.exists(): other.unlink()
    marker.write_text(f"# V837an {decision['diagnosis']}\n\n{_claim(decision)}\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
