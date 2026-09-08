from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.v837aj.af1d_structural_model import model_compute
from experiments.v837_primitive_invention.v837aj.fidelity_calibration import CONFIG, FAMILIES
from experiments.v837_primitive_invention.v837aj.topology import SearchTopology, degree_cosine, edge_jaccard, historical_anchor_topology, topology_edit_distance

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"


def load(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(name: str) -> list[dict]:
    payload = load(HERE / "raw" / name, {}) or {}
    return list(payload.get("rows", []))


def _proxy_runs(name: str) -> list[dict]:
    payload = load(HERE / "raw" / name, {}) or {}
    return list(payload.get("runs", []))


def _family_process(rows: list[dict]) -> dict:
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    out = {}; passes = 0
    for family in FAMILIES:
        fr = sorted(by_family[family], key=lambda r: int(r["run_index"]))
        n = len(fr)
        required = int(math.ceil(float(CONFIG["search_process_family_gate"]["competent_run_fraction_required"]) * n)) if n else 0
        devs = [float(r["development_success"]) for r in fr]
        vals = [float(r["final_validation_success"]) for r in fr]
        competent = sum(bool(r["competent"]) for r in fr)
        med_dev = float(np.median(devs)) if devs else float("nan")
        med_val = float(np.median(vals)) if vals else float("nan")
        passed = bool(n and competent >= required and med_dev >= 0.90 and med_val >= 0.85)
        passes += int(passed)
        out[family] = {
            "runs": n,
            "competent_required": required,
            "competent_hits": competent,
            "development_scores": devs,
            "validation_scores": vals,
            "median_development": med_dev,
            "median_validation": med_val,
            "pass": passed,
        }
    return {"families": out, "families_passing": passes, "structurally_competent": passes >= 4}


def _paired_rows(search_rows: list[dict], random_rows: list[dict]) -> list[dict]:
    rs = {(r["family"], int(r["run_index"])): r for r in search_rows}
    rr = {(r["family"], int(r["run_index"])): r for r in random_rows}
    if set(rs) != set(rr):
        raise RuntimeError("search/random finalization pairing mismatch")
    paired = []
    for key in sorted(rs):
        s, r = rs[key], rr[key]
        if int(s["finalization_seed"]) != int(r["finalization_seed"]):
            raise RuntimeError("search/random finalization seed mismatch")
        paired.append({
            "family": key[0], "run_index": key[1],
            "search_validation": float(s["final_validation_success"]),
            "random_validation": float(r["final_validation_success"]),
            "validation_delta": float(s["final_validation_success"] - r["final_validation_success"]),
            "search_development": float(s["development_success"]),
            "random_development": float(r["development_success"]),
            "development_delta": float(s["development_success"] - r["development_success"]),
            "search_competent": bool(s["competent"]), "random_competent": bool(r["competent"]),
        })
    return paired


def _permutation_test(deltas: list[float]) -> dict:
    values = np.asarray(deltas, dtype=np.float64)
    count = int(CONFIG["superiority_gate"]["permutations"])
    rng = np.random.default_rng(deterministic_int("v837aj-paired-permutation") % (2**63 - 1))
    observed_mean = float(np.mean(values)); observed_median = float(np.median(values))
    one = two = 0; generated = 0
    batch = 10000
    while generated < count:
        n = min(batch, count - generated)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n, len(values)))
        means = np.mean(signs * values[None, :], axis=1)
        one += int(np.sum(means >= observed_mean - 1e-15))
        two += int(np.sum(np.abs(means) >= abs(observed_mean) - 1e-15))
        generated += n
    return {
        "permutations": count,
        "observed_mean_delta": observed_mean,
        "observed_median_delta": observed_median,
        "one_sided_search_gt_random_p": float((one + 1) / (count + 1)),
        "two_sided_p": float((two + 1) / (count + 1)),
        "rng_namespace": "v837aj-paired-permutation",
    }


def _auc(curve: list[dict]) -> float:
    vals = np.asarray([float(row["best_selection_success"]) for row in curve], dtype=np.float64)
    if vals.size < 2:
        return float(vals[0]) if vals.size else 0.0
    return float(np.trapz(vals, dx=1.0) / (len(vals) - 1))


def _threshold_index(curve: list[dict]) -> int | None:
    for row in curve:
        if float(row["best_selection_success"]) >= float(CONFIG["search"]["proxy_threshold"]):
            return int(row["evaluation_index"]) + 1
    return None


def _search_efficiency(search_runs: list[dict], random_runs: list[dict]) -> dict:
    search_auc = [_auc(run["best_so_far"]) for run in search_runs]
    random_auc = [_auc(run["best_so_far"]) for run in random_runs]
    search_threshold = [_threshold_index(run["best_so_far"]) for run in search_runs]
    random_threshold = [_threshold_index(run["best_so_far"]) for run in random_runs]
    return {
        "search_best_so_far_auc": search_auc,
        "random_best_so_far_auc": random_auc,
        "median_search_auc": float(np.median(search_auc)) if search_auc else None,
        "median_random_auc": float(np.median(random_auc)) if random_auc else None,
        "search_evaluations_to_first_proxy_threshold": search_threshold,
        "random_evaluations_to_first_proxy_threshold": random_threshold,
        "median_search_evaluations_to_threshold": float(np.median([x for x in search_threshold if x is not None])) if any(x is not None for x in search_threshold) else None,
        "median_random_evaluations_to_threshold": float(np.median([x for x in random_threshold if x is not None])) if any(x is not None for x in random_threshold) else None,
    }


def _budget_match(search_runs: list[dict], random_runs: list[dict]) -> dict:
    rs = {(r["family"], int(r["run_index"])): r for r in search_runs}
    rr = {(r["family"], int(r["run_index"])): r for r in random_runs}
    slot_checks = []
    for key in sorted(rs.keys() & rr.keys()):
        srows = sorted(rs[key]["records"], key=lambda r: int(r["evaluation_index"]))
        rrows = sorted(rr[key]["records"], key=lambda r: int(r["evaluation_index"]))
        exact = len(srows) == len(rrows) == 64
        complexity = exact and all(
            int(s["topology"]["edge_count"]) == int(r["topology"]["edge_count"])
            and int(s["topology"]["recurrent_edge_count"]) == int(r["topology"]["recurrent_edge_count"])
            and int(s["candidate_initialization_slot"]) == int(r["candidate_initialization_slot"])
            for s, r in zip(srows, rrows)
        )
        slot_checks.append({"family":key[0],"run_index":key[1],"exact_64_each":exact,"slot_complexity_and_initialization_match":complexity})
    return {
        "paired_runs": len(slot_checks),
        "all_exact_64": all(x["exact_64_each"] for x in slot_checks) if slot_checks else False,
        "all_slot_matched": all(x["slot_complexity_and_initialization_match"] for x in slot_checks) if slot_checks else False,
        "runs": slot_checks,
    }


def _topology_diversity(search_rows: list[dict], random_rows: list[dict]) -> dict:
    def summarize(rows: list[dict]) -> dict:
        tops = [SearchTopology.from_dict(row["topology"]) for row in rows]
        jaccard=[]; same=[]; recur=[]; degree=[]; edit=[]
        for i in range(len(tops)):
            for j in range(i+1,len(tops)):
                jaccard.append(edge_jaccard(tops[i],tops[j])); same.append(edge_jaccard(tops[i],tops[j],recurrent=False)); recur.append(edge_jaccard(tops[i],tops[j],recurrent=True)); degree.append(degree_cosine(tops[i],tops[j])); edit.append(topology_edit_distance(tops[i],tops[j]))
        return {
            "champions":len(tops),
            "unique_topologies":len({t.topology_id for t in tops}),
            "median_edge_jaccard":float(np.median(jaccard)) if jaccard else None,
            "median_same_step_jaccard":float(np.median(same)) if same else None,
            "median_recurrent_jaccard":float(np.median(recur)) if recur else None,
            "median_degree_cosine":float(np.median(degree)) if degree else None,
            "median_topology_edit_distance":float(np.median(edit)) if edit else None,
        }
    return {"directed":summarize(search_rows),"random":summarize(random_rows)}


def _family_classifier(rows: list[dict]) -> dict:
    # Simple leave-one-out nearest-centroid classifier on topology descriptor vectors.
    if len(rows) < 10:
        return {"available":False}
    def feat(row):
        d=row["topology"]["descriptors"]
        return np.asarray(d["in_degree"]+d["out_degree"]+d["recurrent_in_degree"]+d["recurrent_out_degree"]+[d["edge_count"],d["recurrent_edge_count"],d["same_step_dag_depth"],d["weakly_connected_components"]],dtype=float)
    X=np.stack([feat(r) for r in rows]); y=np.asarray([FAMILIES.index(r["family"]) for r in rows],dtype=int)
    def score(labels):
        correct=0
        for i in range(len(rows)):
            train=np.arange(len(rows))!=i; centroids=[]
            for c in range(len(FAMILIES)):
                subset=X[train & (labels==c)]
                centroids.append(np.mean(subset,axis=0) if len(subset) else np.zeros(X.shape[1]))
            pred=int(np.argmin([np.linalg.norm(X[i]-c) for c in centroids])); correct += int(pred==labels[i])
        return correct/len(rows)
    observed=score(y); rng=np.random.default_rng(deterministic_int("v837aj-topology-family-permutation")%(2**63-1)); perms=10000; ge=0
    for _ in range(perms): ge += int(score(rng.permutation(y)) >= observed-1e-15)
    return {"available":True,"method":"leave-one-out nearest centroid on degree/edge descriptors","accuracy":float(observed),"permutations":perms,"permutation_p":float((ge+1)/(perms+1))}


def _message_dependence(search_rows: list[dict], random_rows: list[dict]) -> dict:
    def summary(rows):
        comp=[r for r in rows if r["competent"]]
        drops=[float(r["message_dependence"]["success_drop"]) for r in comp]
        changes=[float(r["message_dependence"]["mean_abs_prediction_change"]) for r in comp]
        return {"competent_champions":len(comp),"median_success_drop":float(np.median(drops)) if drops else None,"median_abs_prediction_change":float(np.median(changes)) if changes else None}
    anchor = load(ROOT / "experiments/v837_primitive_invention/v837ai/diagnostics/message_dependence.json", {}) or {}
    return {"directed":summary(search_rows),"random":summary(random_rows),"fixed_af1d_anchor_4x":anchor.get("4x",{})}


def _structural_efficiency_scoreboard(search_rows: list[dict], random_rows: list[dict]) -> dict:
    anchor_payload = load(HERE / "raw/anchor_reproduction.json", {}) or {}
    anchor_rows = list(anchor_payload.get("rows", []))
    anchor_topology = historical_anchor_topology()
    anchor_compute = model_compute(anchor_topology)
    anchor_message = load(ROOT / "experiments/v837_primitive_invention/v837ai/diagnostics/message_dependence.json", {}) or {}
    anchor_message_4x = anchor_message.get("4x", {})

    anchor_family_rows = []
    for family in FAMILIES:
        rows = [row for row in anchor_rows if row.get("family") == family]
        values = [float(row["validation_success"]) for row in rows]
        anchor_family_rows.append({
            "family": family,
            "replicates": len(values),
            "median_final_validation_success": float(np.median(values)) if values else None,
        })
    anchor_family_medians = [
        row["median_final_validation_success"]
        for row in anchor_family_rows
        if row["median_final_validation_success"] is not None
    ]

    def champion_block(rows: list[dict], engine: str) -> dict:
        entries = []
        for row in sorted(rows, key=lambda item: (item["family"], int(item["run_index"]))):
            compute = row["compute"]
            entries.append({
                "engine": engine,
                "family": row["family"],
                "run_index": int(row["run_index"]),
                "competent": bool(row["competent"]),
                "final_validation_success": float(row["final_validation_success"]),
                "message_ablation_success_drop": float(row["message_dependence"]["success_drop"]),
                "edge_count": int(row["topology"]["edge_count"]),
                "active_parameters": int(compute["active_parameters"]),
                "modeled_macs_per_timestep": int(compute["total_modeled_macs_per_timestep"]),
                "search_evaluations_required": int(row["champion_selected_evaluation_index"]) + 1,
            })

        def median(key: str):
            values = [float(entry[key]) for entry in entries]
            return float(np.median(values)) if values else None

        return {
            "engine": engine,
            "champions": len(entries),
            "competent_champions": sum(int(entry["competent"]) for entry in entries),
            "median_final_validation_success": median("final_validation_success"),
            "median_message_ablation_success_drop": median("message_ablation_success_drop"),
            "median_edge_count": median("edge_count"),
            "median_active_parameters": median("active_parameters"),
            "median_modeled_macs_per_timestep": median("modeled_macs_per_timestep"),
            "median_search_evaluations_required": median("search_evaluations_required"),
            "rows": entries,
        }

    return {
        "run": bool(search_rows and random_rows),
        "fixed_af1d_anchor": {
            "engine": "FIXED_AF1D_ANCHOR",
            "families": anchor_family_rows,
            "median_final_validation_success": float(np.median(anchor_family_medians)) if anchor_family_medians else None,
            "message_ablation_success_drop": float(anchor_message_4x["success_drop_median"]) if "success_drop_median" in anchor_message_4x else None,
            "message_ablation_scope": "V837ai 4x global median; family-specific anchor ablation was not rerun in V837aj",
            "edge_count": int(anchor_topology.edge_count),
            "active_parameters": int(anchor_compute["active_parameters"]),
            "modeled_macs_per_timestep": int(anchor_compute["total_modeled_macs_per_timestep"]),
            "search_evaluations_required": None,
            "search_evaluations_required_semantics": "fixed predeclared AF1D anchor; no structural-search evaluations required",
        },
        "directed": champion_block(search_rows, "DIRECTED_STRUCTURAL_SEARCH"),
        "random": champion_block(random_rows, "RANDOM_STRUCTURAL_SAMPLER"),
    }


def _resource_totals() -> dict:
    anchor = load(HERE / "raw/anchor_reproduction.json", {}) or {}
    fidelity = load(HERE / "raw/fidelity_runs.json", {}) or {}
    search_runs = _proxy_runs("search_proxy_runs.json"); random_runs = _proxy_runs("random_proxy_runs.json")
    search_final = _rows("search_finalized.json"); random_final = _rows("random_finalized.json")
    def aggregate(rows):
        keys=("optimizer_steps","processed_examples","forward_calls","backward_calls","environment_interactions","cpu_seconds","wall_seconds","gpu_seconds","modeled_active_mac_volume")
        out={key:0 for key in keys}
        for row in rows:
            for key in keys: out[key]+=row.get(key,0)
        out["fits"]=len(rows); return out
    anchor_rows=anchor.get("rows",[]); fidelity_rows=fidelity.get("rows",[])
    def flatten(runs): return [row for run in runs for row in run.get("records",[])]
    search_proxy=flatten(search_runs); random_proxy=flatten(random_runs)
    primary_search_proxy=flatten([run for run in search_runs if int(run.get("run_index", -1)) < 5])
    primary_random_proxy=flatten([run for run in random_runs if int(run.get("run_index", -1)) < 5])
    extension_search_proxy=flatten([run for run in search_runs if int(run.get("run_index", -1)) >= 5])
    extension_random_proxy=flatten([run for run in random_runs if int(run.get("run_index", -1)) >= 5])
    primary_search_final=[row for row in search_final if int(row.get("run_index", -1)) < 5]
    primary_random_final=[row for row in random_final if int(row.get("run_index", -1)) < 5]
    extension_search_final=[row for row in search_final if int(row.get("run_index", -1)) >= 5]
    extension_random_final=[row for row in random_final if int(row.get("run_index", -1)) >= 5]
    return {
        "anchor":aggregate(anchor_rows),"calibration":aggregate(fidelity_rows),"proxy_directed":aggregate(search_proxy),"proxy_random":aggregate(random_proxy),"final_directed":aggregate(search_final),"final_random":aggregate(random_final),
        "primary_proxy_directed":aggregate(primary_search_proxy),"primary_proxy_random":aggregate(primary_random_proxy),"primary_final_directed":aggregate(primary_search_final),"primary_final_random":aggregate(primary_random_final),
        "robustness_proxy_directed":aggregate(extension_search_proxy),"robustness_proxy_random":aggregate(extension_random_proxy),"robustness_final_directed":aggregate(extension_search_final),"robustness_final_random":aggregate(extension_random_final),
        "primary_stage_b":aggregate(primary_search_proxy + primary_random_proxy + primary_search_final + primary_random_final),
        "robustness_extension":aggregate(extension_search_proxy + extension_random_proxy + extension_search_final + extension_random_final),
        "candidate_evaluations":{"calibration":len(fidelity_rows),"directed":len(search_proxy),"random":len(random_proxy)},
        "union_unique_task_episodes":3200,
    }


def _decision(search_rows: list[dict], random_rows: list[dict], search_runs: list[dict], random_runs: list[dict]) -> dict:
    directed=_family_process(search_rows); random=_family_process(random_rows); paired=_paired_rows(search_rows,random_rows); permutation=_permutation_test([r["validation_delta"] for r in paired])
    d_hit=float(np.mean([r["competent"] for r in search_rows])); r_hit=float(np.mean([r["competent"] for r in random_rows])); advantage=d_hit-r_hit
    median_delta=float(np.median([r["validation_delta"] for r in paired])); win_fraction=float(np.mean([1.0 if r["validation_delta"]>0 else 0.5 if r["validation_delta"]==0 else 0.0 for r in paired]))
    superiority=(directed["structurally_competent"] and advantage>=0.20 and median_delta>=0.08 and permutation["one_sided_search_gt_random_p"]<=0.01)
    random_sufficient=random["structurally_competent"]
    # Proxy overfit example rule: majority of families strong proxy champions but poor final medians.
    overfit_families=0
    for family in FAMILIES:
        sr=[r for r in search_rows if r["family"]==family]
        if sr and float(np.median([r["champion_search_selection_success"] for r in sr]))>=0.90 and float(np.median([r["final_validation_success"] for r in sr]))<0.75: overfit_families+=1
    clear_failure=directed["families_passing"]<=2 and random["families_passing"]<=2 and median_delta<=0.02
    near_boundary=directed["families_passing"] in {3,4} and random["families_passing"] in {3,4}
    rob=CONFIG["robustness_triggers"]
    robustness=(
        directed["families_passing"]==int(rob["directed_family_passes_exact"])
        or float(rob["paired_validation_delta_low"])<=median_delta<float(rob["paired_validation_delta_high"])
        or float(rob["competent_hit_advantage_low"])<=advantage<float(rob["competent_hit_advantage_high"])
        or float(rob["paired_p_low"])<permutation["one_sided_search_gt_random_p"]<=float(rob["paired_p_high"])
        or near_boundary
    )
    if superiority: diagnosis="STRUCTURAL_SEARCH_RECOVERED"; auto=True; evo=True; mining=True; next_program="V837ak_FUNCTIONAL_DYNAMICAL_MOTIF_DISCOVERY"; robustness=False
    elif random_sufficient: diagnosis="RANDOM_STRUCTURAL_DISCOVERY_SUFFICIENT"; auto=True; evo=False; mining=True; next_program="V837ak_FUNCTIONAL_DYNAMICAL_MOTIF_DISCOVERY"; robustness=False
    elif overfit_families>=3: diagnosis="STRUCTURAL_SELECTION_OVERFIT"; auto=False; evo=False; mining=False; next_program="V837ak_SEARCH_SELECTION_PROTOCOL"; robustness=False
    elif clear_failure: diagnosis="STRUCTURAL_DISCOVERY_NOT_RECOVERED"; auto=False; evo=False; mining=False; next_program="V837ak_SEARCH_REPRESENTATION_OR_ALGORITHM_REDESIGN"; robustness=False
    else: diagnosis="STRUCTURAL_DISCOVERY_SUPERIORITY_INCONCLUSIVE"; auto=False; evo=False; mining=False; next_program="V837ak_STRUCTURAL_DISCOVERY_ROBUSTNESS_OR_REDESIGN"
    efficiency=_search_efficiency(search_runs,random_runs)
    return {
        "directed":directed,"random":random,"paired":paired,"permutation":permutation,"directed_competent_hit_rate":d_hit,"random_competent_hit_rate":r_hit,"competent_hit_advantage":advantage,"median_paired_final_validation_delta":median_delta,"search_win_fraction":win_fraction,"superiority_gate_pass":superiority,"random_sufficiency":random_sufficient,"overfit_families":overfit_families,"clear_failure":clear_failure,"robustness_extension_required":bool(robustness),"diagnosis":diagnosis,"automated_structural_discovery":auto,"evolutionary_search_superiority":evo,"primitive_mining_allowed_next":mining,"next_program":next_program,"efficiency":efficiency,
    }


def _placeholder_stage_b(reason: str) -> None:
    for name in ("search_proxy_runs.json","random_proxy_runs.json"):
        if not (HERE/"raw"/name).exists(): write_json(HERE/"raw"/name,{"version":"V837aj","run":False,"reason":reason,"runs":[],"candidate_evaluations":0})
    for name in ("search_champions.json","random_champions.json"):
        if not (HERE/"raw"/name).exists(): write_json(HERE/"raw"/name,{"version":"V837aj","run":False,"reason":reason,"champions":[]})
    for name in ("search_finalized.json","random_finalized.json"):
        if not (HERE/"raw"/name).exists(): write_json(HERE/"raw"/name,{"version":"V837aj","run":False,"reason":reason,"rows":[]})


def _plot_placeholder(path: Path, title: str, reason: str) -> None:
    fig=plt.figure(figsize=(8,5)); plt.axis("off"); plt.text(0.5,0.6,title,ha="center",va="center",fontsize=15); plt.text(0.5,0.4,reason,ha="center",va="center",wrap=True); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def _fidelity_plots() -> None:
    PLOTS.mkdir(exist_ok=True)
    decision=load(HERE/"diagnostics/fidelity_decision.json",{}) or {}; metrics=decision.get("metrics",{})
    labels=["F_LEGACY","F0","F1","F2","F3"]
    costs=[24*16,24*64,48*128,96*256,144*384]
    for key,name,title in (("median_spearman_rho","fidelity_spearman_vs_cost.png","Median Spearman vs proxy cost"),("median_kendall_tau","fidelity_kendall_vs_cost.png","Median Kendall tau-b vs proxy cost"),("median_top4_recall","fidelity_top4_recall_vs_cost.png","Median top-4 recall vs proxy cost")):
        fig=plt.figure(figsize=(7,5)); plt.plot(costs,[metrics.get(f,{}).get(key,np.nan) for f in labels],marker="o"); plt.xscale("log"); plt.xlabel("steps × training episodes"); plt.ylabel(key); plt.title(title); fig.tight_layout(); fig.savefig(PLOTS/name,dpi=160); plt.close(fig)
    fig=plt.figure(figsize=(8,5)); x=np.arange(len(labels)); plt.plot(x,[metrics.get(f,{}).get("median_spearman_rho",np.nan) for f in labels],marker="o",label="median Spearman"); plt.plot(x,[metrics.get(f,{}).get("median_kendall_tau",np.nan) for f in labels],marker="o",label="median Kendall"); plt.plot(x,[metrics.get(f,{}).get("median_pairwise_order_accuracy",np.nan) for f in labels],marker="o",label="pairwise order accuracy"); plt.xticks(x,labels); plt.ylim(-1.0,1.05); plt.ylabel("order-stability metric"); plt.title("Calibration order stability across fidelities"); plt.legend(); fig.tight_layout(); fig.savefig(PLOTS/"calibration_order_stability.png",dpi=160); plt.close(fig)
    raw=load(HERE/"diagnostics/fidelity_raw_scores.json",{}) or {}; agg=raw.get("aggregated_fitness",{})
    panel_ids=CONFIG["calibration"]["panel_ids"]
    selected=decision.get("selected_search_fidelity") or "F0"
    fig=plt.figure(figsize=(7,7)); plotted=False
    for family in FAMILIES:
        if selected not in agg or "F4" not in agg or family not in agg[selected] or family not in agg["F4"]: continue
        proxy=np.asarray([agg[selected][family][panel] for panel in panel_ids],dtype=float); target=np.asarray([agg["F4"][family][panel] for panel in panel_ids],dtype=float)
        proxy_rank=np.argsort(np.argsort(-proxy))+1; target_rank=np.argsort(np.argsort(-target))+1
        plt.scatter(target_rank,proxy_rank,label=family); plotted=True
    if plotted: plt.plot([1,len(panel_ids)],[1,len(panel_ids)],linestyle="--"); plt.legend()
    plt.xlabel("F4 target topology rank"); plt.ylabel(f"{selected} proxy topology rank"); plt.title("Calibration topology ranking preservation"); fig.tight_layout(); fig.savefig(PLOTS/"calibration_topology_ranking.png",dpi=160); plt.close(fig)
    for fidelity,name,title in ((selected,"proxy_vs_target_scatter.png","Selected proxy vs F4 target"),("F_LEGACY","legacy_proxy_vs_target.png","Legacy proxy vs F4 target")):
        x=[]; y=[]
        for family in FAMILIES:
            if fidelity not in agg or "F4" not in agg: continue
            x.extend([agg[fidelity][family][p] for p in panel_ids]); y.extend([agg["F4"][family][p] for p in panel_ids])
        fig=plt.figure(figsize=(6,6)); plt.scatter(x,y,s=18); plt.xlabel(f"{fidelity} fitness"); plt.ylabel("F4 fitness"); plt.title(title); fig.tight_layout(); fig.savefig(PLOTS/name,dpi=160); plt.close(fig)


def _search_plots(search_rows: list[dict], random_rows: list[dict], search_runs: list[dict], random_runs: list[dict], decision: dict | None) -> None:
    PLOTS.mkdir(exist_ok=True)
    names=["best_so_far_search_vs_random.png","final_validation_search_vs_random.png","competent_hit_rate_by_family.png","search_random_paired_deltas.png","champion_edge_counts.png","champion_recurrent_fraction.png","champion_topology_similarity.png","message_dependence_champions.png","candidate_evaluations_to_threshold.png","capability_vs_search_compute.png"]
    if not search_rows or not random_rows or decision is None:
        for name in names: _plot_placeholder(PLOTS/name,name.replace("_"," ").replace(".png","").title(),"V837aj-B not run because no valid Stage-A proxy was authorized.")
        return
    # best-so-far
    fig=plt.figure(figsize=(8,5));
    for runs,label in ((search_runs,"directed"),(random_runs,"random")):
        curves=np.asarray([[r["best_selection_success"] for r in run["best_so_far"]] for run in runs]); plt.plot(np.arange(1,curves.shape[1]+1),np.median(curves,axis=0),label=label)
    plt.xlabel("candidate evaluations"); plt.ylabel("best selection success"); plt.legend(); fig.tight_layout(); fig.savefig(PLOTS/names[0],dpi=160); plt.close(fig)
    # final paired scatter
    paired=decision["paired"]; fig=plt.figure(figsize=(6,6)); plt.scatter([p["random_validation"] for p in paired],[p["search_validation"] for p in paired]); plt.plot([0,1],[0,1],linestyle="--"); plt.xlabel("random final validation"); plt.ylabel("directed final validation"); fig.tight_layout(); fig.savefig(PLOTS/names[1],dpi=160); plt.close(fig)
    # main competent fraction plot
    x=np.arange(len(FAMILIES)); d=[decision["directed"]["families"][f]["competent_hits"]/decision["directed"]["families"][f]["runs"] for f in FAMILIES]; r=[decision["random"]["families"][f]["competent_hits"]/decision["random"]["families"][f]["runs"] for f in FAMILIES]
    fig=plt.figure(figsize=(10,5)); width=.38; plt.bar(x-width/2,d,width,label="directed"); plt.bar(x+width/2,r,width,label="random"); plt.axhline(.6,linestyle="--",label="3/5 threshold"); plt.xticks(x,FAMILIES,rotation=25,ha="right"); plt.ylim(0,1); plt.ylabel("competent champion fraction"); plt.legend(); fig.tight_layout(); fig.savefig(PLOTS/names[2],dpi=160); plt.close(fig)
    # deltas
    fig=plt.figure(figsize=(8,4)); plt.axhline(0,linewidth=1); plt.plot([p["validation_delta"] for p in paired],marker="o"); plt.ylabel("directed - random validation"); fig.tight_layout(); fig.savefig(PLOTS/names[3],dpi=160); plt.close(fig)
    # edge counts
    fig=plt.figure(figsize=(7,5)); plt.boxplot([[r["topology"]["edge_count"] for r in search_rows],[r["topology"]["edge_count"] for r in random_rows]],labels=["directed","random"]); plt.ylabel("champion edge count"); fig.tight_layout(); fig.savefig(PLOTS/names[4],dpi=160); plt.close(fig)
    # recurrent fraction
    fig=plt.figure(figsize=(7,5)); plt.boxplot([[r["topology"]["recurrent_edge_count"]/max(1,r["topology"]["edge_count"]) for r in search_rows],[r["topology"]["recurrent_edge_count"]/max(1,r["topology"]["edge_count"]) for r in random_rows]],labels=["directed","random"]); plt.ylabel("recurrent fraction"); fig.tight_layout(); fig.savefig(PLOTS/names[5],dpi=160); plt.close(fig)
    diversity=_topology_diversity(search_rows,random_rows); fig=plt.figure(figsize=(7,5)); plt.bar(["directed","random"],[diversity["directed"]["median_edge_jaccard"],diversity["random"]["median_edge_jaccard"]]); plt.ylabel("median champion edge Jaccard"); fig.tight_layout(); fig.savefig(PLOTS/names[6],dpi=160); plt.close(fig)
    msg=_message_dependence(search_rows,random_rows); fig=plt.figure(figsize=(7,5)); plt.bar(["directed","random","AF1D anchor"],[msg["directed"]["median_success_drop"] or 0,msg["random"]["median_success_drop"] or 0,float(msg["fixed_af1d_anchor_4x"].get("success_drop_median",0))]); plt.ylabel("median message-ablation success drop"); fig.tight_layout(); fig.savefig(PLOTS/names[7],dpi=160); plt.close(fig)
    eff=decision["efficiency"]; fig=plt.figure(figsize=(7,5)); data=[x for x in eff["search_evaluations_to_first_proxy_threshold"] if x is not None]; data2=[x for x in eff["random_evaluations_to_first_proxy_threshold"] if x is not None]; plt.boxplot([data or [65],data2 or [65]],labels=["directed","random"]); plt.ylabel("evaluations to selection success >=0.85"); fig.tight_layout(); fig.savefig(PLOTS/names[8],dpi=160); plt.close(fig)
    fig=plt.figure(figsize=(7,5)); plt.scatter([r["compute"]["total_modeled_macs_per_timestep"] for r in search_rows],[r["final_validation_success"] for r in search_rows],label="directed"); plt.scatter([r["compute"]["total_modeled_macs_per_timestep"] for r in random_rows],[r["final_validation_success"] for r in random_rows],label="random"); plt.xlabel("modeled MACs/timestep"); plt.ylabel("final validation success"); plt.legend(); fig.tight_layout(); fig.savefig(PLOTS/names[9],dpi=160); plt.close(fig)


def _write_report(results: dict) -> None:
    fidelity=results["fidelity"]; decision=results.get("structural_discovery")
    lines=["# V837 Structural Search Recovery Report","","## 1. Why structural search reopened","","V837ai confirmed AF1D representation adequacy at 4x and explicitly authorized structural-search recovery.","","## 2. Why old V837 search was confounded","","Historical search failure coincided with an inadequate substrate. V837aj fixes the substrate first.","","## 3. Frozen AF1D mechanism","","Exact AF1D: ten 4D cells, rank-4 candidate coupling, one global joint input+state scalar carry controller, and ten independent 6->6 candidate projections.","","## 4. Searchable structural axis","","Only message-edge existence and SAME_STEP/RECURRENT timing are searchable; cell count and all learned mechanisms remain frozen.","","## 5. Validation-leakage correction","","Search fitness never reads seeds 20000-20127. Final validation becomes accessible only after champion topology freeze (plus the isolated AJ0 anchor reproduction control).","","## 6. Initialization-confound correction","","All common non-edge parameters are bit-identical within paired topology evaluations; common semantic edges receive path-independent initialization.","","## 7. Calibration panel","",f"The fixed task-independent panel contains {results['calibration_panel_size']} topologies.","","## 8. Fidelity ladder","","F_LEGACY, F0, F1, F2, F3 are compared against F4 target ranking with two initialization replicates per topology/family/fidelity.","","## 9. Proxy rank results","",json.dumps(fidelity.get("metrics",{}),indent=2),"","## 10. Selected search fidelity","",str(fidelity.get("selected_search_fidelity")),"","## 11. Constructive search design","","Anchor-free (mu+lambda) search starts from the 19-edge minimal topology and evaluates exactly 64 unique candidates/run.","","## 12. Equal-budget random design","","Random sampling matches directed candidate total-edge and recurrent-edge counts slot by slot, with identical initialization slots.","","## 13. Directed search results","",json.dumps(decision.get("directed",{}) if decision else {"run":False},indent=2),"","## 14. Random results","",json.dumps(decision.get("random",{}) if decision else {"run":False},indent=2),"","## 15. Finalized validation results","",json.dumps(decision.get("paired",[]) if decision else [],indent=2),"","## 16. Competent hit rates","",json.dumps({"directed":decision.get("directed_competent_hit_rate") if decision else None,"random":decision.get("random_competent_hit_rate") if decision else None},indent=2),"","## 17. Search-vs-random statistics","",json.dumps(decision.get("permutation",{}) if decision else {},indent=2),"","## 18. Structural diversity","",json.dumps(results.get("topology_diversity",{}),indent=2),"","## 19. Message dependence","",json.dumps(results.get("message_dependence",{}),indent=2),"","## 20. Compute/resource comparison","",json.dumps(results.get("resource_accounting",{}),indent=2),"","## 21. Capability / structure / compute scoreboard","",json.dumps(results.get("capability_structure_compute_scoreboard",{}),indent=2),"","## 22. V837aj diagnosis","",results["diagnosis"],"","## 23. Primitive-mining authorization","",str(results["primitive_mining_allowed_next"]),"","## 24. Strongest scientific claim","",results["strongest_scientific_claim"],"","## 25. Next single program","",results["next_program"],""]
    (ROOT/"docs/V837_STRUCTURAL_SEARCH_RECOVERY_REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def _quest_tracker() -> None:
    text="# V837 Structural Search Quest Tracker\n\nProtected-weirdness reserve; documented only, not implemented in V837aj.\n\n- quality-diversity search\n- functional novelty search\n- dynamical-fingerprint novelty\n- online structural plasticity\n- cell-count evolution\n- module duplication\n- learned structural mutation policy\n\nFuture primitive identity should combine structural signature, dynamical signature, input/output interface, causal lesion profile, and task occurrence profile. No motifs or primitives are mined here.\n"
    (ROOT/"docs/V837_STRUCTURAL_SEARCH_QUEST_TRACKER.md").write_text(text,encoding="utf-8")


def analyze(phase: str) -> int:
    anchor=load(HERE/"diagnostics/anchor_reproduction.json",{}) or {}; fidelity=load(HERE/"diagnostics/fidelity_decision.json",{}) or {}
    if anchor.get("anchor_reproduced") is not True: raise SystemExit("V837aj analysis blocked: anchor invalid")
    _fidelity_plots(); _quest_tracker()
    if fidelity.get("proxy_valid") is not True:
        _placeholder_stage_b("SEARCH_FIDELITY_PROXY_INVALID")
        diagnosis="SEARCH_FIDELITY_PROXY_INVALID"; next_program="V837ak_SEARCH_FIDELITY_REDESIGN"; mining=False
        results={"version":"V837aj","af1d_anchor":anchor,"fidelity":fidelity,"calibration_panel_size":12,"stage_b_run":False,"structural_discovery":None,"diagnosis":diagnosis,"automated_structural_discovery":False,"evolutionary_search_superiority":False,"primitive_mining_allowed_next":False,"fresh_audit_consumed":False,"primitives_promoted":0,"large_persistent_storage_tested":False,"v838_started":False,"next_program":next_program,"strongest_scientific_claim":"No cheaper ranking-preserving search proxy among F0-F3 passed the frozen fidelity gate against F4, so structural discovery was correctly not attempted."}
        resource=_resource_totals(); results["resource_accounting"]=resource; results["topology_diversity"]={"run":False}; results["message_dependence"]={"run":False}; results["capability_structure_compute_scoreboard"]={"run":False,"reason":diagnosis}
        _search_plots([],[],[],[],None); write_json(HERE/"diagnostics/search_random_budget_match.json",{"run":False,"reason":diagnosis}); write_json(HERE/"diagnostics/topology_diversity.json",results["topology_diversity"]); write_json(HERE/"diagnostics/search_vs_random.json",{"run":False,"reason":diagnosis}); write_json(HERE/"diagnostics/message_dependence.json",results["message_dependence"]); write_json(HERE/"diagnostics/compute_efficiency.json",resource); write_json(HERE/"diagnostics/structural_efficiency_scoreboard.json",results["capability_structure_compute_scoreboard"])
        state={"version":"V837aj","af1d_anchor_valid":True,"fidelity_calibration_complete":True,"selected_search_fidelity":None,"search_stage_allowed":False,"constructive_search_run":False,"primary_runs_per_family":5,"robustness_extension_run":False,"directed_family_passes":None,"random_family_passes":None,"directed_competent_hit_rate":None,"random_competent_hit_rate":None,"paired_validation_delta":None,"paired_permutation_p":None,"automated_structural_discovery":False,"evolutionary_search_superiority":False,"diagnosis":diagnosis,"primitive_mining_allowed_next":False,"fresh_audit_consumed":False,"primitives_promoted":0,"v838_started":False,"next_program":next_program}; write_json(HERE/"diagnostics/decision_state.json",state)
    else:
        search_runs=_proxy_runs("search_proxy_runs.json"); random_runs=_proxy_runs("random_proxy_runs.json"); search_rows=_rows("search_finalized.json"); random_rows=_rows("random_finalized.json")
        target_runs=50 if phase=="final" and len(search_rows)>=50 and len(random_rows)>=50 else 25
        search_rows=[r for r in search_rows if int(r["run_index"]) < (10 if target_runs==50 else 5)]; random_rows=[r for r in random_rows if int(r["run_index"]) < (10 if target_runs==50 else 5)]; search_runs=[r for r in search_runs if int(r["run_index"]) < (10 if target_runs==50 else 5)]; random_runs=[r for r in random_runs if int(r["run_index"]) < (10 if target_runs==50 else 5)]
        if len(search_rows)!=target_runs or len(random_rows)!=target_runs: raise SystemExit(f"V837aj {phase} analysis requires {target_runs} finalized rows per engine")
        decision=_decision(search_rows,random_rows,search_runs,random_runs)
        if phase=="primary": write_json(HERE/"diagnostics/primary_decision.json",decision)
        elif decision["robustness_extension_required"] and target_runs==25: raise SystemExit("final analysis blocked: robustness extension required but absent")
        budget=_budget_match(search_runs,random_runs); diversity=_topology_diversity(search_rows,random_rows); classifier={"directed":_family_classifier(search_rows),"random":_family_classifier(random_rows)}; message=_message_dependence(search_rows,random_rows); resource=_resource_totals(); scoreboard=_structural_efficiency_scoreboard(search_rows,random_rows)
        diagnosis=decision["diagnosis"]; mining=decision["primitive_mining_allowed_next"]; next_program=decision["next_program"]
        strongest=("Automated structural discovery is established on the frozen AF1D substrate." if decision["automated_structural_discovery"] else "Automated structural discovery was not established under the frozen V837aj search design.") + f" Diagnosis: {diagnosis}."
        results={"version":"V837aj","af1d_anchor":anchor,"fidelity":fidelity,"calibration_panel_size":12,"stage_b_run":True,"runs_per_family":target_runs//5,"structural_discovery":decision,"diagnosis":diagnosis,"automated_structural_discovery":decision["automated_structural_discovery"],"evolutionary_search_superiority":decision["evolutionary_search_superiority"],"primitive_mining_allowed_next":mining,"fresh_audit_consumed":False,"primitives_promoted":0,"large_persistent_storage_tested":False,"v838_started":False,"next_program":next_program,"search_random_budget_match":budget,"topology_diversity":diversity,"topology_family_classifier":classifier,"message_dependence":message,"capability_structure_compute_scoreboard":scoreboard,"resource_accounting":resource,"strongest_scientific_claim":strongest}
        _search_plots(search_rows,random_rows,search_runs,random_runs,decision); write_json(HERE/"diagnostics/search_random_budget_match.json",budget); write_json(HERE/"diagnostics/topology_diversity.json",{**diversity,"family_classifier":classifier}); write_json(HERE/"diagnostics/search_vs_random.json",decision); write_json(HERE/"diagnostics/message_dependence.json",message); write_json(HERE/"diagnostics/compute_efficiency.json",resource); write_json(HERE/"diagnostics/structural_efficiency_scoreboard.json",scoreboard)
        state={"version":"V837aj","af1d_anchor_valid":True,"fidelity_calibration_complete":True,"selected_search_fidelity":fidelity["selected_search_fidelity"],"search_stage_allowed":True,"constructive_search_run":True,"primary_runs_per_family":5,"robustness_extension_run":target_runs==50,"directed_family_passes":decision["directed"]["families_passing"],"random_family_passes":decision["random"]["families_passing"],"directed_competent_hit_rate":decision["directed_competent_hit_rate"],"random_competent_hit_rate":decision["random_competent_hit_rate"],"paired_validation_delta":decision["median_paired_final_validation_delta"],"paired_permutation_p":decision["permutation"]["one_sided_search_gt_random_p"],"automated_structural_discovery":decision["automated_structural_discovery"],"evolutionary_search_superiority":decision["evolutionary_search_superiority"],"diagnosis":diagnosis,"primitive_mining_allowed_next":mining,"fresh_audit_consumed":False,"primitives_promoted":0,"v838_started":False,"next_program":next_program}; write_json(HERE/"diagnostics/decision_state.json",state)
    write_json(HERE/"results.json",results); write_json(HERE/"v837aj_resource_accounting.json",results["resource_accounting"]); write_json(ROOT/"experiments/v837_primitive_invention/v837aj_resource_accounting.json",results["resource_accounting"]); write_json(ROOT/"experiments/v837_primitive_invention/structural_search_recovery_program_resource_accounting.json",results["resource_accounting"]); write_json(ROOT/"experiments/v837_primitive_invention/structural_search_recovery_program_status.json",{"version":"V837aj","diagnosis":results["diagnosis"],"automated_structural_discovery":results["automated_structural_discovery"],"evolutionary_search_superiority":results["evolutionary_search_superiority"],"primitive_mining_allowed_next":results["primitive_mining_allowed_next"],"fresh_audit_episodes_consumed":0,"primitives_promoted":0,"large_persistent_storage_tested":False,"v838_started":False,"next_program":results["next_program"]})
    _write_report(results)
    pass_result=results["diagnosis"] in {"STRUCTURAL_SEARCH_RECOVERED","RANDOM_STRUCTURAL_DISCOVERY_SUFFICIENT"}
    marker=HERE/("PASS.md" if pass_result else "FAILURE.md"); other=HERE/("FAILURE.md" if pass_result else "PASS.md")
    if other.exists(): other.unlink()
    marker.write_text(f"# V837aj {'PASS' if pass_result else 'CLOSED'}\n\nDiagnosis: `{results['diagnosis']}`.\n\nNext: `{results['next_program']}`.\n",encoding="utf-8")
    print(json.dumps({"diagnosis":results["diagnosis"],"selected_search_fidelity":fidelity.get("selected_search_fidelity"),"stage_b_run":results["stage_b_run"],"automated_structural_discovery":results["automated_structural_discovery"],"evolutionary_search_superiority":results["evolutionary_search_superiority"],"primitive_mining_allowed_next":results["primitive_mining_allowed_next"],"next_program":results["next_program"]},indent=2)); return 0


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=("primary","final"),default="final"); args=parser.parse_args(); return analyze(args.phase)

if __name__=="__main__": raise SystemExit(main())
