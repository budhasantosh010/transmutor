from __future__ import annotations

from collections import defaultdict
import json

import numpy as np

from experiments.v837_primitive_invention.tasks import task_by_name

from .an_a_causal_macrovariables import _carrier_values, _eligibility, _metrics, _semantic_values
from .authorization import PARTITIONS, PRIMARY_PROBES
from .causal_interventions import many_patched_predictions, patched_predictions
from .counterfactual_subspace import fit_difference_subspace, projected_delta
from .instrumented_af1d import load_population_rows
from .ood_diagnostics import compare as ood_compare, fit_reference
from .random_controls import random_subspaces
from .trace_cache import pair_traces
from .utils import HERE, deterministic_seed, range_list, read_json, write_json


def _phase_index(pair, phase: str) -> int:
    labels = list(pair.base_episode.phase_labels)
    if phase == "MID_DELAY":
        idx = [i for i, x in enumerate(labels) if x == "DELAY"]
        return idx[len(idx) // 2] if idx else int(pair.primary_phase)
    if phase == "PRE_QUERY":
        idx = [i for i, x in enumerate(labels) if x == "DELAY"]
        return idx[-1] if idx else max(0, int(pair.primary_phase) - 1)
    if phase == "POST_PAYLOAD_B":
        idx = [i for i, x in enumerate(labels) if x == "PAYLOAD_B"]
        return idx[-1] if idx else int(pair.primary_phase)
    idx = [i for i, x in enumerate(labels) if x == phase]
    return idx[0] if idx else int(pair.primary_phase)


def _fit_record(organism_id: str, config_id: str) -> dict:
    rows = read_json(HERE / "raw/an_a_fit.json")["records"]
    return next(r for r in rows if r["organism_id"] == organism_id and r["config_id"] == config_id)


def _evaluate_fixed_basis_phase(organism: dict, config_id: str, phase: str) -> dict:
    family = organism["family"]
    record = _fit_record(organism["organism_id"], config_id)
    carrier = record["carrier"]
    q = np.asarray(record["q"], dtype=np.float64)
    fit = pair_traces(organism["organism_id"], family, range_list(PARTITIONS["AN_FIT"]))
    select = pair_traces(organism["organism_id"], family, range_list(PARTITIONS["AN_SELECT"]))
    fm, _ = _eligibility(fit, family); sm, _ = _eligibility(select, family)
    fi = np.flatnonzero(fm); si = np.flatnonzero(sm)
    if len(fi) < 48 or len(si) < 32:
        return {"powered": False, "eligible": len(si), "phase": phase}
    ft = np.asarray([_phase_index(p, phase) for p in fit["pairs"]], dtype=np.int64)
    st = np.asarray([_phase_index(p, phase) for p in select["pairs"]], dtype=np.int64)
    fb = _carrier_values(fit["base_trace"], carrier, ft)[fi]
    fc = _carrier_values(fit["cf_trace"], carrier, ft)[fi]
    ref = fit_reference(np.concatenate([fb, fc], axis=0))
    sb = _carrier_values(select["base_trace"], carrier, st)[si]
    sc = _carrier_values(select["cf_trace"], carrier, st)[si]
    delta = sc - sb
    d = projected_delta(delta, q)
    fv = sb + d; rv = sc - d
    model = select["model"]; times = st[si]
    bo = select["base_obs"][si]; bl = select["base_lengths"][si]
    co = select["cf_obs"][si]; cl = select["cf_lengths"][si]
    pf = patched_predictions(model, bo, bl, carrier, fv, times)
    pr = patched_predictions(model, co, cl, carrier, rv, times)
    controls = random_subspaces(q.shape[0], q.shape[1], 32, deterministic_seed("v837an-phase-control", organism["organism_id"], config_id, phase))
    fsets = np.stack([sb + projected_delta(delta, rq) for rq in controls])
    rsets = np.stack([sc + projected_delta(-delta, rq) for rq in controls])
    rf = many_patched_predictions(model, bo, bl, carrier, fsets, times)
    rr = many_patched_predictions(model, co, cl, carrier, rsets, times)
    bp = select["base_prediction"].detach().cpu().numpy()[si]
    cp = select["cf_prediction"].detach().cpu().numpy()[si]
    bt = select["base_targets"].detach().cpu().numpy()[si]
    ct = select["cf_targets"].detach().cpu().numpy()[si]
    m = _metrics(task_by_name(family), bp, cp, pf, pr, bt, ct, rf, rr, ood_compare(fv, sc, ref), ood_compare(rv, sb, ref), deterministic_seed("v837an-phase-p", organism["organism_id"], config_id, phase))
    return {"powered": True, "eligible": len(si), "phase": phase, **{k: v for k, v in m.items() if k not in {"recovery_values", "random_pair_median"}}}


def _routing_selected_value_diagnostic(organism: dict) -> dict:
    family = "conditional_routing"
    fit = pair_traces(organism["organism_id"], family, range_list(PARTITIONS["AN_FIT"]))
    select = pair_traces(organism["organism_id"], family, range_list(PARTITIONS["AN_SELECT"]))
    fm, _ = _eligibility(fit, family); sm, _ = _eligibility(select, family)
    fi = np.flatnonzero(fm); si = np.flatnonzero(sm)
    if len(fi) < 48 or len(si) < 32:
        return {"powered": False, "organism_id": organism["organism_id"]}
    ft = np.asarray([_phase_index(p, "POST_PAYLOAD_B") for p in fit["pairs"]], dtype=np.int64)
    st = np.asarray([_phase_index(p, "POST_PAYLOAD_B") for p in select["pairs"]], dtype=np.int64)
    # Diagnostic carrier follows the primary winning carrier when available;
    # this diagnostic cannot replace the primary ROUTING_CONTROL_STATE winner.
    winner = read_json(HERE / "raw/an_a_selection.json")["family_winners"].get(family)
    if winner is None:
        return {"powered": True, "organism_id": organism["organism_id"], "run": False, "reason": "NO_PRIMARY_ROUTING_CAUSAL_CARRIER"}
    carrier = winner["carrier"]; k = int(winner["k"])
    fb = _carrier_values(fit["base_trace"], carrier, ft)[fi]; fc = _carrier_values(fit["cf_trace"], carrier, ft)[fi]
    q = fit_difference_subspace(fb, fc, k)["q"]
    sb = _carrier_values(select["base_trace"], carrier, st)[si]; sc = _carrier_values(select["cf_trace"], carrier, st)[si]
    delta = sc - sb; d = projected_delta(delta, q); fv = sb + d; rv = sc - d
    ref = fit_reference(np.concatenate([fb, fc], axis=0)); model = select["model"]; times = st[si]
    bo = select["base_obs"][si]; bl = select["base_lengths"][si]; co = select["cf_obs"][si]; cl = select["cf_lengths"][si]
    pf = patched_predictions(model, bo, bl, carrier, fv, times); pr = patched_predictions(model, co, cl, carrier, rv, times)
    controls = random_subspaces(q.shape[0], q.shape[1], 32, deterministic_seed("v837an-routing-selected-control", organism["organism_id"], carrier, k))
    fsets = np.stack([sb + projected_delta(delta, rq) for rq in controls]); rsets = np.stack([sc + projected_delta(-delta, rq) for rq in controls])
    rf = many_patched_predictions(model, bo, bl, carrier, fsets, times); rr = many_patched_predictions(model, co, cl, carrier, rsets, times)
    bp = select["base_prediction"].detach().cpu().numpy()[si]; cp = select["cf_prediction"].detach().cpu().numpy()[si]
    bt = select["base_targets"].detach().cpu().numpy()[si]; ct = select["cf_targets"].detach().cpu().numpy()[si]
    m = _metrics(task_by_name(family), bp, cp, pf, pr, bt, ct, rf, rr, ood_compare(fv, sc, ref), ood_compare(rv, sb, ref), deterministic_seed("v837an-routing-selected-p", organism["organism_id"], carrier, k))
    return {"powered": True, "run": True, "organism_id": organism["organism_id"], "carrier": carrier, "k": k, "probe": "ROUTING_SELECTED_VALUE", **{k0: v for k0, v in m.items() if k0 not in {"recovery_values", "random_pair_median"}}}


def _coupling_factor_diag(organism: dict) -> list[dict]:
    family = organism["family"]
    fit = pair_traces(organism["organism_id"], family, range_list(PARTITIONS["AN_FIT"]))
    select = pair_traces(organism["organism_id"], family, range_list(PARTITIONS["AN_SELECT"]))
    fm, _ = _eligibility(fit, family); sm, _ = _eligibility(select, family)
    fi = np.flatnonzero(fm); si = np.flatnonzero(sm)
    if len(fi) < 48 or len(si) < 32: return []
    ft = np.asarray([p.primary_phase for p in fit["pairs"]], dtype=np.int64); st = np.asarray([p.primary_phase for p in select["pairs"]], dtype=np.int64)
    rows = np.arange(len(fit["pairs"])); xfit = fit["base_trace"].coupling_factor4[rows, ft].detach().cpu().numpy()[fi]
    yfit = _semantic_values(fit["pairs"], ft, False)[fi]
    rows_s = np.arange(len(select["pairs"])); xsel = select["base_trace"].coupling_factor4[rows_s, st].detach().cpu().numpy()[si]
    ysel = _semantic_values(select["pairs"], st, False)[si]
    out=[]
    for k in (1,2,4):
        # Difference basis is descriptive only; COUPLING_FACTOR4 is not an exact causal bus.
        fb = fit["base_trace"].coupling_factor4[rows, ft].detach().cpu().numpy()[fi]
        fc = fit["cf_trace"].coupling_factor4[rows, ft].detach().cpu().numpy()[fi]
        q = fit_difference_subspace(fb, fc, k)["q"]
        proj_fit=xfit@q; proj_sel=xsel@q
        design=np.concatenate([proj_fit,np.ones((len(proj_fit),1))],axis=1); ridge=1e-6*np.eye(design.shape[1]); ridge[-1,-1]=0.0; coef=np.linalg.solve(design.T@design+ridge,design.T@yfit)
        pred=np.concatenate([proj_sel,np.ones((len(proj_sel),1))],axis=1)@coef; ss_res=float(np.sum((ysel-pred)**2)); ss_tot=float(np.sum((ysel-ysel.mean())**2)); r2=1.0-ss_res/max(ss_tot,1e-12)
        out.append({"organism_id":organism["organism_id"],"family":family,"k":k,"decoder_r2":r2,"diagnostic_only":True,"exact_intervention_implemented":False})
    return out


def _single_cell_controls(organism: dict, winner: dict) -> list[dict]:
    if winner is None or winner.get("carrier") != "STATE40": return []
    family=organism["family"]; fit=pair_traces(organism["organism_id"],family,range_list(PARTITIONS["AN_FIT"])); select=pair_traces(organism["organism_id"],family,range_list(PARTITIONS["AN_SELECT"])); fm,_=_eligibility(fit,family); sm,_=_eligibility(select,family); fi=np.flatnonzero(fm); si=np.flatnonzero(sm)
    if len(fi)<48 or len(si)<32:return []
    ft=np.asarray([p.primary_phase for p in fit["pairs"]],dtype=np.int64); st=np.asarray([p.primary_phase for p in select["pairs"]],dtype=np.int64); ffull=_carrier_values(fit["base_trace"],"STATE40",ft)[fi]; fcfull=_carrier_values(fit["cf_trace"],"STATE40",ft)[fi]; sbase=_carrier_values(select["base_trace"],"STATE40",st)[si]; scf=_carrier_values(select["cf_trace"],"STATE40",st)[si]; model=select["model"]; times=st[si]; bo=select["base_obs"][si];bl=select["base_lengths"][si];co=select["cf_obs"][si];cl=select["cf_lengths"][si];bp=select["base_prediction"].detach().cpu().numpy()[si];cp=select["cf_prediction"].detach().cpu().numpy()[si];bt=select["base_targets"].detach().cpu().numpy()[si];ct=select["cf_targets"].detach().cpu().numpy()[si];ref=fit_reference(np.concatenate([ffull,fcfull])); rows=[]
    for cell in range(10):
        lo=4*cell;hi=lo+4;fd=fcfull[:,lo:hi]-ffull[:,lo:hi];sd=scf[:,lo:hi]-sbase[:,lo:hi]
        for k in (1,2,4):
            q=fit_difference_subspace(ffull[:,lo:hi],fcfull[:,lo:hi],k)["q"]; local=projected_delta(sd,q);fv=sbase.copy();rv=scf.copy();fv[:,lo:hi]+=local;rv[:,lo:hi]-=local;pf=patched_predictions(model,bo,bl,"STATE40",fv,times);pr=patched_predictions(model,co,cl,"STATE40",rv,times);controls=random_subspaces(4,k,32,deterministic_seed("v837an-cell-controls",organism["organism_id"],cell,k));fsets=[];rsets=[]
            for rq in controls:
                d=projected_delta(sd,rq);x=sbase.copy();y=scf.copy();x[:,lo:hi]+=d;y[:,lo:hi]-=d;fsets.append(x);rsets.append(y)
            rf=many_patched_predictions(model,bo,bl,"STATE40",np.stack(fsets),times);rr=many_patched_predictions(model,co,cl,"STATE40",np.stack(rsets),times);m=_metrics(task_by_name(family),bp,cp,pf,pr,bt,ct,rf,rr,ood_compare(fv,scf,ref),ood_compare(rv,sbase,ref),deterministic_seed("v837an-cell-p",organism["organism_id"],cell,k));rows.append({"organism_id":organism["organism_id"],"family":family,"cell":cell,"k":k,**{kk:vv for kk,vv in m.items() if kk not in {"recovery_values","random_pair_median"}}})
    return rows


def run_an_a_diagnostics() -> dict:
    selection=read_json(HERE/"raw/an_a_selection.json");population=[r for r in load_population_rows() if bool(r["competent"])];phase_rows=[];routing_selected=[];coupling=[];single=[]
    recall_winner=selection["family_winners"].get("delayed_recall")
    if recall_winner is not None:
        for org in [r for r in population if r["family"]=="delayed_recall"]:
            for phase in ("WRITE","MID_DELAY","PRE_QUERY"):
                row=_evaluate_fixed_basis_phase(org,recall_winner["config_id"],phase);row.update({"organism_id":org["organism_id"],"family":"delayed_recall","config_id":recall_winner["config_id"]});phase_rows.append(row)
    for org in [r for r in population if r["family"]=="conditional_routing"]: routing_selected.append(_routing_selected_value_diagnostic(org))
    for org in population: coupling.extend(_coupling_factor_diag(org)); single.extend(_single_cell_controls(org,selection["family_winners"].get(org["family"])))
    phase_by=defaultdict(list)
    for r in phase_rows:
        if r.get("powered"):phase_by[r["phase"]].append(r.get("median_recovery",np.nan))
    phase_summary={k:float(np.median(v)) if v else None for k,v in phase_by.items()};stable=bool(phase_summary) and all(v is not None and v>=.60 for v in phase_summary.values());conditional=bool(phase_summary) and not stable
    payload={"version":"V837an","phase_rows":phase_rows,"phase_summary":phase_summary,"phase_stable_causal_representation":stable,"phase_conditional_representation":conditional,"routing_selected_value":routing_selected,"coupling_factor_diagnostic":coupling,"single_cell_state_controls":single,"primary_winners_unchanged":True}
    write_json(HERE/"diagnostics/phase_stability.json",payload);write_json(HERE/"diagnostics/coupling_factor4.json",{"version":"V837an","diagnostic_only":True,"rows":coupling});write_json(HERE/"diagnostics/single_cell_state_controls.json",{"version":"V837an","rows":single});return payload


if __name__=="__main__":print(json.dumps(run_an_a_diagnostics(),indent=2)[:12000])
