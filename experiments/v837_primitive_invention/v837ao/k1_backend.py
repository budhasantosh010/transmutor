from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from experiments.v837_primitive_invention.v837an.an_a_causal_macrovariables import _carrier_values, _eligibility
from experiments.v837_primitive_invention.v837an.authorization import PRIMARY_PROBES
from experiments.v837_primitive_invention.v837an.counterfactual_subspace import fit_difference_subspace
from experiments.v837_primitive_invention.v837an.semantic_compiler import fit_semantic_compiler
from experiments.v837_primitive_invention.v837an.trace_cache import pair_traces

from .canonical_reader import fit_full40_reader, fit_k1_reader, read_full40, read_k1, reader_metrics
from .canonical_writer import raw_writer_from_compiler, set_state
from .gauge_fix import gauge_fix_writer
from .phase_backends import BACKEND_ORDER, PHASES, representative_phase_timesteps
from .setpoint_grid import family_grid
from .utils import deterministic_seed


def _semantic(ep, family: str, t: int) -> float:
    return float(ep.macrostate_traces[PRIMARY_PROBES[family]][int(t)])


def _primary_arrays(data: dict, mask: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    idx = np.flatnonzero(mask)
    times = np.asarray([p.primary_phase for p in data["pairs"]], dtype=np.int64)[idx]
    base = _carrier_values(data["base_trace"], "STATE40", np.asarray([p.primary_phase for p in data["pairs"]], dtype=np.int64))[idx]
    cf = _carrier_values(data["cf_trace"], "STATE40", np.asarray([p.primary_phase for p in data["pairs"]], dtype=np.int64))[idx]
    dz = np.asarray([_semantic(data["pairs"][i].counterfactual_episode, data["pairs"][i].family, int(data["pairs"][i].primary_phase)) - _semantic(data["pairs"][i].base_episode, data["pairs"][i].family, int(data["pairs"][i].primary_phase)) for i in idx], dtype=np.float64)
    return base, cf, dz


def _phase_arrays(data: dict, mask: np.ndarray, family: str, phase: str) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    idx = np.flatnonzero(mask)
    bs=[];cs=[];bz=[];cz=[]
    btrace=data["base_trace"].states.detach().cpu().numpy().reshape(len(data["pairs"]), data["base_trace"].states.shape[1], 40)
    ctrace=data["cf_trace"].states.detach().cpu().numpy().reshape(len(data["pairs"]), data["cf_trace"].states.shape[1], 40)
    for i in idx:
        p=data["pairs"][i]; bt=representative_phase_timesteps(p.base_episode).get(phase); ct=representative_phase_timesteps(p.counterfactual_episode).get(phase)
        if bt is None or ct is None: continue
        z0=_semantic(p.base_episode,family,bt);z1=_semantic(p.counterfactual_episode,family,ct)
        if not np.isfinite(z0) or not np.isfinite(z1): continue
        bs.append(btrace[i,bt]);cs.append(ctrace[i,ct]);bz.append(z0);cz.append(z1)
    if not bs:
        return np.empty((0,40)),np.empty((0,40)),np.empty(0),np.empty(0)
    return np.asarray(bs,dtype=np.float64),np.asarray(cs,dtype=np.float64),np.asarray(bz,dtype=np.float64),np.asarray(cz,dtype=np.float64)


def _fit_component(q: np.ndarray, base: np.ndarray, cf: np.ndarray, bz: np.ndarray, cz: np.ndarray, family: str, component_id: str, organism_id: str) -> dict:
    states=np.concatenate([base,cf],axis=0);semantic=np.concatenate([bz,cz],axis=0)
    reader=fit_k1_reader(states,semantic,q);full40=fit_full40_reader(states,semantic)
    delta=cf-base;dz=cz-bz
    compiler=fit_semantic_compiler(dz,delta,np.asarray(q,dtype=np.float64).reshape(40,1))
    raw=raw_writer_from_compiler(q,compiler);gauge=gauge_fix_writer(reader,raw)
    if not gauge["valid"]:
        return {"valid":False,"component_id":component_id,"q":np.asarray(q).reshape(-1).tolist(),"reader":reader,"full40_reader":full40,"semantic_compiler":{"coef":np.asarray(compiler["coef"]).tolist(),"ridge_lambda":1e-6,"intercept":0.0},"raw_writer":raw.tolist(),"gauge":gauge}
    writer=np.asarray(gauge["writer"],dtype=np.float64)
    pred=read_k1(states,reader);grid=family_grid(family);rm=reader_metrics(pred,semantic,grid["semantic_range"],family in {"conditional_routing","delayed_recall"})
    # Deterministic shuffled-semantic sham: same physical q, destroyed semantic correspondence.
    rng=np.random.default_rng(deterministic_seed("v837ao-shuffle",organism_id,component_id));perm=rng.permutation(len(semantic));shuf_reader=fit_k1_reader(states,semantic[perm],q)
    dz_perm=dz[rng.permutation(len(dz))] if len(dz) else dz;shuf_comp=fit_semantic_compiler(dz_perm,delta,np.asarray(q,dtype=np.float64).reshape(40,1));shuf_raw=raw_writer_from_compiler(q,shuf_comp);shuf_gauge=gauge_fix_writer(shuf_reader,shuf_raw)
    # Algebra checks on a bounded deterministic subset.
    sample=states[:min(128,len(states))];z=read_k1(sample,reader);target=float(grid["targets"][0]);same=set_state(sample,z,reader,writer);written=set_state(sample,target,reader,writer);readback=read_k1(written,reader)
    twice=set_state(written,target,reader,writer);first_norm=np.linalg.norm(written-sample,axis=1);second_norm=np.linalg.norm(twice-written,axis=1);ratio=second_norm/np.maximum(first_norm,1e-12)
    overwrite=set_state(set_state(sample,float(grid["targets"][0]),reader,writer),float(grid["targets"][-1]),reader,writer);overwrite_err=np.abs(read_k1(overwrite,reader)-float(grid["targets"][-1]))/max(grid["semantic_range"],1e-12)
    algebra={"zero_change_max_abs":float(np.max(np.abs(same-sample))) if len(sample) else 0.0,"read_after_write_median_nrmse":float(np.median(np.abs(readback-target)/max(grid["semantic_range"],1e-12))) if len(sample) else 0.0,"read_after_write_p99_nrmse":float(np.quantile(np.abs(readback-target)/max(grid["semantic_range"],1e-12),.99)) if len(sample) else 0.0,"idempotence_relative_update_median":float(np.median(ratio)) if len(sample) else 0.0,"overwrite_median_nrmse":float(np.median(overwrite_err)) if len(sample) else 0.0}
    algebra["pass"]=bool(algebra["zero_change_max_abs"]<=1e-10 and algebra["read_after_write_median_nrmse"]<=1e-6 and algebra["read_after_write_p99_nrmse"]<=1e-5 and algebra["idempotence_relative_update_median"]<=1e-8 and algebra["overwrite_median_nrmse"]<=1e-6)
    return {"valid":True,"component_id":component_id,"q":np.asarray(q).reshape(-1).tolist(),"reader":reader,"full40_reader":full40,"semantic_compiler":{"coef":np.asarray(compiler["coef"]).tolist(),"ridge_lambda":1e-6,"intercept":0.0},"raw_writer":raw.tolist(),"gauge":gauge,"writer":writer.tolist(),"fit_reader_metrics":rm,"algebra":algebra,"fit_examples":int(len(states)),"paired_examples":int(len(base)),"shuffled_control":{"reader":shuf_reader,"writer":shuf_gauge.get("writer"),"valid":bool(shuf_gauge.get("valid")),"gamma":shuf_gauge.get("gamma")}}


def fit_backend(organism_id: str, family: str, variant: str, fit_seeds: list[int], *, calibration_n: int | None = None) -> dict:
    if variant not in BACKEND_ORDER: raise KeyError(variant)
    supplied=list(fit_seeds[:calibration_n] if calibration_n is not None else fit_seeds)
    data=pair_traces(organism_id,family,supplied);mask,power=_eligibility(data,family)
    if not np.any(mask):
        return {"version":"V837ao","organism_id":organism_id,"family":family,"engine":data["row"]["engine"],"variant":variant,"valid":False,"failure_code":"K1_READER_NOT_GLOBAL","fit_seeds":[min(supplied),max(supplied)] if supplied else [],"calibration_n":calibration_n,"eligible_pairs":0}
    base,cf,dz=_primary_arrays(data,mask);sub=fit_difference_subspace(base,cf,1);q=np.asarray(sub["q"],dtype=np.float64)[:,0]
    phase_data={p:_phase_arrays(data,mask,family,p) for p in PHASES[family]}
    components={}
    if variant=="B0_GLOBAL_K1":
        bs=[];cs=[];bz=[];cz=[]
        for phase in PHASES[family]:
            b,c,z0,z1=phase_data[phase]
            if len(b):bs.append(b);cs.append(c);bz.append(z0);cz.append(z1)
        component=_fit_component(q,np.concatenate(bs),np.concatenate(cs),np.concatenate(bz),np.concatenate(cz),family,"GLOBAL",organism_id)
        components["GLOBAL"]=component
    elif variant=="B1_PHASE_GAUGE_K1":
        # Same q and same primary semantic compiler direction; phase-specific reader/gauge only.
        pb,pc,pdz=_primary_arrays(data,mask);primary_compiler=fit_semantic_compiler(pdz,pc-pb,q.reshape(40,1));raw=raw_writer_from_compiler(q,primary_compiler)
        for phase in PHASES[family]:
            b,c,z0,z1=phase_data[phase]
            if not len(b):
                components[phase]={"valid":False,"component_id":phase,"failure_code":"K1_READER_NOT_GLOBAL"};continue
            states=np.concatenate([b,c]);semantic=np.concatenate([z0,z1]);reader=fit_k1_reader(states,semantic,q);full40=fit_full40_reader(states,semantic);gauge=gauge_fix_writer(reader,raw);grid=family_grid(family);rm=reader_metrics(read_k1(states,reader),semantic,grid["semantic_range"],family in {"conditional_routing","delayed_recall"})
            if not gauge["valid"]:components[phase]={"valid":False,"component_id":phase,"q":q.tolist(),"reader":reader,"full40_reader":full40,"raw_writer":raw.tolist(),"gauge":gauge};continue
            writer=np.asarray(gauge["writer"]);sample=states[:min(128,len(states))];target=float(grid["targets"][0]);same=set_state(sample,read_k1(sample,reader),reader,writer);written=set_state(sample,target,reader,writer);readback=read_k1(written,reader);twice=set_state(written,target,reader,writer);first=np.linalg.norm(written-sample,axis=1);second=np.linalg.norm(twice-written,axis=1)
            algebra={"zero_change_max_abs":float(np.max(np.abs(same-sample))),"read_after_write_median_nrmse":float(np.median(np.abs(readback-target)/max(grid["semantic_range"],1e-12))),"read_after_write_p99_nrmse":float(np.quantile(np.abs(readback-target)/max(grid["semantic_range"],1e-12),.99)),"idempotence_relative_update_median":float(np.median(second/np.maximum(first,1e-12))),"overwrite_median_nrmse":0.0};algebra["pass"]=bool(algebra["zero_change_max_abs"]<=1e-10 and algebra["read_after_write_median_nrmse"]<=1e-6 and algebra["read_after_write_p99_nrmse"]<=1e-5 and algebra["idempotence_relative_update_median"]<=1e-8)
            rng=np.random.default_rng(deterministic_seed("v837ao-shuffle",organism_id,phase));perm=rng.permutation(len(semantic));shuf_reader=fit_k1_reader(states,semantic[perm],q);phase_dz=z1-z0;phase_delta=c-b;shuf_comp=fit_semantic_compiler(phase_dz[rng.permutation(len(phase_dz))],phase_delta,q.reshape(40,1));shuf_raw=raw_writer_from_compiler(q,shuf_comp);shuf_gauge=gauge_fix_writer(shuf_reader,shuf_raw)
            components[phase]={"valid":True,"component_id":phase,"q":q.tolist(),"reader":reader,"full40_reader":full40,"semantic_compiler":{"coef":np.asarray(primary_compiler["coef"]).tolist(),"ridge_lambda":1e-6,"intercept":0.0},"raw_writer":raw.tolist(),"gauge":gauge,"writer":writer.tolist(),"fit_reader_metrics":rm,"algebra":algebra,"fit_examples":len(states),"paired_examples":len(b),"shuffled_control":{"reader":shuf_reader,"writer":shuf_gauge.get("writer"),"valid":bool(shuf_gauge.get("valid")),"gamma":shuf_gauge.get("gamma")}}
    else:
        for phase in PHASES[family]:
            b,c,z0,z1=phase_data[phase]
            if not len(b):components[phase]={"valid":False,"component_id":phase,"failure_code":"K1_READER_NOT_GLOBAL"};continue
            delta=c-b;semantic_delta=z1-z0;active=np.abs(semantic_delta)>1e-8
            if not np.any(active) or float(np.linalg.norm(delta[active]))<1e-10:
                components[phase]={"valid":False,"component_id":phase,"failure_code":"PHASE_K1_FAIL","reason":"no nonzero paired semantic/state difference at frozen phase"};continue
            qphase=np.asarray(fit_difference_subspace(b[active],c[active],1)["q"])[:,0]
            components[phase]=_fit_component(qphase,b,c,z0,z1,family,phase,organism_id)
    valid=all(components.get(p if variant!="B0_GLOBAL_K1" else "GLOBAL",{}).get("valid",False) for p in (PHASES[family] if variant!="B0_GLOBAL_K1" else ("GLOBAL",)))
    return {"version":"V837ao","organism_id":organism_id,"family":family,"engine":data["row"]["engine"],"variant":variant,"valid":bool(valid),"fit_partition":"AO_BACKEND_FIT","fit_seeds":[min(supplied),max(supplied)] if supplied else [],"calibration_n":calibration_n,"supplied_pairs":len(supplied),"eligible_pairs":int(np.sum(mask)),"components":components,"global_q":q.tolist(),"historical_v837an_backend_loaded":False,"cross_organism_state_alignment":False,"cross_organism_q_alignment":False,"gradient_steps":0}


def component_for_phase(backend: dict, phase: str) -> dict:
    return backend["components"]["GLOBAL"] if backend["variant"]=="B0_GLOBAL_K1" else backend["components"][phase]


def evaluate_reader_on_partition(backend: dict, eval_seeds: list[int]) -> dict:
    oid=backend["organism_id"];family=backend["family"];data=pair_traces(oid,family,eval_seeds);mask,_=_eligibility(data,family);grid=family_grid(family);rows=[]
    for phase in PHASES[family]:
        b,c,z0,z1=_phase_arrays(data,mask,family,phase);component=component_for_phase(backend,phase)
        if not component.get("valid") or not len(b):rows.append({"phase":phase,"pass":False,"n":0});continue
        states=np.concatenate([b,c]);semantic=np.concatenate([z0,z1]);k1=reader_metrics(read_k1(states,component["reader"]),semantic,grid["semantic_range"],family in {"conditional_routing","delayed_recall"});full=reader_metrics(read_full40(states,component["full40_reader"]),semantic,grid["semantic_range"],family in {"conditional_routing","delayed_recall"});rows.append({"phase":phase,"k1":k1,"full40_diagnostic":full,"pass":bool(k1["pass"] and component.get("algebra",{}).get("pass",False)),"n":len(states)})
    return {"organism_id":oid,"family":family,"engine":backend["engine"],"variant":backend["variant"],"phases":rows,"reader_pass":all(r["pass"] for r in rows),"algebra_pass":all(component_for_phase(backend,p).get("algebra",{}).get("pass",False) for p in PHASES[family])}
