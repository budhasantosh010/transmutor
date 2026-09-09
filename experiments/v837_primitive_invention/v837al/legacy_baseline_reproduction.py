from __future__ import annotations

import time

from experiments.v837_primitive_invention.v837ak.boundary_substitution import _fit_alignment,_aligned_replay,_nrmses
from experiments.v837_primitive_invention.v837ak.boundary_traces import boundary_streams,run_full_probe
from experiments.v837_primitive_invention.v837ak.intervention_runtime import randomized_primitive
from experiments.v837_primitive_invention.v837ak.ported_primitive import PortedPrimitive
from experiments.v837_primitive_invention.v837ak.reconstruct_organisms import load_reconstructed_model
from .aligned_primitive import AlignedPortedPrimitive
from .authorization import assert_authorized
from .interface_ports import collect_interface_trace,load_model
from .utils import HERE,ROOT,read_json,write_json

AK=ROOT/"experiments/v837_primitive_invention/v837ak"
HIST_TEST=list(range(20064,20128));HIST_FIT=list(range(20000,20064))


def _metrics_new(replay,trace,nodes):
    # V837ak metric function only needs states/candidates/outputs and a trace with the same fields.
    class T:pass
    t=T();t.states=trace.states;t.candidates=trace.candidates;t.outputs=trace.outputs;t.lengths=trace.lengths
    return _nrmses(replay,t,nodes)


def reproduce_baselines():
    assert_authorized();hist=read_json(AK/"raw/boundary_substitution_results.json");rows=[];max_identity=0.0;max_legacy=0.0;t0=time.perf_counter();cpu0=time.process_time()
    for cls in hist["classes"]:
        for pair in cls.get("pairs",[]):
            rec=pair["recipient"];same=pair["same_class_donor"];diff=pair["different_class_donor"]
            rt=collect_interface_trace(rec["organism_id"],rec["nodes"],HIST_TEST,"validation");rmodel,_=load_model(rec["organism_id"]);selfp=PortedPrimitive(rmodel,rec["nodes"])
            def replay_for(occ,random=False):
                m,_=load_model(occ["organism_id"]);p=PortedPrimitive(m,occ["nodes"])
                if random:p=randomized_primitive(p,int(same["organism_id"][:12],16)%2_000_000_000)
                return AlignedPortedPrimitive(p,selfp).replay_teacher_forced(rt,rec["nodes"],{"ports":{}},"000000")
            new_same=_metrics_new(replay_for(same),rt,rec["nodes"]);new_diff=_metrics_new(replay_for(diff),rt,rec["nodes"]);new_rand=_metrics_new(replay_for(same,True),rt,rec["nodes"])
            identity_diffs=[]
            for key,new,old in (("same",new_same,pair["same"]),("different",new_diff,pair["different"]),("randomized",new_rand,pair["randomized"])):
                for metric in ("state_nrmse","candidate_nrmse","output_nrmse"):identity_diffs.append(abs(float(new[metric])-float(old[metric])))
            iderr=max(identity_diffs);max_identity=max(max_identity,iderr)
            legacy_err=None;legacy_new=None
            if pair.get("alignment_available"):
                rr=run_full_probe(rec["organism_id"],HIST_FIT,cache_key="v837al_legacy_fit64");dd=run_full_probe(same["organism_id"],HIST_FIT,cache_key="v837al_legacy_fit64")
                sm,_=load_model(same["organism_id"]);sprim=PortedPrimitive(sm,same["nodes"]);adapt=_fit_alignment(rr,dd,rec["nodes"],same["nodes"])
                hist_rt=run_full_probe(rec["organism_id"],HIST_TEST,cache_key="v837al_legacy_test64");streams=boundary_streams(rmodel,hist_rt,rec["nodes"]);rstates=hist_rt.states[:,:,list(rec["nodes"]),:]
                legacy_new=_nrmses(_aligned_replay(sprim,streams,rstates,adapt),hist_rt,rec["nodes"]);legacy_err=max(abs(float(legacy_new[m])-float(pair["aligned_same"][m])) for m in ("state_nrmse","candidate_nrmse","output_nrmse"));max_legacy=max(max_legacy,legacy_err)
            rows.append({"class_id":cls["class_id"],"recipient":rec,"same_class_donor":same,"identity":{"same":new_same,"different":new_diff,"randomized":new_rand,"max_abs_metric_delta":iderr},"legacy_orthogonal":{"available":pair.get("alignment_available",False),"new":legacy_new,"historical":pair.get("aligned_same"),"max_abs_metric_delta":legacy_err}})
    payload={"version":"V837al","stage":"AL1","identity_tolerance":1e-9,"legacy_tolerance":1e-9,"identity_reproduced":max_identity<=1e-9,"legacy_orthogonal_reproduced":max_legacy<=1e-9,"max_identity_metric_delta":max_identity,"max_legacy_metric_delta":max_legacy,"legacy_semantics":{"centered_procrustes_fit":True,"translation_applied":False,"message_fit_stream":"total_message","message_apply_stream":"external_message","global_term_map":"state_map"},"rows":rows,"cpu_seconds":time.process_time()-cpu0,"wall_seconds":time.perf_counter()-t0}
    if not payload["identity_reproduced"] or not payload["legacy_orthogonal_reproduced"]: raise RuntimeError("V837AL_BASELINE_REPRODUCTION_FAILURE")
    write_json(HERE/"raw/legacy_baseline_reproduction.json",payload);write_json(HERE/"diagnostics/legacy_alignment_reproduction.json",payload);return payload

if __name__=="__main__": reproduce_baselines()
