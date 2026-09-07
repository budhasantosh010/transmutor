from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from experiments.v837_primitive_invention.common.gates import capacity_demonstrated
from experiments.v837_primitive_invention.common.reference_training import evaluate_sequence_model
from experiments.v837_primitive_invention.common.resource_accounting import ResourceAccounting,WallTimer
from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.tasks import all_tasks,task_by_name
from experiments.v837_primitive_invention.v837n.run_mechanism_ablation import _configure_torch
from experiments.v837_primitive_invention.v837ac.shared_input_transfer import CONDITIONS,ControllerInputTransferY3

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/"config.json").read_text(encoding="utf-8"))
FAMILIES=[task.name for task in all_tasks()]
AC0="AC0_y3_parent"; TRANSFER=["AC1_controller_input_factorization","AC1F_folded_control"]


def _git_blob_sha256(path:str)->str: return hashlib.sha256(subprocess.check_output(["git","show",f"HEAD:{path}"],cwd=ROOT)).hexdigest()

def _assert_locks()->None:
    frozen={"experiments/v837_primitive_invention/v837ab/results.json":CONFIG["v837ab_results_sha256"],"experiments/v837_primitive_invention/v837ab/diagnostics/decision_state.json":CONFIG["v837ab_decision_sha256"],"experiments/v837_primitive_invention/v837y/config.json":CONFIG["v837y_config_sha256"],"experiments/v837_primitive_invention/v837y/candidate_interaction.py":CONFIG["v837y_model_sha256"],"experiments/v837_primitive_invention/v837y/results.json":CONFIG["v837y_results_sha256"],"experiments/v837_primitive_invention/v837y/raw/interaction_runs.json":CONFIG["v837y_raw_sha256"]}
    for p,h in frozen.items():
        if _git_blob_sha256(p)!=h: raise SystemExit(f"frozen V837ac dependency changed: {p}")
    decision=json.loads((ROOT/"experiments/v837_primitive_invention/v837ab/diagnostics/decision_state.json").read_text(encoding="utf-8"))
    if decision.get("authorized_v837ac_mode")!=CONFIG["authorized_mode"] or decision.get("neutral_transfer_allowed") is not True: raise SystemExit("V837ab did not authorize this V837ac mode")
    audit=json.loads((ROOT/"experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
    if audit.get("episodes_consumed")!=0: raise SystemExit("fresh audit consumed")
    if (ROOT/"experiments/v837_primitive_invention/v838").exists(): raise SystemExit("V838 exists")

def _seeds():
    tr=CONFIG["training"]; return list(range(tr["development_seed_range"][0],tr["development_seed_range"][1]+1)),list(range(tr["validation_seed_range"][0],tr["validation_seed_range"][1]+1))

def _base_seed(family,rep): return deterministic_int(CONFIG["training"]["base_initialization_namespace"],family,rep)
def _coupling_seed(rep): return deterministic_int(CONFIG["training"]["coupling_seed_namespace"],CONFIG["training"]["coupling_seed_condition"],rep)
def _projection_seed(family,rep): return deterministic_int(CONFIG["training"]["projection_initialization_namespace"],family,rep)

def _model(condition,family,rep): return ControllerInputTransferY3(high_capacity_generic_graph(rep),condition=condition,coupling_initialization_seed=_coupling_seed(rep),projection_seed=_projection_seed(family,rep))

def _success_rate(task,pred,targets): return float(np.mean([task.success(float(p),float(t)) for p,t in zip(pred.tolist(),targets.tolist())]))
def _active(lengths,steps): return torch.arange(steps).view(1,-1)<lengths.view(-1,1)

def _candidate_map(model,cell):
    W=model.base.cell_wx[cell].detach()
    if model.base.input_access_mode=="none": return torch.zeros_like(W)
    if model.base.input_access_mode=="broadcast": return W
    mask=model.base.input_access_mask[:,cell].detach().view(1,-1)
    return W*mask

def _post_diagnostics(model,task,validation_seeds):
    eps=[task.generate(s,"validation") for s in validation_seeds]; observations,lengths,targets=episodes_to_batch(eps); model.eval()
    with torch.no_grad(): base,trace=model(observations,lengths,return_trace=True); no_msg=model(observations,lengths,disable_messages=True)
    mask=_active(lengths,trace.input_terms.shape[1]); active_inputs=trace.input_terms[mask]; active_candidates=trace.candidate_states[mask]
    cell_norms=[]; temporal=[]; jac=[]; maps=[]
    for i in range(10):
        vals=active_inputs[:,i,:]; cell_norms.append(float(torch.linalg.vector_norm(vals,dim=-1).mean().item())); temporal.append(float(torch.var(vals,dim=0,unbiased=False).mean().item())); W=_candidate_map(model,i); maps.append(W.reshape(-1)); deriv=float((1.0-active_candidates[:,i,:]**2).mean().item()); jac.append(float(torch.linalg.vector_norm(W).item()*deriv))
    cos=[]
    for i in range(10):
        for j in range(i+1,10):
            den=max(float(torch.linalg.vector_norm(maps[i])*torch.linalg.vector_norm(maps[j])),1e-12); cos.append(float(torch.dot(maps[i],maps[j]).item()/den))
    success=_success_rate(task,base,targets); no_success=_success_rate(task,no_msg,targets)
    return {"message_dependency":{"baseline_success":success,"no_message_success":no_success,"success_drop":success-no_success},"input_term":{"mean_norm":float(np.mean(cell_norms)),"cell_norms":cell_norms,"cell_temporal_variances":temporal,"mean_temporal_variance":float(np.mean(temporal)),"candidate_jacobian_norm_approx":jac,"mean_candidate_jacobian_norm_approx":float(np.mean(jac)),"pairwise_effective_candidate_input_cosine_median":float(np.median(cos))},"controller_input":model.projection_diagnostics()},2

def _trajectory(model,step): return {"step":int(step),"projection":model.projection_diagnostics()}

def _gradient_norm(params):
    chunks=[p.grad.detach().reshape(-1) for p in params if p.grad is not None]; return float(torch.linalg.vector_norm(torch.cat(chunks)).item()) if chunks else 0.0

def _train(condition,family,rep):
    _configure_torch(); task=task_by_name(family); train_seeds,val_seeds=_seeds(); tr=CONFIG["training"]; seed=_base_seed(family,rep)
    torch.manual_seed(seed); np.random.seed(seed%(2**32-1)); model=_model(condition,family,rep); params=list(model.parameters()); opt=torch.optim.AdamW(params,lr=tr["learning_rate"],weight_decay=tr["weight_decay"]); loss_fn=nn.MSELoss()
    train_eps=[task.generate(s,"development") for s in train_seeds]; val_eps=[task.generate(s,"validation") for s in val_seeds]; observations,lengths,targets=episodes_to_batch(train_eps)
    res=ResourceAccounting(candidate_evaluations=1,model_fits=1,environment_steps=sum(len(e.observations) for e in train_eps+val_eps),parameter_count=model.parameter_count(),model_parameter_bytes=model.parameter_bytes())
    curve=[]; trajectory=[]; requested=set(tr["curve_steps"]); latest=0.0
    def record(step):
        dev=evaluate_sequence_model(model,task,train_eps); val=evaluate_sequence_model(model,task,val_eps); res.forward_calls+=2; curve.append({"step":step,"training_loss":dev.loss,"training_success":dev.success_rate,"validation_loss":val.loss,"validation_success":val.success_rate,"gradient_norm":latest}); trajectory.append(_trajectory(model,step))
    with WallTimer() as timer:
        if 0 in requested: record(0)
        for step in range(1,tr["steps"]+1):
            model.train(); opt.zero_grad(set_to_none=True); pred=model(observations,lengths); res.forward_calls+=1; loss=loss_fn(pred,targets); loss.backward(); latest=_gradient_norm(params); torch.nn.utils.clip_grad_norm_(params,tr["gradient_clip"]); opt.step(); res.optimizer_steps+=1; res.examples_processed+=len(train_eps)
            if step in requested: record(step)
        dev=evaluate_sequence_model(model,task,train_eps); val=evaluate_sequence_model(model,task,val_eps); res.forward_calls+=2
    res.wall_seconds=timer.seconds; res.cpu_seconds=timer.cpu_seconds
    diag,extra=_post_diagnostics(model,task,val_seeds); res.forward_calls+=extra
    return model,{"seed":seed,"development_success":dev.success_rate,"validation_success":val.success_rate,"development_loss":dev.loss,"validation_loss":val.loss,"loss_curve":curve,"projection_trajectory":trajectory,"diagnostics":diag,"resources":res.to_dict()}

def _worker(condition,family,rep):
    model,t=_train(condition,family,rep)
    return {"version":"V837ac","condition":condition,"family":family,"replicate_id":rep,"initialization_seed":t["seed"],"coupling_initialization_seed":_coupling_seed(rep),"projection_initialization_seed":_projection_seed(family,rep),"development_success":t["development_success"],"validation_success":t["validation_success"],"development_loss":t["development_loss"],"validation_loss":t["validation_loss"],"capacity_demonstrated":capacity_demonstrated(t["development_success"],t["validation_success"]),"loss_curve":t["loss_curve"],"projection_trajectory":t["projection_trajectory"],"diagnostics":t["diagnostics"],"parameter_count":model.parameter_count(),"projection_specific_macs":model.projection_specific_macs,"recurrent_controller_projection_macs":model.total_recurrent_controller_projection_macs,"resources":t["resources"],"processed_examples":t["resources"]["examples_processed"],"fresh_audit_consumed":False,"gpu_seconds":0.0,"v838_started":False}

def _run(conditions):
    jobs=[(c,f,r) for c in conditions for f in FAMILIES for r in range(CONFIG["training"]["replicates"])]; rows=[]
    with ProcessPoolExecutor(max_workers=min(10,os.cpu_count() or 1)) as pool:
        futures={pool.submit(_worker,*j):j for j in jobs}
        for fut in as_completed(futures):
            row=fut.result(); rows.append(row); print(f"{row['condition']} {row['family']} r{row['replicate_id']}: dev={row['development_success']:.3f} val={row['validation_success']:.3f}",flush=True)
    rows.sort(key=lambda r:(r["condition"],r["family"],r["replicate_id"])); return rows

def _summary(rows,condition):
    out={"families_passing":0,"family_validation_medians":{},"family_development_medians":{}}
    for f in FAMILIES:
        rr=[r for r in rows if r["condition"]==condition and r["family"]==f]; dev=float(np.median([r["development_success"] for r in rr])); val=float(np.median([r["validation_success"] for r in rr])); out["family_development_medians"][f]=dev; out["family_validation_medians"][f]=val; out["families_passing"]+=int(capacity_demonstrated(dev,val))
    return out

def _parent_guard(rows):
    expected=json.loads((ROOT/"experiments/v837_primitive_invention/v837y/results.json").read_text(encoding="utf-8"))["conditions"]["Y3_global_control_rank4_candidate"]
    observed=_summary(rows,AC0); expvals={f:float(expected["family_results"][f]["validation"]["median"]) for f in FAMILIES}; deltas={f:abs(observed["family_validation_medians"][f]-expvals[f]) for f in FAMILIES}; valid=observed["families_passing"]==3 and max(deltas.values())<=CONFIG["parent_drift_threshold"]
    return {"parent_reproduced":bool(valid),"expected_families_passing":3,"observed":observed,"expected_family_validation_medians":expvals,"absolute_deltas":deltas,"max_absolute_delta":max(deltas.values())}

def _step0():
    train,_=_seeds(); maxima={k:0.0 for k in ("candidate_input","controller_effective_input","global_gate","candidate_state","next_state","prediction")}; rows=[]
    for family in FAMILIES:
        task=task_by_name(family); eps=[task.generate(s,"development") for s in train[:8]]; observations,lengths,_=episodes_to_batch(eps)
        for rep in range(5):
            seed=_base_seed(family,rep); torch.manual_seed(seed); a=_model("AC1_controller_input_factorization",family,rep); torch.manual_seed(seed); f=_model("AC1F_folded_control",family,rep)
            with torch.no_grad(): pa,ta=a(observations,lengths,return_trace=True); pf,tf=f(observations,lengths,return_trace=True); wa,ba=a.effective_controller_input(); wf,bf=f.effective_controller_input()
            errs={"candidate_input":float((ta.input_terms-tf.input_terms).abs().max()),"controller_effective_input":max(float((wa-wf).abs().max()),float((ba-bf).abs().max())),"global_gate":float((ta.global_gates-tf.global_gates).abs().max()),"candidate_state":float((ta.candidate_states-tf.candidate_states).abs().max()),"next_state":float((ta.states-tf.states).abs().max()),"prediction":float((pa-pf).abs().max())}; rows.append({"family":family,"replicate_id":rep,"errors":errs});
            for k,v in errs.items(): maxima[k]=max(maxima[k],v)
    passed=max(maxima.values())<=CONFIG["step0_tolerance"]; return {"step0_equivalence_proven":bool(passed),"tolerance":CONFIG["step0_tolerance"],"max_errors":maxima,"comparisons":rows,"ac1d_applicable":False}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=("preflight","ac0","transfer","all"),default="all"); args=parser.parse_args(); _assert_locks(); [ (HERE/d).mkdir(exist_ok=True) for d in ("raw","diagnostics","plots") ]
    if args.phase in {"preflight","all"}:
        step=_step0(); write_json(HERE/"diagnostics/step0_equivalence.json",step); print(json.dumps({"step0":step["step0_equivalence_proven"],"max":max(step["max_errors"].values())},indent=2));
        if not step["step0_equivalence_proven"]: return 2
    if args.phase in {"ac0","all"}:
        if not (HERE/"diagnostics/step0_equivalence.json").exists(): raise SystemExit("V837ac preflight missing")
        rows=_run([AC0]); write_json(HERE/"raw/ac0_runs.json",{"rows":rows,"unique_seed_defined_episodes":3200}); guard=_parent_guard(rows); write_json(HERE/"diagnostics/parent_compatibility.json",guard); print(json.dumps(guard,indent=2));
        if not guard["parent_reproduced"]: return 3
    if args.phase in {"transfer","all"}:
        guard=HERE/"diagnostics/parent_compatibility.json";
        if not guard.exists() or not json.loads(guard.read_text(encoding="utf-8")).get("parent_reproduced"): raise SystemExit("transfer blocked until AC0 reproduces Y3")
        rows=_run(TRANSFER); write_json(HERE/"raw/transfer_runs.json",{"rows":rows,"unique_seed_defined_episodes":3200})
    return 0

if __name__=="__main__": raise SystemExit(main())
