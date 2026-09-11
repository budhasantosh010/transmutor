from __future__ import annotations

import numpy as np
from experiments.v837_primitive_invention.v837aq.operator_metrics import response_metrics
from .shared_fit import ridge_fit, per_organism_fit
from .semantic_execution import discovery_ids, engine_for, predict_episodes
from .operator_words import ROLE_SEEDS, iterative_word_cases
from .program_ir import ProgramIR, canonical_bytes
from .program_ir_interpreter import run
from .utils import HERE, read_json, write_json

GRAMMARS=("I0_CONSTANT_RESPONSE","I1_IDENTITY_STATE","I2_AFFINE_UPDATE","I3_QUADRATIC_UPDATE")

def _features(kind,z,x):
    z=np.asarray(z,float);x=np.asarray(x,float)
    if kind=="I0_CONSTANT_RESPONSE": return np.ones((len(z),1))
    if kind=="I1_IDENTITY_STATE": return None
    if kind=="I2_AFFINE_UPDATE": return np.c_[z,x,np.ones(len(z))]
    if kind=="I3_QUADRATIC_UPDATE": return np.c_[np.ones(len(z)),z,x,z*z,z*x,x*x]
    raise KeyError(kind)

def _fit_rows(role="FIT_WORDS"):
    Xz=[];Xx=[];Y=[];ids=[]
    for oid in discovery_ids("iterative_state"):
        eps=[];meta=[]
        for s in ROLE_SEEDS[role]:
            for w in iterative_word_cases(s,role):
                obs=w["episode"].observations
                for t in range(1,len(obs)):
                    from experiments.v837_primitive_invention.common.task_interface import Episode
                    eps.append(Episode(obs[:t+1].copy(),0.0,{"family":"iterative_state","length":t}));meta.append((w,t,float(obs[t,0])))
        preds=predict_episodes(oid,eps);prev={}
        for (w,t,x),y in zip(meta,preds):
            key=(w["seed"],w["name"]);z=0.0 if t==1 else prev[key];prev[key]=float(y)
            Xz.append(z);Xx.append(x);Y.append(float(y));ids.append(oid)
    return np.asarray(Xz),np.asarray(Xx),np.asarray(Y),np.asarray(ids)

def fit_candidates() -> dict:
    z,x,y,ids=_fit_rows("FIT_WORDS");fits={}
    for g in GRAMMARS:
        if g=="I1_IDENTITY_STATE": coef=[];pred=z.copy()
        else:
            X=_features(g,z,x);coef=ridge_fit(X,y,ids).tolist();pred=X@np.asarray(coef)
        fits[g]={"coefficients":coef,"fit_nrmse":float(np.sqrt(np.mean((pred-y)**2))/2.0),"parameter_count":len(coef),"per_organism":({} if g=="I1_IDENTITY_STATE" else per_organism_fit(_features(g,z,x),y,ids))}
    write_json(HERE/"raw/iterative_ir_fits.json",{"version":"V837ar","fits":fits,"fit_rows":len(y),"oracle_coefficients_read_during_fit":False});return fits

def _ir(g,coef,basis_hash):
    if g=="I0_CONSTANT_RESPONSE":a,b,c=0.0,0.0,float(coef[0])
    elif g=="I1_IDENTITY_STATE":a,b,c=1.0,0.0,0.0
    elif g=="I2_AFFINE_UPDATE":a,b,c=map(float,coef)
    else:return None
    return ProgramIR(family="iterative_state",operator_name="UPDATE",operator_class="LINEAR_STATE_UPDATE_1D",granularity="STATE_UPDATE",
                     semantic_inputs=("z","x"),semantic_outputs=("z_next",),predictive_state={"dimension":1,"name":"z"},parameters={"a":a,"b":b,"c":c,"initial_state":0.0},
                     phase_contract=("UPDATE",),composition_contract={"analytic_semigroup_executable":True,"C1_C2_C3_evidence_deferred_to_AR14":True},validity_domain={"x":[-1,1]},response_basis_hash=basis_hash,
                     fit_provenance={"fold":"AQ_DISCOVERY","word_partition":"FIT_WORDS","equal_organism_weighting":True,"oracle_coefficients_used":False},complexity={"continuous_parameters":3},evidence={"archiveable":True}).to_dict()
def _candidate_predict(g,coef,w):
    if g=="I0_CONSTANT_RESPONSE": return float(coef[0])
    if g=="I1_IDENTITY_STATE": return 0.0
    if g=="I2_AFFINE_UPDATE":
        a,b,c=map(float,coef);z=0.0
        for x in w["x"]: z=a*z+b*float(x)+c
        return z
    c=np.asarray(coef,float);z=0.0
    for x in w["x"]: z=float(np.dot(np.asarray([1,z,x,z*z,z*x,x*x]),c))
    return z

def evaluate_role(g,coef,role):
    source=[];oracle=[];pred=[];success=[];rows=[]
    for oid in discovery_ids("iterative_state"):
        words=[w for s in ROLE_SEEDS[role] for w in iterative_word_cases(s,role)];actual=predict_episodes(oid,[w["episode"] for w in words])
        for w,a in zip(words,actual):
            p=_candidate_predict(g,coef,w);source.append(float(a));oracle.append(float(w["oracle"]));pred.append(p);success.append(abs(p-w["oracle"])<=.16);rows.append({"organism_id":oid,"engine":engine_for("iterative_state",oid),"word":w["name"],"source":float(a),"oracle":float(w["oracle"]),"ir":p})
    return {"role":role,"organism_relative":response_metrics(source,pred,success),"oracle_relative":response_metrics(oracle,pred,success),"rows":rows}
def run_reality_gate() -> dict:
    fits=fit_candidates();basis=read_json(HERE/"raw/canonical_response_basis.json")["response_basis_sha256"];selection=[];winner=None
    for g in GRAMMARS:
        r=evaluate_role(g,fits[g]["coefficients"],"SELECT_WORDS");om=r["organism_relative"];orac=r["oracle_relative"]
        passed=bool(om["response_nrmse"]<=.10 and orac["response_nrmse"]<=.10 and orac["pearson"]>=.85 and orac["direction_agreement"]>=.85 and orac["perturbed_task_success"]>=.80)
        selection.append({"grammar":g,"pass":passed,"metrics":r});
        if passed and winner is None: winner=g;break
    selected_ir=_ir(winner,fits[winner]["coefficients"],basis) if winner else None
    executable=selected_ir is not None
    payload={"version":"V837ar","candidates":selection,"selected_grammar":winner,"selected_ir":selected_ir,"pass":bool(winner and executable),"kill_switch_triggered":not bool(winner and executable),"unsupported_compact_grammar":bool(winner and not executable)}
    if winner:
        payload["fitted_parameters"]=(selected_ir or {}).get("parameters",{})
    write_json(HERE/"diagnostics/iterative_reality_gate.json",payload);return payload
