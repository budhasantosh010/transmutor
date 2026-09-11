from __future__ import annotations

import math
import numpy as np
from experiments.v837_primitive_invention.v837aq.operator_metrics import response_metrics
from .shared_fit import ridge_fit,per_organism_fit
from .semantic_execution import discovery_ids,engine_for,predict_episodes
from .operator_words import ROLE_SEEDS,recall_word_cases
from .program_ir import ProgramIR
from .utils import HERE,read_json,write_json

GRID=np.linspace(-1.25,1.25,2049)
ORDER=("M0_CONSTANT","M1_DELAY_TABLE","M2A_RANK1_SIMPLE","M2_RANK1_MEMORY")
def _rows(role):
    ns=[];v=[];y=[];ids=[];meta=[]
    for oid in discovery_ids("delayed_recall"):
        words=[w for s in ROLE_SEEDS[role] for w in recall_word_cases(s,role)];pred=predict_episodes(oid,[w["episode"] for w in words])
        for w,p in zip(words,pred):ns.append(w["delay"]);v.append(w["value"]);y.append(float(p));ids.append(oid);meta.append((oid,w))
    return np.asarray(ns),np.asarray(v),np.asarray(y),np.asarray(ids),meta
def _fit_fixed_lambda(kind,lam,n,v,y,ids):
    q=np.power(float(lam),n.astype(float));
    if kind=="M2A_RANK1_SIMPLE": X=np.c_[q*v,np.ones(len(v))]
    else:
        geom=np.asarray([float(nn) if abs(1-lam)<1e-12 else (1-float(lam)**int(nn))/(1-float(lam)) for nn in n]);X=np.c_[q*v,q,geom,np.ones(len(v))]
    coef=ridge_fit(X,y,ids);pred=X@coef;return float(np.sqrt(np.mean((pred-y)**2))),coef,X
def _brent_refine(kind,best_i,n,v,y,ids):
    """Deterministic bounded Brent minimization inside the winning grid cell."""
    lo=float(GRID[max(0,best_i-1)]);hi=float(GRID[min(len(GRID)-1,best_i+1)])
    def f(x):return _fit_fixed_lambda(kind,float(x),n,v,y,ids)[0]
    golden=0.3819660112501051
    x=w=z=lo+golden*(hi-lo);fx=fw=fz=f(x);d=e=0.0
    eps=np.finfo(float).eps;tol=1e-12;iterations=0
    for iterations in range(1,129):
        mid=.5*(lo+hi);tol1=math.sqrt(eps)*abs(x)+tol/3.0;tol2=2.0*tol1
        if abs(x-mid)<=tol2-.5*(hi-lo):break
        p=q=r=0.0
        if abs(e)>tol1:
            r=(x-w)*(fx-fz);q=(x-z)*(fx-fw);p=(x-z)*q-(x-w)*r;q=2.0*(q-r)
            if q>0:p=-p
            q=abs(q);etemp=e;e=d
            if abs(p)<abs(.5*q*etemp) and p>q*(lo-x) and p<q*(hi-x):
                d=p/q;u=x+d
                if u-lo<tol2 or hi-u<tol2:d=tol1 if mid>=x else -tol1
            else:e=(hi-x) if x<mid else (lo-x);d=golden*e
        else:e=(hi-x) if x<mid else (lo-x);d=golden*e
        u=x+d if abs(d)>=tol1 else x+(tol1 if d>0 else -tol1);fu=f(u)
        if fu<=fx:
            if u<x:hi=x
            else:lo=x
            z,w,x=w,x,u;fz,fw,fx=fw,fx,fu
        else:
            if u<x:lo=u
            else:hi=u
            if fu<=fw or w==x:z,w=w,u;fz,fw=fw,fu
            elif fu<=fz or z==x or z==w:z=u;fz=fu
    err,coef,X=_fit_fixed_lambda(kind,x,n,v,y,ids);return float(x),err,coef,X,iterations

def _refine(kind,best_i,n,v,y,ids):
    lam,err,coef,X,_iters=_brent_refine(kind,best_i,n,v,y,ids);return lam,err,coef,X

def _fit_rank(kind,n,v,y,ids):
    errs=[]
    for lam in GRID:errs.append(_fit_fixed_lambda(kind,float(lam),n,v,y,ids)[0])
    i=int(np.argmin(errs));lam,err,coef,X,iters=_brent_refine(kind,i,n,v,y,ids);return {"lambda":float(lam),"coefficients":coef.tolist(),"fit_rmse":err,"parameter_count":(3 if kind=="M2A_RANK1_SIMPLE" else 5),"grid_points":2049,"refinement":"BOUNDED_BRENT","brent_iterations":int(iters),"per_organism":per_organism_fit(X,y,ids)}
def fit_candidates():
    n,v,y,ids,_=_rows("FIT_WORDS");fits={}
    X=np.ones((len(y),1));coef=ridge_fit(X,y,ids);fits["M0_CONSTANT"]={"coefficients":coef.tolist(),"parameter_count":1,"fit_rmse":float(np.sqrt(np.mean((X@coef-y)**2)))}
    delays=sorted(set(n.tolist()));cols=[]
    for d in delays:cols.extend([(n==d)*v,(n==d).astype(float)])
    X=np.stack(cols,axis=1);coef=ridge_fit(X,y,ids);fits["M1_DELAY_TABLE"]={"coefficients":coef.tolist(),"delays":delays,"parameter_count":len(coef),"fit_rmse":float(np.sqrt(np.mean((X@coef-y)**2)))}
    fits["M2A_RANK1_SIMPLE"]=_fit_rank("M2A_RANK1_SIMPLE",n,v,y,ids);fits["M2_RANK1_MEMORY"]=_fit_rank("M2_RANK1_MEMORY",n,v,y,ids)
    write_json(HERE/"raw/memory_ir_fits.json",{"version":"V837ar","fits":fits,"lambda_grid":{"lo":-1.25,"hi":1.25,"points":2049},"fit_rows":len(y),"read_gain_fixed":1.0});return fits
def _predict(g,f,w):
    n=w["delay"];v=w["value"]
    if g=="M0_CONSTANT":return float(f["coefficients"][0])
    if g=="M1_DELAY_TABLE":
        if n not in f["delays"]:return float("nan")
        i=f["delays"].index(n);return float(f["coefficients"][2*i]*v+f["coefficients"][2*i+1])
    lam=f["lambda"];q=lam**n;c=f["coefficients"]
    if g=="M2A_RANK1_SIMPLE":return float(q*c[0]*v+c[1])
    geom=n if abs(1-lam)<1e-12 else (1-lam**n)/(1-lam);return float(q*c[0]*v+q*c[1]+geom*c[2]+c[3])
def evaluate_role(g,f,role):
    source=[];oracle=[];pred=[];success=[];rows=[]
    for oid in discovery_ids("delayed_recall"):
        words=[w for s in ROLE_SEEDS[role] for w in recall_word_cases(s,role)];actual=predict_episodes(oid,[w["episode"] for w in words])
        for w,a in zip(words,actual):
            p=_predict(g,f,w);source.append(float(a));oracle.append(w["oracle"]);pred.append(p);success.append(bool(np.isfinite(p) and abs(p-w["oracle"])<=.35));rows.append({"organism_id":oid,"engine":engine_for("delayed_recall",oid),"word":w["name"],"source":float(a),"oracle":w["oracle"],"ir":p})
    finite=np.isfinite(pred)
    if not all(finite): return {"role":role,"undefined_unseen":True,"rows":rows}
    return {"role":role,"undefined_unseen":False,"organism_relative":response_metrics(source,pred,success),"oracle_relative":response_metrics(oracle,pred,success),"rows":rows}
def select_memory():
    fits=fit_candidates();basis=read_json(HERE/"raw/canonical_response_basis.json")["response_basis_sha256"];rows=[];winner=None
    for g in ORDER:
        ev=evaluate_role(g,fits[g],"SELECT_WORDS");passed=False
        if not ev.get("undefined_unseen"):
            m=ev["organism_relative"];o=ev["oracle_relative"];passed=bool(m["response_nrmse"]<=.10 and o["response_nrmse"]<=.10 and o["pearson"]>=.85 and o["direction_agreement"]>=.85 and o["perturbed_task_success"]>=.80)
        rows.append({"grammar":g,"pass":passed,"metrics":ev});
        if passed:winner=g;break
    ir=None
    if winner in {"M2A_RANK1_SIMPLE","M2_RANK1_MEMORY"}:
        f=fits[winner];lam=f["lambda"];c=f["coefficients"]
        params={"w":float(c[0]),"lambda":float(lam),"bw":0.0,"bh":0.0,"br":float(c[1])} if winner=="M2A_RANK1_SIMPLE" else {"w":float(c[0]),"lambda":float(lam),"bw":float(c[1]),"bh":float(c[2]),"br":float(c[3])}
        ir=ProgramIR(family="delayed_recall",operator_name="MEMORY",operator_class="LINEAR_PHASE_MACHINE_R1",granularity="PREDICTIVE_RANK1_MEMORY",semantic_inputs=("value",),semantic_outputs=("y",),predictive_state={"dimension":1,"name":"p","gauge":"READ_GAIN_1"},parameters=params,phase_contract=("WRITE","HOLD","READ"),composition_contract={"repeat_hold_executable":True,"hold_power":"lambda^n","C1_C2_C3_evidence_deferred_to_AR14":True},validity_domain={"delay":[4,12],"value":[-1,1]},response_basis_hash=basis,fit_provenance={"fold":"AQ_DISCOVERY","word_partition":"FIT_WORDS","equal_organism_weighting":True},complexity={"continuous_parameters":f["parameter_count"]},evidence={"archiveable":f["parameter_count"]<=256}).to_dict()
    payload={"version":"V837ar","candidates":rows,"selected_grammar":winner,"selected_ir":ir,"pass":winner is not None,"predictive_rank":1 if winner in {"M2A_RANK1_SIMPLE","M2_RANK1_MEMORY"} else None,"lambda":(fits[winner].get("lambda") if winner else None)}
    write_json(HERE/"diagnostics/memory_predictive_state.json",payload);return payload
