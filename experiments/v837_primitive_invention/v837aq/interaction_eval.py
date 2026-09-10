from __future__ import annotations

import numpy as np
import torch

from experiments.v837_primitive_invention.common.task_interface import Episode
from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837an.instrumented_af1d import load_model_by_id, run_instrumented
from experiments.v837_primitive_invention.v837an.oracle_macrostate import instrument_episode

from .data_roles import seeds
from .operator_metrics import RANGE, direction_agreement, family_gate
from .source_folds import discovery_ids
from .utils import HERE, write_json


def _episode(obs,target,family,**meta):return Episode(np.asarray(obs,dtype=np.float32),float(target),{"family":family,**meta})
def _predict(model,episodes):
    obs,lengths,_=episodes_to_batch(episodes)
    with torch.no_grad():return run_instrumented(model,obs,lengths,return_trace=False).detach().cpu().numpy().astype(np.float64)
def _iter_target(x):
    z=0.0
    for v in x:z=.65*z+.35*float(v)
    return float(z)
def _comp_target(initial,g,d):
    z=float(initial)
    for gg,dd in zip(g,d):z=float(np.tanh(float(gg)*z+float(dd)))
    return z


def interaction_bundles(family:str,seed_values:list[int]):
    bundles=[]
    for seed in seed_values:
        b=instrument_episode(family,seed,"development");base=b.as_episode()
        if family=="conditional_routing":
            c=float(b.causal_inputs["control"]);a=float(b.causal_inputs["payload_a"]);bb=float(b.causal_inputs["payload_b"]);aa=float(np.clip(a+.4,-.85,.85))
            oa=b.observations.copy();oa[1,0]=-c;ta=a if -c>0 else bb
            ob=b.observations.copy();ob[2,1]=aa;tb=aa if c>0 else bb
            oab=b.observations.copy();oab[1,0]=-c;oab[2,1]=aa;tab=aa if -c>0 else bb
            A=_episode(oa,ta,family);B=_episode(ob,tb,family);AB=_episode(oab,tab,family)
        elif family=="delayed_recall":
            v=float(b.causal_inputs["value"]);delay=int(b.latent_variables["delay"]);j=(delay-1)//2;t=2+j;old=float(b.observations[t,0])
            oa=b.observations.copy();oa[1,0]=-v;A=_episode(oa,-v,family,delay=delay)
            ob=b.observations.copy();ob[t,0]=-old;B=_episode(ob,v,family,delay=delay)
            oab=b.observations.copy();oab[1,0]=-v;oab[t,0]=-old;AB=_episode(oab,-v,family,delay=delay)
        elif family=="iterative_state":
            x=np.asarray(b.causal_inputs["x_t"],dtype=float);j1=0;j2=max(0,len(x)//2)
            xa=x.copy();xa[j1]=-xa[j1];xb=x.copy();xb[j2]=-xb[j2];xab=x.copy();xab[j1]=-xab[j1];xab[j2]=-xab[j2]
            oa=b.observations.copy();oa[1+j1,0]=xa[j1];ob=b.observations.copy();ob[1+j2,0]=xb[j2];oab=b.observations.copy();oab[1+j1,0]=xab[j1];oab[1+j2,0]=xab[j2]
            A=_episode(oa,_iter_target(xa),family,length=len(x));B=_episode(ob,_iter_target(xb),family,length=len(x));AB=_episode(oab,_iter_target(xab),family,length=len(x))
        elif family=="variable_composition":
            initial=float(b.causal_inputs["initial_value"]);g=np.asarray(b.causal_inputs["gain"],dtype=float);d=np.asarray(b.causal_inputs["drive"],dtype=float);j=0
            ga=float(np.clip(g[j]+.15,.45,.95));db=float(np.clip(d[j]+.15,-.35,.35));gga=g.copy();gga[j]=ga;ddb=d.copy();ddb[j]=db
            oa=b.observations.copy();oa[1+j,1]=ga;ob=b.observations.copy();ob[1+j,2]=db;oab=b.observations.copy();oab[1+j,1]=ga;oab[1+j,2]=db
            A=_episode(oa,_comp_target(initial,gga,d),family,depth=len(g));B=_episode(ob,_comp_target(initial,g,ddb),family,depth=len(g));AB=_episode(oab,_comp_target(initial,gga,ddb),family,depth=len(g))
        else:raise KeyError(family)
        oracle_a=A.target-base.target;oracle_b=B.target-base.target;oracle_ab=AB.target-base.target
        bundles.append({"seed":seed,"base":base,"a":A,"b":B,"ab":AB,"oracle_a":oracle_a,"oracle_b":oracle_b,"oracle_ab":oracle_ab,"oracle_interaction":oracle_ab-oracle_a-oracle_b})
    return bundles


def evaluate_interactions(organism_id:str,family:str,bundles:list[dict])->dict:
    model,row,_,_=load_model_by_id(organism_id);eps=[]
    for b in bundles:eps.extend([b["base"],b["a"],b["b"],b["ab"]])
    pred=_predict(model,eps).reshape(len(bundles),4);pr_a=pred[:,1]-pred[:,0];pr_b=pred[:,2]-pred[:,0];pr_ab=pred[:,3]-pred[:,0];inter=pr_ab-pr_a-pr_b;oracle=np.asarray([b["oracle_interaction"] for b in bundles],dtype=float)
    nrmse=float(np.sqrt(np.mean((inter-oracle)**2))/RANGE);direction=direction_agreement(oracle,inter);med=float(np.median(np.abs(oracle))/RANGE);p90=float(np.quantile(np.abs(oracle),.9)/RANGE);required=bool(med>=.05 or p90>=.10)
    if required:passed=bool(nrmse<=.10 and direction>=.80)
    else:passed=bool(float(np.median(np.abs(inter))/RANGE)<=.10)
    return {"organism_id":organism_id,"family":family,"engine":row["engine"],"oracle_interaction_median_abs_normalized":med,"oracle_interaction_p90_abs_normalized":p90,"second_order_required":required,"interaction_nrmse":nrmse,"interaction_direction":direction,"predicted_interaction_median_abs_normalized":float(np.median(np.abs(inter))/RANGE),"pass":passed}


def run_interaction_program(eligible_families:list[str])->dict:
    payload={"version":"V837aq","stage":"AQ4_SECOND_ORDER_INTERACTION","families":{},"rows":[]}
    for family in eligible_families:
        bundles=interaction_bundles(family,seeds("AQ_INTERACTION"));rows=[evaluate_interactions(oid,family,bundles) for oid in discovery_ids(family)];gate=family_gate(rows);required=bool(rows and np.median([r["oracle_interaction_median_abs_normalized"] for r in rows])>=.05 or rows and np.median([r["oracle_interaction_p90_abs_normalized"] for r in rows])>=.10)
        payload["families"][family]={"second_order_required":required,"gate":gate,"pass":gate["pass"]};payload["rows"].extend(rows)
    write_json(HERE/"raw/interaction_results.json",payload);write_json(HERE/"diagnostics/interaction_order.json",payload);return payload
