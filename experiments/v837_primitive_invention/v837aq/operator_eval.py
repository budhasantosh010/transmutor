from __future__ import annotations

import numpy as np
import torch

from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837an.instrumented_af1d import load_model_by_id, run_instrumented

from .natural_interventions import OperatorCase, build_first_order_cases
from .operator_metrics import response_metrics


def _predict(model, episodes, chunk_size: int = 1024) -> np.ndarray:
    values=[]
    model.eval()
    for i in range(0,len(episodes),chunk_size):
        obs,lengths,_=episodes_to_batch(episodes[i:i+chunk_size])
        with torch.no_grad():
            pred=run_instrumented(model,obs,lengths,return_trace=False)
        values.extend(pred.detach().cpu().numpy().astype(np.float64).tolist())
    return np.asarray(values,dtype=np.float64)


def evaluate_cases(organism_id: str, family: str, cases: list[OperatorCase]) -> dict:
    model,row,_,_=load_model_by_id(organism_id)
    if row["family"] != family:
        raise RuntimeError("V837AQ_FAMILY_MISMATCH")
    task=task_by_name(family)
    bp=_predict(model,[c.base_episode for c in cases]);cp=_predict(model,[c.intervened_episode for c in cases]);resp=cp-bp
    oracle=np.asarray([c.oracle_response for c in cases],dtype=np.float64)
    success=np.asarray([task.success(float(p),float(c.intervened_episode.target)) for p,c in zip(cp,cases)],dtype=float)
    metrics=response_metrics(oracle,resp,success)
    details=[]
    for c,b,p,r,o,s in zip(cases,bp,cp,resp,oracle,success):
        details.append({**c.metadata(),"base_prediction":float(b),"intervened_prediction":float(p),"predicted_response":float(r),"oracle_response":float(o),"abs_response_error":float(abs(r-o)),"task_success":bool(s)})
    return {"version":"V837aq","organism_id":organism_id,"family":family,"engine":row["engine"],"competent":bool(row.get("competent")),"final_validation_success":float(row.get("final_validation_success",0.0)),"metrics":metrics,"rows":details}


def operator_pass(metrics: dict, *, reality: bool = False) -> tuple[bool,list[str]]:
    failed=[]
    if reality:
        checks=(("response_nrmse",0.08,"max"),("pearson",0.90,"min"),("direction_agreement",0.90,"min"),("perturbed_task_success",0.85,"min"))
        for key,thr,kind in checks:
            if (kind=="max" and metrics[key]>thr) or (kind=="min" and metrics[key]<thr):failed.append(key)
        if not (0.75<=metrics["gain_ratio"]<=1.25):failed.append("gain_ratio")
    else:
        checks=(("response_nrmse",0.10,"max"),("pearson",0.85,"min"),("direction_agreement",0.85,"min"),("perturbed_task_success",0.80,"min"),("zero_effect_median_abs",0.10,"max"))
        for key,thr,kind in checks:
            if (kind=="max" and metrics[key]>thr) or (kind=="min" and metrics[key]<thr):failed.append(key)
    return not failed,failed
