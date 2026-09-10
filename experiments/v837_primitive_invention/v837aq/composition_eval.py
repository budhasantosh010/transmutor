from __future__ import annotations

import json
import numpy as np
import torch

from experiments.v837_primitive_invention.common.trainer import episodes_to_batch
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837an.instrumented_af1d import load_model_by_id, run_instrumented

from .data_roles import seeds
from .failure_ledger import add, make_entry
from .interaction_eval import interaction_bundles
from .operator_metrics import RANGE, direction_agreement, family_gate
from .source_folds import discovery_ids
from .utils import HERE, read_json, write_json


def _predict(model,eps):
    obs,lengths,_=episodes_to_batch(eps)
    with torch.no_grad():return run_instrumented(model,obs,lengths,return_trace=False).detach().cpu().numpy().astype(np.float64)


def evaluate_composition_organism(organism_id:str,family:str,seed_values:list[int],operator_order:int)->dict:
    model,row,_,_=load_model_by_id(organism_id);bundles=interaction_bundles(family,seed_values);eps=[]
    for b in bundles:eps.extend([b["base"],b["a"],b["b"],b["ab"]])
    pred=_predict(model,eps).reshape(len(bundles),4);ra=pred[:,1]-pred[:,0];rb=pred[:,2]-pred[:,0];rab=pred[:,3]-pred[:,0]
    oracle_a=np.asarray([b["oracle_a"] for b in bundles]);oracle_b=np.asarray([b["oracle_b"] for b in bundles]);oracle_ab=np.asarray([b["oracle_ab"] for b in bundles]);oracle_i=np.asarray([b["oracle_interaction"] for b in bundles])
    composed=ra+rb+(oracle_i if operator_order==2 else 0.0)
    comp_nrmse=float(np.sqrt(np.mean((composed-rab)**2))/RANGE);oracle_nrmse=float(np.sqrt(np.mean((rab-oracle_ab)**2))/RANGE);direction=direction_agreement(oracle_ab,composed);task=task_by_name(family);success=float(np.mean([task.success(float(pred[i,3]),float(b["ab"].target)) for i,b in enumerate(bundles)]))
    passed=bool(comp_nrmse<=.10 and oracle_nrmse<=.10 and direction>=.85 and success>=.80)
    return {"organism_id":organism_id,"family":family,"engine":row["engine"],"operator_order":operator_order,"composition_nrmse":comp_nrmse,"compound_oracle_nrmse":oracle_nrmse,"direction_agreement":direction,"compound_task_success":success,"cases":len(bundles),"pass":passed}


def run_composition()->dict:
    disc=read_json(HERE/"raw/operator_discovery.json");payload={"version":"V837aq","stage":"AQ8_OPERATOR_COMPOSITION","families":{},"rows":[],"neural_state_used":False}
    for family in disc.get("accepted_families",[]):
        order=int(disc["families"][family]["operator_order"]);rows=[evaluate_composition_organism(oid,family,seeds("AQ_COMPOSITION"),order) for oid in discovery_ids(family)];gate=family_gate(rows);payload["rows"].extend(rows);payload["families"][family]={"operator_order":order,"gate":gate,"pass":gate["pass"]}
        if not gate["pass"]:add(make_entry(failure_id=f"V837aq-AQ8-{family}",stage="AQ8_OPERATOR_COMPOSITION",branch="PROGRAM_COMPOSITION",family=family,operator_order=order,metrics={"gate":gate,"rows":rows},acceptance_gate={"composition_nrmse_max":.10,"oracle_nrmse_max":.10,"direction_min":.85,"task_success_min":.80,"family_pass":.60},failed_conditions=["composition family gate"],scientific_interpretation="The external causal operator did not compose reliably under the frozen compound intervention word.",confounds_ruled_out=["hidden-state alignment","single-endpoint-only evidence"],confounds_remaining=["predictive causal state","history-conditioned operator"],next_justified_experiment="Run the frozen predictive-state fallback diagnostic only."))
    write_json(HERE/"raw/composition_results.json",payload);write_json(HERE/"diagnostics/operator_composition.json",payload);return payload


if __name__=="__main__":print(json.dumps(run_composition(),indent=2))
