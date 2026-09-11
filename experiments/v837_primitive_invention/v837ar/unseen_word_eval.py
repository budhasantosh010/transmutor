from __future__ import annotations

import math
import numpy as np
from experiments.v837_primitive_invention.v837aq.operator_metrics import response_metrics
from .operator_words import ROLE_SEEDS,iterative_word_cases,routing_word_cases,recall_word_cases
from .program_ir_interpreter import run
from .semantic_execution import discovery_ids,engine_for,predict_episodes
from .utils import HERE,read_json,write_json


def ir_predict(family,ir,w):
    if family=="conditional_routing":return float(run(ir,{"control":w["control"],"A":w["A"],"B":w["B"]}))
    if family=="iterative_state":return float(run(ir,{"ops":[{"phase":"UPDATE","x":float(x)} for x in w["x"]]}))
    if family=="delayed_recall":return float(run(ir,{"ops":[{"phase":"WRITE","value":float(w["value"])}]+[{"phase":"HOLD"} for _ in range(int(w["delay"]))]+[{"phase":"READ"}]}))
    raise KeyError(family)
def _success_limit(family):return {"conditional_routing":.25,"iterative_state":.16,"delayed_recall":.35}[family]
def _words(family,role):
    fn={"conditional_routing":routing_word_cases,"iterative_state":iterative_word_cases,"delayed_recall":recall_word_cases}[family]
    return [w for s in ROLE_SEEDS[role] for w in fn(s,role)]
def evaluate_organism(family,ir,oid,role="FINAL_UNSEEN_WORDS"):
    words=_words(family,role);actual=predict_episodes(oid,[w["episode"] for w in words]);pred=[ir_predict(family,ir,w) for w in words];oracle=[float(w["oracle"]) for w in words];success=[abs(p-o)<=_success_limit(family) for p,o in zip(pred,oracle)]
    org=response_metrics(actual,pred,success);orc=response_metrics(oracle,pred,success);passed=bool(org["response_nrmse"]<=.10 and orc["response_nrmse"]<=.10 and orc["pearson"]>=.85 and orc["direction_agreement"]>=.85 and orc["perturbed_task_success"]>=.80)
    return {"organism_id":oid,"engine":engine_for(family,oid),"role":role,"organism_relative":org,"oracle_relative":orc,"pass":passed,"rows":[{"word":w["name"],"seed":w["seed"],"source":float(a),"oracle":float(o),"ir":float(p)} for w,a,o,p in zip(words,actual,oracle,pred)]}
def family_gate(rows,fraction=.60):
    required=int(math.ceil(fraction*len(rows))) if rows else 0;passing=[r for r in rows if r.get("pass")];engines=sorted({r["engine"] for r in passing});return {"organisms":len(rows),"required":required,"passing":len(passing),"pass_fraction":len(passing)/len(rows) if rows else 0.0,"passing_engines":engines,"both_engines":len(engines)>=2,"pass":bool(rows and len(passing)>=required and len(engines)>=2)}
def run_unseen_words():
    frozen=read_json(HERE/"raw/frozen_canonical_program_irs.json");out={"version":"V837ar","freeze_sha256":frozen["freeze_sha256"],"opened_after_freeze":True,"no_refit":True,"families":{}}
    for fam,ir in frozen["families"].items():
        if ir is None:out["families"][fam]={"pass":False,"reason":"NULL_AT_FREEZE","rows":[],"gate":{"pass":False}};continue
        rows=[evaluate_organism(fam,ir,oid) for oid in discovery_ids(fam)];gate=family_gate(rows,.60);out["families"][fam]={"pass":gate["pass"],"gate":gate,"rows":rows}
    write_json(HERE/"raw/final_unseen_word_results.json",out);write_json(HERE/"diagnostics/unseen_word_generalization.json",out);return out
