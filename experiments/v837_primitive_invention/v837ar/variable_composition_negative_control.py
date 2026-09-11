from __future__ import annotations
import numpy as np
from experiments.v837_primitive_invention.tasks import task_by_name
from experiments.v837_primitive_invention.v837an.oracle_macrostate import instrument_episode
from .semantic_execution import discovery_ids,predict_episodes
from .program_ir_interpreter import run
from .utils import HERE,read_json,write_json

def run_negative_control():
    frozen=read_json(HERE/"raw/frozen_canonical_program_irs.json");routing=frozen["families"].get("conditional_routing");rows=[]
    if routing:
        for oid in discovery_ids("variable_composition"):
            eps=[];sem=[]
            for seed in range(11128,11192):
                e=instrument_episode("variable_composition",seed,"development");eps.append(e.as_episode());sem.append((float(e.causal_inputs["initial_value"]),float(e.causal_inputs["gain"][0]),float(e.causal_inputs["drive"][0])))
            actual=predict_episodes(oid,eps);pred=np.asarray([run(routing,{"control":c,"A":a,"B":b}) for c,a,b in sem],float);nrmse=float(np.sqrt(np.mean((pred-actual)**2))/2.0);rows.append({"organism_id":oid,"nrmse":nrmse,"false_universal_pass":nrmse<=.10})
    payload={"version":"V837ar","new_variable_composition_primitive_fit":False,"routing_shape_compatible_probe":rows,"generic_ir_falsely_universal":any(r["false_universal_pass"] for r in rows)}
    write_json(HERE/"raw/variable_composition_negative_control.json",payload);return payload
