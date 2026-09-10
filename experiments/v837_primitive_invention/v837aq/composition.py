from __future__ import annotations

import numpy as np

from .data_roles import seeds
from .natural_interventions import build_compound_cases
from .operator_discovery import run_operator_discovery
from .operator_eval import evaluate_cases, operator_pass
from .operator_metrics import family_gate
from .source_folds import discovery_ids
from .utils import HERE, read_json, write_json


def run_composition()->dict:
    discovery=read_json(HERE/"raw/operator_discovery.json") if (HERE/"raw/operator_discovery.json").is_file() else run_operator_discovery();families=list(discovery.get("accepted_families",[]));payload={"version":"V837aq","stage":"AQ8_OPERATOR_COMPOSITION","families":{},"rows":[],"contract_prediction_source":"frozen abstract task/operator laws; no neural-state coordinate"}
    for family in families:
        cases=build_compound_cases(family,seeds("AQ_COMPOSITION"));rows=[]
        for oid in discovery_ids(family):
            got=evaluate_cases(oid,family,cases);m=got["metrics"];passed=bool(m["response_nrmse"]<=.10 and m["direction_agreement"]>=.85 and m["perturbed_task_success"]>=.80);row={"organism_id":oid,"family":family,"engine":got["engine"],"metrics":m,"pass":passed};rows.append(row);payload["rows"].append(row)
        gate=family_gate(rows);payload["families"][family]={"gate":gate,"pass":gate["pass"],"operator_word":{"conditional_routing":"SELECT o LOAD o CONTROL","delayed_recall":"READ o HOLD^n o WRITE","iterative_state":"U(x2) o U(x1)"}[family]}
    write_json(HERE/"raw/composition_results.json",payload);write_json(HERE/"diagnostics/operator_composition.json",payload);return payload


if __name__=="__main__":
    import json;print(json.dumps(run_composition(),indent=2))
