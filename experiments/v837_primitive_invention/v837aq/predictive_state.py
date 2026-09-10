from __future__ import annotations

from collections import defaultdict
import numpy as np

from .composition import run_composition
from .data_roles import seeds
from .natural_interventions import build_first_order_cases
from .operator_discovery import run_operator_discovery
from .utils import HERE, read_json, write_json

CONFIG=read_json(HERE/"config.json")


def _rank_at_energy(matrix:np.ndarray,energy:float)->tuple[int,list[float],list[float]]:
    if matrix.size==0:return 0,[],[]
    s=np.linalg.svd(matrix,full_matrices=False,compute_uv=False);e=s*s;total=float(e.sum())
    frac=(e/total).tolist() if total>1e-15 else [0.0 for _ in s]
    if total<=1e-15:return 0,[float(x) for x in s],[float(x) for x in frac]
    cumulative=np.cumsum(e)/total;rank=int(np.searchsorted(cumulative,energy)+1);return rank,[float(x) for x in s],[float(x) for x in frac]


def _test_key(case)->tuple:
    # Canonicalize a future test by semantic operation, not by history-dependent sign.
    if case.intervention in {"CONTROL_FLIP","WRITE_FLIP","DISTRACTOR_FLIP","DELAY_SHORTEN","DELAY_EXTEND"}:
        magnitude=None
    elif case.intervention=="INPUT_PERTURB":
        magnitude=None if case.actual_delta is None else round(abs(float(case.actual_delta)),6)
    else:
        magnitude=None if case.requested_delta is None else round(float(case.requested_delta),6)
    return (case.intervention,case.phase,int(case.horizon),magnitude)


def _oracle_hankel(family:str)->dict:
    seed_values=seeds("AQ_META_CONFIRM");cases=build_first_order_cases(family,seed_values);by_seed=defaultdict(dict)
    for c in cases:by_seed[int(c.seed)][_test_key(c)]=float(c.oracle_response)
    common=None
    for seed in seed_values:
        keys=set(by_seed[int(seed)])
        common=keys if common is None else common & keys
    columns=sorted(common or set(),key=str);matrix=np.asarray([[by_seed[int(seed)][k] for k in columns] for seed in seed_values],dtype=float) if columns else np.empty((len(seed_values),0))
    rank,s,energy=_rank_at_energy(matrix,float(CONFIG["predictive_state"]["energy_fraction"]))
    centered=matrix-matrix.mean(axis=0,keepdims=True) if matrix.size else matrix;crank,cs,cenergy=_rank_at_energy(centered,float(CONFIG["predictive_state"]["energy_fraction"]))
    return {"histories":len(seed_values),"future_tests":len(columns),"test_columns":[list(k) for k in columns],"oracle_rank_99":rank,"oracle_singular_values":s,"oracle_energy_fractions":energy,"centered_oracle_rank_99":crank,"centered_oracle_singular_values":cs,"centered_oracle_energy_fractions":cenergy}


def run_predictive_state_diagnostic()->dict:
    discovery=read_json(HERE/"raw/operator_discovery.json") if (HERE/"raw/operator_discovery.json").is_file() else run_operator_discovery();composition=read_json(HERE/"raw/composition_results.json") if (HERE/"raw/composition_results.json").is_file() else run_composition();payload={"version":"V837aq","stage":"AQ9_PREDICTIVE_CAUSAL_STATE","families":{},"non_gating":True,"state_definition":"history x future semantic intervention-response tests; no neural coordinates"}
    for family in discovery.get("accepted_families",[]):
        comp_pass=bool(composition.get("families",{}).get(family,{}).get("pass"));required=not comp_pass;hankel=_oracle_hankel(family);compact=bool(hankel["oracle_rank_99"]<=int(CONFIG["predictive_state"]["compact_rank_max"]))
        payload["families"][family]={"required":required,"reason":"operator identity exists but frozen composition gate failed" if required else "composition closes; predictive-state fallback not required","hankel":hankel,"compact_predictive_rank":compact,"history_state_is_neural_coordinate":False,"can_rescue_composition_failure":False}
    write_json(HERE/"raw/predictive_state_diagnostic.json",payload);write_json(HERE/"diagnostics/predictive_state.json",payload);return payload


if __name__=="__main__":
    import json;print(json.dumps(run_predictive_state_diagnostic(),indent=2))
