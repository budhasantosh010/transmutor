from __future__ import annotations

from .authorization import POWERED_FAMILIES, assert_authorized
from .canonical_ir import canonical_ir
from .discovery_final import run_discovery_final
from .setpoint_grid import family_grid
from .utils import HERE, read_json, sha256_json, write_json


def _geometry_spec(family:str,confirmed:dict)->dict:
    cand=confirmed["candidate"];ir=canonical_ir(family).to_dict();grid=family_grid(family)
    return {
        "family":family,
        "canonical_ir":ir,
        "semantic_variable":ir["semantic_variable"],
        "semantic_dimension":1,
        "k":int(cand["k"]),
        "chart_family":cand["chart_family"],
        "writer_family":cand.get("writer_family","AUTO"),
        "phase_atlas":bool(cand.get("phase_atlas",False)),
        "subspace_algorithm":"paired counterfactual-difference SVD on STATE40, nested K in {1,2,4,8}; no invertibility and no cross-organism alignment",
        "chart_fit_algorithm":"frozen deterministic scalar/projected chart family; ridge lambda=1e-6 where applicable; chart fit only on calibration episodes",
        "writer_fit_algorithm":"direct inverse/prototype, gradient/Newton, or frozen tangent-field family exactly as discovery selected; no source/model gradients",
        "ridge_lambda":1e-6,
        "setpoint_grid":grid["targets"],
        "semantic_range":grid,
        "reader_gate":{"continuous_nrmse":.10,"continuous_abs_pearson":.95,"continuous_spearman":.95,"binary_nrmse":.15,"binary_accuracy":.95,"binary_balanced_accuracy":.95},
        "set_gate":{"continuous_read_after_set_nrmse":.05,"binary_read_after_set_nrmse":.10,"median_recovery":.70,"direction":.80,"task_success":.75,"trajectory_nrmse":.15,"continuous_targets_passing":6,"binary_targets_passing":2,"ood":2.0,"control_margin":.20,"paired_p":.01},
        "quotient_gate":{"median_residual_sensitivity":.20,"p90_residual_sensitivity":.40,"trajectory_disagreement":.10,"final_prediction_disagreement":.10,"task_success_disagreement":.10,"ood":2.0},
        "dynamics_gate":{"one_step_nrmse":.10,"median_abs":.075,"direction_consistency":.90,"multi_step_nrmse":.15,"final_recovery":.80,"task_success":.75,"control_margin":.20,"ood":2.0},
        "discovery_confirmation":confirmed,
        "complexity_order":"smallest k; simpler chart; simpler writer; fewer phase-specific components; lower parameter_count; lower stored bytes; lower MACs",
        "heldout_compilation":"reconstruct this exact geometry from source checkpoint plus N calibration episodes only; no family/branch/k/degree search",
    }


def freeze_family_geometry()->dict:
    path=HERE/"raw/frozen_family_geometry.json"
    if path.is_file():return read_json(path)
    assert_authorized();p=HERE/"raw/discovery_final_confirmation.json";final=read_json(p) if p.is_file() else run_discovery_final();families={}
    for family in POWERED_FAMILIES:
        confirmed=final["confirmed_families"].get(family);families[family]=None if confirmed is None else _geometry_spec(family,confirmed)
    payload={"version":"V837ap","stage":"AP13_FREEZE","families":families,"frozen_before_any_heldout_backend_read":True,"heldout_backend_evidence_read_before_freeze":False,"historical_validation_read_before_freeze":False,"fresh_audit_consumed":False,"primitive_archive_allowed":False,"primitives_promoted":0,"v838_started":False}
    payload["frozen_sha256"]=sha256_json(payload);write_json(path,payload);write_json(HERE/"diagnostics/family_geometry_freeze.json",{"version":"V837ap","pass":True,"frozen_sha256":payload["frozen_sha256"],"candidate_count":sum(v is not None for v in families.values()),"heldout_backend_evidence_read_before_freeze":False});return payload

if __name__=="__main__":
    import json;print(json.dumps(freeze_family_geometry(),indent=2,default=str))
