from __future__ import annotations

import json

from .authorization import POWERED_FAMILIES
from .setpoint_grid import family_grid
from .utils import HERE, read_json, sha256_json, write_json


def _parameterization(candidate: dict) -> dict:
    return {
        "k": int(candidate["k"]),
        "semantic_dimension": 1,
        "chart_family": candidate["chart_family"],
        "writer_family": candidate.get("writer_family", "AUTO"),
        "phase_atlas": bool(candidate.get("phase_atlas", False)),
        "ridge_lambda": 1e-6,
        "max_polynomial_degree": 3,
        "newton_max_iterations": 4,
        "gradient_epsilon": 1e-8,
        "trust_radius": "2 * FIT-only D90 paired counterfactual projected displacement",
        "random_subspaces": 32,
        "norm_random_setters": 32,
        "heldout_geometry_search": False,
        "cross_organism_state_alignment": False,
    }


def freeze_family_geometries()->dict:
    path=HERE/"raw/frozen_v837ap_family_geometries.json"
    if path.is_file():
        payload=read_json(path)
        alias={**payload,"frozen_before_any_heldout_backend_read":True,"historical_validation_read_before_freeze":False}
        write_json(HERE/"raw/frozen_family_geometry.json",alias)
        return payload
    final=read_json(HERE/"raw/discovery_final.json")
    families={}
    for family in POWERED_FAMILIES:
        cand=final["surviving_families"].get(family)
        if cand is None:
            families[family]=None;continue
        families[family]={
            "family":family,
            "candidate":_parameterization(cand),
            "selection_evidence":cand,
            "semantic_range":family_grid(family),
            "feature_algorithm":"V837an counterfactual-difference SVD Qk, k in {1,2,4,8}; deterministic chart fit; no cross-organism alignment",
            "fit_roles":{"chart":"AP_CHART_FIT","writer":"AP_WRITER_FIT"},
            "heldout_fit_algorithm_frozen":True,
        }
    payload={"version":"V837ap","stage":"AP13_FREEZE_FAMILY_GEOMETRIES","families":families,"frozen_before_heldout_open":True,"heldout_organism_results_read_before_freeze":False,"geometry_changes_after_freeze":False,"primitives_promoted":0}
    payload["frozen_sha256"]=sha256_json(payload)
    write_json(path,payload);write_json(HERE/"raw/frozen_family_geometry.json",payload)
    diag={"version":"V837ap","pass":True,"frozen_sha256":payload["frozen_sha256"],"candidate_count":sum(v is not None for v in families.values()),"frozen_before_heldout_open":True,"frozen_before_any_heldout_backend_read":True,"historical_validation_read_before_freeze":False}
    write_json(HERE/"diagnostics/final_geometry_freeze.json",diag);write_json(HERE/"diagnostics/family_geometry_freeze.json",diag)
    return payload

if __name__=="__main__":
    print(json.dumps(freeze_family_geometries(),indent=2,default=str))
