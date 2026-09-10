from __future__ import annotations

import json
from .authorization import POWERED_FAMILIES, V837AO_PATHS, V837AN_FROZEN, assert_authorized
from .data_roles import PARTITIONS, assert_roles
from .failure_ledger import add, initialize, make_entry
from .source_folds import freeze_source_folds
from .utils import HERE, ROOT, git_blob_sha256, read_json, write_json

BACKFILL_ID="V837ap-BACKFILL-V837ao-CAUSAL-DIRECTION-NOT-GLOBAL-COORDINATE"

def _backfill_v837ao()->None:
    add(make_entry(failure_id=BACKFILL_ID,stage="AP0_PREDECESSOR_BACKFILL",branch="V837ao",family=None,organism=None,phase=None,carrier_dimension=1,chart_family="AFFINE_K1",chart_degree_rank=1,writer_family="V837ao gauge-fixed semantic compiler",fit_partition="V837ao AO_BACKEND_FIT",selection_partition="V837ao AO_BACKEND_SELECT",parameter_count=None,stored_bytes=None,mac_estimate=None,metrics={"diagnosis":"CAUSAL_DIRECTION_NOT_GLOBAL_COORDINATE","powered_families":4,"absolute_coordinate_families":0,"canonical_families":0,"quotient_families":0,"dynamic_families":0,"heldout_backend_fits":0,"v837an_k1_causal_steering_remains_valid":True,"null_family_specs_frozen_before_heldout":True},acceptance_gate="historical interpretation only",failed_conditions=["linear/global K1 coordinate not established"],scientific_interpretation="V837ao showed that V837an K1 is a genuine causal steering direction but not a stable affine global semantic coordinate under its frozen family. Null family specifications were deliberately frozen before any heldout backend fitting; this was not a missing run.",confounds_ruled_out=["heldout rescue","fresh-audit leakage","V837an causal steering invalidation"],confounds_remaining=["nonlinear scalar chart","state-dependent tangent","K2/K4/K8 projected chart","phase atlas"],artifact_paths=[V837AO_PATHS["decision"],V837AO_PATHS["report"]]))

def verify_source_integrity()->dict:
    initialize(); auth=assert_authorized(); assert_roles(); folds=freeze_source_folds(); _backfill_v837ao()
    ao=read_json(ROOT/V837AO_PATHS["decision"]); an=read_json(ROOT/"experiments/v837_primitive_invention/v837an/diagnostics/decision_state.json")
    payload={"version":"V837ap","pass":True,"authorization":auth,"source_integrity":True,"v837ao_diagnosis":ao["diagnosis"],"v837ao_next_program":ao["next_program"],"v837an_diagnosis":an["diagnosis"],"v837an_causal_steering_preserved":True,"powered_families":list(POWERED_FAMILIES),"powered_family_count":4,"partial_observation_status":"UNRESOLVED_UNDERPOWERED_PREDECESSOR_FAMILY","v837ao_fold_sha256":folds["v837ao_fold_sha256"],"data_roles":PARTITIONS,"fresh_audit_consumed":False,"primitives_promoted":0,"v838_started":False,"protected_hashes":{"v837ao_decision":git_blob_sha256(V837AO_PATHS["decision"]),"v837ao_report":git_blob_sha256(V837AO_PATHS["report"]),"v837an_frozen":git_blob_sha256(V837AN_FROZEN)}}
    write_json(HERE/"raw/source_state.json",payload); write_json(HERE/"diagnostics/source_integrity.json",payload)
    write_json(HERE/"raw/data_role_lock.json",{"version":"V837ap","roles":PARTITIONS,"within_program_roles_not_globally_fresh":True,"primary_new_generalization_dimension":"heldout organisms","fresh_audit_unused":True})
    write_json(HERE/"diagnostics/data_role_integrity.json",{"version":"V837ap","roles":PARTITIONS,"development_roles_disjoint":True,"fresh_audit_unused":True,"pass":True})
    return payload

if __name__=="__main__": print(json.dumps(verify_source_integrity(),indent=2))
