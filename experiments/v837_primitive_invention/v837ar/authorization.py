from __future__ import annotations

from .utils import HERE, ROOT, START_SHA, REQUIRED_BRANCH, LOCAL_BRANCH_ALIAS, branch_name, head_sha, sha256_json, write_json, read_json, git_blob_sha256
from .source_contracts import aq_import_snapshot, load_aq_contracts, imported_thresholds, EXPECTED_PROGRAM_CONTRACT_SHA
from .word_partitions import PARTITIONS
from .failure_ledger import initialize

GATE = HERE / "frozen_causal_operator_canonicalization_gate.json"


def freeze_gate() -> dict:
    if GATE.is_file(): return read_json(GATE)
    if head_sha()!=START_SHA: raise RuntimeError(f"V837AR_GATE_MUST_FREEZE_AT_START_SHA:{head_sha()}")
    if branch_name() not in {REQUIRED_BRANCH,LOCAL_BRANCH_ALIAS}: raise RuntimeError(f"V837AR_BAD_BRANCH:{branch_name()}")
    initialize(); aq=load_aq_contracts(); imp=aq_import_snapshot(); thr=imported_thresholds()
    gate={
        "version":"V837ar","program":"V837ar_CAUSAL_OPERATOR_CANONICALIZATION_AND_PROGRAM_IR","start_sha":START_SHA,
        "required_branch":REQUIRED_BRANCH,"local_branch_alias":LOCAL_BRANCH_ALIAS,
        "v837aq_results_hash":imp["aq_results_git_blob_sha256"],"v837aq_contract_hash":imp["aq_gate_git_blob_sha256"],
        "v837aq_program_contract_hash":EXPECTED_PROGRAM_CONTRACT_SHA,"v837aq_program_contract_blob_hash":imp["aq_program_contract_git_blob_sha256"],
        "confirmed_families":["conditional_routing","delayed_recall","iterative_state"],"variable_composition_status":"NEGATIVE_CONTROL_ONLY_DO_NOT_RESCUE",
        "aq_source_fold_hash":aq["folds"]["fold_sha256"],
        "canonical_response_ordering":["family","phase","intervention channel","magnitude","secondary intervention channel","secondary magnitude","response horizon","response semantic variable","interaction order"],
        "semantic_normalization_rules":"benchmark semantic units only; never AF1D activation magnitude",
        "IR0_reference_table":"complete equal-organism pooled empirical response object; upper-bound/non-generalizing; never archiveable",
        "iterative_grammar":["I0_CONSTANT_RESPONSE","I1_IDENTITY_STATE","I2_AFFINE_UPDATE","I3_QUADRATIC_UPDATE"],
        "routing_granularity":["R0_FINE","R1_TWO_STAGE","R2_ATOMIC"],"routing_order_ceiling":2,
        "memory_grammar":["M0_CONSTANT","M1_DELAY_TABLE","M2A_RANK1_SIMPLE","M2_RANK1_MEMORY"],"memory_gauge_rule":"READ gain r=1; absorb scale into WRITE gain",
        "operator_word_partitions":PARTITIONS,"shared_ir_equal_organism_weighting":True,"ridge_lambda":1e-6,
        "memory_lambda_bounds":[-1.25,1.25],"memory_lambda_grid_points":2049,"memory_lambda_fit":"deterministic grid then bounded Brent refinement",
        "program_ir_parameter_ceiling":256,"allowed_operator_classes":["LINEAR_STATE_UPDATE_1D","BILINEAR_STATIC_MAP","SECOND_ORDER_STATIC_MAP","LINEAR_PHASE_MACHINE_R1","EMPIRICAL_RESPONSE_TABLE"],
        "meta_rule":"one preselected candidate per family; no refit; no fallback after META",
        "final_unseen_rule":"open only after Program IR freeze; no grammar/coefficient changes; compare independently to organism and oracle",
        "reused_aq_heldout_rule":">=3/4 and both engines when N=4; exact frozen IR; no gain/bias/per-organism calibration",
        "aq_threshold_import":thr,"v837ar_specific_nrmse_control_margin":0.05,
        "failure_memory":"all failed grammar/granularity/order/rank/family/organism/word/compression/control candidates logged; DO NOT RETRY UNCHANGED",
        "fresh_audit_consumed":False,"primitive_archive_population":False,"primitives_promoted":0,"v838_started":False,
        "new_source_model_fits":0,"source_optimizer_steps":0,"source_training_examples":0,"source_architecture_changes":0,
        "neural_program_ir_forbidden":True,"cross_organism_state_alignment":False,"cross_organism_q_alignment":False,
    }
    gate["gate_sha256"]=sha256_json(gate); write_json(GATE,gate); return gate


def assert_authorized() -> dict:
    gate=freeze_gate(); aq=load_aq_contracts();
    if gate["start_sha"]!=START_SHA or gate["v837aq_program_contract_hash"]!=EXPECTED_PROGRAM_CONTRACT_SHA: raise RuntimeError("V837AR_AUTH_DRIFT")
    payload={"version":"V837ar","authorized":True,"start_sha":START_SHA,"branch":branch_name(),"source_integrity":True,
             "confirmed_families":gate["confirmed_families"],"fresh_audit_consumed":False,"primitive_archive_population":False,"primitives_promoted":0,
             "new_source_model_fits":0,"source_optimizer_steps":0,"v838_started":False}
    write_json(HERE/"diagnostics/authorization.json",payload); return payload
