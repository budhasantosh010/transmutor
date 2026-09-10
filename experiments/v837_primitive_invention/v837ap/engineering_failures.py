from __future__ import annotations

from .failure_ledger import add, make_entry, sync_central_ledger


FIXED = (
    (
        "V837ap-ENG-WORKTREE-REBIND",
        "AP0_ENGINEERING",
        "WORKTREE_REBIND",
        "Initial harness task execution landed V837ap scaffolding in the canonical checkout instead of the isolated research worktree.",
        "Rebound the task to the isolated V837ap worktree before predecessor files were modified; copied only new V837ap scaffolding and restored branch discipline.",
    ),
    (
        "V837ap-ENG-K1-ARRAY-ORIENTATION",
        "AP3_ENGINEERING",
        "K1_ARRAY_ORIENTATION",
        "The first K1 affine/logistic chart sweep treated a length-N scalar vector as one row with N features, causing invalid fits.",
        "Changed 1D scalar chart fit/predict inputs to deterministic [N,1] orientation and reran only K1 rows; K2/K4/K8 rows were retained.",
    ),
    (
        "V837ap-ENG-DYNAMICS-SEED-THREADING",
        "AP9_ENGINEERING",
        "DYNAMICS_SEED_THREADING",
        "Held-out chart/writer calibration seed overrides were referenced by interventional dynamics before being threaded through its function signature.",
        "Added explicit chart_fit_seeds and writer_fit_seeds parameters before any held-out dynamics execution.",
    ),
    (
        "V837ap-ENG-QUOTIENT-TASK-TARGET",
        "AP8_ENGINEERING",
        "QUOTIENT_TASK_TARGET_NORMALIZATION",
        "Early quotient code compared task success against the semantic setpoint rather than the abstract final task target for routing/recall.",
        "Normalized quotient effect and task-success checks to the frozen abstract rollout target before quotient science execution.",
    ),
    (
        "V837ap-ENG-COMPLEXITY-ORDER",
        "AP10_ENGINEERING",
        "COMPLEXITY_ORDERING",
        "One unused discovery runner version placed projected K2/K4/K8 evaluation before the AP-C tangent branch.",
        "Corrected hierarchy to K1 simple nonlinear -> K1+tangent -> K2 -> K4 -> K8 -> phase atlas; canonical run_pipeline uses the corrected AP7 selector.",
    ),
    (
        "V837ap-ENG-ZERO-DELTA-IDENTITY",
        "AP7_ENGINEERING",
        "ZERO_DELTA_IDENTITY",
        "The direct 1D inverse setter reported one operation even when the requested semantic target already equaled the current readout.",
        "Added an explicit normalized zero-delta identity guard before inverse/prototype/Newton dispatch.",
    ),
    (
        "V837ap-ENG-CONTROL-VALIDITY-GATE",
        "AP7_ENGINEERING",
        "CONTROL_VALIDITY_GATE",
        "An intermediate SET evaluator encoded failed shuffled/random-subspace control fits as -inf recovery, which could create an artificial infinite candidate margin.",
        "Require a valid shuffled-semantic control and exactly 32 valid random-subspace controls before any AP7 candidate can pass; discard the earlier AP7 run and rerun from frozen reader artifacts.",
    ),
    (
        "V837ap-ENG-HELDOUT-SUBSPACE-RANK-GUARD",
        "AP14_ENGINEERING",
        "HELDOUT_SUBSPACE_RANK_GUARD",
        "Held-out compilation initially checked chart coefficient sample count without an explicit minimum sample guard for reconstructing the frozen k-dimensional projected subspace.",
        "Reject underdetermined calibration budgets before held-out chart evaluation when permitted calibration traces cannot support the frozen projected dimension.",
    ),
    (
        "V837ap-ENG-HELDOUT-TIME-IMPORT",
        "AP14_ENGINEERING",
        "HELDOUT_TIME_IMPORT",
        "The null held-out closure path referenced time.perf_counter for resource accounting before importing time and initializing the stage timer.",
        "Imported time, initialized the stage timer at held-out-program entry, and reran the null held-out closure; no held-out backend science had executed before either failure.",
    ),
    (
        "V837ap-ENG-ACTIVE-VALIDATOR-CSV-NEWLINES",
        "CLOSEOUT_ENGINEERING",
        "ACTIVE_VALIDATOR_CSV_NEWLINE_PORTABILITY",
        "The V836 integrity manifest froze registry/experiments.csv using CRLF bytes while Git stores the same committed CSV content with LF bytes, causing the active validator to fail on this worktree.",
        "Made only the CSV manifest comparison newline-portable by hashing the CRLF-normalized form against the unchanged frozen SHA; all non-CSV preserved artifacts remain exact-byte checks.",
    ),
)


def record_known_engineering_failures() -> dict:
    for failure_id, stage, code, interpretation, fix in FIXED:
        add(make_entry(
            failure_id=failure_id,
            stage=stage,
            branch="ENGINEERING_REPAIR",
            family=None,
            organism=None,
            phase=None,
            carrier_dimension=None,
            chart_family=None,
            chart_degree_rank=None,
            writer_family=None,
            fit_partition=None,
            selection_partition=None,
            parameter_count=0,
            stored_bytes=0,
            mac_estimate=0,
            metrics={"engineering_code": code, "science_contaminated": False},
            acceptance_gate="Implementation correctness; no affected result may be interpreted scientifically until repaired.",
            failed_conditions=[code],
            distance_from_threshold={"not_applicable": True},
            scientific_interpretation=interpretation,
            confounds_ruled_out=["scientific threshold tuning", "fresh-audit use", "source retraining"],
            confounds_remaining=[],
            result_status="FIXED_BEFORE_RELEVANT_SCIENTIFIC_DECISION",
            next_justified_experiment=fix,
            reproduction_command="python -m pytest tests/test_v837ap_global_or_nonlinear_causal_state.py -q",
            artifact_paths=["experiments/v837_primitive_invention/v837ap/raw/failure_ledger.json"],
            failure_type="ENGINEERING_FAILURE",
        ), append_central=False)
    synced = sync_central_ledger()
    return {"version":"V837ap","engineering_failures_recorded":len(FIXED),"central_ledger":synced}


if __name__ == "__main__":
    import json
    print(json.dumps(record_known_engineering_failures(), indent=2))
