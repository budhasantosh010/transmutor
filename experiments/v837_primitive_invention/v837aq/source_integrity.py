from __future__ import annotations

import json
from collections import Counter

from .authorization import POWERED_FAMILIES, assert_authorized, freeze_gate
from .data_roles import PARTITIONS, assert_roles
from .failure_ledger import add, initialize, make_entry
from .natural_interventions import intervention_library_manifest
from .source_folds import freeze_source_folds
from .utils import HERE, ROOT, read_json, write_json

RECON = ROOT / "experiments/v837_primitive_invention/v837ak/raw/reconstruction_results.json"


def verify_source_integrity() -> dict:
    initialize()
    gate = freeze_gate()
    auth = assert_authorized()
    assert_roles()
    folds = freeze_source_folds()
    pop = read_json(RECON)["rows"]
    competent = [r for r in pop if r.get("competent")]
    incompetent = [r for r in pop if not r.get("competent")]
    compatible_incompetent = [r for r in incompetent if r["family"] in POWERED_FAMILIES]
    excluded_partial = [r for r in incompetent if r["family"] == "partial_observation"]
    ap_fail = read_json(ROOT / "experiments/v837_primitive_invention/v837ap/raw/failure_ledger.json")
    engineering = [e for e in ap_fail.get("entries", []) if e.get("failure_type") == "ENGINEERING_FAILURE"]
    scientific = [e for e in ap_fail.get("entries", []) if e.get("failure_type") == "SCIENTIFIC_FAILURE"]
    backfill = make_entry(
        failure_id="V837aq-BACKFILL-V837ap-NO-CAUSAL-SET-CLOSED-GEOMETRY", stage="AQ0_PREDECESSOR_FORENSICS", branch="V837ap",
        metrics={"v837ap_diagnosis": "DECODABLE_LOW_DIMENSIONAL_STATE_NOT_CAUSALLY_CLOSED", "reader_candidate_rows": 293, "setpoint_rows": 30, "quotient_rows": 0, "commutativity_rows": 0, "heldout_rows": 0, "engineering_failure_entries": len(engineering), "scientific_failure_entries": len(scientific)},
        acceptance_gate="historical interpretation only",
        failed_conditions=["no powered family satisfied the complete valid causal SET/control gate"],
        scientific_interpretation="V837ap established that some low-dimensional projected representations decode semantics but no powered family survived its complete valid causal SET/control gate. Quotient, dynamical commutativity, and heldout failures must not be claimed because those stages received zero candidates. Engineering-invalid configurations remain non-scientific rows.",
        confounds_ruled_out=["claiming quotient failure without rows", "claiming commutativity failure without rows", "converting engineering-invalid rows into scientific negatives"],
        confounds_remaining=["coordinate-free trajectory operator", "spatiotemporal causal process", "predictive causal state"],
        next_justified_experiment="Run V837aq natural semantic intervention operator reality gate.",
        artifact_paths=["experiments/v837_primitive_invention/v837ap/diagnostics/decision_state.json", "experiments/v837_primitive_invention/v837ap/raw/failure_ledger.json"],
    )
    add(backfill)
    negative_registry = {
        "version": "V837aq", "historically_incompetent_total": len(incompetent),
        "powered_family_compatible": len(compatible_incompetent), "partial_observation_excluded_primary": len(excluded_partial),
        "rows": [{"organism_id": r["organism_id"], "family": r["family"], "engine": r["engine"], "final_validation_success": r.get("final_validation_success"), "primary_operator_control_eligible": r["family"] in POWERED_FAMILIES} for r in incompetent],
        "interpretation": "All ten historical incompetents are preserved as negative-control evidence; only family-compatible organisms can receive a powered-family oracle operator test without inventing a partial-observation operator contract.",
    }
    write_json(HERE / "raw/incompetent_control_registry.json", negative_registry)
    write_json(HERE / "raw/natural_intervention_library.json", intervention_library_manifest())
    payload = {
        "version": "V837aq", "pass": True, "authorization": auth, "gate_sha256": gate["gate_sha256"],
        "source_population": {"total": len(pop), "competent": len(competent), "incompetent": len(incompetent)},
        "powered_families": list(POWERED_FAMILIES), "fold_sha256": folds["fold_sha256"], "data_roles": PARTITIONS,
        "v837ap_engineering_failures_preserved": len(engineering), "v837ap_scientific_failures_preserved": len(scientific),
        "v837ap_quotient_rows": 0, "v837ap_commutativity_rows": 0, "v837ap_heldout_rows": 0,
        "fresh_audit_consumed": False, "primitive_archive_population": False, "primitives_promoted": 0, "v838_started": False,
    }
    write_json(HERE / "raw/source_state.json", payload)
    write_json(HERE / "diagnostics/source_integrity.json", payload)
    write_json(HERE / "raw/data_role_lock.json", {"version": "V837aq", "roles": PARTITIONS, "development_roles_disjoint": True, "fresh_audit_unused": True})
    write_json(HERE / "diagnostics/data_role_integrity.json", {"version": "V837aq", "pass": True, "roles": PARTITIONS, "fresh_audit_unused": True})
    return payload


if __name__ == "__main__":
    print(json.dumps(verify_source_integrity(), indent=2))
