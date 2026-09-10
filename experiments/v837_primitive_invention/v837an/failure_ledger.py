from __future__ import annotations

from typing import Any

from .authorization import SOURCE
from .utils import HERE, ROOT, START_SHA, git_blob_sha256, now_stamps, read_json, write_json

RAW = HERE / "raw/failure_ledger.json"
DIAG = HERE / "diagnostics/failure_ledger.json"
CENTRAL = ROOT / "docs/V837_FAILURE_LEDGER.md"

REQUIRED = (
    "failure_id", "version", "stage", "timestamp_utc", "timestamp_local", "source_commit", "source_artifact_hashes",
    "scientific_question", "hypothesis", "why_it_was_tested", "exact_implementation", "data_used", "fit_seeds", "selection_seeds",
    "controls", "parameter_count", "mac_cost", "measured_metrics", "predeclared_acceptance_gate", "which_exact_gate_failed",
    "how_far_from_gate", "result_status", "failure_type", "confounds_ruled_out", "confounds_still_possible", "scientific_interpretation",
    "do_not_repeat_unchanged", "uncertainty_remaining", "next_justified_experiment", "reproduction_command", "artifact_paths",
)


def _empty() -> dict:
    return {"version": "V837an", "append_only": True, "entries": []}


def load() -> dict:
    return read_json(RAW) if RAW.is_file() else _empty()


def _sync(payload: dict) -> None:
    write_json(RAW, payload)
    write_json(DIAG, payload)


def add(entry: dict, *, append_central: bool = True) -> dict:
    missing = [key for key in REQUIRED if key not in entry]
    if missing:
        raise RuntimeError(f"V837AN_FAILURE_LEDGER_ENTRY_INCOMPLETE:{missing}")
    payload = load()
    if any(existing["failure_id"] == entry["failure_id"] for existing in payload["entries"]):
        return payload
    payload["entries"].append(entry)
    _sync(payload)
    if append_central:
        CENTRAL.parent.mkdir(parents=True, exist_ok=True)
        with CENTRAL.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"## {entry['failure_id']} — {entry['version']} / {entry['stage']}\n\n")
            handle.write(f"- **Hypothesis:** {entry['hypothesis']}\n")
            handle.write(f"- **Result:** {entry['result_status']} / {entry['failure_type']}\n")
            handle.write(f"- **Failed gate:** {', '.join(entry['which_exact_gate_failed']) if entry['which_exact_gate_failed'] else 'N/A'}\n")
            handle.write(f"- **Meaning:** {entry['scientific_interpretation']}\n")
            handle.write(f"- **Do not repeat unchanged:** {entry['do_not_repeat_unchanged']}\n")
            handle.write(f"- **Source commit:** `{entry['source_commit']}`\n")
            handle.write(f"- **Artifacts:** {', '.join(entry['artifact_paths'])}\n\n")
    return payload


def make_entry(
    *, failure_id: str, stage: str, hypothesis: str, why: str, implementation: Any, data: Any,
    fit_seeds: Any, selection_seeds: Any, controls: Any, parameter_count: Any, mac_cost: Any,
    metrics: dict, gate: Any, failed: list[str], distance: Any, result_status: str, failure_type: str,
    confounds_ruled_out: Any, confounds_remaining: Any, meaning: str, uncertainty: Any, next_experiment: Any,
    reproduce: str, artifacts: list[str], source_hashes: dict | None = None,
) -> dict:
    utc, local = now_stamps()
    return {
        "failure_id": failure_id,
        "version": "V837an",
        "stage": stage,
        "timestamp_utc": utc,
        "timestamp_local": local,
        "source_commit": START_SHA,
        "source_artifact_hashes": source_hashes or {},
        "scientific_question": "What is the lowest-dimensional interventionally faithful causal computation implemented across independently trained AF1D organisms, and how is it routed?",
        "hypothesis": hypothesis,
        "why_it_was_tested": why,
        "exact_implementation": implementation,
        "data_used": data,
        "fit_seeds": fit_seeds,
        "selection_seeds": selection_seeds,
        "controls": controls,
        "parameter_count": parameter_count,
        "mac_cost": mac_cost,
        "measured_metrics": metrics,
        "predeclared_acceptance_gate": gate,
        "which_exact_gate_failed": failed,
        "how_far_from_gate": distance,
        "result_status": result_status,
        "failure_type": failure_type,
        "confounds_ruled_out": confounds_ruled_out,
        "confounds_still_possible": confounds_remaining,
        "scientific_interpretation": meaning,
        "do_not_repeat_unchanged": True,
        "uncertainty_remaining": uncertainty,
        "next_justified_experiment": next_experiment,
        "reproduction_command": reproduce,
        "artifact_paths": artifacts,
    }


def initialize_historical_refinements() -> dict:
    if not RAW.is_file():
        _sync(_empty())
    hashes = {key: git_blob_sha256(rel) for key, rel in SOURCE.items()}
    add(make_entry(
        failure_id="REF-AN-001", stage="HISTORICAL_INTERPRETATION_REFINEMENT",
        hypothesis="V837am AM-A provided valid evidence for all 126 tested context-conditioned interface configurations.",
        why="The V837an forensic handoff identified that state-map invertibility/conditioning was a prerequisite failure for most AM-A configurations.",
        implementation={"historical_program": "V837am", "branch": "AM-A", "total_configs": 126, "invalid_config_count": 105, "reason": "dynamic-state invertibility/conditioning gate"},
        data="Committed V837am AM-A selection artifact; no historical result rewritten.", fit_seeds=[10000, 10063], selection_seeds=[10064, 10127],
        controls=["historical artifact audit"], parameter_count=0, mac_cost=0,
        metrics={"total_configs": 126, "invalid_config_count": 105, "valid_config_count": 21}, gate={"historical_interpretation_must_distinguish_invalid_from_negative": True},
        failed=["105/126 configurations invalid before scientific effect gate"], distance={"invalid_fraction": 105 / 126},
        result_status="interpretation refinement; historical diagnosis preserved", failure_type="INTERPRETATION_REFINEMENT",
        confounds_ruled_out=["silent reinterpretation as 126 definitive negative tests"], confounds_remaining=["whether noninvertible many-to-one causal abstractions exist"],
        meaning="V837am remains valid within its declared interface gate, but most AM-A configurations cannot be cited as direct evidence against noninvertible causal macrovariables.",
        uncertainty="The correct causal granularity was not tested.", next_experiment="V837an many-to-one causal macrovariable interventions without state bijection.",
        reproduce="python scripts/reproduce_v837_recovery.py --variant v837an --stage source --execute",
        artifacts=["experiments/v837_primitive_invention/v837am/raw/am_a_selection.json", "experiments/v837_primitive_invention/v837an/diagnostics/historical_interpretation_refinement.json"], source_hashes=hashes,
    ))
    add(make_entry(
        failure_id="REF-AN-002", stage="HISTORICAL_INTERPRETATION_REFINEMENT",
        hypothesis="V837am AM-C temporal/history candidates obtained valid intervention pairs and then failed the temporal effect gate.",
        why="The committed AM-C artifact shows every temporal and boundary×context candidate had zero valid evaluated pairs.",
        implementation={"historical_program": "V837am", "branch": "AM-C", "configs": 7, "causal_pairs": 38},
        data="Committed V837am AM-C selection artifact; no historical result rewritten.", fit_seeds=[10256, 10319], selection_seeds=[10320, 10383], controls=["historical artifact audit"],
        parameter_count=0, mac_cost=0, metrics={"configs": 7, "row_count_each": 0, "invalid_pairs_each": 38},
        gate={"valid_intervention_pairs_required_before_temporal_effect_claim": True}, failed=["0 valid evaluated pairs for all 7 AM-C configurations"],
        distance={"valid_pairs_short_of_one": 1}, result_status="interpretation refinement; historical diagnosis preserved", failure_type="INTERPRETATION_REFINEMENT",
        confounds_ruled_out=["claim that AM-C produced a powered direct temporal negative"], confounds_remaining=["temporal causal operators without invertible state interface"],
        meaning="The tested temporal-interface formulation could not satisfy its prerequisite state-map validity gate; temporal causal operators were not directly disproven.",
        uncertainty="Whether a noninvertible temporal causal abstraction exists.", next_experiment="Test within-organism causal macrovariables and routing without cross-organism state inversion.",
        reproduce="python scripts/reproduce_v837_recovery.py --variant v837an --stage source --execute",
        artifacts=["experiments/v837_primitive_invention/v837am/raw/am_c_selection.json", "experiments/v837_primitive_invention/v837an/diagnostics/historical_interpretation_refinement.json"], source_hashes=hashes,
    ))
    add(make_entry(
        failure_id="REF-AN-003", stage="HISTORICAL_INTERPRETATION_REFINEMENT",
        hypothesis="V837am's diagnosis can be restated as definitive disproof of temporal causal operators or all shared internal causal abstractions.",
        why="V837an must preserve the exact logical scope of the predecessor evidence before changing primitive level.",
        implementation={"preserved_diagnosis": "DYNAMICAL_RECURRENCE_NOT_OPERATOR_EQUIVALENCE", "forbidden_overstatement": "temporal causal operators definitively disproven"},
        data="Committed V837am decision plus AM-A/AM-C validity diagnostics.", fit_seeds=[], selection_seeds=[], controls=["logical-scope audit"], parameter_count=0, mac_cost=0,
        metrics={"historical_diagnosis_preserved": True, "historical_files_modified": False}, gate={"historical_results_must_not_be_rewritten": True},
        failed=[], distance={"scope_refinement_only": True}, result_status="append-only interpretation refinement", failure_type="INTERPRETATION_REFINEMENT",
        confounds_ruled_out=["rewriting V837am's machine diagnosis"], confounds_remaining=["causal level at which common computation exists"],
        meaning="V837am remains the authority for failure of its tested operator/interface formulation; V837an asks a different question that removes full-state invertibility as an assumption.",
        uncertainty="Whether common causal macrovariables, routing, coalitions, or only behavioral programs are shared.", next_experiment="Run V837an.",
        reproduce="python scripts/reproduce_v837_recovery.py --variant v837an --stage source --execute",
        artifacts=["experiments/v837_primitive_invention/v837am/diagnostics/decision_state.json", "docs/V837_FAILURE_LEDGER.md"], source_hashes=hashes,
    ))
    return load()


if __name__ == "__main__":
    import json
    print(json.dumps(initialize_historical_refinements(), indent=2))
