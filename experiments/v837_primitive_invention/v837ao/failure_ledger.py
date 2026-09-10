from __future__ import annotations

from typing import Any

from .utils import HERE, ROOT, START_SHA, now_stamps, read_json, write_json

RAW = HERE / "raw/failure_ledger.json"
DIAG = HERE / "diagnostics/failure_ledger.json"
CENTRAL = ROOT / "docs/V837_FAILURE_LEDGER.md"

FAILURE_CODES = {
    "AO_SOURCE_INTEGRITY_FAILURE", "AO_ORGANISM_SPLIT_INVALID", "K1_READER_NOT_GLOBAL",
    "READER_WRITER_GAIN_DEGENERATE", "CANONICAL_BACKEND_IMPLEMENTATION_INVALID",
    "ABSOLUTE_SETPOINT_FAIL", "LOCAL_DIRECTION_ONLY", "RANDOM_CONTROL_NOT_SEPARATED",
    "GLOBAL_BACKEND_FAIL", "PHASE_GAUGE_FAIL", "PHASE_K1_FAIL",
    "QUOTIENT_RESIDUAL_SENSITIVITY", "RESIDUAL_SWAP_OOD",
    "CANONICAL_DYNAMICS_ONE_STEP_FAIL", "CANONICAL_DYNAMICS_ROLLOUT_FAIL",
    "META_CONFIRM_FAIL", "HELDOUT_BACKEND_FAIL", "HELDOUT_CALIBRATION_64_INSUFFICIENT",
    "CROSS_ORGANISM_CANONICAL_DISAGREEMENT", "HISTORICAL_ROBUSTNESS_CONTRADICTION",
}
REQUIRED = (
    "failure_id", "version", "stage", "timestamp_utc", "timestamp_local", "source_sha",
    "failure_code", "failure_type", "result_status", "family", "organism_id", "backend_variant",
    "phase", "setpoint", "calibration_budget", "scientific_question", "hypothesis", "why_it_was_tested",
    "exact_configuration", "data_used", "metrics", "predeclared_gate", "failed_conditions",
    "distance_to_gate", "confounds_ruled_out", "remaining_uncertainty", "do_not_repeat_unchanged",
    "next_justified_experiment", "reproduction_command", "artifact_paths", "artifact_hashes",
)


def _empty() -> dict:
    return {"version": "V837ao", "append_only": True, "entries": []}


def initialize_failure_memory() -> dict:
    payload = read_json(RAW) if RAW.is_file() else _empty()
    if not RAW.is_file():
        write_json(RAW, payload)
        write_json(DIAG, payload)
    elif not DIAG.is_file():
        write_json(DIAG, payload)
    if not (HERE / "FAILURE_ANALYSIS.md").is_file():
        raise RuntimeError("AO_FAILURE_ANALYSIS_MUST_EXIST_BEFORE_RUN")
    return payload


def add(entry: dict, *, append_central: bool = True) -> dict:
    missing = [k for k in REQUIRED if k not in entry]
    if missing:
        raise RuntimeError(f"AO_FAILURE_LEDGER_ENTRY_INCOMPLETE:{missing}")
    payload = initialize_failure_memory()
    if any(e["failure_id"] == entry["failure_id"] for e in payload["entries"]):
        return payload
    payload["entries"].append(entry)
    write_json(RAW, payload); write_json(DIAG, payload)
    if append_central:
        CENTRAL.parent.mkdir(parents=True, exist_ok=True)
        with CENTRAL.open("a", encoding="utf-8", newline="\n") as h:
            h.write(f"## {entry['failure_id']} — V837ao / {entry['stage']}\n\n")
            h.write(f"- **Code:** {entry['failure_code']}\n")
            h.write(f"- **Type:** {entry['failure_type']} / {entry['result_status']}\n")
            h.write(f"- **Family / organism:** {entry['family']} / {entry['organism_id']}\n")
            h.write(f"- **Backend / phase / calibration:** {entry['backend_variant']} / {entry['phase']} / {entry['calibration_budget']}\n")
            h.write(f"- **Failed gate:** {entry['failed_conditions']}\n")
            h.write(f"- **Meaning:** {entry['remaining_uncertainty']}\n")
            h.write(f"- **Do not repeat unchanged:** {entry['do_not_repeat_unchanged']}\n")
            h.write(f"- **Artifacts:** {', '.join(entry['artifact_paths'])}\n\n")
    return payload


def make_entry(*, failure_id: str, stage: str, failure_code: str, failure_type: str,
               result_status: str, family: str | None = None, organism_id: str | None = None,
               backend_variant: str | None = None, phase: str | None = None, setpoint: Any = None,
               calibration_budget: int | None = None, hypothesis: str, why: str, configuration: Any,
               data: Any, metrics: Any, gate: Any, failed: Any, distance: Any,
               confounds_ruled_out: Any, uncertainty: Any, next_experiment: Any,
               reproduce: str, artifacts: list[str], artifact_hashes: dict | None = None) -> dict:
    if failure_code not in FAILURE_CODES and failure_type == "SCIENTIFIC_FAILURE":
        raise ValueError(f"unknown V837ao scientific failure code: {failure_code}")
    utc, local = now_stamps()
    return {
        "failure_id": failure_id, "version": "V837ao", "stage": stage,
        "timestamp_utc": utc, "timestamp_local": local, "source_sha": START_SHA,
        "failure_code": failure_code, "failure_type": failure_type, "result_status": result_status,
        "family": family, "organism_id": organism_id, "backend_variant": backend_variant,
        "phase": phase, "setpoint": setpoint, "calibration_budget": calibration_budget,
        "scientific_question": "Do V837an STATE40-K1 abstractions define canonical semantic state variables with stable SET, quotient, dynamics, and backend generalization?",
        "hypothesis": hypothesis, "why_it_was_tested": why, "exact_configuration": configuration,
        "data_used": data, "metrics": metrics, "predeclared_gate": gate, "failed_conditions": failed,
        "distance_to_gate": distance, "confounds_ruled_out": confounds_ruled_out,
        "remaining_uncertainty": uncertainty, "do_not_repeat_unchanged": True,
        "next_justified_experiment": next_experiment, "reproduction_command": reproduce,
        "artifact_paths": artifacts, "artifact_hashes": artifact_hashes or {},
    }
