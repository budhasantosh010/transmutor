from __future__ import annotations

from typing import Any
from .utils import HERE, ROOT, START_SHA, git_blob_bytes, now_stamps, sha256_json, write_json, read_json

RAW = HERE / "raw/failure_ledger.json"
DIAG = HERE / "diagnostics/failure_ledger.json"
CENTRAL = ROOT / "docs/V837_FAILURE_LEDGER.md"
REQUIRED = (
    "failure_id", "version", "stage", "branch", "family", "organism", "intervention", "time_or_phase", "horizon",
    "operator_order", "metrics", "matched_control_metrics", "acceptance_gate", "failed_conditions", "distance_from_threshold",
    "scientific_interpretation", "confounds_ruled_out", "confounds_remaining", "result_status", "do_not_repeat_unchanged",
    "next_justified_experiment", "reproduction_command", "artifact_paths", "artifact_hashes", "failure_type"
)


def initialize() -> dict:
    payload = read_json(RAW) if RAW.is_file() else {"version": "V837aq", "append_only": True, "entries": []}
    write_json(RAW, payload)
    write_json(DIAG, payload)
    return payload


def make_entry(*, failure_id: str, stage: str, branch: str, family: str | None = None, organism: str | None = None,
               intervention: str | None = None, time_or_phase: str | None = None, horizon: int | None = None,
               operator_order: int | None = None, metrics: Any = None, matched_control_metrics: Any = None,
               acceptance_gate: Any = None, failed_conditions: Any = None, distance_from_threshold: Any = None,
               scientific_interpretation: str = "", confounds_ruled_out: Any = None, confounds_remaining: Any = None,
               result_status: str = "DEFINITIVE_WITHIN_FROZEN_SCOPE", next_justified_experiment: str = "Follow the frozen V837aq operator ladder.",
               reproduction_command: str = "python scripts/reproduce_v837_recovery.py --variant v837aq --stage analyze --execute",
               artifact_paths: list[str] | None = None, artifact_hashes: dict | None = None,
               failure_type: str = "SCIENTIFIC_FAILURE") -> dict:
    utc, local = now_stamps()
    metrics = metrics or {}
    return {
        "failure_id": failure_id, "version": "V837aq", "stage": stage, "branch": branch, "family": family,
        "organism": organism, "intervention": intervention, "time_or_phase": time_or_phase, "horizon": horizon,
        "operator_order": operator_order, "metrics": metrics, "matched_control_metrics": matched_control_metrics or {},
        "acceptance_gate": acceptance_gate or {}, "failed_conditions": failed_conditions or [], "distance_from_threshold": distance_from_threshold or {},
        "scientific_interpretation": scientific_interpretation, "confounds_ruled_out": confounds_ruled_out or [],
        "confounds_remaining": confounds_remaining or [], "result_status": result_status, "do_not_repeat_unchanged": True,
        "next_justified_experiment": next_justified_experiment, "reproduction_command": reproduction_command,
        "artifact_paths": artifact_paths or [], "artifact_hashes": artifact_hashes or {"metrics_sha256": sha256_json(metrics)},
        "failure_type": failure_type, "timestamp_utc": utc, "timestamp_local": local, "source_sha": START_SHA,
    }


def add(entry: dict, append_central: bool = False) -> dict:
    missing = [k for k in REQUIRED if k not in entry]
    if missing:
        raise RuntimeError(f"V837AQ_FAILURE_ENTRY_INCOMPLETE:{missing}")
    if entry["failure_type"] not in {"SCIENTIFIC_FAILURE", "ENGINEERING_FAILURE"}:
        raise ValueError("bad failure_type")
    payload = initialize()
    if any(e["failure_id"] == entry["failure_id"] for e in payload["entries"]):
        return payload
    payload["entries"].append(entry)
    write_json(RAW, payload)
    write_json(DIAG, payload)
    if append_central:
        sync_central_ledger()
    return payload


def sync_central_ledger() -> dict:
    base = git_blob_bytes("docs/V837_FAILURE_LEDGER.md", START_SHA).decode("utf-8").rstrip("\n") + "\n"
    entries = initialize()["entries"]
    blocks = []
    for e in entries:
        blocks.append(
            f"\n## {e['failure_id']} - V837aq / {e['stage']}\n\n"
            f"- **Type:** {e['failure_type']} / {e['result_status']}\n"
            f"- **Family / organism:** {e['family']} / {e['organism']}\n"
            f"- **Operator:** order={e['operator_order']} intervention={e['intervention']} phase={e['time_or_phase']} horizon={e['horizon']}\n"
            f"- **Failed conditions:** {e['failed_conditions']}\n"
            f"- **Interpretation:** {e['scientific_interpretation']}\n"
            f"- **Do not repeat unchanged:** {e['do_not_repeat_unchanged']}\n"
            f"- **Artifacts:** {', '.join(e['artifact_paths'])}\n"
        )
    CENTRAL.write_text(base + "".join(blocks), encoding="utf-8", newline="\n")
    return {"version": "V837aq", "base_start_sha": START_SHA, "entries_appended": len(entries), "earlier_content_exact": True}
