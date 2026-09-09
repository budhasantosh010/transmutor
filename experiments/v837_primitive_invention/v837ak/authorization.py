from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DECISION = ROOT / "experiments/v837_primitive_invention/v837aj/diagnostics/decision_state.json"
EXPECTED_SHA = "36d2e9ff5c6b4b1b07d930a85a1658f6326562f287c240b48d015400bcec98e2"
ALLOWED_DIAGNOSES = {"RANDOM_STRUCTURAL_DISCOVERY_SUFFICIENT", "STRUCTURAL_SEARCH_RECOVERED"}


def _git_blob_bytes(relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)


def decision_sha256() -> str:
    relative = DECISION.relative_to(ROOT).as_posix()
    try:
        payload = _git_blob_bytes(relative)
    except Exception:
        payload = DECISION.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def assert_v837ak_authorized(*, write_diagnostic: bool = True) -> dict:
    if not DECISION.is_file():
        raise RuntimeError("V837AK_AUTHORIZATION_MISSING")
    data = json.loads(DECISION.read_text(encoding="utf-8"))
    checks = {
        "version": data.get("version") == "V837aj",
        "automated_structural_discovery": data.get("automated_structural_discovery") is True,
        "primitive_mining_allowed_next": data.get("primitive_mining_allowed_next") is True,
        "diagnosis": data.get("diagnosis") in ALLOWED_DIAGNOSES,
        "fresh_audit_consumed": data.get("fresh_audit_consumed") is False,
        "v838_started": data.get("v838_started") is False,
        "decision_sha256": decision_sha256() == EXPECTED_SHA,
    }
    payload = {
        "version": "V837ak",
        "authorized": all(checks.values()),
        "checks": checks,
        "v837aj_diagnosis": data.get("diagnosis"),
        "primitive_mining_allowed_next": data.get("primitive_mining_allowed_next"),
        "v837aj_decision_sha256": decision_sha256(),
        "legacy_guard_used_as_authority": False,
        "fresh_audit_consumed": False,
        "v838_started": False,
    }
    if write_diagnostic:
        path = HERE / "diagnostics/authorization.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not payload["authorized"]:
        raise RuntimeError("V837AK_NOT_AUTHORIZED: " + json.dumps(checks, sort_keys=True))
    return payload


if __name__ == "__main__":
    print(json.dumps(assert_v837ak_authorized(), indent=2, sort_keys=True))
