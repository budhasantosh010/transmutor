from __future__ import annotations

import hashlib
import json
from typing import Any

V837_CAPACITY_DEVELOPMENT_SUCCESS = 0.90
V837_CAPACITY_VALIDATION_SUCCESS = 0.85
V837_CAPACITY_FAMILIES_REQUIRED = 4
V837_CAPACITY_TOTAL_FAMILIES = 5


def v837_capacity_criterion() -> dict[str, Any]:
    return {
        "development_success_rate": V837_CAPACITY_DEVELOPMENT_SUCCESS,
        "validation_success_rate": V837_CAPACITY_VALIDATION_SUCCESS,
        "families_required": V837_CAPACITY_FAMILIES_REQUIRED,
        "total_families": V837_CAPACITY_TOTAL_FAMILIES,
        "source": "experiments/v837_primitive_invention/failures/run_blocker_diagnostic.py",
        "historical_semantics": "family capable iff development >= 0.90 and validation >= 0.85; representation screen succeeds iff >=4/5 families capable",
    }


def v837_capacity_criterion_sha256() -> str:
    payload = json.dumps(v837_capacity_criterion(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def capacity_demonstrated(development_success: float, validation_success: float) -> bool:
    return (
        float(development_success) >= V837_CAPACITY_DEVELOPMENT_SUCCESS
        and float(validation_success) >= V837_CAPACITY_VALIDATION_SUCCESS
    )
