from __future__ import annotations

PARTITIONS = {
    "AQ_REALITY_GATE": (11000, 11063),
    "AQ_OPERATOR_FIT": (11064, 11127),
    "AQ_OPERATOR_SELECT": (11128, 11191),
    "AQ_INTERACTION": (11192, 11255),
    "AQ_LOCALIZATION": (11256, 11319),
    "AQ_COMPOSITION": (11320, 11383),
    "AQ_META_CONFIRM": (11384, 11447),
    "AQ_HELDOUT_EVAL": (11448, 11511),
    "REUSED_HISTORICAL_VALIDATION": (20000, 20127),
    "FRESH_AUDIT": (90000, 90499),
}

DEVELOPMENT_ROLES = tuple(k for k in PARTITIONS if k not in {"REUSED_HISTORICAL_VALIDATION", "FRESH_AUDIT"})


def assert_roles() -> bool:
    sets = {k: set(range(v[0], v[1] + 1)) for k, v in PARTITIONS.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sets[a] & sets[b]:
                raise RuntimeError(f"V837AQ_DATA_ROLE_OVERLAP:{a}:{b}")
    if PARTITIONS["FRESH_AUDIT"] != (90000, 90499):
        raise RuntimeError("V837AQ_FRESH_AUDIT_DRIFT")
    return True


def seeds(role: str) -> list[int]:
    lo, hi = PARTITIONS[role]
    return list(range(lo, hi + 1))
