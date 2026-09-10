from __future__ import annotations

from .utils import range_list

PARTITIONS = {
    "AO_BACKEND_FIT": [10000, 10063],
    "AO_BACKEND_SELECT": [10064, 10127],
    "AO_QUOTIENT_FIT": [10128, 10191],
    "AO_QUOTIENT_SELECT": [10192, 10255],
    "AO_DYNAMICS_FIT": [10256, 10319],
    "AO_DYNAMICS_SELECT": [10320, 10383],
    "AO_META_CONFIRM": [10384, 10447],
    "AO_HELDOUT_ORGANISM_EVAL": [10448, 10511],
    "HISTORICAL_VALIDATION_ROBUSTNESS": [20000, 20127],
    "FRESH_AUDIT": [90000, 90499],
}

REUSED_HISTORICAL_VALIDATION = True


def seeds(name: str) -> list[int]:
    return range_list(PARTITIONS[name])


def assert_disjoint_development_partitions() -> None:
    names = [k for k in PARTITIONS if k not in {"HISTORICAL_VALIDATION_ROBUSTNESS", "FRESH_AUDIT"}]
    expanded = {k: set(seeds(k)) for k in names}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = expanded[left] & expanded[right]
            if overlap:
                raise RuntimeError(f"AO_DATA_PARTITION_OVERLAP:{left}:{right}:{sorted(overlap)[:3]}")
