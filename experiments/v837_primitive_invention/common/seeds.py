from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3]
GATES_PATH = ROOT / "experiments" / "v837_primitive_invention" / "frozen_gates.json"


def frozen_gates() -> dict:
    return json.loads(GATES_PATH.read_text(encoding="utf-8"))


def gate_sha256() -> str:
    return hashlib.sha256(GATES_PATH.read_bytes()).hexdigest()


def seed_ranges() -> dict[str, tuple[int, int]]:
    raw = frozen_gates()["seed_ranges"]
    return {name: (int(bounds[0]), int(bounds[1])) for name, bounds in raw.items()}


def expand(bounds: Iterable[int]) -> set[int]:
    start, end = [int(v) for v in bounds]
    if end < start:
        raise ValueError(f"invalid range {start}..{end}")
    return set(range(start, end + 1))


def assert_seed_partitions_disjoint() -> None:
    ranges = seed_ranges()
    names = sorted(ranges)
    expanded = {name: expand(ranges[name]) for name in names}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = expanded[left] & expanded[right]
            if overlap:
                raise AssertionError(f"seed overlap: {left}/{right}: {sorted(overlap)[:10]}")


def seeds_for(split: str, count: int, *, offset: int = 0) -> list[int]:
    ranges = seed_ranges()
    if split not in ranges:
        raise KeyError(split)
    start, end = ranges[split]
    first = start + int(offset)
    last = first + int(count) - 1
    if first < start or last > end:
        raise ValueError(f"requested {count} seeds at offset {offset} outside {split}={start}..{end}")
    return list(range(first, last + 1))


def cyclic_seeds(split: str, count: int, *, offset: int = 0) -> list[int]:
    ranges = seed_ranges()
    if split not in ranges:
        raise KeyError(split)
    start, end = ranges[split]
    width = end - start + 1
    if count > width:
        raise ValueError(f"cannot request {count} unique cyclic seeds from {split} width {width}")
    return [start + ((int(offset) + index) % width) for index in range(int(count))]


def deterministic_int(*parts: object, bits: int = 31) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << bits) - 1)


def rng(*parts: object) -> random.Random:
    return random.Random(deterministic_int(*parts))


assert_seed_partitions_disjoint()
