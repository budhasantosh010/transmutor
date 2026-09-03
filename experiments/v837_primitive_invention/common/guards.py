from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATUS_PATH = ROOT / "experiments" / "v837_primitive_invention" / "lineage_status.json"


def read_lineage_status(path: str | Path = DEFAULT_STATUS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def primitive_mining_allowed(path: str | Path = DEFAULT_STATUS_PATH) -> bool:
    status = read_lineage_status(path)
    explicit = status.get("primitive_mining_allowed")
    if explicit is not None:
        return bool(explicit)
    competence = status.get("neutral_substrate_competence")
    return competence == "PASS"


def assert_primitive_mining_allowed(path: str | Path = DEFAULT_STATUS_PATH) -> None:
    if not primitive_mining_allowed(path):
        raise RuntimeError(
            "primitive mining is scientifically blocked until the neutral substrate passes the original full V837 competence gate"
        )
