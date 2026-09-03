from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATUS = ROOT / "experiments" / "v837_primitive_invention" / "lineage_status.json"


def main() -> int:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    if status.get("outcome") != "A_MILESTONE_PASSED":
        raise SystemExit(
            "Fresh audit is locked: the V837 primitive-invention milestone did not pass. "
            "Do not consume fresh-audit seeds for an unsuccessful development lineage."
        )
    raise SystemExit("Fresh-audit implementation is intentionally unavailable until a future lineage passes all prerequisite gates.")


if __name__ == "__main__":
    raise SystemExit(main())
