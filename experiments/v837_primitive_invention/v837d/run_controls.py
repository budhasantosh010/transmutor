from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.v837d.experiment import CONFIG, FAMILIES, paired_family_deltas, run_jobs, summarize_condition

HERE = Path(__file__).resolve().parent
DIAGNOSTICS = HERE / "diagnostics"


def main() -> int:
    broadcast = json.loads((DIAGNOSTICS / "broadcast_capacity.json").read_text(encoding="utf-8"))
    sweep = json.loads((DIAGNOSTICS / "sparse_density_sweep.json").read_text(encoding="utf-8"))
    if broadcast.get("baseline_compatibility_pass") is not True:
        raise SystemExit("broadcast compatibility failed")
    selected = float(sweep["selected_density"])
    selected_rows = [row for row in sweep["rows"] if abs(float(row["requested_density"]) - selected) < 1e-12]
    shuffled_jobs = [
        {"condition": "shuffled_sparse", "family": family, "replicate": replicate, "density": selected, "shuffled": True}
        for family in FAMILIES
        for replicate in range(int(CONFIG["replicates"]))
    ]
    no_message_jobs = [
        {"condition": "no_message", "family": family, "replicate": replicate, "density": selected, "disable_messages": True}
        for family in FAMILIES
        for replicate in range(int(CONFIG["replicates"]))
    ]
    shuffled_rows = run_jobs(shuffled_jobs)
    no_message_rows = run_jobs(no_message_jobs)
    payload = {
        "version": "V837d",
        "selected_density": selected,
        "historical_gate_hash": CONFIG["historical_gate_hash"],
        "fresh_audit_consumed": False,
        "fixed_sparse": summarize_condition(selected_rows, seed=83740),
        "shuffled_sparse": summarize_condition(shuffled_rows, seed=83741),
        "no_message": summarize_condition(no_message_rows, seed=83742),
        "paired_validation_deltas": {
            "shuffled_minus_fixed": paired_family_deltas(shuffled_rows, selected_rows, seed=83743),
            "no_message_minus_fixed": paired_family_deltas(no_message_rows, selected_rows, seed=83744),
            "fixed_minus_broadcast": paired_family_deltas(selected_rows, broadcast["rows"], seed=83745),
        },
        "rows": {"shuffled_sparse": shuffled_rows, "no_message": no_message_rows},
    }
    write_json(DIAGNOSTICS / "controls.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
