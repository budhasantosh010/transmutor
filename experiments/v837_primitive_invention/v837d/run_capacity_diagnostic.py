from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.common.gates import v837_capacity_criterion_sha256
from experiments.v837_primitive_invention.common.seeds import gate_sha256
from experiments.v837_primitive_invention.common.serialization import write_json
from experiments.v837_primitive_invention.v837d.experiment import CONFIG, FAMILIES, paired_family_deltas, run_jobs, summarize_condition

HERE = Path(__file__).resolve().parent
DIAGNOSTICS = HERE / "diagnostics"
EXPECTED_GATE = CONFIG["historical_gate_hash"]
EXPECTED_CAPACITY = CONFIG["capacity_criterion_hash"]


def _assert_frozen() -> None:
    if gate_sha256() != EXPECTED_GATE:
        raise SystemExit("historical V837 gate hash changed")
    if v837_capacity_criterion_sha256() != EXPECTED_CAPACITY:
        raise SystemExit("capacity criterion fingerprint changed")


def run_broadcast() -> dict:
    _assert_frozen()
    jobs = [
        {"condition": "broadcast", "family": family, "replicate": replicate}
        for family in FAMILIES
        for replicate in range(int(CONFIG["replicates"]))
    ]
    rows = run_jobs(jobs)
    summary = summarize_condition(rows, seed=83710)
    historical = json.loads(
        (ROOT / "experiments/v837_primitive_invention/failures/blocker_diagnostic_results.json").read_text(encoding="utf-8")
    )
    family_differences = {}
    substantial = 0
    for family in FAMILIES:
        rerun_first_three = [row for row in rows if row["family"] == family and row["replicate"] < 3]
        rerun_best = max(float(row["validation_success"]) for row in rerun_first_three)
        historical_best = float(historical["per_family"][family]["best_validation_success"])
        difference = abs(rerun_best - historical_best)
        if difference > float(CONFIG["baseline_reproduction_tolerance"]["absolute_family_score_difference"]):
            substantial += 1
        family_differences[family] = {
            "historical_best_validation": historical_best,
            "rerun_best_validation_first_three": rerun_best,
            "absolute_difference": difference,
        }
    tolerance_pass = substantial <= int(CONFIG["baseline_reproduction_tolerance"]["max_families_exceeding_before_stop"])
    payload = {
        "version": "V837d",
        "phase": "historical_broadcast_capacity_rerun",
        "historical_gate_hash": EXPECTED_GATE,
        "capacity_criterion_hash": EXPECTED_CAPACITY,
        "fresh_audit_consumed": False,
        "family_differences": family_differences,
        "families_exceeding_0_10": substantial,
        "baseline_compatibility_pass": tolerance_pass,
        "summary": summary,
        "rows": rows,
    }
    write_json(DIAGNOSTICS / "broadcast_capacity.json", payload)
    if not tolerance_pass:
        raise SystemExit("IMPLEMENTATION_FAILURE: broadcast baseline drift exceeds frozen tolerance")
    return payload


def run_sparse() -> dict:
    _assert_frozen()
    broadcast_path = DIAGNOSTICS / "broadcast_capacity.json"
    if not broadcast_path.exists():
        raise SystemExit("run broadcast phase first")
    broadcast = json.loads(broadcast_path.read_text(encoding="utf-8"))
    if broadcast.get("baseline_compatibility_pass") is not True:
        raise SystemExit("broadcast compatibility failed; sparse interpretation is blocked")
    densities = [float(value) for value in CONFIG["densities"] if float(value) < 1.0]
    jobs = [
        {"condition": "fixed_sparse", "family": family, "replicate": replicate, "density": density}
        for density in densities
        for family in FAMILIES
        for replicate in range(int(CONFIG["replicates"]))
    ]
    rows = run_jobs(jobs)
    by_density = {}
    for density in densities:
        density_rows = [row for row in rows if abs(float(row["requested_density"]) - density) < 1e-12]
        summary = summarize_condition(density_rows, seed=int(83720 + density * 1000))
        summary["paired_validation_delta_vs_broadcast"] = paired_family_deltas(
            density_rows, broadcast["rows"], seed=int(83730 + density * 1000)
        )
        by_density[str(density)] = summary
    selected_density = max(
        densities,
        key=lambda density: (
            by_density[str(density)]["median_families_passing_per_replicate"],
            by_density[str(density)]["worst_family_median_validation"],
            -density,
        ),
    )
    payload = {
        "version": "V837d",
        "phase": "fixed_sparse_density_sweep",
        "historical_gate_hash": EXPECTED_GATE,
        "capacity_criterion_hash": EXPECTED_CAPACITY,
        "fresh_audit_consumed": False,
        "selection_rule": CONFIG["density_selection_rule"],
        "selected_density": selected_density,
        "selected_density_summary": by_density[str(selected_density)],
        "by_density": by_density,
        "rows": rows,
    }
    write_json(DIAGNOSTICS / "sparse_density_sweep.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["broadcast", "sparse", "all"], default="all")
    args = parser.parse_args()
    if args.phase in {"broadcast", "all"}:
        run_broadcast()
    if args.phase in {"sparse", "all"}:
        run_sparse()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
