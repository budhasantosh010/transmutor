from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.v837ak.authorization import assert_v837ak_authorized
from experiments.v837_primitive_invention.v837ak.utils import DIRECTED, RANDOM, sha256_json

HERE = ROOT / "experiments" / "v837_primitive_invention" / "v837ak"
START_SHA = "4e989e474b4d93cd04246219457990746015c33f"


def load(relative: str):
    path = HERE / relative
    if not path.is_file():
        raise RuntimeError(f"V837ak required artifact missing: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git_changed(*paths: str) -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--name-only", START_SHA, "--", *paths], cwd=ROOT, text=True
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def validate() -> None:
    auth = assert_v837ak_authorized()
    require(auth["authorized"] is True, "V837ak authorization failed")
    gate = load("frozen_primitive_discovery_gate.json")
    require(gate["starting_sha"] == START_SHA, "V837ak starting SHA changed")
    require(gate["v837aj_decision_sha256"] == auth["v837aj_decision_sha256"], "V837aj decision hash drift")
    require(gate["source_population_count"] == 50 and gate["competent_count"] == 40 and gate["incompetent_count"] == 10, "frozen source counts changed")
    require(gate["all_subsets_enumeration"] is True and gate["subset_universe"] == [1, 10], "subset universe changed")
    require(gate["fresh_audit_consumed"] is False and gate["archive_promotion"] is False and gate["primitives_promoted"] == 0 and gate["v838_started"] is False, "frozen science locks changed")

    source = load("raw/source_population.json")
    require(source["count"] == 50 and source["directed"] == 25 and source["random"] == 25, "source population engine counts mismatch")
    require(source["competent"] == 40 and source["incompetent"] == 10, "source competence counts mismatch")
    require(len(source["rows"]) == 50 and len({r["organism_id"] for r in source["rows"]}) == 50, "organism IDs are not unique")
    require(Counter(r["engine"] for r in source["rows"]) == Counter({DIRECTED: 25, RANDOM: 25}), "source engines mismatch")

    recon = load("raw/reconstruction_results.json")
    require(recon["complete"] is True and recon["organisms_reconstructed"] == 50, "reconstruction incomplete")
    require(recon["competent"] == 40 and recon["incompetent"] == 10 and len(recon["rows"]) == 50, "reconstruction population mismatch")
    require(all(r["equivalent"] is True and r["old_competent"] == r["competent"] for r in recon["rows"]), "reconstruction equivalence/competence mismatch")
    require(max(abs(float(r["development_delta"])) for r in recon["rows"]) <= 1 / 512 + 1e-12, "development reconstruction tolerance exceeded")
    require(max(abs(float(r["validation_delta"])) for r in recon["rows"]) <= 1 / 128 + 1e-12, "validation reconstruction tolerance exceeded")
    require(all((ROOT / r["checkpoint"]).is_file() for r in recon["rows"]), "reconstructed checkpoint missing")
    accounting = recon["resource_accounting"]
    require(accounting["fits"] == 50 and accounting["optimizer_steps"] == 9600 and accounting["processed_examples"] == 4_915_200, "AK0 training accounting mismatch")

    probe = load("diagnostics/probe_partition.json")
    require(probe["discovery_probe"] == list(range(20000, 20064)) and probe["confirmation_probe"] == list(range(20064, 20128)), "probe ranges changed")
    require(probe["D1"] == list(range(20000, 20032)) and probe["D2"] == list(range(20032, 20064)), "D1/D2 changed")
    require(probe["C1"] == list(range(20064, 20096)) and probe["C2"] == list(range(20096, 20128)), "C1/C2 changed")
    require(probe["disjoint"] is True and probe["union_count"] == 128 and probe["fresh_audit_intersection"] == [], "probe partition/fresh-audit lock failed")
    require(probe["candidate_creation_from_confirmation"] is False, "confirmation may not create candidates")

    replay = load("diagnostics/ported_replay_gate.json")
    require(replay["pass"] is True and replay["occurrences_tested"] >= 20, "ported replay gate failed")
    require(replay["sizes"] == list(range(1, 11)), "ported replay does not cover sizes 1..10")
    require(float(replay["max_error"]) <= 1e-6, "ported replay tolerance exceeded")

    census = load("raw/subset_census_summary.json")
    require(census["organisms"] == 50 and census["subsets_per_organism"] == 1023 and census["total_occurrences"] == 51_150, "exhaustive census count mismatch")
    expected = {str(k): 50 * math.comb(10, k) for k in range(1, 11)}
    require(census["size_distribution"] == expected and census["whole_system_occurrences"] == 50 and census["all_nonempty_subsets"] is True, "subset size distribution mismatch")
    structural = load("raw/structural_classes.json")
    require(structural["class_count_total"] == len(structural["all_class_summaries"]), "structural class summary count mismatch")
    require(all(c["competent_support"] >= 3 for c in structural["classes"]), "candidate-detail structural class below support floor")

    reliability = load("diagnostics/fingerprint_reliability.json")
    thresholds = load("diagnostics/fingerprint_thresholds.json")
    require(set(reliability["sizes"]) == {str(k) for k in range(1, 11)}, "fingerprint size coverage mismatch")
    for k in range(1, 11):
        row = reliability["sizes"][str(k)]
        th = thresholds["sizes"][str(k)]
        require(abs(float(row["tau"]) - float(th["tau"])) <= 1e-12, f"tau mismatch size {k}")
        expected_pass = (
            float(row["self_vs_nonself_roc_auc"]) >= 0.80
            and float(row["median_self_distance"]) <= 0.60 * float(row["median_nonself_distance"]) + 1e-12
            and float(row["p90_self_distance"]) <= float(row["median_nonself_distance"]) + 1e-12
        )
        require(bool(row["eligible"]) == expected_pass, f"fingerprint gate mismatch size {k}")

    frozen = load("raw/frozen_candidate_classes.json")
    core = {k: v for k, v in frozen.items() if k != "frozen_candidate_classes_sha256"}
    require(sha256_json(core) == frozen["frozen_candidate_classes_sha256"], "frozen candidate hash mismatch")
    require(len(frozen["classes"]) <= 12, "candidate cap exceeded")
    stream_counts = Counter(c["stream"] for c in frozen["classes"])
    require(stream_counts["S"] <= 6 and stream_counts["D"] <= 6, "per-stream candidate cap exceeded")
    require(frozen["selection_uses_task_performance"] is False, "candidate ranking used task performance")
    for c in frozen["classes"]:
        require(bool(c["reusable_subsystem_eligible"]) == (int(c["size"]) < 10 and int(c["support"]) >= 6), "reusable-subsystem eligibility mismatch")
        if c["stream"] == "D":
            require(int(c["size"]) in reliability["eligible_sizes"], "unreliable size entered dynamic candidate freeze")

    confirmation = load("raw/confirmed_candidate_classes.json")
    require(confirmation["frozen_candidate_classes_sha256"] == frozen["frozen_candidate_classes_sha256"], "confirmation used different frozen classes")
    require(confirmation["class_count"] == len(frozen["classes"]) and confirmation["candidate_creation_from_confirmation"] is False, "confirmation class count/creation violation")
    for c in confirmation["classes"]:
        if c["confirmed"]:
            require(float(c["retention_fraction"]) >= 0.70 and c["engine_coverage_ok"] is True and c["dynamic_confirmation_ok"] is True, "confirmed class failed held-out gate")

    causal = load("raw/causal_results.json")
    for c in causal["classes"]:
        if int(c["size"]) == 10:
            require(c["causal_specificity_pass"] is False and c.get("whole_system_control") is True, "whole-system control passed local causal gate")
            continue
        for r in c["representatives"]:
            require(r["sham_count"] == 5 and r["sham_control_complete"] is True, "causal representative lacks five matched shams")
            require(all(int(s["relaxation_level"]) <= 2 for s in r["shams"]), "sham relaxation exceeded +/-2")
        if c["causal_specificity_pass"]:
            require(c["matched_sham_controls_complete"] is True and float(c["median_causal_specificity"]) >= 0.03 and float(c["positive_fraction"]) >= 0.60 and float(c["one_sided_p"]) <= 0.05, "causal pass violates frozen gate")

    boundary = load("raw/boundary_substitution_results.json")
    for c in boundary["classes"]:
        for pair in c["pairs"]:
            require(pair["same_class_donor"]["organism_id"] != pair["recipient"]["organism_id"], "same-class donor equals recipient")
            require(pair["different_class_donor"]["organism_id"] != pair["recipient"]["organism_id"], "different-class donor equals recipient")
            require(max(float(v) for v in pair["self_replay"].values()) <= 1e-6, "self boundary replay is not exact")
            require(set(("state_nrmse", "candidate_nrmse", "output_nrmse")) <= set(pair["same"]), "same-class replay metrics missing")
            require(set(("state_nrmse", "candidate_nrmse", "output_nrmse")) <= set(pair["different"]), "different-class replay metrics missing")
            require(set(("state_nrmse", "candidate_nrmse", "output_nrmse")) <= set(pair["randomized"]), "randomized replay metrics missing")

    closed = load("raw/closed_loop_substitution_results.json")
    causal_map = {c["class_id"]: c for c in causal["classes"]}
    boundary_map = {c["class_id"]: c for c in boundary["classes"]}
    expected_eligible = [c for c in boundary["classes"] if c["boundary_interchangeable"] and causal_map.get(c["class_id"], {}).get("causal_specificity_pass")]
    require(closed["eligible_class_count"] == len(expected_eligible) and closed["run"] == bool(expected_eligible), "AK9 conditional authorization mismatch")
    for c in closed["classes"]:
        for row in c["rows"]:
            require(max(float(row["self_transplant_prediction_max_error"]), float(row["self_transplant_state_max_error"])) <= 1e-6, "self transplant reality gate failed")

    decision = load("diagnostics/decision_state.json")
    validated = load("validated_primitive_classes.json")
    require(decision["version"] == "V837ak" and decision["source_population_valid"] is True and decision["organisms_reconstructed"] == 50 and decision["competent_organisms"] == 40, "decision population state invalid")
    require(decision["ported_replay_valid"] is True, "decision lost replay gate")
    require(decision["validated_primitive_classes"] == validated["count"], "validated class count mismatch")
    require(decision["primitives_promoted"] == 0 and validated["primitive_archive_populated"] is False, "primitive archive populated/promoted in V837ak")
    require(decision["primitive_archive_allowed_next"] == (validated["count"] > 0), "archive-next authorization inconsistent with Level-6 classes")
    require(decision["fresh_audit_consumed"] is False and decision["large_persistent_storage_tested"] is False and decision["v838_started"] is False, "final science locks changed")
    status_path = ROOT / "experiments" / "v837_primitive_invention" / "functional_dynamical_motif_discovery_program_status.json"
    require(status_path.is_file(), "V837ak program status missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    require(status["diagnosis"] == decision["diagnosis"] and status["diagnosis_qualifier"] == decision["diagnosis_qualifier"], "program status diagnosis mismatch")
    require(status["validated_primitive_classes"] == validated["count"] and status["primitive_archive_allowed_next"] == decision["primitive_archive_allowed_next"], "program status primitive decision mismatch")
    require(status["fresh_audit_episodes_consumed"] == 0 and status["primitives_promoted"] == 0 and status["large_persistent_storage_tested"] is False and status["v838_started"] is False, "program status science locks changed")
    require(not (ROOT / "experiments" / "v837_primitive_invention" / "v838").exists(), "V838 directory exists")

    protected_changes = []
    protected_changes += git_changed("archive", "registry", "experiments/v837_primitive_invention/v837aj")
    protected_changes += git_changed(
        "experiments/v837_primitive_invention/common/motif.py",
        "experiments/v837_primitive_invention/common/primitive_archive.py",
        "experiments/v837_primitive_invention/common/guards.py",
    )
    require(not protected_changes, f"protected historical science changed: {protected_changes}")


def main() -> int:
    validate()
    print("V837ak functional/dynamical motif validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
