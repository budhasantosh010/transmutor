from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.v837aa.candidate_law_alignment import SIGNED_PERMUTATIONS, validate_signed_permutations
from experiments.v837_primitive_invention.v837aa.run_candidate_law_audit import CONFIG, assert_science_locks

HERE = ROOT / "experiments/v837_primitive_invention/v837aa"
ALLOWED_DIAGNOSES = {
    "DIRECT_COMMON_BASIS",
    "COMMON_LAW_UP_TO_SIGNED_PERMUTATION",
    "STABLE_SMALL_TYPE_VOCABULARY",
    "GENUINELY_DIVERSE_CANDIDATE_LAWS",
    "INCONCLUSIVE",
    "Y3_PARENT_REPRODUCTION_FAILURE",
}
RECOMMENDATIONS = {
    "DIRECT_COMMON_BASIS": "V837ab_DIRECT_SHARED_CANDIDATE_CORE",
    "STABLE_SMALL_TYPE_VOCABULARY": "V837ab_GROUP_SHARED_CANDIDATE_TYPES",
    "GENUINELY_DIVERSE_CANDIDATE_LAWS": "NEXT_AXIS_SHARED_INPUT_REPRESENTATION",
    "INCONCLUSIVE": "NO_ARCHITECTURE_CHANGE_TARGETED_DIAGNOSTIC_REQUIRED",
    "Y3_PARENT_REPRODUCTION_FAILURE": "DIAGNOSE_DETERMINISTIC_OR_ENVIRONMENT_DRIFT",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"missing V837aa artifact: {path.relative_to(ROOT)}")


def validate_framework() -> None:
    for name in ["README.md", "RESEARCH_SPEC.md", "config.json", "frozen_candidate_law_gate.json", "candidate_law_alignment.py", "run_candidate_law_audit.py", "analyze_results.py"]:
        require(HERE / name)
    validate_signed_permutations()
    if len(SIGNED_PERMUTATIONS) != 384:
        raise ValueError("V837aa signed-permutation count changed")
    assert_science_locks()
    if CONFIG.get("audit_only") is not True or CONFIG.get("architecture_modified") is not False:
        raise ValueError("V837aa must remain audit-only")
    if CONFIG.get("rerun_conditions") != ["Y3_global_control_rank4_candidate"]:
        raise ValueError("V837aa may rerun only Y3")
    tr = CONFIG["training"]
    expected = {
        "steps": 192,
        "development_seed_range": [10000, 10511],
        "validation_seed_range": [20000, 20127],
        "development_episodes_per_family": 512,
        "validation_episodes_per_family": 128,
        "learning_rate": 0.005,
        "weight_decay": 0.0001,
        "gradient_clip": 5.0,
    }
    for key, value in expected.items():
        if tr.get(key) != value:
            raise ValueError(f"V837aa frozen training value changed: {key}")
    if CONFIG.get("expected_model_fits") != 25 or CONFIG.get("expected_optimizer_steps") != 4800 or CONFIG.get("expected_processed_training_examples") != 2457600:
        raise ValueError("V837aa resource budget changed")
    if CONFIG.get("unique_seed_defined_task_episodes") != 3200:
        raise ValueError("V837aa unique-data accounting changed")
    if CONFIG.get("synthetic_probes_per_fit") != 4096 or CONFIG.get("empirical_probes_per_fit") != 4096:
        raise ValueError("V837aa probe count changed")
    if CONFIG.get("primary_alignment_view") != "core" or CONFIG.get("parameter_views", {}).get("core") != ["Ws", "Wm", "b"]:
        raise ValueError("V837aa primary candidate-core view changed")
    for key in ("structural_search", "primitive_mining", "fresh_audit_consumed", "v837ab_implemented", "v838_started"):
        if CONFIG.get(key) is not False:
            raise ValueError(f"V837aa science lock changed: {key}")
    if (ROOT / "experiments/v837_primitive_invention/v837ab").exists() or (ROOT / "experiments/v837_primitive_invention/v838").exists():
        raise ValueError("forbidden V837ab/V838 directory exists")


def validate_runs() -> None:
    runs_path = HERE / "raw/runs.json"
    if not runs_path.exists():
        return
    rows = load(runs_path).get("rows", [])
    if len(rows) != 25:
        raise ValueError(f"V837aa requires 25 Y3 fits, found {len(rows)}")
    seen = set()
    for row in rows:
        key = (row.get("family"), int(row.get("replicate_id", -1)))
        if key in seen:
            raise ValueError(f"duplicate V837aa run: {key}")
        seen.add(key)
        if row.get("parent_condition") != "Y3_global_control_rank4_candidate":
            raise ValueError("V837aa reran a non-Y3 condition")
        if int(row.get("parameter_count", -1)) != 1223 or int(row.get("recurrent_controller_macs", -1)) != 846:
            raise ValueError("V837aa Y3 parameter/MAC identity changed")
        resources = row.get("training_resources", {})
        if int(resources.get("optimizer_steps", -1)) != 192 or int(row.get("processed_training_examples", -1)) != 98304:
            raise ValueError("V837aa run training budget changed")
        if row.get("fresh_audit_consumed") is not False or row.get("structural_search_allowed") is not False or row.get("primitive_mining_allowed") is not False or row.get("v838_started") is not False:
            raise ValueError("V837aa run violated science locks")
    families = Counter(row["family"] for row in rows)
    if sorted(families.values()) != [5,5,5,5,5]:
        raise ValueError("V837aa family/replicate coverage incomplete")
    for filename in ("initial_parameter_snapshots.json", "trained_parameter_snapshots.json"):
        payload = load(HERE / "raw" / filename)
        snapshots = payload.get("snapshots", [])
        if len(snapshots) != 25:
            raise ValueError(f"{filename} must contain 25 snapshots")
        for snapshot in snapshots:
            if int(snapshot.get("parameter_count", -1)) != 1223 or int(snapshot.get("active_recurrent_controller_macs", -1)) != 846:
                raise ValueError(f"{filename} contains incompatible Y3 snapshot")
            keys = snapshot.get("state_dict", {})
            required = [f"base.cell_{name}.{i}" for name in ("ws","wm","wx","b","wo") for i in range(10)]
            required += ["global_u", "global_v", "global_ws", "global_wx", "global_b", "base.readout.weight", "base.readout.bias"]
            missing = [key for key in required if key not in keys]
            edge_keys = [f"base.edge_weights.{i}" for i in range(55)]
            missing_edges = [key for key in edge_keys if key not in keys]
            if missing or missing_edges:
                raise ValueError(f"snapshot missing required parameters: {(missing + missing_edges)[:5]}")
    reproduction = load(HERE / "diagnostics/parent_reproduction.json")
    if reproduction.get("parent_reproduction_valid") is not True:
        result_path = HERE / "results.json"
        if not result_path.exists() or load(result_path).get("candidate_law_diagnosis") != "Y3_PARENT_REPRODUCTION_FAILURE":
            raise ValueError("invalid parent reproduction must close as Y3_PARENT_REPRODUCTION_FAILURE")
        return
    if reproduction.get("observed_families_passing") != 3 or reproduction.get("expected_families_passing") != 3:
        raise ValueError("V837aa parent reproduction family count changed")
    if max(reproduction.get("family_validation_median_abs_deltas", {}).values(), default=1.0) > 0.015625 + 1e-12:
        raise ValueError("V837aa family median reproduction drift exceeds gate")
    hash_path = HERE / "diagnostics/snapshot_hashes.json"
    require(hash_path)
    hashes = load(hash_path)
    if hashes.get("initial_parameter_snapshots_sha256") != sha256(HERE / "raw/initial_parameter_snapshots.json"):
        raise ValueError("initial snapshot artifact hash mismatch")
    if hashes.get("trained_parameter_snapshots_sha256") != sha256(HERE / "raw/trained_parameter_snapshots.json"):
        raise ValueError("trained snapshot artifact hash mismatch")


def validate_results() -> None:
    result_path = HERE / "results.json"
    if not result_path.exists():
        return
    result = load(result_path)
    diagnosis = result.get("candidate_law_diagnosis")
    if diagnosis not in ALLOWED_DIAGNOSES:
        raise ValueError(f"invalid V837aa diagnosis: {diagnosis}")
    if diagnosis == "COMMON_LAW_UP_TO_SIGNED_PERMUTATION":
        expected = "V837ab_CANONICAL_SHARED_CORE_WITH_FIXED_BASIS_ADAPTERS" if result.get("relative_basis_stable") else "V837ab_COORDINATE_ADAPTIVE_SHARED_CORE_REQUIRED"
    else:
        expected = RECOMMENDATIONS[diagnosis]
    if result.get("recommended_next_axis") != expected:
        raise ValueError("V837aa recommendation does not match frozen decision mapping")
    if result.get("representation_adequacy") != "still_3_of_5_parent":
        raise ValueError("V837aa may not claim representation recovery")
    for key, expected_value in {
        "fresh_audit_consumed": False,
        "primitive_count": 0,
        "structural_search_allowed": False,
        "primitive_mining_allowed": False,
        "v837ab_implemented": False,
        "v838_started": False,
    }.items():
        if result.get(key) != expected_value:
            raise ValueError(f"V837aa final lock changed: {key}")
    if diagnosis != "Y3_PARENT_REPRODUCTION_FAILURE":
        required_diagnostics = [
            "raw_parameter_similarity.json", "aligned_parameter_similarity.json",
            "functional_similarity_synthetic.json", "functional_similarity_empirical.json",
            "initialization_null_control.json", "signed_permutation_assignments.json",
            "relative_basis_stability.json", "gradient_alignment.json",
            "global_coupling_alignment.json", "cell_type_clustering.json",
            "compute_efficiency.json", "decision_state.json",
        ]
        for name in required_diagnostics:
            require(HERE / "diagnostics" / name)
        for name in ["raw_vs_aligned_similarity.png", "functional_alignment_gain.png", "gradient_cosine_matrix.png", "relative_basis_stability.png", "cell_type_consensus.png"]:
            require(HERE / "plots" / name)
        decision = load(HERE / "diagnostics/decision_state.json")
        if decision.get("candidate_law_diagnosis") != diagnosis or decision.get("recommended_next_axis") != result.get("recommended_next_axis"):
            raise ValueError("V837aa decision/result mismatch")
        resources = result.get("resource_accounting", {})
        if int(resources.get("model_fits", -1)) != 25 or int(resources.get("optimizer_steps", -1)) != 4800 or int(resources.get("processed_training_examples", -1)) != 2457600:
            raise ValueError("V837aa final resource accounting changed")
        if int(resources.get("synthetic_probes", -1)) != 102400 or int(resources.get("empirical_probes", -1)) != 102400:
            raise ValueError("V837aa functional probe accounting changed")
        if int(resources.get("unique_seed_defined_task_episodes", -1)) != 3200 or int(resources.get("fresh_task_episodes", -1)) != 0:
            raise ValueError("V837aa unique/fresh task data accounting changed")
    require(HERE / "VERDICT.md")


def main() -> int:
    validate_framework()
    validate_runs()
    validate_results()
    print("V837aa candidate-law alignment validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
