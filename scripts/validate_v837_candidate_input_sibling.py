from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.v837af.candidate_input_factorization import CONDITIONS

HERE = ROOT / "experiments" / "v837_primitive_invention" / "v837af"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def git_blob_sha256(path: str) -> str:
    return hashlib.sha256(subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    require(set(CONFIG["conditions"]) == CONDITIONS, "V837af condition set changed")
    tr = CONFIG["training"]
    require(tr["development_seed_range"] == [10000, 10511] and tr["validation_seed_range"] == [20000, 20127], "V837af task seeds changed")
    require(tr["steps"] == 192 and tr["train_episodes"] == 512 and tr["validation_episodes"] == 128 and tr["replicates"] == 5, "V837af training regime changed")
    require(abs(tr["learning_rate"] - 0.005) < 1e-12 and abs(tr["weight_decay"] - 0.0001) < 1e-12 and abs(tr["gradient_clip"] - 5.0) < 1e-12, "V837af optimizer settings changed")
    frozen = {
        "experiments/v837_primitive_invention/v837y/config.json": CONFIG["v837y_config_sha256"],
        "experiments/v837_primitive_invention/v837y/candidate_interaction.py": CONFIG["v837y_model_sha256"],
        "experiments/v837_primitive_invention/v837y/results.json": CONFIG["v837y_results_sha256"],
        "experiments/v837_primitive_invention/v837y/raw/interaction_runs.json": CONFIG["v837y_raw_sha256"],
        "experiments/v837_primitive_invention/v837ab/results.json": CONFIG["v837ab_results_sha256"],
        "experiments/v837_primitive_invention/v837ac/results.json": CONFIG["v837ac_results_sha256"],
        "experiments/v837_primitive_invention/v837ad/results.json": CONFIG["v837ad_results_sha256"],
        "experiments/v837_primitive_invention/v837ad/diagnostics/decision_state.json": CONFIG["v837ad_decision_sha256"],
    }
    for path, expected in frozen.items():
        require(git_blob_sha256(path) == expected, f"frozen V837af dependency changed: {path}")
    require(CONFIG["projection_shared_params"] == 42 and CONFIG["projection_shared_macs"] == 36, "shared projection accounting changed")
    require(CONFIG["projection_deshared_params"] == 420 and CONFIG["projection_deshared_macs"] == 360, "deshared projection accounting changed")
    require(CONFIG["unique_seed_defined_episodes"] == 3200, "unique task episode count changed")
    require(CONFIG["fresh_audit_consumed"] is False and CONFIG["structural_search_allowed"] is False and CONFIG["primitive_mining_allowed"] is False, "V837af science locks changed")
    require(CONFIG["primitives_promoted"] == 0 and CONFIG["v838_started"] is False, "V837af promotion/V838 lock changed")
    require(not (ROOT / "experiments/v837_primitive_invention/v837ae").exists(), "V837ae must remain absent")
    require(not (ROOT / "experiments/v837_primitive_invention/v838").exists(), "V838 must remain absent")

    step0_path = HERE / "diagnostics" / "step0_equivalence.json"
    if step0_path.exists():
        step0 = json.loads(step0_path.read_text(encoding="utf-8"))
        require(step0.get("step0_equivalence_proven") is True, "V837af step-zero equivalence failed")
        require(float(step0.get("maximum_error", 1.0)) <= float(CONFIG["step0_tolerance"]), "V837af step-zero tolerance exceeded")

    results_path = HERE / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        decision = json.loads((HERE / "diagnostics" / "decision_state.json").read_text(encoding="utf-8"))
        anchor = json.loads((HERE / "diagnostics" / "anchor_compatibility.json").read_text(encoding="utf-8"))
        require(anchor.get("parent_reproduced") is True and anchor["observed"]["families_passing"] == 3, "AF0 did not reproduce exact Y3")
        require(set(results["conditions"]) == CONDITIONS, "V837af result conditions incomplete")
        require(decision.get("v837af_complete") is True, "V837af decision not complete")
        passes = decision.get("families_passing", {})
        require(set(passes) == CONDITIONS, "V837af family-count map incomplete")
        success = any(int(passes[c]) >= 4 for c in CONDITIONS if c != "AF0_y3_parent")
        require(bool(decision.get("representation_adequacy_pass")) == success, "V837af representation decision inconsistent")
        require(bool(decision.get("sample_efficiency_retest_allowed")) == success, "V837af sample-efficiency gate inconsistent")
        all_fail = all(int(passes[c]) < 4 for c in CONDITIONS if c != "AF0_y3_parent")
        require(bool(decision.get("v837ag_allowed")) == all_fail, "V837ag authorization inconsistent")
        require(decision.get("fresh_audit_consumed") is False and decision.get("v838_started") is False, "V837af decision locks changed")
        resources = results["resource_accounting"]
        require(int(resources["model_fits"]) == 100 and int(resources["optimizer_steps"]) == 19200 and int(resources["processed_training_examples"]) == 9830400, "V837af resource totals changed")
        require(int(resources["unique_seed_defined_episodes"]) == 3200, "V837af unique episode total changed")

    print("V837af candidate-input sibling validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
