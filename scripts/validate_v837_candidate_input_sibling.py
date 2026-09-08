from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.v837af.candidate_input_factorization import CandidateInputFactorizationY3
from experiments.v837_primitive_invention.v837af.run_candidate_input_transfer import _assert_locks

HERE = ROOT / "experiments" / "v837_primitive_invention" / "v837af"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
CONDITIONS = CONFIG["conditions"]


def fail(message: str) -> None:
    raise SystemExit(f"V837af validation failure: {message}")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    _assert_locks()
    if CONFIG["training"]["development_seed_range"] != [10000, 10511]: fail("development seeds changed")
    if CONFIG["training"]["validation_seed_range"] != [20000, 20127]: fail("validation seeds changed")
    if CONFIG["unique_seed_defined_episodes"] != 3200: fail("unique episode count changed")
    if CONFIG["training"]["steps"] != 192 or CONFIG["training"]["replicates"] != 5: fail("training budget changed")
    if CONFIG["training"]["optimizer"] != "AdamW": fail("optimizer changed")
    if CONFIG["training"]["projection_initialization_namespace"] != "v837j-primary-init": fail("projection initialization law changed")
    if CONFIG.get("fresh_audit_consumed") or CONFIG.get("structural_search_allowed") or CONFIG.get("primitive_mining_allowed") or CONFIG.get("v838_started"): fail("science lock changed")
    if (ROOT / "experiments/v837_primitive_invention/v837ae").exists(): fail("V837ae must remain absent")
    if (ROOT / "experiments/v837_primitive_invention/v838").exists(): fail("V838 must remain absent")

    models = {}
    for condition in CONDITIONS:
        torch.manual_seed(1234)
        models[condition] = CandidateInputFactorizationY3(
            high_capacity_generic_graph(0), condition=condition,
            coupling_initialization_seed=99, projection_seed=12345,
        )
    expected = {
        "AF0_y3_parent": (1223, 0, 846),
        "AF1_shared_candidate_input_factorization": (1265, 42, 882),
        "AF1F_folded_candidate_input_control": (1223, 0, 846),
        "AF1D_deshared_candidate_input_factorization": (1643, 420, 1206),
    }
    for condition, model in models.items():
        got = (model.parameter_count(), model.projection_parameter_count, model.total_recurrent_controller_projection_macs)
        if got != expected[condition]: fail(f"accounting mismatch {condition}: {got} != {expected[condition]}")
        if model.graph.to_dict() != models["AF0_y3_parent"].graph.to_dict(): fail(f"graph changed in {condition}")
        if model.coupling.to_dict() != models["AF0_y3_parent"].coupling.to_dict(): fail(f"rank4 branch changed in {condition}")
        if not torch.equal(model.global_ws.detach(), models["AF0_y3_parent"].global_ws.detach()): fail(f"global controller state weights changed in {condition}")
        if not torch.equal(model.global_wx.detach(), models["AF0_y3_parent"].global_wx.detach()): fail(f"global controller input weights changed in {condition}")
        if not torch.equal(model.base.readout.weight.detach(), models["AF0_y3_parent"].base.readout.weight.detach()): fail(f"readout changed in {condition}")

    step0_path = HERE / "diagnostics" / "step0_equivalence.json"
    if step0_path.exists():
        step0 = load(step0_path)
        if not step0.get("step0_equivalence_proven"): fail("step-zero equivalence failed")
        if float(step0.get("maximum_error", 1.0)) > 1e-6: fail("step-zero error exceeds tolerance")
        if step0.get("equivalence_precision") != "float64 replay of exact stored float32 initialization": fail("step-zero equivalence precision changed")

    anchor_path = HERE / "diagnostics" / "anchor_compatibility.json"
    if anchor_path.exists() and not load(anchor_path).get("parent_reproduced"): fail("AF0 parent reproduction failed")

    results_path = HERE / "results.json"
    if results_path.exists():
        results = load(results_path)
        decision = load(HERE / "diagnostics" / "decision_state.json")
        if not decision.get("v837af_complete") or not decision.get("parent_reproduced"): fail("decision state incomplete")
        if decision.get("fresh_audit_consumed") or decision.get("v838_started"): fail("decision state consumed forbidden science")
        if decision.get("structural_search_allowed") or decision.get("primitive_mining_allowed"): fail("search/mining incorrectly opened")
        passing = decision.get("families_passing", {})
        if set(passing) != set(CONDITIONS): fail("condition set mismatch")
        transfer_pass = any(int(passing[c]) >= 4 for c in CONDITIONS[1:])
        if bool(decision.get("representation_adequacy_pass")) != transfer_pass: fail("representation decision mismatch")
        all_failed = all(int(passing[c]) < 4 for c in CONDITIONS[1:])
        if bool(decision.get("v837ag_allowed")) != all_failed: fail("V837ag authorization mismatch")
        if transfer_pass and decision.get("v837ag_allowed"): fail("V837ag authorized after neutral success")
        if bool(decision.get("sample_efficiency_retest_allowed")) != transfer_pass: fail("sample-efficiency gate mismatch")
        resource = results.get("resource_accounting", {})
        if int(resource.get("model_fits", -1)) != 100: fail("expected exactly 100 V837af fits")
        if int(resource.get("optimizer_steps", -1)) != 19200: fail("optimizer-step accounting mismatch")
        if int(resource.get("processed_training_examples", -1)) != 9830400: fail("processed-example accounting mismatch")
        if int(resource.get("unique_seed_defined_episodes", -1)) != 3200: fail("unique episode accounting mismatch")

    print("V837af candidate-input sibling validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
