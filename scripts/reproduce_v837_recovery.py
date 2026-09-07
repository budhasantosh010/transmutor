from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VARIANT_COMMANDS = {
    "v837d": [
        [sys.executable, "experiments/v837_primitive_invention/v837d/run_capacity_diagnostic.py"],
        [sys.executable, "experiments/v837_primitive_invention/v837d/run_controls.py"],
        [sys.executable, "experiments/v837_primitive_invention/v837d/analyze_results.py"],
    ],
    "v837g": [
        [sys.executable, "experiments/v837_primitive_invention/v837g/run_capacity_diagnostic.py"],
    ],
    "v837h": [
        [sys.executable, "experiments/v837_primitive_invention/v837h/run_capacity_diagnostic.py"],
    ],
    "v837j": [
        [sys.executable, "experiments/v837_primitive_invention/v837j/run_reference_calibration.py"],
        [sys.executable, "experiments/v837_primitive_invention/v837j/analyze_results.py"],
    ],
    "v837k": [
        [sys.executable, "experiments/v837_primitive_invention/v837k/run_training_budget_diagnostic.py"],
        [sys.executable, "experiments/v837_primitive_invention/v837k/analyze_results.py"],
    ],
    "v837l": [
        [sys.executable, "experiments/v837_primitive_invention/v837l/run_data_diagnostic.py"],
        [sys.executable, "experiments/v837_primitive_invention/v837l/analyze_results.py"],
    ],
    "v837m": [
        [sys.executable, "experiments/v837_primitive_invention/v837m/run_capacity_diagnostic.py"],
        [sys.executable, "experiments/v837_primitive_invention/v837m/analyze_results.py"],
    ],
    "v837n": [
        [sys.executable, "experiments/v837_primitive_invention/v837n/run_mechanism_ablation.py", "--phase", "full"],
        [sys.executable, "experiments/v837_primitive_invention/v837n/run_mechanism_ablation.py", "--phase", "ablations"],
        [sys.executable, "experiments/v837_primitive_invention/v837n/analyze_results.py"],
    ],
    "v837o": [
        [sys.executable, "experiments/v837_primitive_invention/v837o/run_factorial_localization.py", "--phase", "full"],
        [sys.executable, "experiments/v837_primitive_invention/v837o/run_factorial_localization.py", "--phase", "factorial"],
        [sys.executable, "experiments/v837_primitive_invention/v837o/analyze_results.py"],
    ],
    "v837p": [
        [sys.executable, "experiments/v837_primitive_invention/v837p/run_transfer.py"],
        [sys.executable, "experiments/v837_primitive_invention/v837p/analyze_results.py"],
    ],
    "v837q": [
        [sys.executable, "experiments/v837_primitive_invention/v837q/run_state_organization_diagnostic.py", "--phase", "baseline"],
        [sys.executable, "experiments/v837_primitive_invention/v837q/run_state_organization_diagnostic.py", "--phase", "primary"],
        [sys.executable, "experiments/v837_primitive_invention/v837q/analyze_results.py"],
    ],
    "v837r": [
        [sys.executable, "experiments/v837_primitive_invention/v837r/run_coupling_diagnostic.py", "--phase", "baseline"],
        [sys.executable, "experiments/v837_primitive_invention/v837r/run_coupling_diagnostic.py", "--phase", "screen"],
        [sys.executable, "experiments/v837_primitive_invention/v837r/run_coupling_diagnostic.py", "--phase", "localization"],
        [sys.executable, "experiments/v837_primitive_invention/v837r/analyze_results.py"],
    ],
    "v837s": [
        [sys.executable, "experiments/v837_primitive_invention/v837s/run_interaction.py"],
        [sys.executable, "experiments/v837_primitive_invention/v837s/analyze_results.py"],
    ],
    "v837t": [
        [sys.executable, "experiments/v837_primitive_invention/v837t/run_dynamic_granularity.py", "--phase", "anchors"],
        [sys.executable, "experiments/v837_primitive_invention/v837t/run_dynamic_granularity.py", "--phase", "scalarized"],
        [sys.executable, "experiments/v837_primitive_invention/v837t/analyze_results.py"],
    ],
    "v837u": [
        [sys.executable, "experiments/v837_primitive_invention/v837u/run_neutral_followup.py"],
        [sys.executable, "experiments/v837_primitive_invention/v837u/analyze_results.py"],
    ],
    "v837v": [
        [sys.executable, "experiments/v837_primitive_invention/v837v/run_control_scope.py", "--phase", "v0"],
        [sys.executable, "experiments/v837_primitive_invention/v837v/run_control_scope.py", "--phase", "shared"],
        [sys.executable, "experiments/v837_primitive_invention/v837v/analyze_results.py"],
    ],
    "v837w": [
        [sys.executable, "experiments/v837_primitive_invention/v837w/run_controller_information.py"],
        [sys.executable, "experiments/v837_primitive_invention/v837w/analyze_results.py"],
    ],
    "v837x": [
        [sys.executable, "experiments/v837_primitive_invention/v837x/run_global_scalar_control.py"],
        [sys.executable, "experiments/v837_primitive_invention/v837x/analyze_results.py"],
    ],
    "v837y": [
        [sys.executable, "experiments/v837_primitive_invention/v837y/run_candidate_interaction.py", "--phase", "anchors"],
        [sys.executable, "experiments/v837_primitive_invention/v837y/run_candidate_interaction.py", "--phase", "interaction"],
        [sys.executable, "experiments/v837_primitive_invention/v837y/analyze_results.py"],
    ],
    "v837z": [
        [sys.executable, "experiments/v837_primitive_invention/v837z/run_candidate_stage.py", "--phase", "z0"],
        [sys.executable, "experiments/v837_primitive_invention/v837z/run_candidate_stage.py", "--phase", "z1"],
        [sys.executable, "experiments/v837_primitive_invention/v837z/analyze_results.py"],
    ],
    "v837aa": [
        [sys.executable, "experiments/v837_primitive_invention/v837aa/run_candidate_law_audit.py"],
        [sys.executable, "experiments/v837_primitive_invention/v837aa/analyze_results.py"],
    ],
    "v837ab": [
        [sys.executable, "experiments/v837_primitive_invention/v837ab/run_input_factorization.py", "--phase", "preflight"],
        [sys.executable, "experiments/v837_primitive_invention/v837ab/run_input_factorization.py", "--phase", "ab0"],
        [sys.executable, "experiments/v837_primitive_invention/v837ab/run_input_factorization.py", "--phase", "others"],
        [sys.executable, "experiments/v837_primitive_invention/v837ab/analyze_results.py"],
    ],
    "v837ac": [
        [sys.executable, "experiments/v837_primitive_invention/v837ac/run_input_transfer.py", "--phase", "preflight"],
        [sys.executable, "experiments/v837_primitive_invention/v837ac/run_input_transfer.py", "--phase", "ac0"],
        [sys.executable, "experiments/v837_primitive_invention/v837ac/run_input_transfer.py", "--phase", "transfer"],
        [sys.executable, "experiments/v837_primitive_invention/v837ac/analyze_results.py"],
    ],
    "v837ad": [
        [sys.executable, "experiments/v837_primitive_invention/v837ad/run_candidate_geometry.py", "--phase", "preflight"],
        [sys.executable, "experiments/v837_primitive_invention/v837ad/run_candidate_geometry.py", "--phase", "ad0"],
        [sys.executable, "experiments/v837_primitive_invention/v837ad/run_candidate_geometry.py", "--phase", "ad1"],
        [sys.executable, "experiments/v837_primitive_invention/v837ad/run_candidate_geometry.py", "--phase", "geometry"],
        [sys.executable, "experiments/v837_primitive_invention/v837ad/analyze_results.py"],
    ],
}


def display_command(command: list[str]) -> str:
    return " ".join(command)


def enforce_variant_guard(variant: str) -> None:
    if variant == "v837p":
        parent_path = ROOT / "experiments" / "v837_primitive_invention" / "v837o" / "results.json"
        if not parent_path.is_file():
            raise SystemExit("V837p blocked: V837o results are missing")
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        if parent.get("mechanism_diagnosis") != "DYNAMIC_STATE_MODULATION_REQUIRED":
            raise SystemExit("V837p blocked: V837o did not localize dynamic state modulation")
        if parent.get("neutral_followup_allowed") is not True or parent.get("neutral_followup_type") != "single_dynamic_modulator":
            raise SystemExit("V837p blocked: V837o did not authorize the single dynamic-modulator transfer")
        return
    if variant == "v837s":
        decision_path = ROOT / "experiments" / "v837_primitive_invention" / "v837r" / "diagnostics" / "decision_state.json"
        if not decision_path.is_file():
            raise SystemExit("V837s blocked: V837r decision state is missing")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("v837r_complete") is not True or decision.get("interaction_followup_allowed") is not True:
            raise SystemExit("V837s blocked: V837r did not authorize the interaction follow-up")
        if decision.get("best_condition") != "R3_rank4":
            raise SystemExit("V837s blocked: frozen best V837r condition is not R3_rank4")
        return
    if variant == "v837u":
        decision_path = ROOT / "experiments" / "v837_primitive_invention" / "v837t" / "diagnostics" / "decision_state.json"
        if not decision_path.is_file():
            raise SystemExit("V837u blocked: V837t decision state is missing")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        allowed = {"DYNAMIC_SCALAR_CARRY", "POST_TRANSFORM_SCALAR_MODULATION", "DUAL_SCALAR_DYNAMIC_PATHWAYS", "DYNAMIC_VECTOR_STATE_MODULATION"}
        if decision.get("v837t_complete") is not True or decision.get("positive_controls_pass") is not True or decision.get("neutral_followup_allowed") is not True:
            raise SystemExit("V837u blocked: V837t did not authorize a neutral follow-up")
        if decision.get("authorized_v837u_mode") not in allowed:
            raise SystemExit("V837u blocked: V837t authorized mode is invalid")
        return
    if variant == "v837v":
        decision_path = ROOT / "experiments" / "v837_primitive_invention" / "v837u" / "diagnostics" / "decision_state.json"
        if not decision_path.is_file():
            raise SystemExit("V837v blocked: V837u decision state is missing")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("v837u_complete") is not True or decision.get("diagnosis") != "DYNAMIC_SCALAR_CARRY_INSUFFICIENT" or decision.get("representation_adequacy_pass") is not False:
            raise SystemExit("V837v blocked: V837u frontier is incompatible")
        return
    if variant == "v837w":
        decision_path = ROOT / "experiments" / "v837_primitive_invention" / "v837v" / "diagnostics" / "decision_state.json"
        if not decision_path.is_file():
            raise SystemExit("V837w blocked: V837v decision state is missing")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("representation_adequacy_pass") is not False or decision.get("v837w_allowed") is not True:
            raise SystemExit("V837w blocked: V837v restored representation or did not authorize information localization")
        return
    if variant == "v837x":
        decision_path = ROOT / "experiments" / "v837_primitive_invention" / "v837w" / "diagnostics" / "decision_state.json"
        if not decision_path.is_file():
            raise SystemExit("V837x blocked: V837w decision state is missing")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("neutral_global_controller_allowed") is not True:
            raise SystemExit("V837x blocked: V837w did not authorize a global controller")
        config = json.loads((ROOT / "experiments" / "v837_primitive_invention" / "v837x" / "config.json").read_text(encoding="utf-8"))
        if decision.get("authorized_v837x_mode") != config.get("authorized_controller_mode"):
            raise SystemExit("V837x blocked: config mode differs from V837w authorization")
        return
    if variant == "v837y":
        decision_path = ROOT / "experiments" / "v837_primitive_invention" / "v837x" / "diagnostics" / "decision_state.json"
        if not decision_path.is_file():
            raise SystemExit("V837y blocked: V837x decision state is missing")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("v837x_complete") is not True or decision.get("representation_adequacy_pass") is not False:
            raise SystemExit("V837y blocked: V837x frontier is incompatible")
        return
    if variant == "v837z":
        decision_path = ROOT / "experiments" / "v837_primitive_invention" / "v837y" / "diagnostics" / "decision_state.json"
        if not decision_path.is_file():
            raise SystemExit("V837z blocked: V837y decision state is missing")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("v837y_complete") is not True or decision.get("representation_adequacy_pass") is not False or decision.get("v837z_allowed") is not True:
            raise SystemExit("V837z blocked: V837y did not authorize the stage diagnostic")
        config = json.loads((ROOT / "experiments" / "v837_primitive_invention" / "v837z" / "config.json").read_text(encoding="utf-8"))
        if decision.get("selected_v837z_parent") != config.get("selected_parent"):
            raise SystemExit("V837z blocked: selected parent differs from V837y decision")
        return
    if variant == "v837aa":
        status_path = ROOT / "experiments" / "v837_primitive_invention" / "candidate_organization_program_status.json"
        if not status_path.is_file():
            raise SystemExit("V837aa blocked: candidate-organization closure status is missing")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("v837y_diagnosis") != "GLOBAL_CONTROL_X_CANDIDATE_MIXING_INSUFFICIENT" or status.get("v837z_diagnosis") != "HISTORICAL_WITHIN_STEP_CASCADE_BENEFICIAL" or status.get("representation_adequacy") != "FAIL":
            raise SystemExit("V837aa blocked: V837y/V837z frontier is incompatible")
        if status.get("fresh_audit_episodes_consumed") != 0 or status.get("v838_started") is not False:
            raise SystemExit("V837aa blocked: fresh-audit/V838 lock changed")
        return
    if variant == "v837ab":
        decision_path = ROOT / "experiments" / "v837_primitive_invention" / "v837aa" / "diagnostics" / "decision_state.json"
        if not decision_path.is_file():
            raise SystemExit("V837ab blocked: V837aa decision state is missing")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("candidate_law_diagnosis") != "GENUINELY_DIVERSE_CANDIDATE_LAWS" or decision.get("recommended_next_axis") != "NEXT_AXIS_SHARED_INPUT_REPRESENTATION":
            raise SystemExit("V837ab blocked: V837aa did not authorize the input axis")
        if decision.get("fresh_audit_consumed") is not False or decision.get("v838_started") is not False:
            raise SystemExit("V837ab blocked: fresh-audit/V838 lock changed")
        return
    if variant == "v837ac":
        decision_path = ROOT / "experiments" / "v837_primitive_invention" / "v837ab" / "diagnostics" / "decision_state.json"
        if not decision_path.is_file():
            raise SystemExit("V837ac blocked: V837ab decision state is missing")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        config = json.loads((ROOT / "experiments" / "v837_primitive_invention" / "v837ac" / "config.json").read_text(encoding="utf-8"))
        if decision.get("neutral_transfer_allowed") is not True or decision.get("authorized_v837ac_mode") != config.get("authorized_mode"):
            raise SystemExit("V837ac blocked: V837ab authorization does not match V837ac config")
        if decision.get("fresh_audit_consumed") is not False or decision.get("v838_started") is not False:
            raise SystemExit("V837ac blocked: fresh-audit/V838 lock changed")
        return
    if variant == "v837ad":
        status_path = ROOT / "experiments" / "v837_primitive_invention" / "input_factorization_program_status.json"
        if not status_path.is_file():
            raise SystemExit("V837ad blocked: input-factorization closure status is missing")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("v837ab_diagnosis") != "SINGLE_PATH_INPUT_FACTORIZATION_SUFFICIENT" or status.get("v837ac_diagnosis") != "INPUT_ORGANIZATION_TRANSFER_INSUFFICIENT" or status.get("input_axis_closed") is not True:
            raise SystemExit("V837ad blocked: V837ab/V837ac frontier is incompatible")
        if status.get("fresh_audit_episodes_consumed") != 0 or status.get("v838_started") is not False:
            raise SystemExit("V837ad blocked: fresh-audit/V838 lock changed")
        return


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Show or explicitly execute the preserved V837 representation-recovery and learned-reference calibration entrypoints. "
            "By default this command is read-only and does not launch research runs."
        )
    )
    parser.add_argument("--variant", choices=sorted(VARIANT_COMMANDS), required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the variant. Without this flag, only print the preserved commands.",
    )
    args = parser.parse_args()
    enforce_variant_guard(args.variant)

    commands = VARIANT_COMMANDS[args.variant]
    config = ROOT / "experiments" / "v837_primitive_invention" / args.variant / "config.json"
    if not config.is_file():
        raise SystemExit(f"missing preserved config: {config.relative_to(ROOT)}")

    print(f"variant: {args.variant}")
    print(f"config: {config.relative_to(ROOT)}")
    print("commands:")
    for command in commands:
        print(f"  {display_command(command)}")

    if not args.execute:
        print("dry run only; pass --execute to launch the preserved experiment entrypoint(s)")
        return 0

    for command in commands:
        print(f"executing: {display_command(command)}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
