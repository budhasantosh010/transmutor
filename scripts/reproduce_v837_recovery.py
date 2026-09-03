from __future__ import annotations

import argparse
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
}


def display_command(command: list[str]) -> str:
    return " ".join(command)


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
