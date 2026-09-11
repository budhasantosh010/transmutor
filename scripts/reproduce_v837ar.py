from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v837_primitive_invention.v837ar.run_pipeline import STAGES, run_all, run_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce V837ar causal-operator canonicalization / Program IR.")
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    stages = (args.stage,) if args.stage else STAGES
    for stage in stages:
        print(f"{sys.executable} -m experiments.v837_primitive_invention.v837ar.run_pipeline --stage {stage}")
    if not args.execute:
        print("dry run only; pass --execute to run")
        return 0
    if args.stage:
        run_stage(args.stage, {})
    else:
        # Keep the full run in one process so stage wall times survive through
        # the final analysis/resource-accounting stage.
        run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
