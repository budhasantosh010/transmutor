from __future__ import annotations

import argparse
import json

from .authorization import assert_authorized
from .source_integrity import verify_source_integrity
from .failure_ledger import initialize_historical_refinements
from .oracle_macrostate import verify_oracle_instrumentation
from .counterfactual_tasks import verify_counterfactual_constructors
from .instrumented_af1d import verify_runtime_equivalence
from .an_a_causal_macrovariables import run_an_a
from .an_a_diagnostics import run_an_a_diagnostics
from .an_b_causal_routing import run_an_b
from .an_c_distributed_synergy import run_an_c
from .meta_confirm import run_meta
from .final_dev_confirm import run_final_dev
from .final_validation import run_final_validation
from .utils import HERE, read_json

STAGES = (
    "source",
    "oracle",
    "counterfactuals",
    "traces",
    "macrovariables",
    "routing",
    "synergy",
    "meta",
    "dev-confirm",
    "final",
    "analyze",
)


def _need(rel: str, message: str):
    path = HERE / rel
    if not path.is_file():
        raise RuntimeError(message)
    return read_json(path) if path.suffix == ".json" else path


def run(stage: str):
    assert_authorized()
    if stage == "source":
        initialize_historical_refinements()
        return verify_source_integrity()

    _need("diagnostics/source_integrity.json", "V837AN_SOURCE_STAGE_REQUIRED")
    if stage == "oracle":
        return verify_oracle_instrumentation()

    _need("diagnostics/oracle_equivalence.json", "V837AN_ORACLE_STAGE_REQUIRED")
    if stage == "counterfactuals":
        return verify_counterfactual_constructors()

    _need("diagnostics/counterfactual_validity.json", "V837AN_COUNTERFACTUAL_STAGE_REQUIRED")
    if stage == "traces":
        return verify_runtime_equivalence()

    _need("diagnostics/instrumented_runtime_equivalence.json", "V837AN_TRACE_STAGE_REQUIRED")
    if stage == "macrovariables":
        result = run_an_a()
        run_an_a_diagnostics()
        return result

    _need("raw/an_a_selection.json", "V837AN_AN_A_REQUIRED")
    if stage == "routing":
        return run_an_b()

    _need("raw/routing_selection.json", "V837AN_AN_B_REQUIRED")
    if stage == "synergy":
        return run_an_c()

    _need("raw/synergy_results.json", "V837AN_AN_C_REQUIRED")
    if stage == "meta":
        return run_meta()

    _need("raw/meta_confirmed_family_abstractions.json", "V837AN_META_REQUIRED")
    if stage == "dev-confirm":
        return run_final_dev()

    _need("raw/frozen_family_abstractions.json", "V837AN_FINAL_FREEZE_REQUIRED")
    if stage == "final":
        return run_final_validation()

    if stage == "analyze":
        _need("raw/final_validation.json", "V837AN_FINAL_VALIDATION_STAGE_REQUIRED")
        from .analyze_results import analyze
        return analyze()

    raise ValueError(stage)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    result = run(args.stage)
    print(json.dumps(result, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
