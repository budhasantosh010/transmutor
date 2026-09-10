from __future__ import annotations

import argparse
from pathlib import Path

from .utils import HERE


def _need(path:str,msg:str)->None:
    if not (HERE/path).is_file():raise RuntimeError(msg)


def run(stage:str):
    if stage=="source":
        from .source_integrity import verify_source_integrity;return verify_source_integrity()
    if stage=="reality":
        from .reality_gate import run_reality_gate;return run_reality_gate()
    if stage=="operator":
        from .operator_discovery import run_operator_discovery;return run_operator_discovery()
    if stage=="response-tensors":
        from .response_tensors import materialize_response_tensors;return materialize_response_tensors()
    if stage=="localize":
        _need("raw/operator_discovery.json","AQ6 requires operator discovery")
        from .localization import run_localization;return run_localization()
    if stage=="composition":
        _need("raw/operator_discovery.json","AQ8 requires operator discovery")
        from .composition import run_composition;return run_composition()
    if stage=="predictive":
        _need("raw/composition_results.json","AQ9 requires composition results")
        from .predictive_state import run_predictive_state_diagnostic;return run_predictive_state_diagnostic()
    if stage=="meta":
        _need("raw/localization_results.json","META requires localization")
        _need("raw/composition_results.json","META requires composition")
        from .meta_confirm import run_meta_confirm;return run_meta_confirm()
    if stage=="freeze":
        _need("raw/meta_confirmation.json","freeze requires META")
        from .freeze_operator_contracts import freeze_operator_contracts;return freeze_operator_contracts()
    if stage=="heldout":
        _need("raw/frozen_program_operator_contracts.json","heldout requires frozen contracts")
        from .heldout import run_heldout_confirmation;return run_heldout_confirmation()
    if stage=="robustness":
        _need("raw/frozen_program_operator_contracts.json","robustness requires frozen contracts")
        from .historical_robustness import run_historical_robustness;return run_historical_robustness()
    if stage=="analyze":
        from .analyze_results import analyze;return analyze()
    raise KeyError(stage)


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("source","reality","operator","response-tensors","localize","composition","predictive","meta","freeze","heldout","robustness","analyze"),required=True);a=p.parse_args();run(a.stage);return 0


if __name__=="__main__":raise SystemExit(main())
