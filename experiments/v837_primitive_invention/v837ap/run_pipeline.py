from __future__ import annotations

import argparse
import json

from .baseline_reproduction import reproduce_v837ao_anchor
from .causal_closure import run_causal_closure
from .chart_discovery import run_discovery_reader_ladder
from .cross_organism_agreement import run_cross_organism_agreement
from .discovery_final import run_discovery_final
from .freeze_geometry import freeze_family_geometries
from .geometry_discovery import run_discovery_geometry_selection
from .heldout_program import run_heldout_program
from .historical_robustness import run_historical_robustness
from .law_recovery import run_law_recovery
from .meta_confirm import run_meta_confirm
from .projected_causal_spaces import run_projected_spaces
from .source_integrity import verify_source_integrity
from .utils import HERE


def _need(*paths:str)->None:
    missing=[p for p in paths if not (HERE/p).is_file()]
    if missing:raise RuntimeError(f"V837AP_PREDECESSOR_ARTIFACT_MISSING:{missing}")


def run(stage:str):
    if stage=="source":return verify_source_integrity()
    if stage=="baseline":
        _need("diagnostics/source_integrity.json");return reproduce_v837ao_anchor()
    if stage in {"k1-charts","projected-charts"}:
        _need("raw/v837ao_baseline_reproduction.json");run_projected_spaces();return run_discovery_reader_ladder()
    if stage in {"tangent","atlas","setpoints"}:
        _need("diagnostics/chart_conditioning.json");return run_discovery_geometry_selection()
    if stage in {"quotient","dynamics"}:
        _need("raw/discovery_family_geometry_winners.json");return run_causal_closure()
    if stage=="meta":
        _need("raw/model_complexity_adjudication.json");return run_meta_confirm()
    if stage=="final":
        _need("raw/meta_confirmation.json");return run_discovery_final()
    if stage=="freeze":
        _need("raw/discovery_final.json");return freeze_family_geometries()
    if stage=="heldout":
        _need("raw/frozen_v837ap_family_geometries.json");return run_heldout_program()
    if stage=="agreement":
        _need("raw/heldout_calibration_frontier.json");return run_cross_organism_agreement()
    if stage=="robustness":
        _need("raw/heldout_calibration_frontier.json");return run_historical_robustness()
    if stage=="law-audit":
        _need("raw/frozen_v837ap_family_geometries.json");return run_law_recovery()
    if stage=="analyze":
        _need("raw/cross_organism_agreement.json","raw/historical_validation_robustness.json","raw/law_recovery.json")
        from .analyze_results import analyze
        return analyze()
    raise KeyError(stage)


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",required=True,choices=("source","baseline","k1-charts","projected-charts","tangent","atlas","setpoints","quotient","dynamics","meta","final","freeze","heldout","agreement","robustness","law-audit","analyze"));a=p.parse_args();print(json.dumps(run(a.stage),indent=2,default=str))

if __name__=="__main__":main()
