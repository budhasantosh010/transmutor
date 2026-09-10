from __future__ import annotations

import argparse
from pathlib import Path

from .analyze_results import analyze
from .authorization import assert_authorized
from .cross_organism_agreement import run_cross_organism_agreement
from .discovery_backends import run_discovery_backends
from .freeze_canonical_specs import freeze_canonical_specs
from .heldout_backend_calibration import run_heldout_calibration
from .historical_robustness import run_historical_robustness
from .law_recovery import run_law_recovery
from .meta_confirm import run_meta_confirm
from .organism_folds import freeze_organism_folds
from .setpoint_grid import freeze_target_grids
from .source_integrity import verify_source_integrity
from .utils import HERE


def _need(*paths: str) -> None:
    missing=[p for p in paths if not (HERE/p).is_file()]
    if missing:raise RuntimeError(f"V837AO_PREDECESSOR_ARTIFACT_MISSING:{missing}")


def run(stage: str):
    if stage=='source':return verify_source_integrity()
    if stage=='folds':
        _need('diagnostics/source_integrity.json');out=freeze_organism_folds();freeze_target_grids();return out
    if stage in {'backends','setpoints','phases','quotient','dynamics'}:
        _need('raw/frozen_organism_folds.json','raw/canonical_target_grids.json');return run_discovery_backends()
    if stage=='meta':
        _need('raw/discovery_family_backend_winners.json');return run_meta_confirm()
    if stage=='freeze':
        _need('raw/meta_confirmation.json');return freeze_canonical_specs()
    if stage=='heldout':
        _need('raw/frozen_canonical_family_specs.json');return run_heldout_calibration()
    if stage=='agreement':
        _need('raw/heldout_calibration_frontier.json');return run_cross_organism_agreement()
    if stage=='robustness':
        _need('raw/frozen_canonical_family_specs.json','raw/heldout_calibration_frontier.json');return run_historical_robustness()
    if stage=='law-audit':
        _need('raw/frozen_canonical_family_specs.json');return run_law_recovery()
    if stage=='analyze':
        _need('raw/meta_confirmation.json','raw/frozen_canonical_family_specs.json','raw/heldout_calibration_frontier.json','raw/cross_organism_agreement.json','raw/historical_validation_robustness.json','raw/law_recovery.json');return analyze()
    raise KeyError(stage)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=('source','folds','backends','setpoints','phases','quotient','dynamics','meta','freeze','heldout','agreement','robustness','law-audit','analyze'));args=parser.parse_args();out=run(args.stage)
    import json;print(json.dumps(out,indent=2,default=str))


if __name__=='__main__':main()
