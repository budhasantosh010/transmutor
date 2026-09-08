from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/v837_primitive_invention/v837ai"


def load(path: Path) -> dict:
    if not path.is_file(): raise RuntimeError(f"missing {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    cfg=load(HERE/'config.json')
    parent=load(ROOT/'experiments/v837_primitive_invention/v837af/diagnostics/decision_state.json')
    if parent.get('representation_adequacy_pass') is not True or parent.get('sample_efficiency_retest_allowed') is not True:
        raise RuntimeError('V837af did not authorize V837ai')
    if parent.get('best_passing_condition') != 'AF1D_deshared_candidate_input_factorization':
        raise RuntimeError('V837ai parent is not accepted AF1D')
    for name,key in (('architecture_lock.json','compatible'),('ai4_anchor_reuse.json','compatible'),('initialization_pairing.json','pairing_exact'),('historical_v837l_comparison.json','compatible')):
        obj=load(HERE/'diagnostics'/name)
        if obj.get(key) is not True: raise RuntimeError(f'{name} failed {key}')
    nesting=load(HERE/'diagnostics/data_nesting.json')
    if nesting.get('exact') is not True or nesting.get('strict_nesting') is not True or nesting.get('development_validation_overlap') != 0:
        raise RuntimeError('data nesting invalid')
    if any(nesting.get('duplicates',{}).values()): raise RuntimeError('duplicate data seeds')
    anchor=load(HERE/'raw/ai4_reused_anchor.json')
    if len(anchor.get('rows',[])) != 25: raise RuntimeError('AI4 anchor does not contain 25 rows')
    for regime,multiplier,ntrain in (('ai1_runs.json',1,128),('ai2_runs.json',2,256)):
        raw_path=HERE/'raw'/regime
        if not raw_path.is_file():
            # Preflight-only validation is allowed before new runs exist.
            continue
        raw=load(raw_path); rows=raw.get('rows',[])
        if len(rows) != 25: raise RuntimeError(f'{regime} does not contain 25 rows')
        for row in rows:
            if row.get('data_multiplier') != multiplier or row.get('train_episodes') != ntrain or row.get('validation_episodes') != 128:
                raise RuntimeError(f'{regime} data protocol changed')
            if row.get('parameter_count') != 1643 or row.get('active_macs_per_timestep') != 1206:
                raise RuntimeError(f'{regime} architecture/compute changed')
            if row.get('fresh_audit_consumed') is not False or row.get('structural_search_executed') is not False or row.get('primitive_mining_allowed') is not False or row.get('v838_started') is not False:
                raise RuntimeError(f'{regime} science lock changed')
    results_path=HERE/'results.json'
    if results_path.is_file():
        results=load(results_path); decision=load(HERE/'diagnostics/decision_state.json')
        if results.get('version') != 'V837ai' or decision.get('version') != 'V837ai': raise RuntimeError('V837ai result version mismatch')
        if decision.get('architecture') != 'AF1D_deshared_candidate_input_factorization' or decision.get('architecture_frozen') is not True:
            raise RuntimeError('V837ai architecture decision changed')
        if decision.get('families_passing',{}).get('4x') != 4 or decision.get('representation_adequacy_confirmed') is not True:
            raise RuntimeError('V837ai lost accepted 4x representation anchor')
        if decision.get('fresh_audit_consumed') is not False or decision.get('primitive_mining_allowed') is not False or decision.get('v838_started') is not False:
            raise RuntimeError('V837ai final science locks changed')
        if decision.get('structural_search_recovery_allowed') is not True:
            raise RuntimeError('valid completed V837ai must authorize structural-search recovery')
    audit=load(ROOT/'experiments/v837_primitive_invention/audit/audit_results.json')
    if audit.get('episodes_consumed') != 0: raise RuntimeError('fresh audit consumed')
    for variant in ('v837ae','v837ag','v837ah','v838'):
        if (ROOT/'experiments/v837_primitive_invention'/variant).exists(): raise RuntimeError(f'unauthorized {variant} exists')
    print('V837ai AF1D sample-efficiency validation: PASS')
    return 0

if __name__ == '__main__': raise SystemExit(main())
