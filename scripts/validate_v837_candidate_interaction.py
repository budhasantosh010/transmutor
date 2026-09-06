from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'experiments/v837_primitive_invention'; HERE=BASE/'v837y'
def fail(m): raise ValueError(m)
def main():
    if not HERE.exists(): return 0
    c=json.loads((HERE/'config.json').read_text()); x=json.loads((BASE/'v837x/diagnostics/decision_state.json').read_text())
    if c.get('parent')!='V837x': fail('parent mismatch')
    if x.get('v837x_complete') is not True or x.get('representation_adequacy_pass') is not False or x.get('diagnosis')!='GLOBAL_SCALAR_CONTROL_PARTIAL_BENEFIT': fail('V837y not authorized by frozen V837x failure')
    expected=['Y0_historical','Y1_global_control','Y2_rank4_candidate','Y3_global_control_rank4_candidate','Y3C_global_control_matched_local']
    if list(c.get('conditions',{}))!=expected: fail('condition set/order drift')
    tr=c['training']
    if (tr['steps'],tr['train_episodes'],tr['validation_episodes'],tr['development_seed_range'],tr['validation_seed_range'],tr['replicates'])!=(192,512,128,[10000,10511],[20000,20127],5): fail('training/data drift')
    if c.get('unique_seed_defined_episodes')!=3200: fail('unique data drift')
    if c.get('state_layout')!='local_10x4' or c.get('total_state_dim')!=40 or c.get('input_projection')!='historical_per_cell' or c.get('message_schedule')!='historical_mixed_same_step_and_recurrent': fail('substrate organization drift')
    if c.get('candidate_coupling_rank')!=4 or any('dense' in str(v.get('candidate_coupling')) for v in c['conditions'].values()): fail('coupling rank/mode drift')
    if c.get('global_controller_mode')!='JOINT_INPUT_STATE_GLOBAL_SCALAR' or c.get('global_controller_parameters')!=47 or c.get('global_controller_macs')!=46: fail('global controller drift')
    if any(c.get(k) is not False for k in ('vector_gates','shared_state','global_controller_message_input','global_controller_candidate_input','structural_search','primitive_mining','fresh_audit_consumed','v838_started')): fail('science lock violated')
    src=(HERE/'candidate_interaction.py').read_text()
    for token in ('snapshot_states = prev_states','global_terms = self._global_terms(snapshot_states','gate = self._global_gate(snapshot_states, x_t)','candidate = torch.tanh(preactivation)','proposed_state = gate * snapshot_states[cell_index] + (1.0 - gate) * candidate'):
        if token not in src: fail('required composition/timing semantics missing: '+token)
    if (HERE/'results.json').exists():
        rows=json.loads((HERE/'raw/runs.json').read_text()).get('rows',[])
        if len(rows)!=125: fail('raw run count !=125')
        if len({(r['family'],r['replicate_id']) for r in rows})!=25: fail('paired family/replicate grid drift')
        if any(r.get('fresh_audit_consumed') is not False or r.get('structural_search_allowed') is not False or r.get('primitive_mining_allowed') is not False or r.get('v838_started') is not False for r in rows): fail('raw science lock violation')
        compat=json.loads((HERE/'diagnostics/baseline_compatibility.json').read_text())
        d=json.loads((HERE/'diagnostics/decision_state.json').read_text())
        allowed={'GLOBAL_CONTROL_X_CROSS_CELL_CANDIDATE_INTERACTION_SUFFICIENT','CANDIDATE_CAPACITY_INTERACTION_SUFFICIENT','GLOBAL_CONTROL_X_CANDIDATE_MIXING_INSUFFICIENT','GLOBAL_CONTROL_X_CANDIDATE_MIXING_INTERFERENCE','INTERACTION_SPECIFICITY_NOT_ESTABLISHED','CANDIDATE_INTERACTION_BASELINE_DRIFT'}
        if d.get('diagnosis') not in allowed: fail('invalid diagnosis')
        if compat.get('compatible') is not d.get('anchors_reproduced'): fail('anchor decision mismatch')
        if d.get('representation_adequacy_pass') is True and (d.get('sample_efficiency_retest_allowed') is not True or d.get('v837z_allowed') is not False): fail('pass gate state invalid')
        if d.get('representation_adequacy_pass') is False and d.get('anchors_reproduced') is True and d.get('v837z_allowed') is not True: fail('authorized V837z missing after valid failure')
        res=json.loads((HERE/'results.json').read_text())
        if res.get('unique_seed_defined_episodes')!=3200 or res.get('fresh_audit_consumed') is not False or res.get('v838_started') is not False: fail('result locks/data drift')
        if int(res['conditions']['Y3_global_control_rank4_candidate']['parameter_count'])!=int(res['conditions']['Y3C_global_control_matched_local']['parameter_count']): fail('Y3/Y3C parameter matching failed')
    print('V837y candidate interaction validation: PASS'); return 0
if __name__=='__main__': raise SystemExit(main())
