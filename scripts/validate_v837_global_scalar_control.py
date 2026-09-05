from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'experiments/v837_primitive_invention'; HERE=BASE/'v837x'
def fail(m): raise ValueError(m)
def main():
    if not HERE.exists(): return 0
    c=json.loads((HERE/'config.json').read_text()); w=json.loads((BASE/'v837w/diagnostics/decision_state.json').read_text())
    if c.get('parent')!='V837w': fail('parent mismatch')
    if w.get('neutral_global_controller_allowed') is not True or w.get('authorized_v837x_mode')!=c.get('authorized_controller_mode'): fail('authorization mismatch')
    if c.get('conditions')!=['X0_historical_direct','X1_local_scalar_carry','X2_global_scalar_carry','X2C_global_scale_candidate_control']: fail('condition set drift')
    if c.get('authorized_controller_mode') not in {'INPUT_ONLY_GLOBAL_SCALAR','STATE_ONLY_GLOBAL_SCALAR','JOINT_INPUT_STATE_GLOBAL_SCALAR'}: fail('invalid mode')
    tr=c['training']
    if (tr['steps'],tr['train_episodes'],tr['validation_episodes'],tr['development_seed_range'],tr['validation_seed_range'],tr['replicates'])!=(192,512,128,[10000,10511],[20000,20127],5): fail('training/data drift')
    if c.get('unique_seed_defined_episodes')!=3200: fail('unique data drift')
    if c.get('global_recurrent_coupling') is not False or c.get('vector_modulation') is not False or c.get('messages_into_global_controller') is not False or c.get('candidate_states_into_global_controller') is not False: fail('global controller scope widened')
    if any(c.get(k) is not False for k in ('fresh_audit_consumed','structural_search_allowed','primitive_mining_allowed','v838_started')): fail('science lock violated')
    src=(HERE/'global_scalar_control.py').read_text()
    for token in ('torch.cat(prev_states,dim=1)','global_gate=self._global_gate(prev_states,x_t)','torch.sigmoid(terms+self.global_b)','g*prev_states[i]+(1-g)*cand','else g*cand'):
        if token not in src: fail('required timing/control semantics missing: '+token)
    if (HERE/'results.json').exists():
        rows=json.loads((HERE/'raw/runs.json').read_text()).get('rows',[])
        if len(rows)!=100: fail('raw run count !=100')
        if any(r.get('fresh_audit_consumed') is not False or r.get('task_family_label_in_model_input') is not False for r in rows): fail('raw science lock violation')
        if json.loads((HERE/'diagnostics/anchor_guard.json').read_text()).get('compatible') is not True: fail('historical anchors drifted')
        d=json.loads((HERE/'diagnostics/decision_state.json').read_text()); allowed={'GLOBAL_TEMPORAL_CONTROL_WITH_ADAPTIVE_CARRY_SUFFICIENT','GLOBAL_DYNAMIC_SIGNAL_SUFFICIENT','GLOBAL_SCALAR_CONTROL_PARTIAL_BENEFIT','GLOBAL_SCALAR_CONTROLLER_TRANSFER_FAILURE'}
        if d.get('diagnosis') not in allowed: fail('invalid diagnosis')
        if d.get('representation_adequacy_pass') is True and d.get('sample_efficiency_retest_allowed') is not True: fail('sample efficiency not opened after pass')
    print('V837x global scalar control validation: PASS'); return 0
if __name__=='__main__': raise SystemExit(main())
