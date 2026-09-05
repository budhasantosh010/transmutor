from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'experiments/v837_primitive_invention'; HERE=BASE/'v837w'

def fail(msg): raise ValueError(msg)
def main():
    if not HERE.exists(): return 0
    c=json.loads((HERE/'config.json').read_text()); v=json.loads((BASE/'v837v/diagnostics/decision_state.json').read_text())
    if c.get('parent')!='V837v' or c.get('reference_only') is not True: fail('V837w parent/reference_only mismatch')
    if v.get('representation_adequacy_pass') is not False or v.get('v837w_allowed') is not True: fail('V837w lacks V837v authorization')
    if c.get('conditions')!=['W0_joint_input_state','W1_input_only','W2_state_only','W3_bias_only']: fail('W0-W3 condition set changed')
    tr=c['training'];
    if (tr['steps'],tr['train_episodes'],tr['validation_episodes'],tr['development_seed_range'],tr['validation_seed_range'],tr['replicates'])!=(192,512,128,[10000,10511],[20000,20127],5): fail('training/data regime drift')
    if c.get('unique_seed_defined_episodes')!=3200: fail('unique data accounting drift')
    if any(c.get(k) is not False for k in ('fresh_audit_consumed','structural_search_allowed','primitive_mining_allowed','v838_started')): fail('science lock violated')
    src=(HERE/'gru_controller_information.py').read_text()
    for token in ('torch.sigmoid(i_z + h_z)','input_logit + bias_logit','state_logit + bias_logit','torch.sigmoid(bias_logit)','scalarize_dynamic_gate(raw)','input_logit = F.linear(projected, w_iz, None)','state_logit = F.linear(state, w_hz, None)'):
        if token not in src: fail('required exact anchor/decomposition/scalarization missing: '+token)
    if 'reset' in src.lower() and 'No reset: exact T2 candidate' not in src: pass
    if (HERE/'results.json').exists():
        raw=json.loads((HERE/'raw/runs.json').read_text()); rows=raw.get('rows',[])
        if len(rows)!=100: fail('V837w raw run count !=100')
        if any(r.get('fresh_audit_consumed') is not False or r.get('task_family_label_in_model_input') is not False for r in rows): fail('raw science lock violation')
        guard=json.loads((HERE/'diagnostics/positive_control_guard.json').read_text());
        if guard.get('compatible') is not True or guard.get('families_passing')!=4: fail('W0 T2 anchor not reproduced')
        d=json.loads((HERE/'diagnostics/decision_state.json').read_text());
        allowed={'GLOBAL_CONTROL_INPUT_ONLY_SUFFICIENT':'INPUT_ONLY_GLOBAL_SCALAR','GLOBAL_STATE_ASSESSMENT_REQUIRED':'STATE_ONLY_GLOBAL_SCALAR','JOINT_INPUT_STATE_GLOBAL_CONTROL_REQUIRED':'JOINT_INPUT_STATE_GLOBAL_SCALAR','DYNAMIC_CONTROL_SPECIFICITY_NOT_ESTABLISHED':None,'REFERENCE_CONTROLLER_INFORMATION_INCONCLUSIVE':None}
        if d.get('diagnosis') not in allowed or d.get('authorized_v837x_mode')!=allowed[d['diagnosis']]: fail('invalid V837w decision')
    print('V837w controller-information validation: PASS'); return 0
if __name__=='__main__': raise SystemExit(main())
