from __future__ import annotations
import json, unittest
from pathlib import Path
import torch
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel
from experiments.v837_primitive_invention.v837u.dynamic_control import NeutralDynamicCarryModel
from experiments.v837_primitive_invention.v837x.global_scalar_control import GlobalScalarNeutralModel
ROOT=Path(__file__).resolve().parents[1]; HERE=ROOT/'experiments/v837_primitive_invention/v837x'; CONFIG=json.loads((HERE/'config.json').read_text())
class V837xTests(unittest.TestCase):
    def test_requires_v837w_authorization(self):
        w=json.loads((ROOT/'experiments/v837_primitive_invention/v837w/diagnostics/decision_state.json').read_text()); self.assertTrue(w['neutral_global_controller_allowed']); self.assertEqual(w['authorized_v837x_mode'],CONFIG['authorized_controller_mode'])
    def test_only_authorized_mode_executes(self): self.assertEqual(CONFIG['authorized_controller_mode'],'JOINT_INPUT_STATE_GLOBAL_SCALAR')
    def test_joint_controller_has_47_params(self):
        m=GlobalScalarNeutralModel(high_capacity_generic_graph(0),condition='X2_global_scalar_carry',authorized_mode='JOINT_INPUT_STATE_GLOBAL_SCALAR'); self.assertEqual(m.controller_param_count,47); self.assertEqual(m.controller_macs,46); self.assertEqual(m.global_ws.numel(),40); self.assertEqual(m.global_wx.numel(),6)
    def test_input_only_has_7_params_and_never_reads_state(self):
        m=GlobalScalarNeutralModel(high_capacity_generic_graph(0),condition='X2_global_scalar_carry',authorized_mode='INPUT_ONLY_GLOBAL_SCALAR'); self.assertEqual(m.controller_param_count,7); self.assertIsNone(m.global_ws)
    def test_state_only_has_41_params_and_never_reads_input(self):
        m=GlobalScalarNeutralModel(high_capacity_generic_graph(0),condition='X2_global_scalar_carry',authorized_mode='STATE_ONLY_GLOBAL_SCALAR'); self.assertEqual(m.controller_param_count,41); self.assertIsNone(m.global_wx)
    def test_one_scalar_broadcast_all_cells(self):
        m=GlobalScalarNeutralModel(high_capacity_generic_graph(0),condition='X2_global_scalar_carry'); _,tr=m(torch.randn(2,6,6),return_trace=True); self.assertTrue(torch.equal(tr.state_modulators,tr.state_modulators[:,:,:1,:].expand_as(tr.state_modulators)))
    def test_global_scalar_computed_from_previous_state_once(self):
        m=GlobalScalarNeutralModel(high_capacity_generic_graph(0),condition='X2_global_scalar_carry'); x=torch.randn(2,6); prev=[torch.randn(2,4) for _ in range(10)]; g=m._global_gate(prev,x); self.assertEqual(tuple(g.shape),(2,1)); self.assertTrue(torch.isfinite(g).all())
    def test_x2c_same_controller_parameters_as_x2(self):
        torch.manual_seed(7); a=GlobalScalarNeutralModel(high_capacity_generic_graph(0),condition='X2_global_scalar_carry'); torch.manual_seed(7); b=GlobalScalarNeutralModel(high_capacity_generic_graph(0),condition='X2C_global_scale_candidate_control'); self.assertTrue(torch.equal(a.global_ws,b.global_ws)); self.assertTrue(torch.equal(a.global_wx,b.global_wx)); self.assertTrue(torch.equal(a.global_b,b.global_b)); self.assertEqual(a.parameter_count(),b.parameter_count())
    def test_x2c_has_no_old_state_carry(self):
        g=high_capacity_generic_graph(0); torch.manual_seed(7); a=GlobalScalarNeutralModel(g,condition='X2_global_scalar_carry'); torch.manual_seed(7); b=GlobalScalarNeutralModel(g,condition='X2C_global_scale_candidate_control'); b.load_state_dict(a.state_dict()); x=torch.randn(2,5,6); _,ta=a(x,return_trace=True); _,tb=b(x,return_trace=True); self.assertTrue(torch.equal(ta.state_modulators[:,0],tb.state_modulators[:,0])); self.assertFalse(torch.allclose(ta.states,tb.states))
    def test_x0_matches_historical_direct(self):
        g=high_capacity_generic_graph(0); torch.manual_seed(5); a=GlobalScalarNeutralModel(g,condition='X0_historical_direct'); torch.manual_seed(99); b=NeutralGraphModel(g,obs_dim=6,state_dim=4,message_dim=4,state_update_mode='direct',interaction_mode='none',state_modulation_mode='none'); b.load_state_dict(a.state_dict()); x=torch.randn(3,7,6); l=torch.tensor([7,6,5]); self.assertTrue(torch.equal(a(x,l),b(x,l)))
    def test_x1_matches_v837u_u2(self):
        g=high_capacity_generic_graph(0); torch.manual_seed(5); a=GlobalScalarNeutralModel(g,condition='X1_local_scalar_carry'); torch.manual_seed(99); b=NeutralDynamicCarryModel(g,condition='U2_dynamic_scalar_carry'); b.load_state_dict(a.state_dict()); x=torch.randn(2,6,6); l=torch.tensor([6,5]); pa,ta=a(x,l,return_trace=True); pb,tb=b(x,l,return_trace=True); self.assertTrue(torch.equal(pa,pb)); self.assertTrue(torch.equal(ta.states,tb.states)); self.assertTrue(torch.equal(ta.state_modulators,tb.state_modulators))
    def test_science_locks(self):
        self.assertFalse(CONFIG['global_recurrent_coupling']); self.assertFalse(CONFIG['vector_modulation']); self.assertFalse(CONFIG['messages_into_global_controller']); self.assertFalse(CONFIG['candidate_states_into_global_controller']); self.assertFalse(CONFIG['fresh_audit_consumed']); self.assertFalse(CONFIG['structural_search_allowed']); self.assertFalse(CONFIG['primitive_mining_allowed']); self.assertFalse(CONFIG['v838_started'])
if __name__=='__main__': unittest.main()
