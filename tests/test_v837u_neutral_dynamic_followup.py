from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.v837u.dynamic_control import NeutralDynamicCarryModel

ROOT=Path(__file__).resolve().parents[1]; HERE=ROOT/"experiments/v837_primitive_invention/v837u"; CONFIG=json.loads((HERE/"config.json").read_text(encoding="utf-8"))


class V837uTests(unittest.TestCase):
    def test_v837u_requires_v837t_authorization(self):
        d=json.loads((ROOT/"experiments/v837_primitive_invention/v837t/diagnostics/decision_state.json").read_text(encoding="utf-8")); self.assertTrue(d["neutral_followup_allowed"]); self.assertEqual(d["authorized_v837u_mode"],CONFIG["authorized_mode"])
    def test_only_authorized_mode_can_execute(self): self.assertEqual(CONFIG["authorized_mode"],"DYNAMIC_SCALAR_CARRY")
    def test_state_layout_remains_local_10x4(self): self.assertEqual(CONFIG["state_layout"],"local_10x4")
    def test_global_coupling_disabled(self): self.assertFalse(CONFIG["global_coupling_allowed"])
    def test_total_state_dim_40(self): self.assertEqual(CONFIG["total_state_dim"],40)
    def test_graph_unchanged(self): self.assertEqual((CONFIG["graph_cells"],CONFIG["graph_edges"]),(10,55))
    def test_training_regime_unchanged(self): self.assertEqual((CONFIG["training"]["steps"],CONFIG["unique_seed_defined_episodes"]),(192,3200))

    def test_historical_no_modulation_forward_unchanged(self):
        g=high_capacity_generic_graph(0); torch.manual_seed(5); a=NeutralDynamicCarryModel(g,condition="U0_historical_direct"); torch.manual_seed(99); b=NeutralGraphModel(g,obs_dim=6,state_dim=4,message_dim=4,state_update_mode="direct",interaction_mode="none",state_modulation_mode="none"); b.load_state_dict(a.state_dict()); x=torch.randn(3,7,6); l=torch.tensor([7,6,5]); self.assertTrue(torch.equal(a(x,l),b(x,l)))
    def test_v837p_scalar_candidate_forward_unchanged(self):
        g=high_capacity_generic_graph(0); a=NeutralDynamicCarryModel(g,condition="U1_v837p_scalar_candidate"); b=NeutralGraphModel(g,obs_dim=6,state_dim=4,message_dim=4,state_update_mode="direct",interaction_mode="none",state_modulation_mode="dynamic_scalar_candidate"); b.load_state_dict(a.state_dict()); x=torch.randn(2,5,6); self.assertTrue(torch.equal(a(x),b(x)))
    def test_parameter_counts(self):
        g=high_capacity_generic_graph(0); self.assertEqual(NeutralDynamicCarryModel(g,condition="U0_historical_direct").parameter_count(),856); self.assertEqual(NeutralDynamicCarryModel(g,condition="U2_dynamic_scalar_carry").parameter_count(),1006); self.assertEqual(NeutralDynamicCarryModel(g,condition="U2C_scalar_scale_candidate_control").parameter_count(),1006)
    def test_same_controller_count_carry_and_control(self):
        g=high_capacity_generic_graph(0); a=NeutralDynamicCarryModel(g,condition="U2_dynamic_scalar_carry"); b=NeutralDynamicCarryModel(g,condition="U2C_scalar_scale_candidate_control"); self.assertEqual(a.controller_parameter_count,b.controller_parameter_count); self.assertEqual(a.parameter_count(),b.parameter_count())
    def test_dynamic_scalar_carry_is_time_varying(self):
        g=high_capacity_generic_graph(0); m=NeutralDynamicCarryModel(g,condition="U2_dynamic_scalar_carry"); _,tr=m(torch.randn(2,6,6),return_trace=True); self.assertGreater(float(tr.state_modulators.var(dim=1).mean()),0)
    def test_carry_and_control_differ_only_update_equation(self):
        g=high_capacity_generic_graph(0); a=NeutralDynamicCarryModel(g,condition="U2_dynamic_scalar_carry"); b=NeutralDynamicCarryModel(g,condition="U2C_scalar_scale_candidate_control"); b.load_state_dict(a.state_dict()); x=torch.randn(2,5,6); _,ta=a(x,return_trace=True); _,tb=b(x,return_trace=True); self.assertTrue(torch.allclose(ta.state_modulators[:,0,0],tb.state_modulators[:,0,0])); self.assertFalse(torch.allclose(ta.states,tb.states))
    def test_controller_macs(self):
        g=high_capacity_generic_graph(0); self.assertEqual(NeutralDynamicCarryModel(g,condition="U2_dynamic_scalar_carry").controller_macs_per_timestep,140)
    def test_science_locks(self):
        self.assertFalse(CONFIG["structural_search_allowed"]); self.assertFalse(CONFIG["primitive_mining_allowed"]); self.assertFalse(CONFIG["fresh_audit_consumed"]); self.assertFalse(CONFIG["v838_started"])


if __name__=="__main__": unittest.main()
