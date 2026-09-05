from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.v837u.dynamic_control import NeutralDynamicCarryModel
from experiments.v837_primitive_invention.v837v.control_scope import DOMAIN_SPECS, NeutralControlScopeModel

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/v837_primitive_invention/v837v"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


class V837vControlScopeTests(unittest.TestCase):
    def test_domain_counts(self):
        self.assertEqual([DOMAIN_SPECS[n].domain_count for n in ("V0_10_domains","V1_5_domains","V2_2_domains","V3_1_domain")],[10,5,2,1])

    def test_every_cell_in_exactly_one_domain(self):
        for spec in DOMAIN_SPECS.values():
            flat=[c for d in spec.domains for c in d]
            self.assertEqual(sorted(flat),list(range(10)))
            self.assertEqual(len(flat),len(set(flat)))

    def test_each_source_belongs_and_is_earliest(self):
        for spec in DOMAIN_SPECS.values():
            for domain,source in zip(spec.domains,spec.source_cells):
                self.assertIn(source,domain); self.assertEqual(source,min(domain))

    def test_domain_layout_task_independent_and_deterministic(self):
        for name,spec in DOMAIN_SPECS.items():
            self.assertEqual(spec,DOMAIN_SPECS[name]); self.assertNotIn("family",repr(spec).lower())

    def test_v0_matches_v837u_u2_forward_and_trace(self):
        graph=high_capacity_generic_graph(0); torch.manual_seed(17)
        u2=NeutralDynamicCarryModel(graph,condition="U2_dynamic_scalar_carry")
        v0=NeutralControlScopeModel(graph,domain_spec=DOMAIN_SPECS["V0_10_domains"])
        v0.load_state_dict(u2.state_dict())
        x=torch.randn(3,8,6); lengths=torch.tensor([8,7,5])
        pu,tu=u2(x,lengths,return_trace=True); pv,tv=v0(x,lengths,return_trace=True)
        self.assertTrue(torch.allclose(pu,pv,rtol=0,atol=1e-7))
        self.assertTrue(torch.allclose(tu.states,tv.states,rtol=0,atol=1e-7))
        self.assertTrue(torch.allclose(tu.state_modulators,tv.state_modulators,rtol=0,atol=1e-7))
        self.assertEqual(u2.parameter_count(),v0.parameter_count())

    def test_source_gate_controls_all_followers_and_v3_identical(self):
        graph=high_capacity_generic_graph(0); model=NeutralControlScopeModel(graph,domain_spec=DOMAIN_SPECS["V3_1_domain"])
        _,trace=model(torch.randn(2,6,6),return_trace=True)
        gates=trace.state_modulators
        for cell in range(1,10): self.assertTrue(torch.equal(gates[:,:,0],gates[:,:,cell]))

    def test_different_domains_can_use_different_sources(self):
        graph=high_capacity_generic_graph(0); model=NeutralControlScopeModel(graph,domain_spec=DOMAIN_SPECS["V2_2_domains"])
        _,trace=model(torch.randn(2,7,6),return_trace=True)
        self.assertTrue(torch.equal(trace.state_modulators[:,:,0],trace.state_modulators[:,:,4]))
        self.assertTrue(torch.equal(trace.state_modulators[:,:,5],trace.state_modulators[:,:,9]))

    def test_follower_controller_not_used_primary_update(self):
        graph=high_capacity_generic_graph(0); model=NeutralControlScopeModel(graph,domain_spec=DOMAIN_SPECS["V3_1_domain"])
        x=torch.randn(2,6,6); baseline=model(x)
        with torch.no_grad():
            for i in range(1,10):
                model.cell_gs[i].fill_(999); model.cell_gm[i].fill_(-999); model.cell_gx[i].fill_(777); model.cell_gb[i].fill_(555)
        self.assertTrue(torch.equal(baseline,model(x)))

    def test_active_controller_counts_params_macs(self):
        expected={"V0_10_domains":(10,150,140),"V1_5_domains":(5,75,70),"V2_2_domains":(2,30,28),"V3_1_domain":(1,15,14)}
        graph=high_capacity_generic_graph(0)
        for name,(count,params,macs) in expected.items():
            model=NeutralControlScopeModel(graph,domain_spec=DOMAIN_SPECS[name])
            self.assertEqual(model.active_controller_count,count); self.assertEqual(model.active_controller_parameter_count,params); self.assertEqual(model.controller_macs_per_timestep,macs)
            self.assertEqual(model.nominal_controller_parameter_count,150)

    def test_no_message_control_zeroes_messages_but_keeps_recurrence(self):
        graph=high_capacity_generic_graph(0); model=NeutralControlScopeModel(graph,domain_spec=DOMAIN_SPECS["V1_5_domains"])
        _,trace=model(torch.randn(2,5,6),return_trace=True,disable_messages=True)
        self.assertEqual(float(trace.messages.abs().max()),0.0); self.assertGreater(float(trace.recurrent_terms.abs().sum()),0.0); self.assertGreater(float(trace.input_terms.abs().sum()),0.0)

    def test_science_locks_and_no_pooling(self):
        self.assertEqual(CONFIG["data_regime"],"4x_unique"); self.assertEqual(CONFIG["unique_seed_defined_episodes"],3200); self.assertEqual(CONFIG["training"]["steps"],192)
        self.assertFalse(CONFIG["global_state_visibility"]); self.assertFalse(CONFIG["global_controller"]); self.assertFalse(CONFIG["gate_pooling"]); self.assertFalse(CONFIG["vector_modulation"]); self.assertFalse(CONFIG["global_recurrent_coupling"])
        self.assertFalse(CONFIG["structural_search"]); self.assertFalse(CONFIG["primitive_mining"]); self.assertFalse(CONFIG["fresh_audit_consumed"]); self.assertFalse(CONFIG["v838_started"])
        source=(HERE/"control_scope.py").read_text(encoding="utf-8")
        self.assertNotIn("mean(g_",source); self.assertNotIn("stack(source_gates).mean",source)


if __name__ == "__main__": unittest.main()
