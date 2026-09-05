from __future__ import annotations
import json, unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.v837_primitive_invention.v837t.gru_dynamic_granularity import DynamicGranularityGRU
from experiments.v837_primitive_invention.v837w.gru_controller_information import ControllerInformationGRU
ROOT=Path(__file__).resolve().parents[1]; HERE=ROOT/'experiments/v837_primitive_invention/v837w'; CONFIG=json.loads((HERE/'config.json').read_text())

class V837wTests(unittest.TestCase):
    def test_authorized_only_after_failed_v837v(self):
        d=json.loads((ROOT/'experiments/v837_primitive_invention/v837v/diagnostics/decision_state.json').read_text()); self.assertFalse(d['representation_adequacy_pass']); self.assertTrue(d['v837w_allowed'])
    def test_w0_equation_matches_t2(self):
        torch.manual_seed(11); a=DynamicGranularityGRU(13,6,condition='T2_scalarized_update_no_reset'); torch.manual_seed(99); b=ControllerInformationGRU(13,6,condition='W0_joint_input_state'); b.load_state_dict(a.state_dict()); x=torch.randn(3,7,6); l=torch.tensor([7,6,5]); pa,ta=a(x,l,return_trace=True); pb,tb=b(x,l,return_trace=True); self.assertTrue(torch.equal(pa,pb)); self.assertTrue(torch.equal(ta.states,tb.states)); self.assertTrue(torch.equal(ta.updates,tb.updates)); self.assertTrue(torch.equal(ta.candidates,tb.candidates))
    def test_update_logit_joint_matches_t2(self):
        m=ControllerInformationGRU(13,6,condition='W0_joint_input_state'); x=torch.randn(4,6); h=torch.randn(4,13); p=m.input_projection(x); v=m._decomposed_components(p,h); gi=F.linear(p,m.weight_ih,m.bias_ih); gh=F.linear(h,m.weight_hh,m.bias_hh); _,iz,_=gi.chunk(3,1); _,hz,_=gh.chunk(3,1); self.assertTrue(torch.allclose(v['raw_update'],torch.sigmoid(iz+hz),rtol=0,atol=1e-7))
    def test_input_only_removes_state_only(self):
        m=ControllerInformationGRU(13,6,condition='W1_input_only'); p=m.input_projection(torch.randn(2,6)); h1=torch.randn(2,13); h2=torch.randn(2,13); self.assertTrue(torch.equal(m._decomposed_components(p,h1)['update'],m._decomposed_components(p,h2)['update']))
    def test_state_only_removes_input_only(self):
        m=ControllerInformationGRU(13,6,condition='W2_state_only'); h=torch.randn(2,13); p1=m.input_projection(torch.randn(2,6)); p2=m.input_projection(torch.randn(2,6)); self.assertTrue(torch.equal(m._decomposed_components(p1,h)['update'],m._decomposed_components(p2,h)['update']))
    def test_bias_only_is_time_invariant(self):
        m=ControllerInformationGRU(13,6,condition='W3_bias_only'); _,tr=m(torch.randn(2,8,6),return_trace=True); self.assertEqual(float(tr.updates.var(dim=1).max()),0.0)
    def test_biases_preserved_and_scalarization_after_sigmoid(self):
        m=ControllerInformationGRU(13,6,condition='W0_joint_input_state'); p=m.input_projection(torch.randn(2,6)); h=torch.randn(2,13); v=m._decomposed_components(p,h); self.assertTrue(torch.allclose(v['update'][:,:,] if v['update'].ndim==3 else v['update'],v['raw_update'].mean(-1,keepdim=True).expand_as(v['raw_update']))); self.assertTrue(torch.any(v['bias_logit']!=0))
    def test_no_reset_same_candidate_all_conditions(self):
        base=ControllerInformationGRU(13,6,condition='W0_joint_input_state'); x=torch.randn(2,5,6); _,t0=base(x,return_trace=True)
        for c in ('W1_input_only','W2_state_only','W3_bias_only'):
            m=ControllerInformationGRU(13,6,condition=c); m.load_state_dict(base.state_dict()); _,tr=m(x,return_trace=True); self.assertEqual(tr.candidates.shape,t0.candidates.shape)
    def test_same_nominal_parameters(self):
        vals=[ControllerInformationGRU(13,6,condition=c).nominal_parameter_count() for c in CONFIG['conditions']]; self.assertEqual(len(set(vals)),1)
    def test_science_locks(self):
        self.assertTrue(CONFIG['reference_only']); self.assertFalse(CONFIG['fresh_audit_consumed']); self.assertFalse(CONFIG['structural_search_allowed']); self.assertFalse(CONFIG['primitive_mining_allowed']); self.assertFalse(CONFIG['v838_started'])

if __name__=='__main__': unittest.main()
