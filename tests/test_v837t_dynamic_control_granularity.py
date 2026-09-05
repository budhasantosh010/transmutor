from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.v837_primitive_invention.v837o.factorial_gru import FactorialGRUReferenceModel
from experiments.v837_primitive_invention.v837t.gru_dynamic_granularity import DynamicGranularityGRU, scalarize_dynamic_gate

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/v837_primitive_invention/v837t"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
START = CONFIG["start_sha"]


class DynamicGranularityTests(unittest.TestCase):
    def _paired_models(self, t_condition: str, old_condition: str):
        torch.manual_seed(1234); a = DynamicGranularityGRU(condition=t_condition)
        torch.manual_seed(9999); b = FactorialGRUReferenceModel(condition=old_condition)
        b.load_state_dict(a.state_dict(), strict=True)
        x = torch.randn(4, 7, 6); lengths = torch.tensor([7, 6, 5, 4])
        return a, b, x, lengths

    def test_t0_matches_frozen_full_gru(self):
        a,b,x,l=self._paired_models("T0_full_vector_gru","G0_full_dynamic")
        self.assertTrue(torch.equal(a(x,l),b(x,l)))

    def test_t1_matches_frozen_vector_update_no_reset(self):
        a,b,x,l=self._paired_models("T1_vector_update_no_reset","G1_dynamic_update_no_reset")
        self.assertTrue(torch.equal(a(x,l),b(x,l)))

    def test_t3_matches_frozen_no_update_vector_reset(self):
        a,b,x,l=self._paired_models("T3_no_update_vector_reset","G2_no_update_dynamic_reset")
        self.assertTrue(torch.equal(a(x,l),b(x,l)))

    def test_reset_modulation_remains_post_hidden_transform(self):
        torch.manual_seed(7); m=DynamicGranularityGRU(condition="T3_no_update_vector_reset")
        x=torch.randn(3,6); h=torch.randn(3,13); p=m.input_projection(x); v=m._components(p,h)
        gi=F.linear(p,m.weight_ih,m.bias_ih); gh=F.linear(h,m.weight_hh,m.bias_hh)
        i_r,_,i_n=gi.chunk(3,1); h_r,_,h_n=gh.chunk(3,1); r=torch.sigmoid(i_r+h_r)
        self.assertTrue(torch.allclose(v["candidate"],torch.tanh(i_n+r*h_n)))

    def test_update_convention_matches_frozen_gru(self):
        torch.manual_seed(8); m=DynamicGranularityGRU(condition="T1_vector_update_no_reset")
        x=torch.randn(2,1,6); _,tr=m(x,return_trace=True); z=tr.updates[:,0]; n=tr.candidates[:,0]
        self.assertTrue(torch.allclose(tr.states[:,0],(1-z)*n))

    def test_scalarization_preserves_shape(self):
        g=torch.rand(3,13); self.assertEqual(scalarize_dynamic_gate(g).shape,g.shape)

    def test_scalarization_makes_all_gate_dimensions_equal(self):
        g=torch.rand(3,13); s=scalarize_dynamic_gate(g); self.assertTrue(torch.allclose(s,s[:,:1].expand_as(s)))

    def test_scalarization_preserves_post_sigmoid_mean(self):
        logits=torch.randn(5,13); g=torch.sigmoid(logits); s=scalarize_dynamic_gate(g); self.assertTrue(torch.allclose(s.mean(-1),g.mean(-1)))

    def test_scalarization_is_time_varying(self):
        m=DynamicGranularityGRU(condition="T2_scalarized_update_no_reset"); _,tr=m(torch.randn(3,8,6),return_trace=True)
        self.assertGreater(float(tr.updates.var(dim=1).mean()),0.0)

    def test_scalarization_gradient_reaches_all_vector_outputs(self):
        g=torch.rand(4,13,requires_grad=True); scalarize_dynamic_gate(g).sum().backward(); self.assertTrue(torch.all(g.grad!=0))

    def test_scalarized_condition_keeps_same_nominal_parameters(self):
        counts={DynamicGranularityGRU(condition=c).nominal_parameter_count() for c in CONFIG["conditions"]}; self.assertEqual(counts,{875})

    def test_t2_has_scalarized_update_and_no_reset(self):
        m=DynamicGranularityGRU(condition="T2_scalarized_update_no_reset"); _,tr=m(torch.randn(2,4,6),return_trace=True)
        self.assertTrue(torch.allclose(tr.updates,tr.updates[:,:,:1].expand_as(tr.updates))); self.assertTrue(torch.equal(tr.resets,torch.ones_like(tr.resets)))

    def test_t4_has_no_update_and_scalarized_reset(self):
        m=DynamicGranularityGRU(condition="T4_no_update_scalarized_reset"); _,tr=m(torch.randn(2,4,6),return_trace=True)
        self.assertTrue(torch.equal(tr.updates,torch.zeros_like(tr.updates))); self.assertTrue(torch.allclose(tr.resets,tr.resets[:,:,:1].expand_as(tr.resets)))

    def test_t5_has_two_independent_scalarized_dynamic_paths(self):
        m=DynamicGranularityGRU(condition="T5_dual_scalarized"); _,tr=m(torch.randn(2,5,6),return_trace=True)
        self.assertTrue(torch.allclose(tr.updates,tr.updates[:,:,:1].expand_as(tr.updates))); self.assertTrue(torch.allclose(tr.resets,tr.resets[:,:,:1].expand_as(tr.resets))); self.assertFalse(torch.allclose(tr.updates,tr.resets))

    def test_no_static_gate_substitution(self):
        for c in CONFIG["conditions"]:
            self.assertFalse(hasattr(DynamicGranularityGRU(condition=c),"static_update_logit"))

    def test_v837t_uses_4x_unique_data(self): self.assertEqual(CONFIG["data_regime"],"4x_unique")
    def test_v837t_uses_3200_unique_family_seed_episodes(self): self.assertEqual(CONFIG["unique_seed_defined_episodes"],3200)
    def test_v837t_uses_192_steps(self): self.assertEqual(CONFIG["training"]["steps"],192)
    def test_v837t_uses_paired_replicates(self): self.assertEqual(CONFIG["training"]["replicates"],5)
    def test_task_labels_not_inputs(self): self.assertFalse(CONFIG["task_family_label_allowed"])
    def test_structural_search_locked(self): self.assertFalse(CONFIG["structural_search_allowed"])
    def test_primitive_mining_locked(self): self.assertFalse(CONFIG["primitive_mining_allowed"])
    def test_fresh_audit_unused(self): self.assertFalse(CONFIG["fresh_audit_consumed"])
    def test_v838_not_started(self): self.assertFalse(CONFIG["v838_started"])

    def test_historical_results_unchanged(self):
        paths=["archive","registry"]+[f"experiments/v837_primitive_invention/{v}" for v in ["v837","v837b","v837c","v837d","v837g","v837h","v837j","v837k","v837l","v837m","v837n","v837o","v837p","v837q","v837r","v837s"]]
        out=subprocess.check_output(["git","diff","--name-only",START,"--",*paths],cwd=ROOT,text=True).strip(); self.assertEqual(out,"")

    def test_v837t_result_decision_is_machine_authorized_scalar_carry(self):
        result=json.loads((HERE/"results.json").read_text(encoding="utf-8")); decision=json.loads((HERE/"diagnostics/decision_state.json").read_text(encoding="utf-8"))
        self.assertEqual({name:int(row["families_passing"]) for name,row in result["conditions"].items()},{"T0_full_vector_gru":5,"T1_vector_update_no_reset":5,"T2_scalarized_update_no_reset":4,"T3_no_update_vector_reset":5,"T4_no_update_scalarized_reset":3,"T5_dual_scalarized":3})
        self.assertEqual(result["diagnosis"],"DYNAMIC_VECTOR_GRANULARITY_NOT_REQUIRED"); self.assertEqual(decision["authorized_v837u_mode"],"DYNAMIC_SCALAR_CARRY")


if __name__ == "__main__": unittest.main()
