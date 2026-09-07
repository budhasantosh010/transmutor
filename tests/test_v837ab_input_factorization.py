from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.v837ab.input_factorization import (
    CONDITIONS,
    HIDDEN_SIZE,
    INPUT_DIM,
    InputFactorizationT2,
    fold_full_weight_ih,
    fold_linear,
    max_trace_errors,
)
from experiments.v837_primitive_invention.v837t.gru_dynamic_granularity import DynamicGranularityGRU

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "experiments/v837_primitive_invention/v837ab/config.json").read_text(encoding="utf-8"))


def source(seed: int = 17) -> DynamicGranularityGRU:
    torch.manual_seed(seed)
    return DynamicGranularityGRU(HIDDEN_SIZE, INPUT_DIM, condition="T2_scalarized_update_no_reset")


def models_from_same_source(seed: int = 17):
    s = source(seed)
    state = s.state_dict()
    return s, {condition: InputFactorizationT2(condition=condition, source_state=state) for condition in CONDITIONS}


class V837abInputFactorizationTests(unittest.TestCase):
    def test_linear_projection_folding_exact(self):
        torch.manual_seed(1)
        A = torch.randn(6, 6); a = torch.randn(6); W = torch.randn(13, 6); b = torch.randn(13); x = torch.randn(11, 6)
        We, be = fold_linear(W, b, A, a)
        self.assertLessEqual(float((torch.nn.functional.linear(torch.nn.functional.linear(x, A, a), W, b) - torch.nn.functional.linear(x, We, be)).abs().max()), 1e-5)

    def test_full_weight_ih_folding_exact(self):
        s = source(2); x = torch.randn(9, 6)
        We, be = fold_full_weight_ih(s.weight_ih, s.bias_ih, s.input_projection.weight, s.input_projection.bias)
        lhs = torch.nn.functional.linear(s.input_projection(x), s.weight_ih, s.bias_ih)
        rhs = torch.nn.functional.linear(x, We, be)
        self.assertLessEqual(float((lhs-rhs).abs().max()), 1e-6)

    def test_candidate_slice_folding_exact(self):
        s = source(3); w = s.weight_ih.chunk(3,0)[2]; b=s.bias_ih.chunk(3,0)[2]; x=torch.randn(7,6)
        We,be=fold_linear(w,b,s.input_projection.weight,s.input_projection.bias)
        self.assertLessEqual(float((torch.nn.functional.linear(s.input_projection(x),w,b)-torch.nn.functional.linear(x,We,be)).abs().max()),1e-6)

    def test_update_slice_folding_exact(self):
        s = source(4); w = s.weight_ih.chunk(3,0)[1]; b=s.bias_ih.chunk(3,0)[1]; x=torch.randn(7,6)
        We,be=fold_linear(w,b,s.input_projection.weight,s.input_projection.bias)
        self.assertLessEqual(float((torch.nn.functional.linear(s.input_projection(x),w,b)-torch.nn.functional.linear(x,We,be)).abs().max()),1e-6)

    def test_projection_bias_folding_exact(self):
        s=source(5); w=s.weight_ih.chunk(3,0)[2]; b=s.bias_ih.chunk(3,0)[2]
        _,be=fold_linear(w,b,s.input_projection.weight,s.input_projection.bias)
        self.assertTrue(torch.allclose(be,b+w@s.input_projection.bias,atol=0,rtol=0))

    def _trained_pair(self):
        s=source(6); opt=torch.optim.AdamW(s.parameters(),lr=0.005,weight_decay=0.0001)
        torch.manual_seed(77); x=torch.randn(12,8,6); lengths=torch.tensor([8,8,8,8,7,7,6,6,5,5,4,3]); target=torch.tanh(torch.randn(12))
        for _ in range(3):
            opt.zero_grad(set_to_none=True); pred=s(x,lengths); loss=torch.mean((pred-target)**2); loss.backward(); opt.step()
        return s,InputFactorizationT2.from_trained_t2(s),x,lengths

    def test_trained_t2_folded_prediction_equivalence(self):
        s,f,x,l=self._trained_pair(); self.assertLessEqual(max_trace_errors(s,f,x,l)["prediction"],1e-6)
    def test_trained_t2_folded_state_equivalence(self):
        s,f,x,l=self._trained_pair(); self.assertLessEqual(max_trace_errors(s,f,x,l)["state"],1e-6)
    def test_trained_t2_folded_candidate_equivalence(self):
        s,f,x,l=self._trained_pair(); self.assertLessEqual(max_trace_errors(s,f,x,l)["candidate"],1e-6)
    def test_trained_t2_folded_update_equivalence(self):
        s,f,x,l=self._trained_pair(); e=max_trace_errors(s,f,x,l); self.assertLessEqual(max(e["update"],e["raw_update"]),1e-6)
    def test_trained_t2_folded_raw_reset_equivalence(self):
        s,f,x,l=self._trained_pair(); self.assertLessEqual(max_trace_errors(s,f,x,l)["raw_reset"],1e-6)

    def _step0_errors(self, condition: str):
        _,models=models_from_same_source(10); a=models["AB0_exact_factorized_t2"]; b=models[condition]
        torch.manual_seed(90); x=torch.randn(5,9,6); lengths=torch.tensor([9,8,7,6,5])
        with torch.no_grad(): pa,ta=a(x,lengths,return_trace=True); pb,tb=b(x,lengths,return_trace=True)
        return {
            "candidate": float((ta.candidates-tb.candidates).abs().max()),
            "update": float((ta.updates-tb.updates).abs().max()),
            "state": float((ta.states-tb.states).abs().max()),
            "prediction": float((pa-pb).abs().max()),
        }

    def test_ab0_ab1_step0_identical(self): self.assertLessEqual(max(self._step0_errors("AB1_fully_folded_equivalent").values()),1e-6)
    def test_ab0_ab2_step0_identical(self): self.assertLessEqual(max(self._step0_errors("AB2_candidate_factorized_update_folded").values()),1e-6)
    def test_ab0_ab3_step0_identical(self): self.assertLessEqual(max(self._step0_errors("AB3_candidate_folded_update_factorized").values()),1e-6)
    def test_ab0_ab4_step0_identical(self): self.assertLessEqual(max(self._step0_errors("AB4_frozen_shared_projection").values()),1e-6)

    def test_ab2_projection_affects_candidate_path_only(self): self.assertEqual(InputFactorizationT2(condition="AB2_candidate_factorized_update_folded").factorized_gates,frozenset({"n"}))
    def test_ab2_update_uses_folded_direct_map(self): self.assertNotIn("z",InputFactorizationT2(condition="AB2_candidate_factorized_update_folded").factorized_gates)
    def test_ab3_projection_affects_update_path_only(self): self.assertEqual(InputFactorizationT2(condition="AB3_candidate_folded_update_factorized").factorized_gates,frozenset({"z"}))
    def test_ab3_candidate_uses_folded_direct_map(self): self.assertNotIn("n",InputFactorizationT2(condition="AB3_candidate_folded_update_factorized").factorized_gates)
    def test_ab4_projection_requires_grad_false(self):
        m=InputFactorizationT2(condition="AB4_frozen_shared_projection"); self.assertFalse(m.projection_weight.requires_grad); self.assertFalse(m.projection_bias.requires_grad)

    def test_v837ab_uses_t2_reference(self): self.assertEqual(CONFIG["reference_condition"],"T2_scalarized_update_no_reset")
    def test_v837ab_uses_4x_unique_data(self): self.assertEqual(CONFIG["data_regime"],"4x_unique")
    def test_v837ab_uses_3200_unique_seed_episodes(self): self.assertEqual(CONFIG["unique_seed_defined_episodes"],3200)
    def test_v837ab_uses_192_steps(self): self.assertEqual(CONFIG["training"]["steps"],192)
    def test_v837ab_five_replicates(self): self.assertEqual(CONFIG["training"]["replicates"],5)
    def test_no_fresh_audit(self): self.assertFalse(CONFIG["fresh_audit_consumed"])
    def test_no_structural_search(self): self.assertFalse(CONFIG["structural_search_allowed"])
    def test_no_primitive_mining(self): self.assertFalse(CONFIG["primitive_mining_allowed"])
    def test_no_v838(self): self.assertFalse(CONFIG["v838_started"])


if __name__ == "__main__":
    unittest.main()
