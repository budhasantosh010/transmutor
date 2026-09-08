from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.common.graph import build_input_access_spec
from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.v837af.analyze_results import AF1, AF1D, AF1F, _decision
from experiments.v837_primitive_invention.v837af.candidate_input_factorization import (
    CandidateInputFactorizationY3,
    exact_t2_projection_from_seed,
)

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments" / "v837_primitive_invention" / "v837af"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def model(condition: str, *, seed: int = 1234, projection_seed: int = 5678, graph=None):
    torch.manual_seed(seed)
    return CandidateInputFactorizationY3(
        high_capacity_generic_graph(0) if graph is None else graph,
        condition=condition,
        coupling_initialization_seed=777,
        projection_seed=projection_seed,
    )


class V837afCandidateInputSiblingTests(unittest.TestCase):
    def test_af0_exact_y3_shape_and_counts(self):
        m = model("AF0_y3_parent")
        self.assertEqual(m.parameter_count(), 1223)
        self.assertEqual(m.total_recurrent_controller_projection_macs, 846)
        self.assertEqual(m.projection_parameter_count, 0)

    def test_af1_shared_projection_count(self):
        m = model("AF1_shared_candidate_input_factorization")
        self.assertIsNotNone(m.shared_candidate_projection)
        self.assertIsNone(m.cell_candidate_projections)
        self.assertEqual(m.parameter_count(), 1265)
        self.assertEqual(m.projection_parameter_count, 42)
        self.assertEqual(m.projection_specific_macs, 36)
        self.assertEqual(m.total_recurrent_controller_projection_macs, 882)

    def test_af1d_deshared_projection_count(self):
        m = model("AF1D_deshared_candidate_input_factorization")
        self.assertIsNone(m.shared_candidate_projection)
        self.assertEqual(len(m.cell_candidate_projections), 10)
        self.assertEqual(m.parameter_count(), 1643)
        self.assertEqual(m.projection_parameter_count, 420)
        self.assertEqual(m.projection_specific_macs, 360)
        self.assertEqual(m.total_recurrent_controller_projection_macs, 1206)

    def test_af1f_has_no_runtime_projection(self):
        m = model("AF1F_folded_candidate_input_control")
        self.assertFalse(m.runtime_projection_active)
        self.assertIsNone(m.shared_candidate_projection)
        self.assertIsNone(m.cell_candidate_projections)
        self.assertEqual(m.parameter_count(), 1223)
        self.assertEqual(m.projection_specific_macs, 0)

    def test_projection_initialization_exact_frozen_t2_law(self):
        projection_seed = 99117
        expected_w, expected_b = exact_t2_projection_from_seed(projection_seed)
        m = model("AF1_shared_candidate_input_factorization", projection_seed=projection_seed)
        self.assertTrue(torch.equal(m.shared_candidate_projection.weight.detach(), expected_w))
        self.assertTrue(torch.equal(m.shared_candidate_projection.bias.detach(), expected_b))

    def test_af1d_all_projections_bit_identical_to_af1(self):
        projection_seed = 8131
        a = model("AF1_shared_candidate_input_factorization", projection_seed=projection_seed)
        d = model("AF1D_deshared_candidate_input_factorization", projection_seed=projection_seed)
        for layer in d.cell_candidate_projections:
            self.assertTrue(torch.equal(a.shared_candidate_projection.weight.detach(), layer.weight.detach()))
            self.assertTrue(torch.equal(a.shared_candidate_projection.bias.detach(), layer.bias.detach()))
        self.assertEqual(len({layer.weight.data_ptr() for layer in d.cell_candidate_projections}), 10)

    def test_af1_effective_candidate_maps_match_folded_control_at_step0(self):
        seed = 2026
        pseed = 3026
        shared = model("AF1_shared_candidate_input_factorization", seed=seed, projection_seed=pseed)
        folded = model("AF1F_folded_candidate_input_control", seed=seed, projection_seed=pseed)
        for cell in range(10):
            w, b = shared.effective_candidate_input_map(cell)
            self.assertLessEqual(float(torch.max(torch.abs(w - folded.base.cell_wx[cell]))), 1e-6)
            self.assertLessEqual(float(torch.max(torch.abs(b - folded.base.cell_b[cell]))), 1e-6)

    def test_af1d_effective_candidate_maps_match_shared_at_step0(self):
        seed = 2027
        pseed = 3027
        shared = model("AF1_shared_candidate_input_factorization", seed=seed, projection_seed=pseed)
        deshared = model("AF1D_deshared_candidate_input_factorization", seed=seed, projection_seed=pseed)
        for cell in range(10):
            ws, bs = shared.effective_candidate_input_map(cell)
            wd, bd = deshared.effective_candidate_input_map(cell)
            self.assertLessEqual(float(torch.max(torch.abs(ws - wd))), 1e-6)
            self.assertLessEqual(float(torch.max(torch.abs(bs - bd))), 1e-6)

    def test_af1_vs_af1f_forward_step0(self):
        seed = 2127
        pseed = 3127
        a = model("AF1_shared_candidate_input_factorization", seed=seed, projection_seed=pseed)
        f = model("AF1F_folded_candidate_input_control", seed=seed, projection_seed=pseed)
        torch.manual_seed(44)
        x = torch.randn(7, 1, 6)
        lengths = torch.ones(7, dtype=torch.long)
        with torch.no_grad():
            pa, ta = a(x, lengths, return_trace=True)
            pf, tf = f(x, lengths, return_trace=True)
        for lhs, rhs in [
            (ta.candidate_states, tf.candidate_states),
            (ta.global_gates, tf.global_gates),
            (ta.states, tf.states),
            (ta.outputs, tf.outputs),
            (pa, pf),
        ]:
            self.assertLessEqual(float(torch.max(torch.abs(lhs - rhs))), 1e-6)
        with torch.no_grad():
            for t in range(x.shape[1]):
                for cell in range(10):
                    va = a.historical_visible_input(x[:, t, :], cell)
                    vf = f.historical_visible_input(x[:, t, :], cell)
                    ia = a.candidate_input_affine_term(va, cell)
                    iff = f.candidate_input_affine_term(vf, cell)
                    self.assertLessEqual(float(torch.max(torch.abs(ia - iff))), 1e-6)

    def test_af1_vs_af1d_forward_step0(self):
        seed = 2128
        pseed = 3128
        a = model("AF1_shared_candidate_input_factorization", seed=seed, projection_seed=pseed)
        d = model("AF1D_deshared_candidate_input_factorization", seed=seed, projection_seed=pseed)
        torch.manual_seed(45)
        x = torch.randn(7, 8, 6)
        lengths = torch.tensor([8, 8, 7, 6, 5, 4, 3])
        with torch.no_grad():
            pa, ta = a(x, lengths, return_trace=True)
            pd, td = d(x, lengths, return_trace=True)
        for lhs, rhs in [
            (ta.input_terms, td.input_terms),
            (ta.candidate_states, td.candidate_states),
            (ta.global_gates, td.global_gates),
            (ta.states, td.states),
            (ta.outputs, td.outputs),
            (pa, pd),
        ]:
            self.assertLessEqual(float(torch.max(torch.abs(lhs - rhs))), 1e-6)

    def test_projected_visible_input_equivalence(self):
        seed = 2211
        pseed = 3211
        a = model("AF1_shared_candidate_input_factorization", seed=seed, projection_seed=pseed)
        f = model("AF1F_folded_candidate_input_control", seed=seed, projection_seed=pseed)
        d = model("AF1D_deshared_candidate_input_factorization", seed=seed, projection_seed=pseed)
        x = torch.randn(5, 6)
        for cell in range(10):
            visible = a.historical_visible_input(x, cell)
            pa = a.diagnostic_projected_visible_input(visible, cell)
            pf = f.diagnostic_projected_visible_input(visible, cell)
            pd = d.diagnostic_projected_visible_input(visible, cell)
            self.assertLessEqual(float(torch.max(torch.abs(pa - pf))), 1e-7)
            self.assertLessEqual(float(torch.max(torch.abs(pa - pd))), 1e-7)

    def test_candidate_projection_does_not_change_global_controller(self):
        seed = 2321
        pseed = 3321
        parent = model("AF0_y3_parent", seed=seed, projection_seed=pseed)
        for condition in ["AF1_shared_candidate_input_factorization", "AF1F_folded_candidate_input_control", "AF1D_deshared_candidate_input_factorization"]:
            m = model(condition, seed=seed, projection_seed=pseed)
            self.assertTrue(torch.equal(parent.global_ws.detach(), m.global_ws.detach()))
            self.assertTrue(torch.equal(parent.global_wx.detach(), m.global_wx.detach()))
            self.assertTrue(torch.equal(parent.global_b.detach(), m.global_b.detach()))

    def test_rank4_branch_unchanged(self):
        seed = 2421
        pseed = 3421
        parent = model("AF0_y3_parent", seed=seed, projection_seed=pseed)
        shared = model("AF1_shared_candidate_input_factorization", seed=seed, projection_seed=pseed)
        self.assertTrue(torch.equal(parent.global_u.detach(), shared.global_u.detach()))
        self.assertTrue(torch.equal(parent.global_v.detach(), shared.global_v.detach()))
        self.assertTrue(torch.equal(parent.cross_block_mask, shared.cross_block_mask))

    def test_message_graph_and_readout_unchanged(self):
        seed = 2521
        pseed = 3521
        parent = model("AF0_y3_parent", seed=seed, projection_seed=pseed)
        shared = model("AF1_shared_candidate_input_factorization", seed=seed, projection_seed=pseed)
        self.assertEqual(parent.graph.to_dict(), shared.graph.to_dict())
        for a, b in zip(parent.base.edge_weights, shared.base.edge_weights):
            self.assertTrue(torch.equal(a.detach(), b.detach()))
        self.assertTrue(torch.equal(parent.base.readout.weight.detach(), shared.base.readout.weight.detach()))
        self.assertTrue(torch.equal(parent.base.readout.bias.detach(), shared.base.readout.bias.detach()))

    def test_visibility_mask_applied_before_projection(self):
        graph = high_capacity_generic_graph(0)
        graph.input_access = build_input_access_spec("fixed_sparse", 6, 10, density=0.5, seed=71)
        m = model("AF1_shared_candidate_input_factorization", graph=graph, seed=2711, projection_seed=3711)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        cell = 0
        visible = m.historical_visible_input(x, cell)
        expected_visible = x * m.base.input_access_mask[:, cell].view(1, -1)
        self.assertTrue(torch.equal(visible, expected_visible))
        actual = m.diagnostic_projected_visible_input(visible, cell)
        expected = torch.nn.functional.linear(expected_visible, m.shared_candidate_projection.weight, m.shared_candidate_projection.bias)
        self.assertTrue(torch.equal(actual, expected))

    def test_af1_shared_projection_is_computed_as_one_shared_parameterization(self):
        m = model("AF1_shared_candidate_input_factorization")
        names = [name for name, _ in m.named_parameters() if "candidate_projection" in name]
        self.assertEqual(set(names), {"shared_candidate_projection.weight", "shared_candidate_projection.bias"})

    def test_af1d_projections_are_independent_parameters(self):
        m = model("AF1D_deshared_candidate_input_factorization")
        names = [name for name, _ in m.named_parameters() if "candidate_projections" in name]
        self.assertEqual(len(names), 20)
        self.assertEqual(len({id(p) for n, p in m.named_parameters() if "candidate_projections" in n}), 20)

    def test_training_regime_locked(self):
        tr = CONFIG["training"]
        self.assertEqual(CONFIG["data_regime"], "4x_unique")
        self.assertEqual(CONFIG["unique_seed_defined_episodes"], 3200)
        self.assertEqual(tr["steps"], 192)
        self.assertEqual(tr["train_episodes"], 512)
        self.assertEqual(tr["validation_episodes"], 128)
        self.assertEqual(tr["development_seed_range"], [10000, 10511])
        self.assertEqual(tr["validation_seed_range"], [20000, 20127])
        self.assertEqual(tr["replicates"], 5)
        self.assertEqual(tr["learning_rate"], 0.005)
        self.assertEqual(tr["weight_decay"], 0.0001)
        self.assertEqual(tr["gradient_clip"], 5.0)

    def test_projection_namespace_matches_reference_factorization_law(self):
        self.assertEqual(CONFIG["training"]["projection_initialization_namespace"], "v837j-primary-init")
        seed = deterministic_int("v837j-primary-init", "conditional_routing", 0)
        w, b = exact_t2_projection_from_seed(seed)
        self.assertEqual(tuple(w.shape), (6, 6)); self.assertEqual(tuple(b.shape), (6,))

    def test_decision_folded_success_has_priority(self):
        s = {AF1: {"families_passing": 2}, AF1F: {"families_passing": 4}, AF1D: {"families_passing": 4}}
        self.assertEqual(_decision(s), ("CANDIDATE_COMPOSED_EFFECTIVE_INPUT_MAPPING_SUFFICIENT", [], True, AF1F, False))

    def test_decision_shared_specific(self):
        s = {AF1: {"families_passing": 4}, AF1F: {"families_passing": 3}, AF1D: {"families_passing": 3}}
        self.assertEqual(_decision(s), ("SHARED_CANDIDATE_INPUT_FACTORIZATION_SPECIFICALLY_SUFFICIENT", [], True, AF1, False))

    def test_decision_sharedness_not_established(self):
        s = {AF1: {"families_passing": 4}, AF1F: {"families_passing": 3}, AF1D: {"families_passing": 4}}
        self.assertEqual(_decision(s), ("CANDIDATE_INPUT_FACTORIZATION_SUFFICIENT", ["SHAREDNESS_NOT_ESTABLISHED"], True, AF1, False))

    def test_decision_deshared_specific(self):
        s = {AF1: {"families_passing": 3}, AF1F: {"families_passing": 3}, AF1D: {"families_passing": 4}}
        self.assertEqual(_decision(s), ("DESHARED_CANDIDATE_INPUT_FACTORIZATION_SUFFICIENT", ["SHARED_INPUT_BASIS_HARMFUL"], True, AF1D, False))

    def test_decision_all_fail_authorizes_v837ag(self):
        s = {AF1: {"families_passing": 3}, AF1F: {"families_passing": 3}, AF1D: {"families_passing": 3}}
        self.assertEqual(_decision(s), ("CANDIDATE_INPUT_FACTORIZATION_TRANSFER_INSUFFICIENT", [], False, None, True))

    def test_science_locks(self):
        self.assertFalse(CONFIG["fresh_audit_consumed"])
        self.assertFalse(CONFIG["structural_search_allowed"])
        self.assertFalse(CONFIG["primitive_mining_allowed"])
        self.assertEqual(CONFIG["primitives_promoted"], 0)
        self.assertFalse(CONFIG["v838_started"])
        audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["episodes_consumed"], 0)
        self.assertFalse((ROOT / "experiments/v837_primitive_invention/v837ae").exists())
        self.assertFalse((ROOT / "experiments/v837_primitive_invention/v838").exists())


if __name__ == "__main__":
    unittest.main()
