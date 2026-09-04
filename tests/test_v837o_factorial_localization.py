from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.v837_primitive_invention.v837n.gru_reference_explicit import ExplicitGRUReferenceModel
from experiments.v837_primitive_invention.v837o.factorial_gru import CONDITION_FACTORS, FactorialGRUReferenceModel

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments" / "v837_primitive_invention" / "v837o"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


class V837oFactorialEquationTests(unittest.TestCase):
    def _obs(self):
        torch.manual_seed(123)
        return torch.randn(3, 7, 6), torch.tensor([7, 5, 6])

    def test_full_gru_condition_matches_v837n_equation(self):
        torch.manual_seed(700)
        prior = ExplicitGRUReferenceModel(13, 6, condition="full_gru")
        torch.manual_seed(700)
        current = FactorialGRUReferenceModel(13, 6, condition="G0_full_dynamic")
        obs, lengths = self._obs()
        self.assertEqual(prior.parameter_count(), current.nominal_parameter_count())
        self.assertEqual(current.nominal_parameter_count(), 875)
        self.assertTrue(torch.allclose(prior(obs, lengths), current(obs, lengths), atol=1e-7, rtol=1e-7))

    def test_dynamic_update_no_reset_semantics(self):
        model = FactorialGRUReferenceModel(condition="G1_dynamic_update_no_reset")
        obs, lengths = self._obs()
        _, trace = model(obs, lengths, return_trace=True)
        self.assertTrue(torch.equal(trace.resets, torch.ones_like(trace.resets)))
        self.assertGreater(float(trace.updates.var()), 0.0)

    def test_no_update_dynamic_reset_semantics(self):
        model = FactorialGRUReferenceModel(condition="G2_no_update_dynamic_reset")
        obs, lengths = self._obs()
        _, trace = model(obs, lengths, return_trace=True)
        self.assertTrue(torch.equal(trace.updates, torch.zeros_like(trace.updates)))
        self.assertGreater(float(trace.resets.var()), 0.0)

    def test_static_update_vector_time_independent(self):
        model = FactorialGRUReferenceModel(condition="G3_static_update_vector_no_reset")
        obs, lengths = self._obs()
        _, trace = model(obs, lengths, return_trace=True)
        self.assertTrue(torch.allclose(trace.updates[:, 1:, :], trace.updates[:, :-1, :], atol=0, rtol=0))

    def test_static_reset_vector_time_independent(self):
        model = FactorialGRUReferenceModel(condition="G4_no_update_static_reset_vector")
        obs, lengths = self._obs()
        _, trace = model(obs, lengths, return_trace=True)
        self.assertTrue(torch.allclose(trace.resets[:, 1:, :], trace.resets[:, :-1, :], atol=0, rtol=0))

    def test_static_static_condition_has_no_dynamic_gate_dependency(self):
        model = FactorialGRUReferenceModel(condition="G5_static_update_vector_static_reset_vector")
        obs, lengths = self._obs()
        _, trace = model(obs, lengths, return_trace=True)
        self.assertEqual(float(trace.update_input_components.abs().max()), 0.0)
        self.assertEqual(float(trace.update_state_components.abs().max()), 0.0)
        self.assertEqual(float(trace.reset_input_components.abs().max()), 0.0)
        self.assertEqual(float(trace.reset_state_components.abs().max()), 0.0)

    def test_static_scalar_shapes(self):
        model = FactorialGRUReferenceModel(condition="G8_static_update_scalar_static_reset_scalar")
        update = model.static_coefficient("update")
        reset = model.static_coefficient("reset")
        self.assertEqual(tuple(update.shape), (13,))
        self.assertEqual(tuple(reset.shape), (13,))
        self.assertEqual(torch.unique(update).numel(), 1)
        self.assertEqual(torch.unique(reset).numel(), 1)
        self.assertEqual(model.active_parameter_count(), 331)

    def test_both_off_equals_dense_recurrent_control(self):
        model = FactorialGRUReferenceModel(condition="G9_no_update_no_reset")
        obs, lengths = self._obs()
        state = torch.zeros(obs.shape[0], 13)
        for t in range(obs.shape[1]):
            projected = model.input_projection(obs[:, t, :])
            gi = F.linear(projected, model.weight_ih, model.bias_ih)
            gh = F.linear(state, model.weight_hh, model.bias_hh)
            _, _, i_n = gi.chunk(3, dim=1)
            _, _, h_n = gh.chunk(3, dim=1)
            proposed = torch.tanh(i_n + h_n)
            active = (t < lengths).float().unsqueeze(1)
            state = active * proposed + (1.0 - active) * state
        expected = torch.tanh(model.readout(state)).squeeze(-1)
        self.assertTrue(torch.allclose(model(obs, lengths), expected, atol=1e-7, rtol=1e-7))

    def test_nominal_parameter_tensors_retained(self):
        for condition in CONFIG["conditions"]:
            model = FactorialGRUReferenceModel(condition=condition)
            self.assertEqual(model.nominal_parameter_count(), 875, condition)
            names = dict(model.named_parameters())
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                self.assertIn(name, names)


class V837oFactorialIntegrityTests(unittest.TestCase):
    def test_update_factor_levels_are_distinct(self):
        self.assertEqual({x[0] for x in CONDITION_FACTORS.values()}, {"dynamic", "static_vector", "static_scalar", "off"})

    def test_reset_factor_levels_are_distinct(self):
        self.assertEqual({x[1] for x in CONDITION_FACTORS.values()}, {"dynamic", "static_vector", "static_scalar", "off"})

    def test_condition_matrix_complete(self):
        self.assertEqual(set(CONFIG["conditions"]), set(CONDITION_FACTORS))
        self.assertEqual(len(CONFIG["conditions"]), 10)

    def test_g5_static_static_has_no_input_conditioned_gate_values(self):
        model = FactorialGRUReferenceModel(condition="G5_static_update_vector_static_reset_vector")
        obs = torch.randn(2, 5, 6)
        _, trace = model(obs, return_trace=True)
        self.assertEqual(float(trace.update_input_components.var()), 0.0)
        self.assertEqual(float(trace.reset_input_components.var()), 0.0)

    def test_all_conditions_use_same_4x_data(self):
        training = CONFIG["training"]
        self.assertEqual(training["train_episodes"], 512)
        self.assertEqual(training["development_seed_range"], [10000, 10511])
        self.assertEqual(training["validation_seed_range"], [20000, 20127])

    def test_all_conditions_use_same_optimizer_steps(self):
        self.assertEqual(CONFIG["training"]["steps"], 192)
        self.assertEqual(CONFIG["training"]["optimizer"], "AdamW")

    def test_all_conditions_use_same_task_seeds(self):
        self.assertEqual(CONFIG["training"]["train_episodes"], CONFIG["training"]["development_seed_range"][1] - CONFIG["training"]["development_seed_range"][0] + 1)
        self.assertEqual(CONFIG["training"]["validation_episodes"], CONFIG["training"]["validation_seed_range"][1] - CONFIG["training"]["validation_seed_range"][0] + 1)

    def test_fresh_audit_unused(self):
        self.assertFalse(CONFIG["fresh_audit_consumed"])
        self.assertFalse(CONFIG["fresh_audit_allowed"])

    def test_primitive_mining_locked(self):
        self.assertFalse(CONFIG["primitive_mining_allowed"])
        self.assertEqual(CONFIG["primitives_promoted"], 0)

    def test_positive_control_when_result_exists(self):
        path = HERE / "diagnostics" / "full_gru_positive_control.json"
        if not path.exists():
            self.skipTest("V837o positive control not run yet")
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(result["compatible"])
        self.assertGreaterEqual(result["families_passing"], 4)


if __name__ == "__main__":
    unittest.main()
