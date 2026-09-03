from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.common.reference_models import GRUReferenceModel
from experiments.v837_primitive_invention.v837n.gru_reference_explicit import ExplicitGRUReferenceModel

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "v837_primitive_invention" / "v837n"


class ExplicitGRUEquationTests(unittest.TestCase):
    def test_explicit_gru_matches_reference_parameter_count(self):
        explicit = ExplicitGRUReferenceModel(13, 6, condition="full_gru")
        reference = GRUReferenceModel(13, 6)
        self.assertEqual(reference.parameter_count(), 875)
        self.assertEqual(explicit.parameter_count(), reference.parameter_count())

    def test_explicit_gru_forward_equations(self):
        seed = 837013
        torch.manual_seed(seed)
        reference = GRUReferenceModel(13, 6)
        torch.manual_seed(seed)
        explicit = ExplicitGRUReferenceModel(13, 6, condition="full_gru")
        observations = torch.randn(9, 11, 6)
        lengths = torch.tensor([11, 10, 9, 8, 7, 6, 5, 4, 3])
        with torch.no_grad():
            expected = reference(observations, lengths)
            actual = explicit(observations, lengths)
        self.assertLessEqual(float(torch.max(torch.abs(expected - actual)).item()), 1e-6)

    def test_initialization_matches_framework_reference(self):
        seed = 555123
        torch.manual_seed(seed)
        reference = GRUReferenceModel(13, 6)
        torch.manual_seed(seed)
        explicit = ExplicitGRUReferenceModel(13, 6, condition="full_gru")
        self.assertTrue(torch.equal(reference.input_projection.weight, explicit.input_projection.weight))
        self.assertTrue(torch.equal(reference.cell.weight_ih, explicit.weight_ih))
        self.assertTrue(torch.equal(reference.cell.weight_hh, explicit.weight_hh))
        self.assertTrue(torch.equal(reference.cell.bias_ih, explicit.bias_ih))
        self.assertTrue(torch.equal(reference.cell.bias_hh, explicit.bias_hh))
        self.assertTrue(torch.equal(reference.readout.weight, explicit.readout.weight))

    def test_static_update_is_time_independent(self):
        model = ExplicitGRUReferenceModel(13, 6, condition="static_update_vector")
        observations = torch.randn(4, 8, 6)
        _, trace = model(observations, return_trace=True)
        self.assertTrue(torch.equal(trace.updates[:, 1:, :], trace.updates[:, :-1, :]))

    def test_no_update_removes_carry_path(self):
        model = ExplicitGRUReferenceModel(13, 6, condition="no_update")
        observations = torch.randn(4, 8, 6)
        _, trace = model(observations, return_trace=True)
        self.assertTrue(torch.allclose(trace.states, trace.candidates, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.count_nonzero(trace.updates).item() == 0)

    def test_no_reset_sets_reset_to_one(self):
        model = ExplicitGRUReferenceModel(13, 6, condition="no_reset")
        observations = torch.randn(4, 8, 6)
        _, trace = model(observations, return_trace=True)
        self.assertTrue(torch.equal(trace.resets, torch.ones_like(trace.resets)))

    def test_disabled_gate_parameters_do_not_affect_forward(self):
        observations = torch.randn(5, 7, 6)
        model = ExplicitGRUReferenceModel(13, 6, condition="no_update")
        with torch.no_grad():
            baseline = model(observations)
            h = model.hidden_size
            model.weight_ih[h:2*h].add_(100.0)
            model.weight_hh[h:2*h].sub_(100.0)
            model.bias_ih[h:2*h].add_(50.0)
            model.bias_hh[h:2*h].sub_(50.0)
            changed = model(observations)
        self.assertTrue(torch.allclose(baseline, changed, atol=1e-7, rtol=0.0))

        model = ExplicitGRUReferenceModel(13, 6, condition="no_reset")
        with torch.no_grad():
            baseline = model(observations)
            h = model.hidden_size
            model.weight_ih[:h].add_(100.0)
            model.weight_hh[:h].sub_(100.0)
            model.bias_ih[:h].add_(50.0)
            model.bias_hh[:h].sub_(50.0)
            changed = model(observations)
        self.assertTrue(torch.allclose(baseline, changed, atol=1e-7, rtol=0.0))


class V837nScientificBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))

    def test_all_ablation_conditions_use_same_data(self):
        training = self.config["training"]
        self.assertEqual(training["train_episodes"], 512)
        self.assertEqual(training["development_seed_range"], [10000, 10511])
        self.assertEqual(training["validation_seed_range"], [20000, 20127])
        self.assertEqual(training["validation_episodes"], 128)

    def test_all_ablation_conditions_use_same_optimizer_steps(self):
        training = self.config["training"]
        self.assertEqual(training["optimizer"], "AdamW")
        self.assertEqual(training["steps"], 192)
        self.assertEqual(training["learning_rate"], 0.005)
        self.assertEqual(training["weight_decay"], 0.0001)

    def test_fresh_audit_unused(self):
        self.assertFalse(self.config["fresh_audit_allowed"])
        self.assertFalse(self.config["fresh_audit_consumed"])
        audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["episodes_consumed"], 0)

    def test_primitive_mining_locked(self):
        self.assertFalse(self.config["primitive_mining_allowed"])
        self.assertFalse(self.config["structural_search_allowed"])

    def test_full_gru_control_reproduces_reference_regime_when_result_exists(self):
        path = BASE / "diagnostics" / "full_gru_positive_control.json"
        if not path.exists():
            self.skipTest("positive-control run not completed yet")
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(result["compatible"])
        self.assertGreaterEqual(result["families_passing"], 4)
        self.assertEqual(result["parameter_count"], 875)


if __name__ == "__main__":
    unittest.main()
