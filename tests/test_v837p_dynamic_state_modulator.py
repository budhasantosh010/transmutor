from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments" / "v837_primitive_invention" / "v837p"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


class V837pDynamicModulatorTests(unittest.TestCase):
    def _model(self, mode: str, replicate: int = 0) -> NeutralGraphModel:
        torch.manual_seed(1234)
        return NeutralGraphModel(
            high_capacity_generic_graph(replicate),
            obs_dim=6,
            state_dim=4,
            message_dim=4,
            state_modulation_mode=mode,
        )

    def test_dynamic_scalar_modulator_shape(self):
        model = self._model("dynamic_scalar_candidate")
        observations = torch.randn(3, 5, 6)
        _, trace = model(observations, torch.tensor([5, 4, 3]), return_trace=True)
        self.assertIsNotNone(trace.state_modulators)
        self.assertEqual(tuple(trace.state_modulators.shape), (3, 5, 10, 1))

    def test_dynamic_scalar_depends_on_input(self):
        model = self._model("dynamic_scalar_candidate")
        with torch.no_grad():
            for gs, gm, gx, gb in zip(model.cell_gs, model.cell_gm, model.cell_gx, model.cell_gb):
                gs.zero_(); gm.zero_(); gx.zero_(); gb.zero_()
            model.cell_gx[0][0] = 2.0
        a = torch.zeros(1, 1, 6)
        b = a.clone(); b[0, 0, 0] = 1.0
        _, ta = model(a, return_trace=True)
        _, tb = model(b, return_trace=True)
        self.assertNotEqual(float(ta.state_modulators[0, 0, 0, 0]), float(tb.state_modulators[0, 0, 0, 0]))

    def test_dynamic_scalar_can_depend_on_state_and_message(self):
        model = self._model("dynamic_scalar_candidate")
        self.assertEqual(tuple(model.cell_gs[0].shape), (4,))
        self.assertEqual(tuple(model.cell_gm[0].shape), (4,))
        self.assertEqual(tuple(model.cell_gx[0].shape), (6,))
        self.assertEqual(tuple(model.cell_gb[0].shape), (1,))

    def test_parameter_matched_control_exact_count(self):
        dynamic = self._model("dynamic_scalar_candidate")
        control = self._model("dynamic_scalar_matched_additive")
        self.assertEqual(dynamic.parameter_count(), 1006)
        self.assertEqual(control.parameter_count(), 1006)

    def test_historical_mode_parameter_count_unchanged(self):
        self.assertEqual(self._model("none").parameter_count(), 856)

    def test_historical_mode_forward_unchanged_by_explicit_none(self):
        graph = high_capacity_generic_graph(0)
        torch.manual_seed(987)
        default = NeutralGraphModel(graph)
        torch.manual_seed(987)
        explicit_none = NeutralGraphModel(graph, state_modulation_mode="none")
        observations = torch.randn(2, 4, 6)
        lengths = torch.tensor([4, 3])
        self.assertTrue(torch.equal(default(observations, lengths), explicit_none(observations, lengths)))

    def test_scalar_persistence_count_preserved(self):
        graph = high_capacity_generic_graph(0)
        model = NeutralGraphModel(graph, state_update_mode="learned_leaky")
        self.assertEqual(model.parameter_count(), 866)


class V837pScientificBoundaryTests(unittest.TestCase):
    def test_parent_diagnosis_authorizes_only_dynamic_modulator(self):
        parent = json.loads((ROOT / "experiments/v837_primitive_invention/v837o/results.json").read_text(encoding="utf-8"))
        self.assertEqual(parent["mechanism_diagnosis"], "DYNAMIC_STATE_MODULATION_REQUIRED")
        self.assertTrue(parent["neutral_followup_allowed"])
        self.assertEqual(parent["neutral_followup_type"], "single_dynamic_modulator")

    def test_all_conditions_use_same_4x_data(self):
        training = CONFIG["training"]
        self.assertEqual(training["train_episodes"], 512)
        self.assertEqual(training["development_seed_range"], [10000, 10511])
        self.assertEqual(training["validation_seed_range"], [20000, 20127])

    def test_all_conditions_use_same_optimizer_steps(self):
        self.assertEqual(CONFIG["training"]["steps"], 192)
        self.assertEqual(CONFIG["training"]["optimizer"], "AdamW")

    def test_fresh_audit_unused(self):
        self.assertFalse(CONFIG["fresh_audit_consumed"])
        audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["episodes_consumed"], 0)

    def test_primitive_mining_locked(self):
        self.assertFalse(CONFIG["primitive_mining_allowed"])

    def test_result_gate_when_present(self):
        result_path = HERE / "results.json"
        if not result_path.exists():
            self.skipTest("V837p result not generated yet")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(result["version"], "V837p")
        self.assertFalse(result["fresh_audit_consumed"])
        self.assertFalse(result["primitive_mining_allowed"])
        self.assertEqual(result["parameter_matching"]["dynamic_scalar_state_modulation"], 1006)
        self.assertEqual(result["parameter_matching"]["parameter_matched_dynamic_additive"], 1006)
        self.assertTrue(result["parameter_matching"]["exact_match"])


if __name__ == "__main__":
    unittest.main()
