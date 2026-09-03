from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel, clone_with_state
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph

ROOT = Path(__file__).resolve().parents[1]


class V837mLinearTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = high_capacity_generic_graph(0)
        self.config = json.loads((ROOT / "experiments/v837_primitive_invention/v837m/config.json").read_text(encoding="utf-8"))

    def test_historical_default_parameter_count_unchanged(self) -> None:
        self.assertEqual(NeutralGraphModel(self.graph).parameter_count(), 856)

    def test_transport_and_additive_control_parameter_counts_match_exactly(self) -> None:
        transport = NeutralGraphModel(self.graph, state_update_mode="linear_transport")
        additive = NeutralGraphModel(self.graph, state_update_mode="transport_matched_additive")
        self.assertEqual(transport.parameter_count(), 1016)
        self.assertEqual(additive.parameter_count(), 1016)
        self.assertEqual(transport.parameter_count(), additive.parameter_count())

    def test_scalar_persistence_parameter_count_remains_866(self) -> None:
        self.assertEqual(NeutralGraphModel(self.graph, state_update_mode="learned_leaky").parameter_count(), 866)

    def test_transport_spectral_norm_is_bounded_by_rho(self) -> None:
        model = NeutralGraphModel(self.graph, state_update_mode="linear_transport", transport_rho=0.95)
        for row in model.transport_diagnostics():
            self.assertLessEqual(row["spectral_norm"], 0.950001)
            self.assertLessEqual(row["spectral_radius"], 0.950001)

    def test_clone_round_trip_preserves_transport_mode_and_parameters(self) -> None:
        model = NeutralGraphModel(self.graph, state_update_mode="linear_transport", transport_rho=0.95)
        with torch.no_grad():
            model.cell_transport_raw[0][0, 1] = 0.125
        clone = clone_with_state(model)
        self.assertEqual(clone.state_update_mode, "linear_transport")
        self.assertEqual(clone.transport_rho, 0.95)
        self.assertTrue(torch.equal(model.cell_transport_raw[0], clone.cell_transport_raw[0]))

    def test_candidate_trace_is_available_without_changing_historical_output(self) -> None:
        torch.manual_seed(7)
        model = NeutralGraphModel(self.graph)
        x = torch.randn(2, 5, 6)
        lengths = torch.tensor([5, 4])
        plain = model(x, lengths)
        traced, trace = model(x, lengths, return_trace=True)
        self.assertTrue(torch.equal(plain, traced))
        self.assertEqual(trace.candidate_states.shape, trace.states.shape)

    def test_v837m_uses_calibrated_data_and_keeps_science_locks(self) -> None:
        training = self.config["training"]
        self.assertEqual(training["steps"], 192)
        self.assertEqual(training["train_episodes"], 512)
        self.assertEqual(training["development_seed_range"], [10000, 10511])
        self.assertEqual(training["validation_seed_range"], [20000, 20127])
        self.assertFalse(self.config["fresh_audit_allowed"])
        self.assertFalse(self.config["primitive_mining_allowed"])


if __name__ == "__main__":
    unittest.main()
