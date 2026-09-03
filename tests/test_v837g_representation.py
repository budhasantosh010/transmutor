from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.common.graph import initial_graph
from experiments.v837_primitive_invention.common.serialization import load_model_bundle, save_model_bundle
from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel


class StateUpdateCompatibilityTests(unittest.TestCase):
    def test_direct_mode_is_exact_historical_default(self):
        graph = initial_graph()
        torch.manual_seed(123)
        legacy = NeutralGraphModel(graph)
        torch.manual_seed(123)
        direct = NeutralGraphModel(graph, state_update_mode="direct")
        direct.load_state_dict(legacy.state_dict())
        x = torch.randn(5, 7, 6, generator=torch.Generator().manual_seed(77))
        lengths = torch.tensor([7, 6, 5, 4, 3])
        with torch.no_grad():
            a = legacy(x, lengths)
            b = direct(x, lengths)
        self.assertLessEqual(float(torch.max(torch.abs(a - b))), 1e-8)
        self.assertEqual(direct.state_update_coefficients(), [1.0, 1.0])

    def test_learned_alpha_adds_one_scalar_per_cell_only(self):
        graph = initial_graph()
        direct = NeutralGraphModel(graph, state_update_mode="direct")
        learned = NeutralGraphModel(graph, state_update_mode="learned_leaky", alpha_init=0.5)
        self.assertEqual(learned.parameter_count() - direct.parameter_count(), len(graph.cells))
        self.assertEqual(len(learned.cell_alpha_logits), len(graph.cells))
        for alpha in learned.state_update_coefficients():
            self.assertAlmostEqual(alpha, 0.5, places=7)

    def test_state_update_mode_serialization_round_trip(self):
        model = NeutralGraphModel(initial_graph(), state_update_mode="learned_leaky", alpha_init=0.5)
        with torch.no_grad():
            model.cell_alpha_logits[0].fill_(1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model_bundle(path, model)
            restored, _ = load_model_bundle(path)
        self.assertEqual(restored.state_update_mode, "learned_leaky")
        self.assertEqual(restored.parameter_count(), model.parameter_count())
        self.assertAlmostEqual(restored.state_update_coefficients()[0], model.state_update_coefficients()[0], places=7)


if __name__ == "__main__":
    unittest.main()
