from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.v837_primitive_invention.common.serialization import load_model_bundle, save_model_bundle
from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph


class InteractionBasisTests(unittest.TestCase):
    def test_parameter_matched_additive_equals_multiplicative_parameter_count(self):
        graph = high_capacity_generic_graph(0)
        additive = NeutralGraphModel(graph, interaction_mode="parameter_matched_additive", interaction_rank=2)
        multiplicative = NeutralGraphModel(graph, interaction_mode="low_rank_multiplicative", interaction_rank=2)
        self.assertEqual(additive.parameter_count(), multiplicative.parameter_count())

    def test_rank_two_interaction_increases_parameters_but_not_cells_or_edges(self):
        graph = high_capacity_generic_graph(0)
        historical = NeutralGraphModel(graph)
        multiplicative = NeutralGraphModel(graph, interaction_mode="low_rank_multiplicative", interaction_rank=2)
        self.assertGreater(multiplicative.parameter_count(), historical.parameter_count())
        self.assertEqual(multiplicative.graph.descriptors()["cell_count"], historical.graph.descriptors()["cell_count"])
        self.assertEqual(multiplicative.graph.descriptors()["edge_count"], historical.graph.descriptors()["edge_count"])

    def test_interaction_mode_serialization_round_trip(self):
        model = NeutralGraphModel(high_capacity_generic_graph(0), interaction_mode="low_rank_multiplicative", interaction_rank=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model_bundle(path, model)
            restored, _ = load_model_bundle(path)
        self.assertEqual(restored.interaction_mode, "low_rank_multiplicative")
        self.assertEqual(restored.interaction_rank, 2)
        self.assertEqual(restored.parameter_count(), model.parameter_count())


if __name__ == "__main__":
    unittest.main()
