from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.v837_primitive_invention.common.gates import (
    capacity_demonstrated,
    v837_capacity_criterion_sha256,
)
from experiments.v837_primitive_invention.common.graph import (
    GraphSpec,
    build_fixed_sparse_mask,
    build_input_access_spec,
    initial_graph,
)
from experiments.v837_primitive_invention.common.guards import assert_primitive_mining_allowed, primitive_mining_allowed
from experiments.v837_primitive_invention.common.motif import begin_scientific_motif_pipeline
from experiments.v837_primitive_invention.common.seeds import gate_sha256
from experiments.v837_primitive_invention.common.serialization import load_model_bundle, save_model_bundle
from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel
from experiments.v837_primitive_invention.common.trainer import train_graph
from experiments.v837_primitive_invention.tasks.delayed_recall import DelayedRecallTask

ROOT = Path(__file__).resolve().parents[1]
GATE_HASH = "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867"
CAPACITY_HASH = "7178eed701ad50a298f172e867c73db47c03ecb28767de2add61feb34a61a3aa"
HISTORICAL_HASHES = {
    "archive/preserved_artifacts/transmutor_experiments_v836plus/v836_results.json": "0ed63ee1e1c5903c1c90b58942aaf968b747df19d4c4a51c1d73a6b36f91527d",
    "experiments/v837_primitive_invention/v837/results.json": "5fed69cc990be5c6f64a5229f59ff7f27af0c1fc26398bdfbe80ee46255eef14",
    "experiments/v837_primitive_invention/v837b/results.json": "f131110969e7700ec0cd9a82825e8554a51a9c05bb308d54625452db54e35cb0",
    "experiments/v837_primitive_invention/v837c/results.json": "994195fdd0e32e12ec44521ea782c1fc3561b8f596fd4a70e9d59f335fe7d009",
    "experiments/v837_primitive_invention/BLOCKER_ANALYSIS.md": "4eea85cbfb2fb9e379675765038527daf2ad6a49aa0721e861c9cc61b0155a20",
    "experiments/v837_primitive_invention/final_resource_accounting.json": "c712ea3c0771ebc398e4ccb80a4d0ffe0d8ead946d42fd460633577fbb3d9b37",
}


def legacy_forward(model: NeutralGraphModel, observations: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
    batch, steps, _ = observations.shape
    n = len(model.graph.cells)
    prev_states = [torch.zeros(batch, model.state_dim) for _ in range(n)]
    prev_outputs = [torch.zeros(batch, model.message_dim) for _ in range(n)]
    for t in range(steps):
        x_t = observations[:, t, :]
        current_states = []
        current_outputs = []
        for cell_index in range(n):
            message = torch.zeros(batch, model.message_dim)
            for edge_index, edge in enumerate(model.graph.edges):
                if edge.dst != cell_index:
                    continue
                if edge.recurrent or edge.src >= len(current_outputs):
                    source = prev_outputs[edge.src]
                else:
                    source = current_outputs[edge.src]
                message = message + model.edge_weights[edge_index] * source
            proposed_state = torch.tanh(
                prev_states[cell_index] @ model.cell_ws[cell_index].T
                + message @ model.cell_wm[cell_index].T
                + x_t @ model.cell_wx[cell_index].T
                + model.cell_b[cell_index]
            )
            proposed_output = proposed_state @ model.cell_wo[cell_index].T
            if lengths is not None:
                active = (t < lengths).to(observations.dtype).unsqueeze(1)
                state = active * proposed_state + (1.0 - active) * prev_states[cell_index]
                output = active * proposed_output + (1.0 - active) * prev_outputs[cell_index]
            else:
                state = proposed_state
                output = proposed_output
            current_states.append(state)
            current_outputs.append(output)
        prev_states = current_states
        prev_outputs = current_outputs
    stacked = torch.cat(prev_states, dim=1)
    return torch.tanh(model.readout(stacked)).squeeze(-1)


class InputAccessTests(unittest.TestCase):
    def test_broadcast_mask_all_ones(self):
        spec = build_input_access_spec("broadcast", 6, 10)
        mask = np.asarray(spec.mask)
        self.assertEqual(mask.shape, (6, 10))
        self.assertTrue(np.all(mask == 1.0))
        self.assertEqual(spec.edge_count, 60)

    def test_sparse_mask_deterministic(self):
        a = build_fixed_sparse_mask(6, 10, 0.25, 837001)
        b = build_fixed_sparse_mask(6, 10, 0.25, 837001)
        self.assertTrue(np.array_equal(a, b))

    def test_sparse_mask_density(self):
        mask = build_fixed_sparse_mask(6, 10, 0.25, 837001)
        self.assertEqual(int(mask.sum()), 15)
        low = build_fixed_sparse_mask(6, 10, 0.125, 837001)
        self.assertEqual(int(low.sum()), 10)  # minimum needed to connect all 10 cells
        self.assertAlmostEqual(float(low.mean()), 1.0 / 6.0, places=7)

    def test_every_input_connected(self):
        mask = build_fixed_sparse_mask(6, 10, 0.125, 837002)
        self.assertTrue(np.all(mask.sum(axis=1) >= 1))

    def test_every_cell_receives_input_when_feasible(self):
        mask = build_fixed_sparse_mask(6, 10, 0.125, 837003)
        self.assertTrue(np.all(mask.sum(axis=0) >= 1))

    def test_sparse_mask_has_no_task_label_dependency(self):
        params = inspect.signature(build_fixed_sparse_mask).parameters
        forbidden = {name for name in params if any(token in name.lower() for token in ("task", "family", "domain", "label"))}
        self.assertFalse(forbidden)

    def test_broadcast_refactor_preserves_forward_output(self):
        torch.manual_seed(1234)
        graph = initial_graph()
        model = NeutralGraphModel(graph)
        observations = torch.randn(5, 7, 6, generator=torch.Generator().manual_seed(99))
        lengths = torch.tensor([7, 6, 5, 7, 4])
        with torch.no_grad():
            historical = legacy_forward(model, observations, lengths)
            refactored = model(observations, lengths)
        self.assertLessEqual(float(torch.max(torch.abs(historical - refactored))), 1e-7)

        broadcast_graph = graph.clone()
        broadcast_graph.input_access = build_input_access_spec("broadcast", 6, len(graph.cells))
        explicit = NeutralGraphModel(broadcast_graph)
        explicit.load_state_dict(model.state_dict())
        with torch.no_grad():
            explicit_output = explicit(observations, lengths)
        self.assertLessEqual(float(torch.max(torch.abs(refactored - explicit_output))), 1e-7)
        self.assertEqual(graph.graph_id, broadcast_graph.graph_id)

    def test_serialization_round_trip_preserves_input_mask(self):
        graph = initial_graph()
        graph.input_access = build_input_access_spec("fixed_sparse", 6, len(graph.cells), density=0.5, seed=837004)
        model = NeutralGraphModel(graph)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model_bundle(path, model, {"test": True})
            loaded, metadata = load_model_bundle(path)
        self.assertEqual(metadata["test"], True)
        self.assertEqual(loaded.graph.input_access.to_dict(), graph.input_access.to_dict())
        self.assertTrue(torch.equal(loaded.input_access_mask, model.input_access_mask))


class ResourceAndControlTests(unittest.TestCase):
    def test_input_edges_counted(self):
        graph = initial_graph(); graph.input_access = build_input_access_spec("fixed_sparse", 6, 2, density=0.5, seed=1)
        model = NeutralGraphModel(graph)
        self.assertEqual(model.input_edge_count, int(np.asarray(graph.input_access.mask).sum()))

    def test_message_edges_counted(self):
        model = NeutralGraphModel(initial_graph())
        self.assertEqual(model.internal_message_edge_count, len(model.graph.edges))

    def test_parameter_count_matches_model(self):
        model = NeutralGraphModel(initial_graph())
        self.assertEqual(model.parameter_count(), sum(parameter.numel() for parameter in model.parameters()))

    def test_no_message_control_records_disabled_message_edges(self):
        task = DelayedRecallTask()
        result = train_graph(
            initial_graph(), task, [30000, 30001], [30200, 30201],
            run_seed=123, steps=1, training_scope="full_adamw", forward_options={"disable_messages": True}
        )
        self.assertEqual(result.resources.disabled_message_edges, len(result.graph.edges))
        self.assertEqual(result.resources.internal_message_edges, len(result.graph.edges))

    def test_no_message_control_zeroes_messages_but_not_raw_or_recurrence(self):
        graph = initial_graph(); graph.input_access = build_input_access_spec("fixed_sparse", 6, 2, density=0.5, seed=837005)
        model = NeutralGraphModel(graph)
        observations = torch.randn(4, 5, 6, generator=torch.Generator().manual_seed(7))
        lengths = torch.tensor([5, 5, 5, 5])
        with torch.no_grad():
            _, trace = model(observations, lengths, disable_messages=True, return_trace=True)
        self.assertEqual(float(torch.max(torch.abs(trace.messages))), 0.0)
        self.assertEqual(float(torch.max(torch.abs(trace.message_terms))), 0.0)
        self.assertGreater(float(torch.max(torch.abs(trace.input_terms))), 0.0)
        self.assertGreater(float(torch.max(torch.abs(trace.recurrent_terms[:, 1:]))), 0.0)


class ScientificGuardTests(unittest.TestCase):
    def test_v837_gate_hash_unchanged(self):
        self.assertEqual(gate_sha256(), GATE_HASH)
        self.assertEqual(v837_capacity_criterion_sha256(), CAPACITY_HASH)
        self.assertTrue(capacity_demonstrated(0.90, 0.85))

    def test_v837_historical_results_unchanged(self):
        for relative, expected in HISTORICAL_HASHES.items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_v836_archive_hash_unchanged(self):
        path = ROOT / "archive/preserved_artifacts/transmutor_experiments_v836plus/v836_results.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), HISTORICAL_HASHES[str(path.relative_to(ROOT)).replace('\\','/')])

    def test_fresh_audit_still_unused(self):
        status = json.loads((ROOT / "experiments/v837_primitive_invention/lineage_status.json").read_text(encoding="utf-8"))
        self.assertIs(status["fresh_audit_seed_range_used"], False)
        audit = json.loads((ROOT / "experiments/v837_primitive_invention/audit/audit_results.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["episodes_consumed"], 0)

    def test_primitive_promotion_blocked_during_representation_recovery(self):
        self.assertFalse(primitive_mining_allowed())
        with self.assertRaises(RuntimeError):
            assert_primitive_mining_allowed()

    def test_motif_pipeline_refuses_failed_substrate(self):
        with self.assertRaises(RuntimeError):
            begin_scientific_motif_pipeline()


if __name__ == "__main__":
    unittest.main()
