from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.common.gates import v837_capacity_criterion_sha256
from experiments.v837_primitive_invention.common.seeds import gate_sha256
from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.v837q.state_organization_models import (
    TOTAL_STATE_DIM,
    SharedStateNeutralGraphModel,
    StateLayoutSpec,
    build_fixed_state_projection,
    group_write_normalization,
    projection_norm_error,
    standard_state_layout,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "v837_primitive_invention"
HERE = BASE / "v837q"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
START_SHA = "f20316fd2aca8751b32226d653ac5b5f6976c7b3"
CONDITIONS = ("Q0_local_10x4", "Q1_group5_5x8", "Q2_group2_2x20", "Q3_shared_1x40")


class StateLayoutTests(unittest.TestCase):
    def _layout(self, name: str):
        return standard_state_layout(name, projection_seed=CONFIG["projection_seed"])

    def test_local_layout_10x4(self):
        spec = self._layout("Q0_local_10x4")
        self.assertEqual(spec.num_state_groups, 10)
        self.assertEqual(spec.group_dims, (4,) * 10)
        self.assertEqual(spec.group_assignment, tuple(range(10)))

    def test_group5_layout_5x8(self):
        spec = self._layout("Q1_group5_5x8")
        self.assertEqual(spec.num_state_groups, 5)
        self.assertEqual(spec.group_dims, (8,) * 5)
        self.assertEqual(spec.group_assignment, (0,0,1,1,2,2,3,3,4,4))

    def test_group2_layout_2x20(self):
        spec = self._layout("Q2_group2_2x20")
        self.assertEqual(spec.num_state_groups, 2)
        self.assertEqual(spec.group_dims, (20,20))
        self.assertEqual(spec.group_assignment, (0,0,0,0,0,1,1,1,1,1))

    def test_shared_layout_1x40(self):
        spec = self._layout("Q3_shared_1x40")
        self.assertEqual(spec.num_state_groups, 1)
        self.assertEqual(spec.group_dims, (40,))
        self.assertEqual(spec.group_assignment, (0,) * 10)

    def test_total_state_dim_always_40(self):
        for name in CONDITIONS:
            spec = self._layout(name)
            self.assertEqual(spec.total_state_dim, TOTAL_STATE_DIM)
            self.assertEqual(sum(spec.group_dims), 40)

    def test_group_assignments_deterministic_and_task_independent(self):
        for name in CONDITIONS:
            first = self._layout(name)
            second = self._layout(name)
            self.assertEqual(first, second)
            self.assertNotIn("task", first.to_dict())

    def test_layout_serialization_roundtrip(self):
        for name in CONDITIONS:
            spec = self._layout(name)
            self.assertEqual(StateLayoutSpec.from_dict(spec.to_dict()), spec)


class ProjectionTests(unittest.TestCase):
    def test_projection_deterministic(self):
        a = build_fixed_state_projection(40, 4, 3, 837040)
        b = build_fixed_state_projection(40, 4, 3, 837040)
        self.assertTrue(torch.equal(a, b))

    def test_projection_shape_and_norm_stable(self):
        for group_dim in (8, 20, 40):
            projection = build_fixed_state_projection(group_dim, 4, 1, 837040)
            self.assertEqual(tuple(projection.shape), (4, group_dim))
            self.assertLess(projection_norm_error(projection), 1e-5)

    def test_projection_seed_changes_projection(self):
        a = build_fixed_state_projection(40, 4, 2, 837040)
        b = build_fixed_state_projection(40, 4, 2, 837041)
        self.assertFalse(torch.equal(a, b))

    def test_write_normalization_predeclared_and_scale_matched(self):
        self.assertEqual(group_write_normalization(8, 2), 1.0)
        self.assertEqual(group_write_normalization(20, 5), 1.0)
        self.assertEqual(group_write_normalization(40, 10), 1.0)

    def test_projection_has_no_trainable_parameters(self):
        graph = high_capacity_generic_graph(0)
        model = SharedStateNeutralGraphModel(graph, standard_state_layout("Q3_shared_1x40", projection_seed=837040))
        projection_ids = {id(model.projection(i)) for i in range(10)}
        self.assertTrue(projection_ids.isdisjoint({id(parameter) for parameter in model.parameters()}))
        self.assertEqual(model.parameter_count(), model.base.parameter_count())


class HistoricalCompatibilityTests(unittest.TestCase):
    def test_local_state_layout_matches_historical_forward(self):
        torch.manual_seed(777)
        graph = high_capacity_generic_graph(0)
        model = SharedStateNeutralGraphModel(graph, standard_state_layout("Q0_local_10x4", projection_seed=837040))
        observations = torch.randn(3, 7, 6)
        lengths = torch.tensor([7, 5, 3])
        actual = model(observations, lengths)
        expected = model.base(observations, lengths)
        self.assertTrue(torch.equal(actual, expected))

    def test_historical_parameter_count_unchanged(self):
        graph = high_capacity_generic_graph(0)
        direct = NeutralGraphModel(graph, obs_dim=6, state_dim=4, message_dim=4)
        local = SharedStateNeutralGraphModel(graph, standard_state_layout("Q0_local_10x4", projection_seed=837040))
        self.assertEqual(direct.parameter_count(), 856)
        self.assertEqual(local.parameter_count(), 856)

    def test_historical_serialization_keys_unchanged(self):
        graph = high_capacity_generic_graph(0)
        direct = NeutralGraphModel(graph, obs_dim=6, state_dim=4, message_dim=4)
        local = SharedStateNeutralGraphModel(graph, standard_state_layout("Q0_local_10x4", projection_seed=837040))
        self.assertEqual(set(direct.state_dict()), set(local.base.state_dict()))

    def test_all_primary_conditions_have_same_trainable_parameter_count(self):
        graph = high_capacity_generic_graph(0)
        counts = []
        for name in CONDITIONS:
            model = SharedStateNeutralGraphModel(graph, standard_state_layout(name, projection_seed=837040))
            counts.append(model.parameter_count())
            self.assertEqual(model.readout_input_width, 40)
        self.assertEqual(counts, [856, 856, 856, 856])


class SharedStateSemanticsTests(unittest.TestCase):
    def _run(self, condition: str, disabled=None):
        torch.manual_seed(123)
        graph = high_capacity_generic_graph(0)
        model = SharedStateNeutralGraphModel(graph, standard_state_layout(condition, projection_seed=837040))
        observations = torch.randn(2, 4, 6)
        lengths = torch.tensor([4, 4])
        prediction, trace = model(observations, lengths, disabled_contribution_cells=disabled, return_trace=True)
        return model, prediction, trace

    def test_cells_in_same_group_read_same_underlying_state(self):
        model, _, trace = self._run("Q1_group5_5x8")
        previous_group0 = trace.states[:, 0, :8]
        for cell in (0, 1):
            expected = previous_group0 @ model.projection(cell).T
            self.assertTrue(torch.allclose(trace.local_views[:, 1, cell], expected, atol=1e-6, rtol=0.0))

    def test_cells_in_different_groups_read_different_state(self):
        model, _, trace = self._run("Q1_group5_5x8")
        group0 = trace.states[:, 0, :8]
        group1 = trace.states[:, 0, 8:16]
        self.assertFalse(torch.equal(group0, group1))
        expected0 = group0 @ model.projection(0).T
        expected2 = group1 @ model.projection(2).T
        self.assertTrue(torch.allclose(trace.local_views[:, 1, 0], expected0, atol=1e-6, rtol=0.0))
        self.assertTrue(torch.allclose(trace.local_views[:, 1, 2], expected2, atol=1e-6, rtol=0.0))

    def test_group_update_committed_simultaneously_and_no_same_step_write_leakage(self):
        _, _, normal = self._run("Q3_shared_1x40")
        _, _, disabled = self._run("Q3_shared_1x40", disabled={0})
        # Disabling a write contribution changes committed future state but not
        # any same-timestep local view, because every path reads the snapshot.
        self.assertTrue(torch.equal(normal.local_views[:, 0], disabled.local_views[:, 0]))
        self.assertFalse(torch.equal(normal.states[:, 0], disabled.states[:, 0]))

    def test_shared_state_has_single_underlying_tensor_width_40(self):
        _, _, trace = self._run("Q3_shared_1x40")
        self.assertEqual(trace.states.ndim, 3)
        self.assertEqual(trace.states.shape[-1], 40)

    def test_readout_input_width_is_40_all_conditions(self):
        graph = high_capacity_generic_graph(0)
        for name in CONDITIONS:
            model = SharedStateNeutralGraphModel(graph, standard_state_layout(name, projection_seed=837040))
            self.assertEqual(model.readout_input_width, 40)

    def test_output_interface_unchanged(self):
        model, prediction, _ = self._run("Q3_shared_1x40")
        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertEqual(model.message_dim, 4)

    def test_historical_message_edges_preserved_and_no_40d_message_bypass(self):
        model, _, trace = self._run("Q3_shared_1x40")
        self.assertEqual(model.internal_message_edge_count, 55)
        self.assertEqual(trace.messages.shape[-1], 4)
        self.assertEqual(trace.outputs.shape[-1], 4)

    def test_no_message_control_zeroes_only_messages(self):
        torch.manual_seed(123)
        graph = high_capacity_generic_graph(0)
        model = SharedStateNeutralGraphModel(graph, standard_state_layout("Q3_shared_1x40", projection_seed=837040))
        observations = torch.randn(2, 3, 6)
        lengths = torch.tensor([3, 3])
        _, trace = model(observations, lengths, disable_messages=True, return_trace=True)
        self.assertEqual(float(trace.messages.abs().max().item()), 0.0)
        self.assertGreater(float(trace.cell_candidates.abs().sum().item()), 0.0)


class ScientificBoundaryTests(unittest.TestCase):
    def test_v837q_uses_4x_data_and_existing_capacity_criterion(self):
        self.assertEqual(CONFIG["data_regime"], "4x_unique")
        self.assertEqual(CONFIG["training"]["train_episodes"], 512)
        self.assertEqual(CONFIG["training"]["development_seed_range"], [10000, 10511])
        self.assertEqual(CONFIG["training"]["validation_seed_range"], [20000, 20127])
        self.assertEqual(CONFIG["capacity_criterion_hash"], v837_capacity_criterion_sha256())
        self.assertEqual(CONFIG["historical_gate_hash"], gate_sha256())

    def test_v837q_does_not_enable_dynamic_modulation_or_downstream_science(self):
        self.assertFalse(CONFIG["dynamic_modulation_allowed"])
        self.assertFalse(CONFIG["structural_search_allowed"])
        self.assertFalse(CONFIG["primitive_mining_allowed"])
        self.assertFalse(CONFIG["fresh_audit_consumed"])

    def test_v837_through_v837p_unchanged(self):
        protected = ["archive", "registry"] + [f"experiments/v837_primitive_invention/v837{s}" for s in ("", "b", "c", "d", "g", "h", "j", "k", "l", "m", "n", "o", "p")]
        completed = subprocess.run(["git", "diff", "--quiet", START_SHA, "--", *protected], cwd=ROOT, check=False)
        self.assertEqual(completed.returncode, 0)

    def test_result_gate_when_present(self):
        path = HERE / "results.json"
        if not path.exists():
            self.skipTest("V837q result not generated yet")
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result["version"], "V837q")
        self.assertEqual(result["parent"], "V837p")
        self.assertFalse(result["fresh_audit_consumed"])
        self.assertFalse(result["primitive_mining_allowed"])
        self.assertFalse(result["structural_search_allowed"])


if __name__ == "__main__":
    unittest.main()
