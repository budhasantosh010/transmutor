from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel
from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.v837r.recurrent_coupling import GloballyCoupledNeutralGraphModel, RecurrentCouplingSpec

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "v837_primitive_invention"
HERE = BASE / "v837s"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


class V837sMechanismTests(unittest.TestCase):
    def test_local_dynamic_scalar_exactly_matches_v837p_mechanism(self):
        graph = high_capacity_generic_graph(0)
        seed = 12345
        torch.manual_seed(seed)
        reference = NeutralGraphModel(
            graph,
            obs_dim=6,
            state_dim=4,
            message_dim=4,
            state_update_mode="direct",
            interaction_mode="none",
            state_modulation_mode="dynamic_scalar_candidate",
        )
        torch.manual_seed(seed)
        wrapped = GloballyCoupledNeutralGraphModel(
            graph,
            RecurrentCouplingSpec(mode="none", initialization_seed=0),
            obs_dim=6,
            state_modulation_mode="dynamic_scalar_candidate",
        )
        observations = torch.randn(4, 7, 6)
        lengths = torch.tensor([7, 6, 5, 4])
        self.assertEqual(reference.parameter_count(), wrapped.parameter_count())
        self.assertTrue(torch.equal(reference(observations, lengths), wrapped(observations, lengths)))

    def test_rank4_no_modulation_preserves_v837r_semantics(self):
        graph = high_capacity_generic_graph(1)
        spec = RecurrentCouplingSpec(mode="low_rank", rank=4, cross_block_only=True, initialization_seed=991)
        torch.manual_seed(44)
        default_model = GloballyCoupledNeutralGraphModel(graph, spec, obs_dim=6)
        torch.manual_seed(44)
        explicit_model = GloballyCoupledNeutralGraphModel(graph, spec, obs_dim=6, state_modulation_mode="none")
        observations = torch.randn(3, 6, 6)
        lengths = torch.tensor([6, 5, 4])
        self.assertTrue(torch.equal(default_model(observations, lengths), explicit_model(observations, lengths)))

    def test_true_and_matched_dynamic_controls_have_exact_parameter_count(self):
        graph = high_capacity_generic_graph(0)
        spec = RecurrentCouplingSpec(mode="low_rank", rank=4, cross_block_only=True, initialization_seed=7)
        true_model = GloballyCoupledNeutralGraphModel(graph, spec, state_modulation_mode="dynamic_scalar_candidate")
        control_model = GloballyCoupledNeutralGraphModel(graph, spec, state_modulation_mode="dynamic_scalar_matched_additive")
        self.assertEqual(true_model.parameter_count(), 1326)
        self.assertEqual(control_model.parameter_count(), 1326)

    def test_dynamic_scalar_does_not_modify_global_coupling_operator(self):
        graph = high_capacity_generic_graph(2)
        spec = RecurrentCouplingSpec(mode="low_rank", rank=4, cross_block_only=True, initialization_seed=19)
        a = GloballyCoupledNeutralGraphModel(graph, spec, state_modulation_mode="none")
        b = GloballyCoupledNeutralGraphModel(graph, spec, state_modulation_mode="dynamic_scalar_candidate")
        self.assertTrue(torch.equal(a.effective_global_matrix(), b.effective_global_matrix()))
        snapshot = [torch.randn(5, 4) for _ in range(10)]
        terms_a = a._global_terms(snapshot)
        terms_b = b._global_terms(snapshot)
        for left, right in zip(terms_a, terms_b):
            self.assertTrue(torch.equal(left, right))

    def test_rank4_remains_cross_block_only(self):
        graph = high_capacity_generic_graph(0)
        spec = RecurrentCouplingSpec(mode="low_rank", rank=4, cross_block_only=True, initialization_seed=3)
        model = GloballyCoupledNeutralGraphModel(graph, spec, state_modulation_mode="dynamic_scalar_candidate")
        matrix = model.effective_global_matrix().detach()
        for cell in range(10):
            start = 4 * cell
            self.assertEqual(float(torch.sum(torch.abs(matrix[start:start+4, start:start+4])).item()), 0.0)

    def test_modulator_trace_is_dynamic(self):
        graph = high_capacity_generic_graph(0)
        spec = RecurrentCouplingSpec(mode="low_rank", rank=4, cross_block_only=True, initialization_seed=5)
        model = GloballyCoupledNeutralGraphModel(graph, spec, state_modulation_mode="dynamic_scalar_candidate")
        observations = torch.randn(3, 8, 6)
        _, trace = model(observations, torch.tensor([8, 7, 6]), return_trace=True)
        self.assertIsNotNone(trace.state_modulators)
        self.assertEqual(tuple(trace.state_modulators.shape), (3, 8, 10, 1))
        self.assertGreater(float(trace.state_modulators.var().item()), 0.0)


class V837sScientificBoundaryTests(unittest.TestCase):
    def test_parent_machine_guard_authorizes_v837s(self):
        decision = json.loads((BASE / "v837r" / "diagnostics" / "decision_state.json").read_text(encoding="utf-8"))
        self.assertTrue(decision["v837r_complete"])
        self.assertEqual(decision["diagnosis"], "GLOBAL_COUPLING_PARTIAL_BENEFIT")
        self.assertEqual(decision["best_condition"], "R3_rank4")
        self.assertTrue(decision["interaction_followup_allowed"])

    def test_factorial_matrix_and_control_complete(self):
        self.assertEqual(set(CONFIG["conditions"]), {
            "S0_local_no_modulation",
            "S1_local_dynamic_scalar",
            "S2_rank4_no_modulation",
            "S3_rank4_dynamic_scalar",
            "S3C_rank4_matched_dynamic_additive",
        })
        self.assertFalse(CONFIG["conditions"]["S0_local_no_modulation"]["global_coupling"])
        self.assertFalse(CONFIG["conditions"]["S1_local_dynamic_scalar"]["global_coupling"])
        self.assertTrue(CONFIG["conditions"]["S2_rank4_no_modulation"]["global_coupling"])
        self.assertTrue(CONFIG["conditions"]["S3_rank4_dynamic_scalar"]["global_coupling"])

    def test_all_conditions_use_same_4x_budget_and_paired_seeds(self):
        training = CONFIG["training"]
        self.assertEqual(training["steps"], 192)
        self.assertEqual(training["train_episodes"], 512)
        self.assertEqual(training["development_seed_range"], [10000, 10511])
        self.assertEqual(training["validation_episodes"], 128)
        self.assertEqual(training["validation_seed_range"], [20000, 20127])
        self.assertEqual(training["replicates"], 5)
        self.assertEqual(training["initialization_seed_namespace"], "v837j-primary-init")
        self.assertEqual(training["coupling_seed_condition_key"], "R3_rank4")

    def test_state_and_graph_boundary_unchanged(self):
        self.assertEqual(CONFIG["state_layout"], "local_10x4")
        self.assertEqual(CONFIG["total_state_dim"], 40)
        self.assertEqual(CONFIG["local_state_dim"], 4)
        self.assertEqual(CONFIG["message_dim"], 4)
        self.assertEqual(CONFIG["coupling_rank"], 4)

    def test_science_locks(self):
        self.assertFalse(CONFIG["fresh_audit_allowed"])
        self.assertFalse(CONFIG["structural_search_allowed"])
        self.assertFalse(CONFIG["primitive_mining_allowed"])
        self.assertFalse(CONFIG["task_family_label_allowed"])
        self.assertFalse(CONFIG["v838_allowed"])

    def test_result_gate_when_present(self):
        path = HERE / "results.json"
        if not path.exists():
            return
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result["version"], "V837s")
        self.assertEqual(result["parent"], "V837r")
        self.assertFalse(result["fresh_audit_consumed"])
        self.assertFalse(result["structural_search_allowed"])
        self.assertFalse(result["primitive_mining_allowed"])
        self.assertFalse(result["v838_started"])
        self.assertEqual(result["primitives_promoted"], 0)

    def test_v837t_starts_from_closed_v837s_frontier_without_reopening_coupling(self):
        decision = json.loads((HERE / "diagnostics" / "decision_state.json").read_text(encoding="utf-8"))
        self.assertTrue(decision["v837s_complete"])
        self.assertEqual(decision["diagnosis"], "GLOBAL_COUPLING_X_DYNAMIC_CONTROL_INSUFFICIENT")
        self.assertFalse(decision["coupling_compression_allowed"])
        t_config = json.loads((BASE / "v837t" / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(t_config["parent"], "V837s")
        self.assertEqual(t_config["question"], "Does successful recurrent computation require dimension-specific dynamic modulation, or are scalarized dynamic pathways sufficient?")
        self.assertFalse(t_config["structural_search_allowed"])
        self.assertFalse(t_config["primitive_mining_allowed"])

    def test_coupling_seed_matches_v837r_rank4(self):
        for replicate in range(5):
            expected = deterministic_int("v837r-coupling-init", "R3_rank4", replicate)
            actual = deterministic_int(CONFIG["training"]["coupling_seed_namespace"], CONFIG["training"]["coupling_seed_condition_key"], replicate)
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
