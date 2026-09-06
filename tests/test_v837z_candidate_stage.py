from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.v837y.candidate_interaction import CandidateInteractionSpec, ControlledCandidateInteractionModel
from experiments.v837_primitive_invention.v837z.candidate_stage import CandidateStageModel, historical_candidate_depths, synchronous_candidate_depths

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "experiments/v837_primitive_invention/v837z/config.json").read_text())


def coupling_seed(rep: int = 0) -> int:
    tr = CONFIG["training"]
    return deterministic_int(tr["coupling_seed_namespace"], tr["coupling_seed_condition"], rep)


class V837zCandidateStageTests(unittest.TestCase):
    def setUp(self):
        self.graph = high_capacity_generic_graph(0)
        self.seed = coupling_seed(0)

    def test_z0_matches_selected_v837y_parent(self):
        torch.manual_seed(123)
        parent = ControlledCandidateInteractionModel(
            self.graph,
            spec=CandidateInteractionSpec(True, "rank4_cross_block", coupling_rank=4),
            coupling_initialization_seed=self.seed,
        )
        torch.manual_seed(123)
        z0 = CandidateStageModel(self.graph, stage_mode="historical_mixed", coupling_initialization_seed=self.seed)
        z0.load_state_dict(parent.state_dict())
        x = torch.randn(3, 6, 6)
        lengths = torch.tensor([6, 5, 4])
        yp, tp = parent(x, lengths, return_trace=True)
        yz, tz = z0(x, lengths, return_trace=True)
        self.assertTrue(torch.equal(yp, yz))
        self.assertTrue(torch.equal(tp.states, tz.states))
        self.assertTrue(torch.equal(tp.candidate_states, tz.candidate_states))
        self.assertTrue(torch.equal(tp.messages, tz.messages))
        self.assertTrue(torch.equal(tp.global_gates, tz.global_gates))

    def test_z1_graph_edges_unchanged(self):
        z1 = CandidateStageModel(self.graph, stage_mode="fully_synchronous", coupling_initialization_seed=self.seed)
        self.assertEqual(z1.graph.to_dict(), self.graph.to_dict())
        self.assertEqual(len(z1.graph.edges), len(self.graph.edges))

    def test_z1_message_dimension_unchanged(self):
        z1 = CandidateStageModel(self.graph, stage_mode="fully_synchronous", coupling_initialization_seed=self.seed)
        self.assertEqual(z1.message_dim, 4)
        self.assertEqual(z1.state_dim, 4)

    def test_z1_candidate_controller_coupling_unchanged(self):
        z0 = CandidateStageModel(self.graph, stage_mode="historical_mixed", coupling_initialization_seed=self.seed)
        z1 = CandidateStageModel(self.graph, stage_mode="fully_synchronous", coupling_initialization_seed=self.seed)
        z1.load_state_dict(z0.state_dict())
        self.assertEqual(z0.parameter_count(), z1.parameter_count())
        self.assertEqual(z0.controller_param_count, z1.controller_param_count)
        self.assertEqual(z0.candidate_branch_param_count, z1.candidate_branch_param_count)
        self.assertTrue(torch.equal(z0.effective_global_matrix(), z1.effective_global_matrix()))
        self.assertEqual(z0.recurrent_controller_macs, z1.recurrent_controller_macs)

    def test_z1_all_messages_use_previous_outputs_leakage(self):
        edge = next(e for e in self.graph.edges if not e.recurrent and e.src < e.dst)
        src, dst = edge.src, edge.dst
        torch.manual_seed(7)
        a = CandidateStageModel(self.graph, stage_mode="fully_synchronous", coupling_initialization_seed=self.seed)
        b = CandidateStageModel(self.graph, stage_mode="fully_synchronous", coupling_initialization_seed=self.seed)
        b.load_state_dict(a.state_dict())
        with torch.no_grad():
            b.base.cell_wo[src].add_(0.75)
        x = torch.randn(2, 3, 6)
        _, ta = a(x, return_trace=True)
        _, tb = b(x, return_trace=True)
        self.assertTrue(torch.equal(ta.candidate_states[:, 0, dst], tb.candidate_states[:, 0, dst]))
        self.assertFalse(torch.allclose(ta.candidate_states[:, 1, dst], tb.candidate_states[:, 1, dst]))

    def test_z0_same_timestep_source_can_leak(self):
        edge = next(e for e in self.graph.edges if not e.recurrent and e.src < e.dst)
        src, dst = edge.src, edge.dst
        torch.manual_seed(9)
        a = CandidateStageModel(self.graph, stage_mode="historical_mixed", coupling_initialization_seed=self.seed)
        b = CandidateStageModel(self.graph, stage_mode="historical_mixed", coupling_initialization_seed=self.seed)
        b.load_state_dict(a.state_dict())
        with torch.no_grad():
            b.base.cell_wo[src].add_(0.75)
        x = torch.randn(2, 2, 6)
        _, ta = a(x, return_trace=True)
        _, tb = b(x, return_trace=True)
        self.assertFalse(torch.allclose(ta.candidate_states[:, 0, dst], tb.candidate_states[:, 0, dst]))

    def test_candidate_stage_depth(self):
        z0 = historical_candidate_depths(self.graph)
        z1 = synchronous_candidate_depths(self.graph)
        self.assertEqual(len(z0), 10)
        self.assertEqual(z1, [1] * 10)
        self.assertGreaterEqual(max(z0), 1)

    def test_science_locks(self):
        tr = CONFIG["training"]
        self.assertEqual((tr["steps"], tr["train_episodes"], tr["validation_episodes"], tr["replicates"]), (192, 512, 128, 5))
        self.assertEqual(CONFIG["unique_seed_defined_episodes"], 3200)
        self.assertEqual(CONFIG["state_layout"], "local_10x4")
        self.assertEqual(CONFIG["selected_parent"], "Y3_global_control_rank4_candidate")
        self.assertFalse(CONFIG["fresh_audit_consumed"])
        self.assertFalse(CONFIG["structural_search_allowed"])
        self.assertFalse(CONFIG["primitive_mining_allowed"])
        self.assertFalse(CONFIG["v838_started"])


if __name__ == "__main__":
    unittest.main()
