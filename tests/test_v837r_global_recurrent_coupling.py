from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.common.substrate import NeutralGraphModel
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.v837r.recurrent_coupling import (
    GloballyCoupledNeutralGraphModel,
    RecurrentCouplingSpec,
    coupling_actual_macs,
    coupling_core_macs,
    cross_block_mask,
    local_recurrent_macs,
)

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments" / "v837_primitive_invention" / "v837r"
BASE = ROOT / "experiments" / "v837_primitive_invention"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FROZEN = json.loads((HERE / "frozen_global_coupling_gate.json").read_text(encoding="utf-8"))


def git_blob_sha(relative: str) -> str:
    payload = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    return hashlib.sha256(payload).hexdigest()


class CouplingMaskTests(unittest.TestCase):
    def test_cross_block_mask_zeroes_local_blocks(self):
        mask = cross_block_mask()
        for i in range(10):
            block = mask[i * 4 : (i + 1) * 4, i * 4 : (i + 1) * 4]
            self.assertTrue(torch.equal(block, torch.zeros_like(block)))

    def test_cross_block_mask_preserves_cross_blocks(self):
        mask = cross_block_mask()
        self.assertTrue(torch.equal(mask[0:4, 4:8], torch.ones(4, 4)))
        self.assertEqual(int(mask.sum().item()), 1440)

    def test_dense_cross_block_shape_40x40(self):
        graph = high_capacity_generic_graph(0)
        model = GloballyCoupledNeutralGraphModel(graph, RecurrentCouplingSpec(mode="dense", initialization_seed=1))
        matrix = model.effective_global_matrix()
        self.assertEqual(tuple(matrix.shape), (40, 40))
        self.assertTrue(torch.equal(matrix[:4, :4], torch.zeros(4, 4)))


class LowRankTests(unittest.TestCase):
    def _model(self, rank: int):
        graph = high_capacity_generic_graph(0)
        return GloballyCoupledNeutralGraphModel(graph, RecurrentCouplingSpec(mode="low_rank", rank=rank, initialization_seed=77))

    def test_low_rank_shapes(self):
        for rank in (1, 2, 4, 8):
            model = self._model(rank)
            self.assertEqual(tuple(model.global_u.shape), (40, rank))
            self.assertEqual(tuple(model.global_v.shape), (40, rank))

    def test_rank1_factorization(self):
        self.assertEqual(self._model(1).coupling.rank, 1)

    def test_rank2_factorization(self):
        self.assertEqual(self._model(2).coupling.rank, 2)

    def test_rank4_factorization(self):
        self.assertEqual(self._model(4).coupling.rank, 4)

    def test_effective_matrix_respects_cross_block_mask(self):
        model = self._model(4)
        matrix = model.effective_global_matrix()
        for i in range(10):
            self.assertLessEqual(float(torch.max(torch.abs(matrix[i * 4 : (i + 1) * 4, i * 4 : (i + 1) * 4])).item()), 0.0)


class HistoricalCompatibilityTests(unittest.TestCase):
    def test_no_coupling_matches_historical_forward(self):
        graph = high_capacity_generic_graph(2)
        observations = torch.randn(3, 5, 6)
        lengths = torch.tensor([5, 4, 3])
        torch.manual_seed(12345)
        historical = NeutralGraphModel(graph, obs_dim=6, state_dim=4, message_dim=4, state_update_mode="direct", interaction_mode="none", state_modulation_mode="none")
        torch.manual_seed(12345)
        wrapped = GloballyCoupledNeutralGraphModel(graph, RecurrentCouplingSpec(mode="none"), obs_dim=6)
        with torch.no_grad():
            expected = historical(observations, lengths)
            actual = wrapped(observations, lengths)
        self.assertTrue(torch.equal(expected, actual))

    def test_no_coupling_parameter_count_matches_historical(self):
        graph = high_capacity_generic_graph(0)
        model = GloballyCoupledNeutralGraphModel(graph, RecurrentCouplingSpec(mode="none"))
        self.assertEqual(model.parameter_count(), 856)
        self.assertEqual(model.added_parameter_count(), 0)

    def test_historical_results_unchanged(self):
        self.assertEqual(git_blob_sha("experiments/v837_primitive_invention/v837q/results.json"), FROZEN["parent_result_git_blob_sha256"])
        self.assertEqual(git_blob_sha("experiments/v837_primitive_invention/v837p/results.json"), FROZEN["v837p_result_git_blob_sha256"])


class TimestepSemanticsTests(unittest.TestCase):
    def test_global_coupling_reads_previous_state_only(self):
        graph = high_capacity_generic_graph(0)
        model = GloballyCoupledNeutralGraphModel(graph, RecurrentCouplingSpec(mode="dense", initialization_seed=2))
        observations = torch.randn(2, 2, 6)
        with torch.no_grad():
            _, trace = model(observations, return_trace=True)
        self.assertTrue(torch.equal(trace.global_recurrent_terms[:, 0], torch.zeros_like(trace.global_recurrent_terms[:, 0])))

    def test_no_same_timestep_global_state_leakage(self):
        graph = high_capacity_generic_graph(0)
        model = GloballyCoupledNeutralGraphModel(graph, RecurrentCouplingSpec(mode="low_rank", rank=2, initialization_seed=3))
        a = torch.randn(1, 1, 6)
        b = a.clone()
        with torch.no_grad():
            _, ta = model(a, return_trace=True)
            _, tb = model(b, return_trace=True)
        self.assertTrue(torch.equal(ta.global_recurrent_terms[:, 0], tb.global_recurrent_terms[:, 0]))
        self.assertTrue(torch.equal(ta.global_recurrent_terms[:, 0], torch.zeros_like(ta.global_recurrent_terms[:, 0])))

    def test_messages_preserve_historical_order(self):
        graph = high_capacity_generic_graph(3)
        observations = torch.randn(2, 4, 6)
        torch.manual_seed(44)
        historical = NeutralGraphModel(graph, obs_dim=6, state_dim=4, message_dim=4)
        torch.manual_seed(44)
        wrapped = GloballyCoupledNeutralGraphModel(graph, RecurrentCouplingSpec(mode="none"))
        with torch.no_grad():
            _, htrace = historical(observations, return_trace=True)
            _, wtrace = wrapped(observations, return_trace=True)
        self.assertTrue(torch.equal(htrace.messages, wtrace.messages))


class ParameterControlTests(unittest.TestCase):
    def test_parameter_matched_control_declared(self):
        for primary in ("R1_rank1", "R2_rank2", "R3_rank4", "R4_rank8", "R5_dense_cross_block"):
            self.assertTrue(any(row["matches"] == primary for row in CONFIG["matched_controls"].values()))

    def test_parameter_count_delta_reported_and_exact(self):
        graph = high_capacity_generic_graph(0)
        pairs = [(1, 80), (2, 160), (4, 320), (8, 640)]
        for rank, delta in pairs:
            global_model = GloballyCoupledNeutralGraphModel(graph, RecurrentCouplingSpec(mode="low_rank", rank=rank, initialization_seed=1))
            local_model = GloballyCoupledNeutralGraphModel(graph, RecurrentCouplingSpec(mode="parameter_matched_local", matched_local_rank=rank, initialization_seed=1))
            self.assertEqual(global_model.added_parameter_count(), delta)
            self.assertEqual(local_model.added_parameter_count(), delta)
            self.assertEqual(global_model.parameter_count(), local_model.parameter_count())
        dense = GloballyCoupledNeutralGraphModel(graph, RecurrentCouplingSpec(mode="dense", initialization_seed=1))
        dense_control = GloballyCoupledNeutralGraphModel(graph, RecurrentCouplingSpec(mode="parameter_matched_local", matched_local_rank=20, initialization_seed=1))
        self.assertEqual(dense.added_parameter_count(), 1600)
        self.assertEqual(dense.parameter_count(), dense_control.parameter_count())

    def test_matched_local_control_has_no_cross_cell_state_access(self):
        graph = high_capacity_generic_graph(0)
        model = GloballyCoupledNeutralGraphModel(graph, RecurrentCouplingSpec(mode="parameter_matched_local", matched_local_rank=4, initialization_seed=8))
        states = [torch.randn(2, 4) for _ in range(10)]
        first = model._matched_local_terms(states)
        changed = [s.clone() for s in states]
        changed[7] = changed[7] + 10.0
        second = model._matched_local_terms(changed)
        for i in range(10):
            if i != 7:
                self.assertTrue(torch.equal(first[i], second[i]))


class EnergyAccountingTests(unittest.TestCase):
    def test_recurrent_mac_estimate_local(self):
        self.assertEqual(local_recurrent_macs(), 160)
        self.assertEqual(coupling_core_macs(RecurrentCouplingSpec(mode="none")), 0)

    def test_recurrent_mac_estimate_low_rank(self):
        for rank in (1, 2, 4, 8):
            spec = RecurrentCouplingSpec(mode="low_rank", rank=rank)
            self.assertEqual(coupling_core_macs(spec), 80 * rank)
            self.assertEqual(coupling_actual_macs(spec), 160 * rank)

    def test_recurrent_mac_estimate_dense(self):
        spec = RecurrentCouplingSpec(mode="dense")
        self.assertEqual(coupling_core_macs(spec), 1600)
        self.assertEqual(coupling_actual_macs(spec), 1440)


class ScientificBoundaryTests(unittest.TestCase):
    def test_v837r_dynamic_modulation_disabled(self):
        self.assertFalse(CONFIG["dynamic_modulation_allowed"])
        graph = high_capacity_generic_graph(0)
        model = GloballyCoupledNeutralGraphModel(graph, RecurrentCouplingSpec(mode="low_rank", rank=2))
        self.assertEqual(model.base.state_modulation_mode, "none")

    def test_v837r_interaction_mode_none(self):
        self.assertEqual(CONFIG["interaction_mode"], "none")

    def test_v837r_state_layout_local(self):
        self.assertEqual(CONFIG["state_layout"], "local_10x4")
        self.assertFalse(CONFIG["shared_state_allowed"])

    def test_v837r_total_state_dim_40(self):
        self.assertEqual(CONFIG["total_state_dim"], 40)

    def test_v837r_uses_4x_data(self):
        self.assertEqual(CONFIG["data_regime"], "4x_unique")
        self.assertEqual(CONFIG["training"]["train_episodes"], 512)
        self.assertEqual(CONFIG["training"]["validation_episodes"], 128)
        self.assertEqual(CONFIG["training"]["steps"], 192)

    def test_v837r_uses_existing_capacity_gate(self):
        self.assertEqual(CONFIG["historical_gate_hash"], "a1f587b268fec51c236c710ca5028933c1ba864064bb1275652f12bd13906867")
        self.assertEqual(CONFIG["capacity_criterion_hash"], "7178eed701ad50a298f172e867c73db47c03ecb28767de2add61feb34a61a3aa")

    def test_structural_search_locked(self):
        self.assertFalse(CONFIG["structural_search_allowed"])

    def test_primitive_mining_locked(self):
        self.assertFalse(CONFIG["primitive_mining_allowed"])

    def test_fresh_audit_unused(self):
        self.assertFalse(CONFIG["fresh_audit_allowed"])
        self.assertFalse(CONFIG["fresh_audit_consumed"])
        audit = json.loads((BASE / "audit" / "audit_results.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["episodes_consumed"], 0)

    def test_v838_not_started(self):
        self.assertFalse((BASE / "v838").exists())


if __name__ == "__main__":
    unittest.main()
