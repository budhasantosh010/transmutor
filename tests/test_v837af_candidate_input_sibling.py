from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.common.graph import InputAccessSpec
from experiments.v837_primitive_invention.common.seeds import deterministic_int
from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.v837y.candidate_interaction import ControlledCandidateInteractionModel
from experiments.v837_primitive_invention.v837af.candidate_input_factorization import CONDITIONS, CandidateInputFactorizationY3

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "experiments/v837_primitive_invention/v837af/config.json").read_text(encoding="utf-8"))


def model(condition: str, family: str = "conditional_routing", replicate: int = 0, graph=None):
    base_seed = deterministic_int(CONFIG["training"]["base_initialization_namespace"], family, replicate)
    coupling = deterministic_int(CONFIG["training"]["coupling_seed_namespace"], CONFIG["training"]["coupling_seed_condition"], replicate)
    projection = deterministic_int(CONFIG["training"]["projection_initialization_namespace"], family, replicate)
    torch.manual_seed(base_seed)
    return CandidateInputFactorizationY3(
        graph or high_capacity_generic_graph(replicate),
        condition=condition,
        coupling_initialization_seed=coupling,
        projection_seed=projection,
    )


class V837afCandidateInputSiblingTests(unittest.TestCase):
    def test_exact_condition_set(self):
        self.assertEqual(set(CONFIG["conditions"]), CONDITIONS)

    def test_y3_parent_hash_frozen(self):
        self.assertEqual(CONFIG["v837y_results_sha256"], "e4cb9004c7c68a25d5feb84cb6c25263459a0610a74350f08b3f1d3b999fa7e6")

    def test_af0_exact_counts(self):
        m = model("AF0_y3_parent")
        self.assertEqual(m.parameter_count(), 1223)
        self.assertEqual(m.recurrent_controller_macs, 846)
        self.assertEqual(m.projection_specific_macs, 0)

    def test_af1_shared_projection_counts(self):
        m = model("AF1_shared_candidate_input_factorization")
        self.assertEqual(m.parameter_count(), 1265)
        self.assertEqual(m.projection_parameter_count, 42)
        self.assertEqual(m.projection_specific_macs, 36)
        self.assertEqual(m.total_recurrent_controller_projection_macs, 882)

    def test_af1d_deshared_projection_counts(self):
        m = model("AF1D_deshared_candidate_input_factorization")
        self.assertEqual(m.parameter_count(), 1643)
        self.assertEqual(m.projection_parameter_count, 420)
        self.assertEqual(m.projection_specific_macs, 360)
        self.assertEqual(m.total_recurrent_controller_projection_macs, 1206)

    def test_af1f_has_no_runtime_projection_parameters(self):
        m = model("AF1F_folded_candidate_input_control")
        self.assertFalse(m.runtime_projection_active)
        self.assertEqual(m.projection_parameter_count, 0)
        self.assertEqual(m.projection_specific_macs, 0)
        self.assertEqual(m.parameter_count(), 1223)

    def test_shared_projection_computed_once_per_broadcast_timestep(self):
        m = model("AF1_shared_candidate_input_factorization")
        calls = []
        hook = m.shared_candidate_projection.register_forward_hook(lambda *args: calls.append(1))
        m(torch.randn(2, 7, 6), torch.tensor([7, 6]))
        hook.remove()
        self.assertEqual(len(calls), 7)

    def test_deshared_projection_executes_independently(self):
        m = model("AF1D_deshared_candidate_input_factorization")
        counts = [0] * 10
        hooks = []
        for i, layer in enumerate(m.cell_candidate_projections):
            hooks.append(layer.register_forward_hook(lambda *args, i=i: counts.__setitem__(i, counts[i] + 1)))
        m(torch.randn(2, 5, 6), torch.tensor([5, 4]))
        for hook in hooks: hook.remove()
        self.assertEqual(counts, [5] * 10)

    def test_visibility_mask_applied_before_projection(self):
        graph = high_capacity_generic_graph(0)
        mask = [[1.0 if (d + c) % 2 == 0 else 0.0 for c in range(10)] for d in range(6)]
        graph.input_access = InputAccessSpec(mode="fixed_sparse", observation_dim=6, num_cells=10, mask=mask, density=0.5, seed=1)
        m = model("AF1_shared_candidate_input_factorization", graph=graph)
        x = torch.arange(6, dtype=torch.float32).view(1, 6)
        visible = ControlledCandidateInteractionModel._visible_input(m, x, 0)
        expected = x * torch.tensor([row[0] for row in mask]).view(1, -1)
        self.assertTrue(torch.equal(visible, expected))
        projected = m._visible_input(x, 0)
        self.assertTrue(torch.allclose(projected, m.shared_candidate_projection(expected)))
        self.assertEqual(m.projection_specific_macs, 360)

    def test_af1_af0_effective_candidate_maps_match_at_step0(self):
        a = model("AF0_y3_parent")
        f = model("AF1_shared_candidate_input_factorization")
        for i in range(10):
            wa, ba = a.effective_candidate_input_map(i)
            wf, bf = f.effective_candidate_input_map(i)
            self.assertLessEqual(float((wa - wf).abs().max()), 1e-6)
            self.assertLessEqual(float((ba - bf).abs().max()), 1e-6)

    def _equivalence(self, other: str) -> dict[str, float]:
        # Replay the exact float32 initialization in float64 so the gate tests
        # algebraic implementation equivalence rather than matmul association noise.
        a = model("AF1_shared_candidate_input_factorization").double()
        b = model(other).double()
        x = torch.randn(4, 8, 6).double()
        lengths = torch.tensor([8, 7, 6, 5])
        with torch.no_grad():
            pa, ta = a(x, lengths, return_trace=True)
            pb, tb = b(x, lengths, return_trace=True)
            pva = []
            pvb = []
            for t in range(x.shape[1]):
                ca = []
                cb = []
                for cell in range(10):
                    va = ControlledCandidateInteractionModel._visible_input(a, x[:, t], cell)
                    vb = ControlledCandidateInteractionModel._visible_input(b, x[:, t], cell)
                    ca.append(a.diagnostic_projected_visible_input(va, cell))
                    cb.append(b.diagnostic_projected_visible_input(vb, cell))
                pva.append(torch.stack(ca, dim=1)); pvb.append(torch.stack(cb, dim=1))
        bias_a = torch.stack(list(a.base.cell_b), dim=0).view(1, 1, 10, 4)
        bias_b = torch.stack(list(b.base.cell_b), dim=0).view(1, 1, 10, 4)
        active = (torch.arange(x.shape[1]).view(1, -1) < lengths.view(-1, 1)).to(x.dtype).view(x.shape[0], x.shape[1], 1, 1)
        return {
            "projected_visible_input": float((torch.stack(pva, dim=1) - torch.stack(pvb, dim=1)).abs().max()),
            "candidate_input_term": float(((ta.input_terms + active * bias_a) - (tb.input_terms + active * bias_b)).abs().max()),
            "candidate": float((ta.candidate_states - tb.candidate_states).abs().max()),
            "global_gate": float((ta.global_gates - tb.global_gates).abs().max()),
            "next_state": float((ta.states - tb.states).abs().max()),
            "output": float((ta.outputs - tb.outputs).abs().max()),
            "prediction": float((pa - pb).abs().max()),
        }

    def test_af1_af1f_step0_equivalence(self):
        self.assertLessEqual(max(self._equivalence("AF1F_folded_candidate_input_control").values()), 1e-6)

    def test_af1_af1d_step0_equivalence(self):
        self.assertLessEqual(max(self._equivalence("AF1D_deshared_candidate_input_factorization").values()), 1e-6)

    def test_af1d_projections_bit_identical_at_initialization(self):
        m = model("AF1D_deshared_candidate_input_factorization")
        for layer in m.cell_candidate_projections[1:]:
            self.assertTrue(torch.equal(layer.weight, m.cell_candidate_projections[0].weight))
            self.assertTrue(torch.equal(layer.bias, m.cell_candidate_projections[0].bias))

    def test_non_candidate_components_paired(self):
        a = model("AF0_y3_parent")
        b = model("AF1_shared_candidate_input_factorization")
        for name in ("global_ws", "global_wx", "global_b", "global_u", "global_v"):
            self.assertTrue(torch.equal(getattr(a, name), getattr(b, name)))
        self.assertTrue(torch.equal(a.base.readout.weight, b.base.readout.weight))
        self.assertTrue(torch.equal(a.base.readout.bias, b.base.readout.bias))
        for ea, eb in zip(a.base.edge_weights, b.base.edge_weights): self.assertTrue(torch.equal(ea, eb))
        for i in range(10):
            self.assertTrue(torch.equal(a.base.cell_ws[i], b.base.cell_ws[i]))
            self.assertTrue(torch.equal(a.base.cell_wm[i], b.base.cell_wm[i]))
            self.assertTrue(torch.equal(a.base.cell_wo[i], b.base.cell_wo[i]))

    def test_rank4_branch_unchanged(self):
        m = model("AF1_shared_candidate_input_factorization")
        self.assertEqual(m.interaction_spec.candidate_coupling_mode, "rank4_cross_block")
        self.assertEqual(m.interaction_spec.coupling_rank, 4)
        self.assertEqual(m.candidate_branch_param_count, 320)

    def test_global_controller_unchanged(self):
        m = model("AF1_shared_candidate_input_factorization")
        self.assertTrue(m.global_scalar_control)
        self.assertEqual(m.controller_param_count, 47)
        self.assertEqual(m.controller_macs, 46)

    def test_message_graph_unchanged(self):
        m = model("AF1_shared_candidate_input_factorization")
        self.assertEqual(len(m.graph.edges), 55)
        self.assertEqual(sum(int(edge.recurrent) for edge in m.graph.edges), 10)

    def test_frozen_data_and_science_locks(self):
        tr = CONFIG["training"]
        self.assertEqual(tr["development_seed_range"], [10000, 10511])
        self.assertEqual(tr["validation_seed_range"], [20000, 20127])
        self.assertEqual(CONFIG["unique_seed_defined_episodes"], 3200)
        self.assertFalse(CONFIG["fresh_audit_consumed"])
        self.assertFalse(CONFIG["structural_search_allowed"])
        self.assertFalse(CONFIG["primitive_mining_allowed"])
        self.assertEqual(CONFIG["primitives_promoted"], 0)
        self.assertTrue(CONFIG["v837ae_absent"])
        self.assertFalse(CONFIG["v838_started"])


if __name__ == "__main__":
    unittest.main()
