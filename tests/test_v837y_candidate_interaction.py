from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.v837_primitive_invention.failures.run_blocker_diagnostic import high_capacity_generic_graph
from experiments.v837_primitive_invention.v837r.recurrent_coupling import GloballyCoupledNeutralGraphModel, RecurrentCouplingSpec, cross_block_mask
from experiments.v837_primitive_invention.v837x.global_scalar_control import GlobalScalarNeutralModel
from experiments.v837_primitive_invention.v837y.candidate_interaction import CandidateInteractionSpec, ControlledCandidateInteractionModel

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "experiments/v837_primitive_invention/v837y/config.json").read_text())


def model(condition: str, replicate: int = 0):
    row = CONFIG["conditions"][condition]
    mode = row["candidate_coupling"]
    spec = CandidateInteractionSpec(
        global_scalar_control=bool(row["global_scalar_control"]),
        candidate_coupling_mode=mode,
        coupling_rank=4 if mode == "rank4_cross_block" else None,
        matched_local_rank=4 if mode == "rank4_matched_local" else None,
    )
    from experiments.v837_primitive_invention.common.seeds import deterministic_int
    seed = deterministic_int(CONFIG["training"]["coupling_seed_namespace"], CONFIG["training"]["coupling_seed_condition"], replicate)
    return ControlledCandidateInteractionModel(high_capacity_generic_graph(replicate), spec=spec, coupling_initialization_seed=seed)


class V837yTests(unittest.TestCase):
    def test_y0_matches_historical_neutral(self):
        torch.manual_seed(11); y = model("Y0_historical")
        torch.manual_seed(11); base = GloballyCoupledNeutralGraphModel(high_capacity_generic_graph(0), RecurrentCouplingSpec(mode="none", initialization_seed=0))
        base.load_state_dict(y.state_dict(), strict=True)
        x = torch.randn(3, 7, 6); lengths = torch.tensor([7, 6, 5])
        py, ty = y(x, lengths, return_trace=True); pb, tb = base(x, lengths, return_trace=True)
        self.assertTrue(torch.equal(py, pb)); self.assertTrue(torch.equal(ty.states, tb.states)); self.assertTrue(torch.equal(ty.candidate_states, tb.candidate_states))

    def test_y1_matches_v837x_x2_forward(self):
        torch.manual_seed(13); x2 = GlobalScalarNeutralModel(high_capacity_generic_graph(0), condition="X2_global_scalar_carry", authorized_mode="JOINT_INPUT_STATE_GLOBAL_SCALAR")
        torch.manual_seed(99); y = model("Y1_global_control")
        base_state = {k: v for k, v in x2.state_dict().items() if not k.startswith("global_")}
        y.base.load_state_dict(base_state, strict=True)
        with torch.no_grad(): y.global_ws.copy_(x2.global_ws); y.global_wx.copy_(x2.global_wx); y.global_b.copy_(x2.global_b)
        obs = torch.randn(4, 8, 6); lengths = torch.tensor([8, 7, 6, 5])
        px, tx = x2(obs, lengths, return_trace=True); py, ty = y(obs, lengths, return_trace=True)
        self.assertTrue(torch.equal(px, py)); self.assertTrue(torch.equal(tx.states, ty.states)); self.assertTrue(torch.equal(tx.candidate_states, ty.candidate_states)); self.assertTrue(torch.equal(tx.state_modulators, ty.state_modulators))

    def test_y1_matches_v837x_x2_parameter_count(self):
        self.assertEqual(model("Y1_global_control").parameter_count(), 903)
        self.assertEqual(model("Y1_global_control").controller_param_count, 47)

    def test_y1_matches_v837x_controller_semantics(self):
        m = model("Y1_global_control"); prev = [torch.randn(2,4) for _ in range(10)]; x = torch.randn(2,6)
        expected = torch.sigmoid(torch.sum(torch.cat(prev,1)*m.global_ws.view(1,-1),1,keepdim=True)+torch.sum(x*m.global_wx.view(1,-1),1,keepdim=True)+m.global_b)
        self.assertTrue(torch.equal(m._global_gate(prev,x), expected))

    def test_y2_matches_v837r_rank4_forward(self):
        from experiments.v837_primitive_invention.common.seeds import deterministic_int
        seed = deterministic_int("v837r-coupling-init", "R3_rank4", 0)
        torch.manual_seed(7); r = GloballyCoupledNeutralGraphModel(high_capacity_generic_graph(0), RecurrentCouplingSpec(mode="low_rank",rank=4,cross_block_only=True,scaling=1.0,initialization_seed=seed))
        torch.manual_seed(55); y = model("Y2_rank4_candidate"); y.load_state_dict(r.state_dict(), strict=True)
        obs=torch.randn(3,6,6); lengths=torch.tensor([6,5,4]); pr,tr=r(obs,lengths,return_trace=True); py,ty=y(obs,lengths,return_trace=True)
        self.assertTrue(torch.equal(pr,py)); self.assertTrue(torch.equal(tr.states,ty.states)); self.assertTrue(torch.equal(tr.global_recurrent_terms,ty.global_recurrent_terms))

    def test_y2_matches_v837r_rank4_coupling_matrix(self):
        m=model("Y2_rank4_candidate"); self.assertEqual(m.coupling.rank,4); self.assertTrue(torch.equal(m.cross_block_mask,cross_block_mask())); self.assertTrue(torch.allclose(m.effective_global_matrix(),(m.global_u@m.global_v.T)*m.cross_block_mask,rtol=0,atol=0))

    def test_y2_matches_v837r_rank4_parameter_count(self): self.assertEqual(model("Y2_rank4_candidate").parameter_count(),1176)
    def test_y3_uses_exact_v837x_global_controller(self): self.assertEqual(model("Y3_global_control_rank4_candidate").controller_param_count,47)
    def test_y3_uses_exact_v837r_rank4_candidate_branch(self):
        m=model("Y3_global_control_rank4_candidate"); self.assertEqual(m.coupling.mode,"low_rank"); self.assertEqual(m.coupling.rank,4); self.assertEqual(m.candidate_branch_param_count,320)
    def test_y3_controller_and_candidate_branch_read_same_previous_state_snapshot(self):
        m=model("Y3_global_control_rank4_candidate"); obs=torch.randn(2,2,6); _,tr=m(obs,return_trace=True); self.assertEqual(tuple(tr.global_gates.shape),(2,2,1)); self.assertEqual(tuple(tr.global_recurrent_terms.shape),(2,2,10,4))
    def test_y3_rank4_term_inside_candidate_preactivation(self):
        m=model("Y3_global_control_rank4_candidate"); obs=torch.randn(2,3,6); _,tr=m(obs,return_trace=True); pre=tr.recurrent_terms+tr.global_recurrent_terms+tr.matched_local_terms+tr.message_terms+tr.input_terms+m.base.cell_b[0].new_zeros(1)
        # Bias differs per cell; recompute explicitly.
        b=torch.stack([p for p in m.base.cell_b],0).view(1,1,10,4); self.assertTrue(torch.allclose(tr.candidate_states,torch.tanh(pre+b),atol=1e-6,rtol=1e-6))
    def test_y3_global_scalar_applied_after_candidate(self):
        m=model("Y3_global_control_rank4_candidate"); obs=torch.randn(2,2,6); _,tr=m(obs,return_trace=True); g=tr.global_gates[:,0,:].unsqueeze(1); expected=(1-g)*tr.candidate_states[:,0,:,:]; self.assertTrue(torch.allclose(tr.states[:,0,:,:],expected,atol=1e-6,rtol=1e-6))
    def test_y3_no_vector_gate(self): self.assertEqual(tuple(model("Y3_global_control_rank4_candidate").global_ws.shape),(40,))
    def test_y3_no_global_controller_message_input(self): self.assertFalse(CONFIG["global_controller_message_input"])
    def test_y3c_matches_y3_parameter_count(self): self.assertEqual(model("Y3C_global_control_matched_local").parameter_count(),model("Y3_global_control_rank4_candidate").parameter_count())
    def test_y3c_extra_candidate_branch_cannot_read_other_cell_states(self):
        m=model("Y3C_global_control_matched_local"); states=[torch.randn(2,4) for _ in range(10)]; a=m._matched_local_terms(states); states2=[s.clone() for s in states]; states2[0]+=10; b=m._matched_local_terms(states2); self.assertTrue(torch.equal(a[1],b[1])); self.assertFalse(torch.equal(a[0],b[0]))
    def test_y3c_global_controller_identical_to_y3(self): self.assertEqual(model("Y3C_global_control_matched_local").controller_param_count,model("Y3_global_control_rank4_candidate").controller_param_count)
    def test_y3c_same_training_configuration(self): self.assertEqual(CONFIG["training"]["steps"],192)
    def test_v837y_uses_4x_unique_data(self): self.assertEqual(CONFIG["data_regime"],"4x_unique")
    def test_v837y_unique_episode_count_3200(self): self.assertEqual(CONFIG["unique_seed_defined_episodes"],3200)
    def test_v837y_optimizer_steps_192(self): self.assertEqual(CONFIG["training"]["steps"],192)
    def test_v837y_five_replicates(self): self.assertEqual(CONFIG["training"]["replicates"],5)
    def test_input_projection_unchanged(self): self.assertEqual(CONFIG["input_projection"],"historical_per_cell")
    def test_message_schedule_unchanged(self): self.assertEqual(CONFIG["message_schedule"],"historical_mixed_same_step_and_recurrent")
    def test_state_layout_local_10x4(self): self.assertEqual(CONFIG["state_layout"],"local_10x4")
    def test_total_state_dim_40(self): self.assertEqual(CONFIG["total_state_dim"],40)
    def test_no_shared_state(self): self.assertFalse(CONFIG["shared_state"])
    def test_no_vector_modulation(self): self.assertFalse(CONFIG["vector_gates"])
    def test_no_extra_coupling_rank(self): self.assertEqual(CONFIG["candidate_coupling_rank"],4)
    def test_no_dense_coupling(self): self.assertNotIn("dense",[v["candidate_coupling"] for v in CONFIG["conditions"].values()])
    def test_no_structural_search(self): self.assertFalse(CONFIG["structural_search"])
    def test_no_primitive_mining(self): self.assertFalse(CONFIG["primitive_mining"])
    def test_fresh_audit_unused(self): self.assertFalse(CONFIG["fresh_audit_consumed"])
    def test_v838_not_started(self): self.assertFalse(CONFIG["v838_started"])


if __name__ == "__main__": unittest.main()
